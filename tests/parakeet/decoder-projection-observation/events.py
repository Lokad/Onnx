"""Reconcile complete diagnostic calls and exported events; never create a score."""
import base64
from bisect import bisect_left
from collections import Counter, defaultdict
import math

CUSTOM = 'Lokad-Parakeet-MatMul-Diagnostic'
CLR = 'Microsoft-Windows-DotNETRuntime'
SAMPLE = 'Microsoft-DotNETCore-SampleProfiler'


def calls(value):
    assert value['passed'] and value['diagnostic_only']
    assert value['protocol'] == 'parakeet-decoder-projection-observation-v1'
    assert value['mode'] in ['control', 'trace'] and value['flags'] == {}
    assert (value['controls'], value['calls'], value['warmups'], value['observations']) == (3, 1280, 256, 1024)
    assert len(value['clocks']) == 1280 and type(value['frequency']) is int and value['frequency'] > 0
    for field in ['every_output_exact', 'inputs_unchanged', 'held_outputs_unchanged',
                  'initializers_unchanged', 'ordinary_outputs_only']:
        assert value[field], field
    assert value['counter_scope'] == 'Complete decoder execution; never per-node allocation or copy attribution'
    previous = 0
    for index, row in enumerate(value['clocks']):
        assert row['iteration'] == index and row['warmup'] == (index < 256)
        assert previous < row['marker'] <= row['start'] < row['stop']
        assert row['ticks'] == row['stop'] - row['start']
        previous = row['stop']
        for field in ['decoder_allocated_bytes', 'decoder_copy_bytes', 'decoder_scratch_bytes']:
            assert type(row[field]) is int and row[field] >= 0, field
        assert len(row['decoder_gc_collections']) == 3
        assert all(type(n) is int and n >= 0 for n in row['decoder_gc_collections'])
    return dict(calls=1280, warmups=256, observations=1024,
        decoder_allocated_bytes=sum(r['decoder_allocated_bytes'] for r in value['clocks']),
        decoder_copy_bytes=sum(r['decoder_copy_bytes'] for r in value['clocks']),
        decoder_scratch_bytes=sum(r['decoder_scratch_bytes'] for r in value['clocks']),
        decoder_gc_calls=sum(any(r['decoder_gc_collections']) for r in value['clocks']),
        diagnostic_only=True, per_node_counters=False)


def reconcile(value, events, summary):
    result = calls(value)
    assert value['mode'] == 'trace'
    assert summary['complete'] and summary['protocol'] == 'all-event-records-v1'
    assert summary['lost'] == 0 and summary['runtime'] == '10.0.8'
    assert len(events) == summary['recorded'] == sum(summary['allCounts'].values())
    assert dict(Counter(e['provider'] + ':' + e['name'] for e in events)) == summary['allCounts']
    assert sum(e['provider'] == CLR for e in events) == summary['clr_events'] > 0
    assert any(e['provider'] == SAMPLE for e in events), 'No sample-profiler events'
    for index, event in enumerate(events):
        assert event['index'] == index and event['pointerSize'] == 8
        assert len(base64.b64decode(event['rawBase64'], validate=True)) == event['rawLength']
        assert math.isfinite(event['ms']) and event['ms'] >= 0
        assert event['pid'] == value['pid'], 'Mixed process trace'
    markers = [e for e in events if e['provider'] == CUSTOM]
    assert len(markers) == 2560
    assert all(e['thread'] == value['native_thread'] for e in markers)
    assert all(a['ms'] <= b['ms'] for a, b in zip(markers, markers[1:]))
    intervals = []
    for index, row in enumerate(value['clocks']):
        pair = markers[2 * index:2 * index + 2]
        for event, identifier, counter in zip(pair, [1, 2], [row['marker'], row['stop']], strict=True):
            assert event['id'] == identifier
            assert {k: int(v) for k, v in event['payload'].items()} == dict(fixture=0, iteration=index, counter=counter)
        intervals.append(dict(iteration=index, warmup=row['warmup'], begin_ms=pair[0]['ms'], end_ms=pair[1]['ms'],
                              execute_ms=row['ticks'] * 1000 / value['frequency']))
    # Markers include their small recording overhead. Keep their intervals intact;
    # do not subtract it or promote the Execute clocks to a release measurement.
    return dict(**result, markers=len(markers), records=len(events), intervals=intervals)


def classify(stack, frames):
    names = '\n'.join(frames[index]['name'] for index in stack)
    if 'mm_m1_kblocked' in names:
        return 'one-row-row-major'
    if 'PackedFinalRowKernel' in names:
        return 'prepared-final-row'
    if 'PackPanelsB' in names:
        return 'packing'
    if 'mm_unsafe_vectorized_intrinsics' in names:
        return 'other-matmul-kernel'
    return 'other' if stack else 'empty'


def stacks(document, intervals, thread):
    """Conserve every exported interval, including outside calls and empty stacks.

    Speedscope reconstructs sampled stacks; durations are estimates, not direct
    kernel timers or sample counts. Frame names cannot resolve every JIT version.
    """
    assert document['$schema'] == 'https://www.speedscope.app/file-format-schema.json'
    frames = document['shared']['frames']
    assert frames and all(isinstance(f['name'], str) for f in frames)
    assert intervals and all(0 <= c['begin_ms'] <= c['end_ms'] for c in intervals)
    assert all(a['end_ms'] <= b['begin_ms'] for a, b in zip(intervals, intervals[1:]))
    starts = [c['begin_ms'] for c in intervals]
    ends = [c['end_ms'] for c in intervals]
    inside, outside, leaves = defaultdict(float), defaultdict(float), defaultdict(float)
    profiles, rounding = [], []
    assert len({p['name'] for p in document['profiles']}) == len(document['profiles'])
    assert sum(p['name'] == f'Thread ({thread})' for p in document['profiles']) == 1
    for profile in document['profiles']:
        assert profile['type'] == 'evented' and profile['unit'] == 'milliseconds'
        stack, previous = [], profile['startValue']
        assert math.isfinite(previous) and 0 <= previous <= profile['endValue']
        worker = profile['name'] == f'Thread ({thread})'
        for event in profile['events']:
            at = event['at']
            assert math.isfinite(at) and profile['startValue'] - .001 <= at <= profile['endValue'] + .001
            if at < previous:
                assert previous - at <= .001
                rounding.append(previous - at)
                at = previous
            frame = event['frame']
            assert type(frame) is int and 0 <= frame < len(frames)
            category = classify(stack, frames)
            used = 0.
            if worker and at > previous:
                index = max(0, bisect_left(ends, previous))
                while index < len(intervals) and starts[index] < at:
                    overlap = max(0., min(at, ends[index]) - max(previous, starts[index]))
                    if overlap:
                        inside[(index, category)] += overlap
                        leaf = frames[stack[-1]]['name'] if stack else '<empty>'
                        leaves[(index, category, leaf)] += overlap
                        used += overlap
                    index += 1
            outside[(profile['name'], category)] += at - previous - used
            if event['type'] == 'O':
                stack.append(frame)
            else:
                assert event['type'] == 'C' and stack and stack.pop() == frame
            previous = at
        assert not stack and abs(previous - profile['endValue']) <= .001
        profiles.append(dict(name=profile['name'], start=profile['startValue'], end=previous,
                             milliseconds=previous - profile['startValue']))
    assert abs(sum(inside.values()) + sum(outside.values()) - sum(p['milliseconds'] for p in profiles)) < 1e-6
    assert abs(sum(inside.values()) - sum(leaves.values())) < 1e-6
    return dict(profiles=profiles, rounding_adjustments_ms=rounding,
        inside=[dict(iteration=i, category=c, estimated_thread_ms=v) for (i, c), v in sorted(inside.items())],
        outside=[dict(thread=t, category=c, estimated_thread_ms=v) for (t, c), v in sorted(outside.items())],
        leaves=[dict(iteration=i, category=c, frame=f, estimated_thread_ms=v) for (i, c, f), v in sorted(leaves.items())],
        scope='Reconstructed sampled-thread intervals; no kernel timer, sample count, or per-call JIT-version proof.',
        per_call_jit_version_proved=False, diagnostic_only=True)


def method_records(events):
    """Retain method load/rundown identities without guessing a tier per sample."""
    # The retained exporter does not give every rundown event a typed name.
    # Keep that provider in full, including its unparsed raw records.
    return [e for e in events if e['provider'] == CLR + 'Rundown'
            or (e['provider'] == CLR and e['name'].startswith('Method/'))]
