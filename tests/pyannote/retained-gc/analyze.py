"""Independently attribute runtime suspension to complete saved request intervals."""
import collections
import math
from common import *

STAGES = ['GC/SuspendEEStart', 'GC/SuspendEEStop', 'GC/RestartEEStart', 'GC/RestartEEStop']
GC_REASONS = {'SuspendForGC', 'SuspendForGCPrep'}


def pair_pauses(events):
    pending, result = {}, []
    previous = -math.inf
    for event in events:
        assert math.isfinite(event['ms']) and event['ms'] >= previous, 'Unordered events'
        previous = event['ms']
        if event['name'] not in STAGES:
            continue
        key = (event['pid'], event['payload']['ClrInstanceID'])
        step = STAGES.index(event['name'])
        if step == 0:
            assert key not in pending, 'Overlapping suspension'
            pending[key] = []
        assert key in pending and len(pending[key]) == step, 'Missing or reordered suspension event'
        pending[key].append(event)
        if step == 3:
            start, stopped, restart, end = pending.pop(key)
            result.append(dict(pid=key[0], clr_instance=key[1], reason=start['payload']['Reason'],
                count=int(start['payload']['Count']), start_ms=start['ms'], suspended_ms=stopped['ms'],
                restart_ms=restart['ms'], end_ms=end['ms'], indices=[r['index'] for r in [start, stopped, restart, end]]))
    assert not pending, 'Unclosed suspension'
    return result


def pair_requests(events):
    pending, result, seen = {}, [], set()
    for event in events:
        if event['provider'] != 'Lokad-Pyannote-Diagnostic':
            continue
        assert event['name'] == 'Boundary' and event['id'] == 1
        value = event['payload']
        key = (event['pid'], event['thread'])
        identity = (value['name'], int(value['pass']))
        if value['phase'] == 'begin':
            assert key not in pending and identity not in seen, 'Overlapping or repeated request'
            pending[key] = event
            seen.add(identity)
        else:
            assert value['phase'] == 'end' and key in pending, 'Missing request begin'
            first = pending.pop(key)
            assert identity == (first['payload']['name'], int(first['payload']['pass'])) and event['ms'] > first['ms']
            result.append(dict(name=identity[0], pass_index=identity[1], pid=key[0], thread=key[1],
                start_ms=first['ms'], end_ms=event['ms'], indices=[first['index'], event['index']]))
    assert not pending, 'Missing request end'
    return result


def union_length(intervals, start, end):
    assert math.isfinite(start) and math.isfinite(end) and start <= end
    clipped = []
    for a, b in intervals:
        assert math.isfinite(a) and math.isfinite(b) and a <= b
        a, b = max(start, a), min(end, b)
        if b > a:
            clipped.append((a, b))
    total, current = 0., None
    for a, b in sorted(clipped):
        if current is None:
            current = (a, b)
        elif a <= current[1]:
            current = (current[0], max(current[1], b))
        else:
            total += current[1] - current[0]
            current = (a, b)
    if current is not None:
        total += current[1] - current[0]
    assert 0 <= total <= end - start + 1e-8
    return total


def collections_from(events):
    pending, result, seen = {}, [], set()
    for event in events:
        if event['name'] not in ['GC/Start', 'GC/Stop']:
            continue
        value = event['payload']
        key = (event['pid'], value['ClrInstanceID'], int(value['Count']))
        if event['name'] == 'GC/Start':
            assert key not in pending and key not in seen, 'Repeated GC start'
            pending[key] = event
            seen.add(key)
        else:
            assert key in pending, 'Missing GC start'
            first = pending.pop(key)
            depth = int(value['Depth'])
            assert depth == int(first['payload']['Depth']) and depth in [0, 1, 2] and event['ms'] >= first['ms']
            result.append(dict(count=key[2], depth=depth, type=first['payload']['Type'], reason=first['payload']['Reason'],
                start_ms=first['ms'], end_ms=event['ms'], indices=[first['index'], event['index']]))
    assert not pending, 'Missing GC stop'
    return result


def inspect(name):
    path = BASE / name
    summary = read(path / 'summary.json')
    assert summary['complete'] and summary['lost'] == 0 and summary['input_sha256'] == pin(INPUT / name / 'capture.nettrace')['sha256']
    events = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
    assert len(events) == summary['recorded'] and [e['index'] for e in events] == list(range(len(events)))
    ready, result = read(INPUT / name / 'ready.json'), read(INPUT / name / 'result.json')
    assert {e['pid'] for e in events} == {ready['pid']}
    counts = collections.Counter(e['name'] for e in events if e['provider'] == 'Microsoft-Windows-DotNETRuntime')
    assert counts == {key: value for key, value in summary['counts'].items() if key.startswith('GC/')}
    pauses, requests, gcs = pair_pauses(events), pair_requests(events), collections_from(events)
    assert len(gcs) == summary['counts']['GC/Start'] == summary['counts']['GC/Stop'] > 0
    assert len(pauses) == summary['counts'][STAGES[0]] > 0 and len(requests) == 12
    assert {p['reason'] for p in pauses} == GC_REASONS | {'SuspendOther'}
    # These captures have one runtime and non-overlapping suspension envelopes.
    assert len({p['clr_instance'] for p in pauses}) == 1
    assert all(a['end_ms'] <= b['start_ms'] for a, b in zip(pauses, pauses[1:]))
    records = [row for row in result['records'] if row['phase'] == 'measured']
    rows = []
    for request, record in zip(requests, records, strict=True):
        assert request['name'] == record['name'] and request['pass_index'] == record['pass']
        assert request['thread'] == record['thread_id'] == ready['thread_id']
        start, end = request['start_ms'], request['end_ms']
        delta_ms = end - start - record['seconds'] * 1000
        assert 0 <= delta_ms <= 2, 'Marker overhead outside the declared 2ms bound'
        expected = [b - a for a, b in zip(record['gc_before'], record['gc_after'], strict=True)]
        begun = [g for g in gcs if start <= g['start_ms'] <= end]
        completed = [g for g in gcs if start <= g['end_ms'] <= end]
        assert [g['count'] for g in begun] == [g['count'] for g in completed], 'GC crosses request boundary'
        actual = [sum(g['depth'] >= generation for g in completed) for generation in range(3)]
        assert actual == expected, (name, request['name'], request['pass_index'], actual, expected)
        by_reason = {}
        for reason in sorted(GC_REASONS | {'SuspendOther'}):
            selected = [p for p in pauses if p['reason'] == reason and p['end_ms'] > start and p['start_ms'] < end]
            by_reason[reason] = dict(envelopes=len(selected),
                envelope_ms=union_length([(p['start_ms'], p['end_ms']) for p in selected], start, end),
                suspension_handshake_ms=union_length([(p['start_ms'], p['suspended_ms']) for p in selected], start, end),
                suspended_until_restart_ms=union_length([(p['suspended_ms'], p['restart_ms']) for p in selected], start, end),
                restart_ms=union_length([(p['restart_ms'], p['end_ms']) for p in selected], start, end))
            values = by_reason[reason]
            assert abs(values['envelope_ms'] - sum(values[k] for k in ['suspension_handshake_ms', 'suspended_until_restart_ms', 'restart_ms'])) < 1e-8
        gc_ms = sum(by_reason[reason]['envelope_ms'] for reason in GC_REASONS)
        all_ms = union_length([(p['start_ms'], p['end_ms']) for p in pauses], start, end)
        assert abs(all_ms - sum(value['envelope_ms'] for value in by_reason.values())) < 1e-8
        rows.append(dict(**request, wall_seconds=record['seconds'], marker_extra_ms=delta_ms, gc_generation_delta=actual,
            gc_envelope_ms=gc_ms, gc_envelope_to_wall=gc_ms / (record['seconds'] * 1000),
            all_suspension_envelope_ms=all_ms, by_reason=by_reason))
    totals = []
    for case in dict.fromkeys(row['name'] for row in rows):
        selected = [row for row in rows if row['name'] == case]
        assert len(selected) == 3
        wall = sum(row['wall_seconds'] for row in selected)
        gc_ms = sum(row['gc_envelope_ms'] for row in selected)
        totals.append(dict(name=case, calls=3, wall_seconds=wall, gc_envelope_ms=gc_ms,
            gc_envelope_to_wall=gc_ms / (wall * 1000), other_envelope_ms=sum(row['by_reason']['SuspendOther']['envelope_ms'] for row in selected)))
    return dict(name=name, captured_core=result['core_sha256'], captured_data=result['data_sha256'], summary=summary,
        reason_counts=dict(collections.Counter(p['reason'] for p in pauses)), pauses=pauses, collections=gcs, requests=rows, totals=totals)


def main():
    assert not (BASE / 'closed.json').exists()
    inputs, prepared, state = read(BASE / 'inputs.json'), read(BASE / 'prepared.json'), read(BASE / 'processes.json')
    verify(inputs['files'])
    verify(prepared['files'])
    assert state['complete'] and state['code'] == 0
    assert [r['name'] for r in state['runs']] == ['restore', 'build', 'sampled-a', 'sampled-b']
    identities, resources = [state['supervisor']], []
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0 and run['seconds'] < 300
        assert run['preflight']['available'] >= (8 if run['name'] in ['restore', 'build'] else 2) * 1024**3
        samples = [json.loads(line) for line in (BASE / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == run['samples'] > 0 and max(r['rss'] for r in samples) == run['peak_rss']
        for row in samples:
            assert row['seconds'] < 300 and row['rss'] < 2 * 1024**3 and row['available'] >= 1024**3
            assert row['disk'] >= 20 * 1024**3 and row['output_bytes'] <= 1024**3
            assert row['rss'] == sum(p['rss'] for p in row['members'])
            assert all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in row['members'])
            if run['name'].startswith('sampled'):
                assert len(row['members']) <= 1
        identities.extend(dict(pid=int(pid), birth=birth) for pid, birth in run['members'].items())
        resources.append(dict(name=run['name'], samples=len(samples), peak_rss=run['peak_rss'], seconds=run['seconds']))
    tests = read(BASE / 'unit-tests.json')
    assert tests['passed'] and tests['tests'] >= 8 and tests['failures'] == tests['errors'] == tests['skips'] == 0
    assert tests['analysis'] == pin(Path(__file__))
    identities.append(tests['identity'])
    for identity in identities:
        terminal(identity)
    captures = [inspect(name) for name in ['sampled-a', 'sampled-b']]
    for capture in captures:
        run = next(r for r in state['runs'] if r['name'] == capture['name'])
        assert capture['summary']['exporter_pid'] == run['worker']['pid']
    analysis = dict(passed=True, inference_executed=False, captures=captures, tests=tests, identities=identities,
        resources=resources, resource_samples=sum(r['samples'] for r in resources), peak_rss=max(r['peak_rss'] for r in resources),
        scope='Request-aligned GC suspension upper envelopes from two retained older Windows captures; no new inference, current-build/AMD attribution, or latency comparison.')
    save(BASE / 'analysis.json', analysis)
    files = dict(prepared['files'])
    for path in [*BASE.rglob('*'), *TOOLS.iterdir()]:
        if path.is_file() and (not path.is_relative_to(BASE) or 'obj' not in path.relative_to(BASE).parts):
            files[rel(path)] = pin(path)
    save(BASE / 'closed.json', dict(passed=True, files=files, analysis=pin(BASE / 'analysis.json'), identities=identities))
    print(json.dumps(dict(closed=pin(BASE / 'closed.json'), totals=[dict(capture=c['name'], totals=c['totals']) for c in captures])))


if __name__ == '__main__':
    main()
