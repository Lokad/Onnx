"""Close only complete, bit-preserving diagnostic processes; retain all counters."""
from collections import defaultdict
import json
import statistics
from common import *


def main():
    assert not (BASE / 'closed.json').exists()
    prepared = read(BASE / 'prepared.json'); verify(prepared['files'])
    state = read(BASE / 'processes.json'); assert state['complete'] and state['code'] == 0
    terminal(state['supervisor']); identities = [state['supervisor']]
    assert [r['name'] for r in state['runs']] == ['build', 'forward', 'reverse']
    samples = 0
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0 and run['preflight']['available'] >= 10 * 1024**3
        for pid, birth in run['members'].items():
            item = dict(pid=int(pid), birth=birth); terminal(item); identities.append(item)
        resource = [json.loads(line) for line in (BASE / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(resource) == run['samples'] > 0 and max(s['rss'] for s in resource) == run['peak_rss']
        for row in resource:
            assert row['seconds'] < 900 and row['rss'] < 8 * 1024**3 and row['available'] >= 1024**3 and row['disk'] >= 20 * 1024**3
            assert row['output_bytes'] <= 1024**3 and row['rss'] == sum(p['rss'] for p in row['members'])
            assert all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in row['members'])
        samples += len(resource)
    cases = read(BASE / 'manifest.json')['cases']
    all_rows = []; summaries = []; admitted = True; values = 0
    for order in prepared['orders']:
        result = read(BASE / order / 'result.json')
        assert result['passed'] and result['inputs_and_held_outputs_unchanged'] and result['runtime'] == '.NET 10.0.12'
        assert result['affinity'] == 4 and result['processor_count'] == 1 and result['order'] == order
        for key, name in [('core_sha256', 'Lokad.Onnx.dll'), ('data_sha256', 'Lokad.Onnx.Data.dll'), ('runner_sha256', 'Probe.dll')]:
            assert result[key] == pin(BASE / 'bin' / name)['sha256']
        assert result['manifest_sha256'] == pin(BASE / 'manifest.json')['sha256']
        modes = ['fresh', 'reuse', 'no-cache'] if order == 'forward' else ['no-cache', 'reuse', 'fresh']
        expected = [(mode, p, c) for mode in modes for p in range(3) for c in cases]
        assert len(result['records']) == len(expected) == 54
        grouped = defaultdict(list); previous = None
        for index, (row, (mode, p, case)) in enumerate(zip(result['records'], expected, strict=True)):
            assert row == read(BASE / order / f'{index:03}.json')
            assert (row['name'], row['graph'], row['mode'], row['pass']) == (case['name'], case['graph'], mode, p)
            assert row['phase'] == ('first' if p == 0 else 'repeat') and row['ownership']
            assert row['context_created'] == (mode == 'fresh' or (p == 0 and index % 6 < 2))
            assert row['end_ticks'] > row['start_ticks'] and row['frequency'] > 0
            assert row['allocated_after'] >= row['allocated_before'] and row['allocated_bytes'] == row['allocated_after'] - row['allocated_before']
            assert row['pause_after_ticks'] >= row['pause_before_ticks'] and row['pause_ticks'] == row['pause_after_ticks'] - row['pause_before_ticks']
            assert row['pause_frequency'] == 10000000 and all(b >= a for a, b in zip(row['gc_before'], row['gc_after']))
            if previous:
                assert row['start_ticks'] >= previous['end_ticks'] and row['allocated_before'] >= previous['allocated_after']
                assert row['pause_before_ticks'] >= previous['pause_after_ticks'] and all(b >= a for a, b in zip(previous['gc_after'], row['gc_before']))
            previous = row
            assert all(row[k] >= 0 for k in ['pool_new_bytes', 'pool_reused_bytes', 'pool_peak_outstanding_bytes'])
            assert row['input_sha256'] == case['input']['sha256'] and row['output']['shape'] == case['expected']['shape']
            actual = BASE / order / row['output']['file']; wanted = ROOT / case['expected']['path']
            assert actual.read_bytes() == wanted.read_bytes() and pin(actual)['sha256'] == row['output']['sha256']
            assert actual.stat().st_size == row['output']['values'] * 4; values += row['output']['values']
            grouped[(row['graph'], mode, row['phase'])].append(row)
            all_rows.append(dict(order=order, **row))
        for (graph, mode, phase), rows in grouped.items():
            summaries.append(dict(order=order, graph=graph, mode=mode, phase=phase, calls=len(rows),
                allocated_mean=statistics.fmean(r['allocated_bytes'] for r in rows),
                pool_new_mean=statistics.fmean(r['pool_new_bytes'] for r in rows),
                pool_reused_mean=statistics.fmean(r['pool_reused_bytes'] for r in rows),
                seconds_mean=statistics.fmean((r['end_ticks'] - r['start_ticks']) / r['frequency'] for r in rows),
                pause_seconds_sum=sum(r['pause_ticks'] for r in rows) / 10000000,
                gc_delta_sum=[sum(r['gc_after'][g] - r['gc_before'][g] for r in rows) for g in range(3)]))
        fresh = statistics.fmean(r['allocated_bytes'] for r in grouped[('embedding', 'fresh', 'repeat')])
        reused = statistics.fmean(r['allocated_bytes'] for r in grouped[('embedding', 'reuse', 'repeat')])
        admitted = admitted and reused < fresh
    analysis = dict(passed=True, calls=len(all_rows), values=values, rows=all_rows, summaries=summaries,
        request_scoped_prototype_admitted=admitted, resource_samples=samples, peak_rss=max(r['peak_rss'] for r in state['runs']),
        identities=identities, scope='Graph diagnostic with precise process allocations and cumulative pause counters. No public/ORT latency claim.')
    save(BASE / 'analysis.json', analysis)
    files = dict(prepared['files'])
    for path in BASE.rglob('*'):
        if path.is_file(): files[rel(path)] = pin(path)
    save(BASE / 'closed.json', dict(passed=True, files=files, analysis=pin(BASE / 'analysis.json'), identities=identities))
    print(json.dumps({k: analysis[k] for k in ['passed', 'calls', 'values', 'request_scoped_prototype_admitted', 'resource_samples', 'peak_rss']}))
    print(json.dumps(dict(closed=pin(BASE / 'closed.json'))))


if __name__ == '__main__': main()
