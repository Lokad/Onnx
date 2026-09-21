"""Apply unchanged per-emitter/union/counter gates to every new captured request."""
from gc_common import *


def main():
    assert not (GC_BASE / 'closed.json').exists()
    prepared, state = read(GC_BASE / 'prepared.json'), read(GC_BASE / 'processes.json')
    verify(prepared['files'])
    assert prepared['passed'] and not prepared['inference_executed'] and state['complete'] and state['code'] == 0
    assert [run['name'] for run in state['runs']] == ['sampled-a', 'sampled-b']
    identities, resources = [state['supervisor']], []
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0 and run['seconds'] < 300 and run['preflight']['available'] >= 2 * 1024**3
        samples = [json.loads(line) for line in (GC_BASE / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == run['samples'] > 0 and max(r['rss'] for r in samples) == run['peak_rss']
        for row in samples:
            assert row['seconds'] < 300 and row['rss'] < 2 * 1024**3 and row['available'] >= 1024**3
            assert row['disk'] >= 20 * 1024**3 and row['output_bytes'] <= 1024**3 and len(row['members']) <= 1
            assert row['rss'] == sum(p['rss'] for p in row['members'])
            assert all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in row['members'])
        identities.extend(dict(pid=int(pid), birth=birth) for pid, birth in run['members'].items())
        resources.append(dict(name=run['name'], samples=len(samples), peak_rss=run['peak_rss'], seconds=run['seconds']))
    for identity in identities:
        terminal(identity)
    inherited_tests = read(OLD_GC / 'unit-tests-v3.json')
    assert inherited_tests['passed'] and inherited_tests['tests'] == 17
    assert inherited_tests['analysis'] == pin(GC_TOOLS / 'analyze_v3.py')
    reader = parser()
    captures = [reader.inspect(name) for name in ['sampled-a', 'sampled-b']]
    for capture, run in zip(captures, state['runs'], strict=True):
        assert capture['name'] == run['name'] and capture['summary']['exporter_pid'] == run['worker']['pid']
        assert capture['summary']['runtime'] == '10.0.12' and capture['captured_core'] == CORE and capture['captured_data'] == DATA
    analysis = dict(passed=True, inference_executed=False, captures=captures, identities=identities, resources=resources,
        resource_samples=sum(r['samples'] for r in resources), peak_rss=max(r['peak_rss'] for r in resources),
        scope='GC suspension upper envelopes for the exact integrated Windows candidate; no background GC CPU attribution or performance comparison.')
    save(GC_BASE / 'analysis.json', analysis)
    files = dict(prepared['files'])
    for path in GC_BASE.rglob('*'):
        if path.is_file():
            files[rel(path)] = pin(path)
    save(GC_BASE / 'closed.json', dict(passed=True, files=files, identities=identities, analysis=pin(GC_BASE / 'analysis.json')))
    print(json.dumps(dict(closed=pin(GC_BASE / 'closed.json'), resources=analysis['resource_samples'], totals=[c['totals'] for c in captures])))


if __name__ == '__main__':
    main()
