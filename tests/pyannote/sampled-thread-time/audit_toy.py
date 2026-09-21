"""Require terminal bounded capture, resolved scopes and agreeing stack exports."""
from common import *
from stacks import inspect, cross_export


def main():
    assert not (BASE / 'toy-closed.json').exists()
    spec = read(BASE / 'prepared.json')
    verify_spec(spec)
    identities, resources = [], []
    for file, names in [('preparation.json', ['version', 'profiles', 'collect-help', 'convert-help', 'restore', 'build']),
                        ('toy-processes.json', ['toy', 'export-speedscope', 'export-chromium'])]:
        state = read(BASE / file)
        assert state['complete'] and state['code'] == 0 and [r['name'] for r in state['runs']] == names
        identities.append(state['supervisor'])
        for run in state['runs']:
            assert run['complete'] and run['code'] == 0 and run['preflight']['available'] >= 8*1024**3
            paired = run['name'] == 'toy'
            identities.extend(run['processes'].values() if paired else [dict(pid=int(pid), birth=birth) for pid, birth in run['members'].items()])
            samples = [json.loads(s) for s in (BASE / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
            assert len(samples) == run['samples'] > 0 and max(s['rss'] for s in samples) == run['peak_rss']
            assert samples[-1]['seconds'] <= run['seconds'] < 900
            for s in samples:
                assert s['rss'] < 8*1024**3 and s['available'] >= 1024**3 and s['disk'] >= 20*1024**3 and s['output_bytes'] <= 1024**3
                assert s['rss'] == sum(p['rss'] for p in s['members'])
                for p in s['members']:
                    record = run['members'][str(p['pid'])]
                    if paired:
                        assert record['birth'] == p['birth'] and record['affinity'] == p['affinity'] == ([2] if p['role']=='target' else [0])
                    else:
                        assert record == p['birth'] and p['affinity'] == [2]
            resources.append(dict(name=run['name'], samples=len(samples), peak_rss=run['peak_rss']))
    for identity in identities:
        terminal(identity)
    output = BASE / 'toy-output'
    result = read(output / 'result.json')
    assert result['passed'] and result['sampled'] and result['runtime'] == '10.0.12' and not result['flags']
    assert [r['pass'] for r in result['records']] == [0, 1]
    assert all(2 <= r['seconds'] < 2.1 and r['cpu_ticks'] > 0 for r in result['records'])
    speedscope, chromium = read(output / 'speedscope.speedscope.json'), read(output / 'chromium.chromium.json')
    assert not any('ToyMarkers.Warmup' in f['name'] for f in speedscope['shared']['frames'])
    parsed = inspect(speedscope, {'measured': '!ToyMarkers.Measured(', 'validation': '!ToyMarkers.Validation('})
    assert abs(parsed['selected_seconds']['measured']/sum(r['seconds'] for r in result['records'])-1) < .1
    assert .8 < parsed['selected_seconds']['validation'] < 1.2
    exports = cross_export(speedscope, chromium)
    analysis = dict(passed=True, no_model_inference=True, exports=exports, parsed=parsed, resources=resources,
        resource_samples=sum(r['samples'] for r in resources), identities=identities)
    save(BASE / 'toy-analysis.json', analysis)
    files = dict(spec['files'])
    for folder in [BASE, TOOLS]:
        for p in folder.rglob('*'):
            if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(folder).parts):
                files[rel(p)] = pin(p)
    save(BASE / 'toy-closed.json', dict(passed=True, files=files, external_files=spec['external_files'], identities=identities,
        analysis=pin(BASE / 'toy-analysis.json')))
    print(dict(passed=True, sampled_seconds=parsed['selected_seconds'], resource_samples=analysis['resource_samples'], closed=pin(BASE / 'toy-closed.json')))


if __name__ == '__main__':
    main()
