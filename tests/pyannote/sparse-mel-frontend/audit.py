"""Independently verify all real-audio feature bytes, support indices and process evidence."""
from prepare import *

if __name__ == '__main__':
    assert not (BASE / 'closed.json').exists()
    spec = read(BASE / 'prepared.json')
    verify(spec['files'])
    identities, resources = [], []
    for filename, jobs in [('preparation.json', ['restore', 'build']), ('processes.json', spec['jobs'])]:
        state = read(BASE / filename)
        assert state['complete'] and state['code'] == 0 and [r['name'] for r in state['runs']] == jobs
        identities.append(state['supervisor'])
        for run in state['runs']:
            assert run['complete'] and run['code'] == 0 and run['seconds'] < 900
            minimum = 8 if filename == 'preparation.json' else 10
            assert run['preflight']['available'] >= minimum * 1024**3
            samples = [json.loads(line) for line in (BASE / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
            assert len(samples) == run['samples'] > 0 and max(r['rss'] for r in samples) == run['peak_rss']
            for row in samples:
                assert row['seconds'] < 900 and row['rss'] < 4 * 1024**3 and row['available'] >= 1024**3 and row['disk'] >= 20 * 1024**3
                assert row['output_bytes'] <= 1024**3 and row['rss'] == sum(p['rss'] for p in row['members'])
                assert all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in row['members'])
            identities.extend(dict(pid=int(pid), birth=birth) for pid, birth in run['members'].items())
            resources.append(dict(name=run['name'], samples=len(samples), peak_rss=run['peak_rss']))
    for identity in identities:
        terminal(identity)
    import numpy as np
    for case in read(INPUT)['cases']:
        original = np.load(ROOT / case['pcm']['path'], allow_pickle=False)
        assert (BASE / 'inputs' / (case['name'] + '.f32')).read_bytes() == original.tobytes()
    records, coefficients = [], None
    for name in spec['jobs']:
        result = read(BASE / name / 'result.json')
        assert result['passed'] and result['runtime'] == '.NET 10.0.12' and result['processor_count'] == 1 and result['flags'] == []
        assert result['core'] == spec['manifest']['core'] and result['data'] == spec['manifest']['data']
        assert result['baseline_data'] == '1d34666456a5da749dc3b40ee25621af0bed806c9167f3ad12e96d0736dca662'
        assert result['executable'] == pin(BASE / 'runtime/Probe.dll')['sha256'] and result['manifest'] == pin(BASE / 'manifest.json')['sha256']
        assert result['calls'] == 50 and result['inputs_and_held_outputs_unchanged'] and len(result['records']) == 25
        job = next(r for r in read(BASE / 'processes.json')['runs'] if r['name'] == name)
        assert result['pid'] == job['worker']['pid'] and result['order'] == name
        selected = {k: result[k] for k in ['coefficient_sha256', 'dense_terms_per_frame', 'retained_terms_per_frame', 'nonzero_terms_per_frame', 'ranges']}
        if coefficients is None:
            coefficients = selected
        else:
            assert selected == coefficients
        for index, (row, fixture) in enumerate(zip(result['records'], spec['manifest']['cases'], strict=True)):
            expected_values = (1 + (fixture['samples'] - 400) // 160) * 80
            assert row['ordinal'] == index and row['name'] == fixture['name'] and row['shape'] == [1, expected_values // 80, 80]
            assert row['values'] == expected_values and row['inputs_unchanged']
            left, right = BASE / name / f'{index}.dense.f32', BASE / name / f'{index}.sparse.f32'
            assert pin(left) == pin(right) == dict(bytes=expected_values * 4, sha256=row['sha256'])
            records.append(dict(order=name, **row))
    assert len(records) == spec['expected_pairs'] and sum(r['values'] for r in records) == spec['expected_values']
    analysis = dict(passed=True, pairs=len(records), calls=100, values=sum(r['values'] for r in records), coefficients=coefficients,
        records=records, resources=resources, resource_samples=sum(r['samples'] for r in resources), identities=identities,
        scope='Exact real-audio frontend qualification and coefficient census, no timing result.')
    save(BASE / 'analysis.json', analysis)
    files = dict(spec['files'])
    for path in BASE.rglob('*'):
        if path.is_file() and not {'obj', 'packages'}.intersection(path.relative_to(BASE).parts):
            files[rel(path)] = pin(path)
    save(BASE / 'closed.json', dict(passed=True, files=files, analysis=pin(BASE / 'analysis.json'), identities=identities))
    print(json.dumps(dict(closed=pin(BASE / 'closed.json'), pairs=analysis['pairs'], values=analysis['values'], coefficients=coefficients['retained_terms_per_frame'], resources=analysis['resource_samples'])))
