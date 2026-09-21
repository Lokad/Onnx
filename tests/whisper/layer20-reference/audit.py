"""Audit complete saved reference outputs and preserve every signed decomposition."""
import json
import time
from protocol import *


def audit():
    spec = read(BASE/'manifest.json'); state = read(BASE/'processes.json')
    assert state['complete'] is True and state['code'] == 0 and state['manifest'] == pin(BASE/'manifest.json')
    assert len(state['runs']) == 48 and [r['job'] for r in state['runs']] == spec['jobs']
    births = [state['supervisor']]+[r['worker'] for r in state['runs']]
    assert all(absent(b) for b in births)
    for name, wanted in spec['files'].items():
        assert pin(ROOT/name) == wanted, name
    for name, wanted in spec['numerical_files'].items():
        assert pin(name) == wanted, name
    outputs = {}; resources = []; repeated = {}; previous = state['started']
    for run in state['runs']:
        job = run['job']; request = spec['requests'][job['request']]
        assert run['complete'] is True and run['code'] == 0 and previous <= run['started'] < run['ended'] <= state['ended']
        previous = run['ended']; assert 0 < run['seconds'] < LIMITS['seconds']
        folder = BASE/'outputs'/job['id']; result = read(folder/'result.json')
        assert result['complete'] is True and result['input_unchanged'] is True and result['job'] == job
        assert result['manifest'] == pin(BASE/'manifest.json') and result['input_sha256'] == raw(load(job['input']))
        runtime = result['runtime']
        assert runtime['pid'] == run['worker']['pid'] and runtime['birth'] == run['worker']['birth']
        assert runtime['affinity'] == [2] and runtime['blas_threads'] == 1 and runtime['native_loaded'] is (job['engine'] == 'ort')
        for name, wanted in runtime['loaded'].items():
            assert pin(name) == wanted, name
        assert len(result['records']) == (48 if job['engine'] == 'numpy' else 3)
        if job['engine'] == 'ort':
            for record in result['records']:
                if record['kind'] == 'ort':
                    settings = record['settings']
                    assert settings['intra'] == settings['inter'] == 1
                    assert settings['execution'] == 'ExecutionMode.ORT_SEQUENTIAL'
                    assert settings['optimizations'] == 'GraphOptimizationLevel.ORT_DISABLE_ALL'
                    assert settings['intra_spinning'] == settings['inter_spinning'] == '0'
        assert len(result['outputs']) == 12
        assert {p.name for p in folder.iterdir()} == {'result.json'} | {f'{i:02}.f64' for i in range(12)}
        for i, row in enumerate(result['outputs']):
            desc = spec['outputs'][i]
            assert row['index'] == i and row['name'] == desc['name'] and row['shape'] == desc['shape']
            path = folder/row['file']; assert pin(path) == row['pin'] and row['pin']['bytes'] == int(np.prod(desc['shape']))*8
            key = (request['features'], job['incoming'], job['engine'], i)
            if request['selected_request'] == 0:
                repeated[key] = row['pin']
            if request['selected_request'] == 3:
                assert row['pin'] == repeated[key]
            outputs[job['request'], job['incoming'], job['engine'], i] = dict(file=rel(path), pin=row['pin'], dtype='<f8', shape=row['shape'])
        samples = [json.loads(line) for line in (BASE/'process'/job['id']/'samples.jsonl').read_text().splitlines()]
        assert len(samples) == run['samples'] > 0; last = -1
        for sample in samples:
            assert last <= sample['seconds'] <= run['seconds']; last = sample['seconds']
            assert sample['pid'] == run['worker']['pid'] and sample['birth'] == run['worker']['birth'] and sample['affinity'] == [2]
            assert sample['rss'] < LIMITS['rss'] and sample['available'] >= LIMITS['available']
        assert max(s['rss'] for s in samples) == run['peak_rss']
        resources.append(dict(job=job['id'], samples=len(samples), peak_rss=run['peak_rss'], minimum_available=min(s['available'] for s in samples)))
    assert len(state['pairs']) == 24
    pairs = [dict(request=r['request'], incoming=incoming, maximum=check_pair(spec, r['request'], incoming))
             for r in spec['requests'] for incoming in ['reference', 'managed', 'native']]
    assert pairs == state['pairs']
    rows = []; started = time.monotonic(); own = psutil.Process(); own.cpu_affinity([2])
    analysis_resources = []
    for request in spec['requests']:
        for cell in ['MM', 'MN', 'NM', 'NN']:
            incoming = 'managed' if cell[1] == 'M' else 'native'
            for engine in ['numpy', 'ort']:
                for index, desc in enumerate(spec['outputs']):
                    actual = load(request['actual'][cell][index]); local = load(outputs[request['request'], incoming, engine, index])
                    ideal = load(outputs[request['request'], 'reference', engine, index])
                    rows.append(dict(request=request['request'], selected_request=request['selected_request'], features=request['features'],
                                     name=request['name'], cell=cell, reference=engine, index=index, output=desc['name'],
                                     actual=request['actual'][cell][index], own=outputs[request['request'], incoming, engine, index],
                                     ideal=outputs[request['request'], 'reference', engine, index], metrics=decompose(actual, local, ideal)))
                    del actual, local, ideal
                    resource = dict(seconds=time.monotonic()-started, rss=own.memory_info().rss, available=psutil.virtual_memory().available)
                    assert resource['seconds'] < 1800 and resource['rss'] < LIMITS['rss'] and resource['available'] >= LIMITS['available']
                    analysis_resources.append(resource)
    assert len(rows) == 768
    return dict(passed=True, manifest=pin(BASE/'manifest.json'), rows=rows, pairs=pairs, resources=resources,
                analysis_resources=analysis_resources, births=births, reference_calls=48, arrays=576, original_limit=1e-4, reference_limit=1e-9)


def main():
    assert not (BASE/'audit.json').exists()
    result = audit(); write(BASE/'audit.json', result)
    print(json.dumps(dict(passed=True, rows=len(result['rows']), maximum_reference_difference=max(r['maximum'] for r in result['pairs']),
                          local_maximum=max(r['metrics']['local_own']['max_scaled'] for r in result['rows']))))


if __name__ == '__main__':
    main()
