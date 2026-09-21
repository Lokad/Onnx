"""Recompute every comparison and close only complete, unchanged evidence."""
from common import *
import math
import numpy as np


def metrics(actual, reference):
    assert actual.shape == reference.shape and actual.dtype == reference.dtype == np.float32
    a = actual.astype(np.float64); b = reference.astype(np.float64)
    errors = np.abs(a-b)/np.maximum(1., np.abs(b))
    result = dict(max_scaled=float(errors.max()), failures=int(np.count_nonzero(errors > 1e-4)),
                  l2=float(np.linalg.norm(a-b)), values=int(a.size), worst_index=int(errors.argmax()))
    # Independent scalar arithmetic over every value: no vector reduction or
    # previously reported error/count is trusted for the acceptance decision.
    maximum = 0.; failures = 0
    for x,y in zip(actual.reshape(-1), reference.reshape(-1), strict=True):
        error = abs(float(x)-float(y))/max(1., abs(float(y)))
        maximum = max(maximum, error); failures += error > 1e-4
    assert maximum == result['max_scaled'] and failures == result['failures']
    return result


def main():
    spec = read(BASE/'manifest.json'); verify(spec)
    state = read(BASE/'processes.json')
    assert state['complete'] and state['code'] == 0 and absent(state['supervisor'])
    expected_jobs = [*spec['jobs'], dict(id='decode', engine='native')]
    assert [r['job'] for r in state['runs']] == expected_jobs
    resource_samples = 0; peak = 0
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0 and absent(run['worker']) and not run.get('error')
        assert run['preflight']['available'] >= LIMITS['preflight_available'] and run['preflight']['disk'] >= LIMITS['disk']
        samples = [json.loads(line) for line in (BASE/'process'/run['job']['id']/'samples.jsonl').read_text().splitlines()]
        assert len(samples) == run['samples'] and samples
        for sample in samples:
            assert {k:sample[k] for k in ('pid','birth')} == run['worker'] and sample['affinity'] == [2]
            assert sample['seconds'] < LIMITS['seconds'] and sample['rss'] < LIMITS['rss']
            assert sample['available'] >= LIMITS['available'] and sample['disk'] >= LIMITS['disk']
        assert run['peak_rss'] == max(s['rss'] for s in samples)
        resource_samples += len(samples); peak = max(peak, run['peak_rss'])
    records = {}; arrays = {}; checks = []
    for job in spec['jobs']:
        folder = BASE/'outputs'/job['id']; result = read(folder/'result.json'); records[job['id']] = result
        assert result['complete'] and result['job'] == job and result['manifest_sha256'] == state['manifest']['sha256']
        assert result['inputs_unchanged'] and result['held_outputs_unchanged'] and result['affinity'] == [2]
        assert [r['name'] for r in result['outputs']] == spec['outputs'][job['mode']]
        if job['engine'] == 'managed':
            assert result['core_sha256'] == CORE and result['runtime'] == '10.0.12' and not result['native_loaded']
            assert result['runner_sha256'] == pin(BASE/'bin/ParakeetLayerTrace.dll')['sha256']
            assert result['nodes_sha256'] == pin(folder/'nodes.json')['sha256']
        else:
            assert result['onnxruntime'] == spec['onnxruntime'] and result['numpy'] == spec['numpy']
            assert result['settings'] == dict(threads=1, execution='sequential', optimization='all', spinning=False)
        arrays[job['id']] = {r['name']:array(folder/r['file'],r) for r in result['outputs']}
        for name,value in arrays[job['id']].items():
            expected = (1,) if name == 'encoded_lengths' else (1,1024,74) if name == 'outputs' else (1,74,1024)
            assert value.shape == expected, (name,value.shape)
            assert value.dtype == (np.int64 if name == 'encoded_lengths' else np.float32), (name,value.dtype)
    def check(name, left, right):
        ok = left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()
        checks.append(dict(name=name, passed=ok)); return ok
    for prefix,path in spec['controls'].items():
        check('original-'+prefix, arrays[prefix+'-plain']['outputs'], np.load(ROOT/path, allow_pickle=False))
    for engine in ('managed','native'):
        for kind in ('native','managed'):
            prefix = engine+'-'+kind
            for name in spec['outputs']['plain']:
                check('plain/trace-'+prefix+'-'+name, arrays[prefix+'-plain'][name], arrays[prefix+'-trace'][name])
            for name in spec['outputs']['trace']:
                check('repeat-'+prefix+'-'+name, arrays[prefix+'-trace'][name], arrays[prefix+'-trace-repeat'][name])
            if engine == 'managed':
                paths = [BASE/'outputs'/(prefix+'-'+suffix)/'nodes.json' for suffix in ('plain','trace','trace-repeat')]
                nodes = [read(path) for path in paths]
                checks.append(dict(name='optimized-nodes-'+prefix, passed=nodes[0] == nodes[1] == nodes[2]))
    qualified = all(c['passed'] for c in checks)
    comparisons = []
    if qualified:
        for route,left,right in [('native-features','managed-native-trace','native-native-trace'),
                                 ('managed-features','managed-managed-trace','native-managed-trace'),
                                 ('natural-inputs','managed-managed-trace','native-native-trace')]:
            for name in spec['outputs']['trace']:
                if name == 'encoded_lengths':
                    assert arrays[left][name].tobytes() == arrays[right][name].tobytes(); continue
                comparisons.append(dict(route=route, name=name, **metrics(arrays[left][name],arrays[right][name])))
    decoder = read(BASE/'outputs/decode/result.json'); assert decoder['complete'] and decoder['inputs_unchanged'] and decoder['held_outputs_unchanged']
    assert decoder['manifest_sha256'] == state['manifest']['sha256'] and decoder['onnxruntime'] == spec['onnxruntime'] and decoder['affinity'] == [2]
    assert [row['job'] for row in decoder['rows']] == spec['jobs']
    decoding = []
    for row in decoder['rows']:
        assert [r['name'] for r in row['outputs']] == list(spec['decoder']['expected'])
        folder = BASE/'outputs/decode'/row['job']['id']
        for record in row['outputs']:
            value = array(folder/record['file'], record)
            reference = np.load(ROOT/spec['decoder']['expected'][record['name']], allow_pickle=False)
            if value.dtype == np.float32:
                item = dict(job=row['job']['id'], name=record['name'], **metrics(value, reference))
                if record['name'] == 'outputs':
                    assert value.shape == (1,1,1,8198)
                    item['duration_zero'] = float(value.reshape(-1)[8193]); item['reference_duration_zero'] = float(reference.reshape(-1)[8193])
                    item['token'] = int(value.reshape(-1)[:8193].argmax()); item['duration'] = int(value.reshape(-1)[8193:].argmax())
                decoding.append(item)
            else:
                assert value.dtype == reference.dtype and value.tobytes() == reference.tobytes()
    result = dict(protocol=PROTOCOL, trace_qualified=qualified, checks=checks, comparisons=comparisons,
                  decoder=decoding, resource_samples=resource_samples, peak_rss=peak,
                  encoder_calls=12, decoder_calls=12, encoder_arrays=sum(len(r['outputs']) for r in records.values()),
                  decoder_arrays=sum(len(r['outputs']) for r in decoder['rows']), terminal_identities=14)
    write(BASE/'analysis.json', result)
    files = {rel(p):pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    write(BASE/'closed.json', dict(protocol=PROTOCOL, trace_qualified=qualified, files=files, external_files=spec['files'],
                                 identities=[state['supervisor']]+[r['worker'] for r in state['runs']]))
    print(json.dumps(dict(trace_qualified=qualified, failed_checks=[c['name'] for c in checks if not c['passed']],
                         comparisons=len(comparisons), decoder_comparisons=len(decoding), peak_rss=peak, files=len(files), closure=pin(BASE/'closed.json'))))
    if not qualified: raise SystemExit(1)


if __name__ == '__main__':
    main()
