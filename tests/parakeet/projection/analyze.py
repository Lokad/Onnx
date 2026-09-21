"""Original-bit controls and independent local/inherited error decomposition."""
from common import *
import math
import onnx


def capture_checks(spec):
    checks = []; values = {}
    def check(name, passed): checks.append(dict(name=name, passed=bool(passed)))
    for job in CAPTURES:
        folder = BASE/'outputs'/job['id']; result = read(folder/'result.json')
        check('complete-'+job['id'], result['complete'] and result['job'] == job)
        check('ownership-'+job['id'], result['inputs_unchanged'] and result['held_outputs_unchanged'])
        check('coverage-'+job['id'], [r['name'] for r in result['outputs']] == OUTPUTS)
        if job['engine'] == 'managed':
            check('manifest-'+job['id'], result['manifest_sha256'] == pin(BASE/'manifest.json')['sha256'])
            check('core-'+job['id'], result['core_sha256'] == 'd1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4')
            check('runtime-'+job['id'], result['runtime'] == '10.0.12' and result['processor_count'] == 1 and not result['native_loaded'])
            check('nodes-'+job['id'], read(folder/'nodes.json') == read(ROOT/spec['nodes']['managed']))
        else:
            check('manifest-'+job['id'], result['manifest'] == pin(BASE/'manifest.json'))
            check('runtime-'+job['id'], result['onnxruntime'] == spec['onnxruntime'] and result['affinity'] == [2]
                  and result['settings'] == dict(threads=1, execution='sequential', optimization='all', spinning=False))
        values[job['id']] = tensors(folder)
        route = job['id'].removesuffix('-repeat')
        for name, record in spec['controls'][route].items():
            a, b = values[job['id']][name], array(ROOT/record['file'], record)
            check('original-'+job['id']+'-'+name, a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes())
        x = values[job['id']][RESHAPE]
        check('reshape-'+job['id'], x.dtype == np.float32 and x.shape == (1,74,4096))
    for engine in ('managed', 'native'):
        for name in OUTPUTS:
            a, b = values[engine+'-native'][name], values[engine+'-native-repeat'][name]
            check('repeat-'+engine+'-'+name, a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes())
    model = onnx.load(BASE/'outputs/native-native/optimized.onnx', load_external_data=False)
    nodes = [dict(name=n.name, op=n.op_type, domain=n.domain, inputs=list(n.input), outputs=list(n.output),
                  attributes=[dict(name=a.name, sha256=hashlib.sha256(a.SerializeToString()).hexdigest()) for a in n.attribute])
             for n in model.graph.node]
    check('native-optimized-nodes', nodes == read(ROOT/spec['nodes']['native'])['nodes'])
    return checks


def scalar_metric_check(a, b, reported, limit):
    # Independent scalar iteration verifies count, maximum and sum of squares.
    pairs = [(float(x),float(y)) for x,y in zip(a.flat,b.flat,strict=True)]
    scaled = [abs(x-y)/max(1.,abs(y)) for x,y in pairs]
    assert max(scaled) == reported['max_scaled']
    assert sum(v > limit for v in scaled) == reported['failed']
    assert max(abs(x-y) for x,y in pairs) == reported['max_absolute']
    rms = math.sqrt(math.fsum((x-y)*(x-y) for x,y in pairs)/len(pairs))
    assert math.isclose(rms, reported['rms'], rel_tol=1e-14, abs_tol=1e-20)


def main():
    spec = read(BASE/'manifest.json'); verify(spec); state = read(BASE/'processes.json')
    assert state['complete'] and state['code'] == 0 and absent(state['supervisor'])
    assert [r['job'] for r in state['runs']] == JOBS
    samples = 0; identities = [state['supervisor']]
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0 and absent(run['worker']) and run['samples'] > 0
        identities.append(run['worker']); preflight = run['preflight']
        assert preflight['available'] >= LIMITS['preflight_available'] and preflight['disk'] >= LIMITS['disk']
        rows = [json.loads(line) for line in (BASE/'process'/run['job']['id']/'samples.jsonl').read_text().splitlines()]
        assert len(rows) == run['samples'] and max(r['rss'] for r in rows) == run['peak_rss']
        for s in rows:
            assert s['seconds'] < LIMITS['seconds'] and s['rss'] < LIMITS['rss'] and s['available'] >= LIMITS['available']
            assert s['disk'] >= LIMITS['disk'] and s['affinity'] == [2]
            assert s['pid'] == run['worker']['pid'] and s['birth'] == run['worker']['birth']
        samples += len(rows)
    checks = capture_checks(spec); assert checks == read(BASE/'capture-checks.json') and all(c['passed'] for c in checks)
    refs = {}; agreements = []; dots = 0; max_dot_error = 0.
    w = np.load(ROOT/spec['weights']['projection.weight']['file'], allow_pickle=False)
    bias = np.load(ROOT/spec['weights']['projection.bias']['file'], allow_pickle=False).astype(np.float64)
    captured = {route:tensors(BASE/'outputs'/route) for route in ROUTES}
    for engine in ('numpy', 'torch'):
        folder = BASE/'outputs'/engine; result = read(folder/'result.json')
        assert result['complete'] and result['manifest'] == pin(BASE/'manifest.json')
        assert result['job'] == next(j for j in JOBS if j['id'] == engine)
        assert result['inputs_unchanged'] and result['weights_unchanged'] and result['held_outputs_unchanged']
        assert result['openblas_threads'] == 1 and not result['native_ort_loaded'] and result['numpy'] == spec['numpy']
        for path, expected in result['libraries'].items(): assert spec['numerical_libraries'][path] == expected == pin(path)
        assert [r['route'] for r in result['rows']] == ROUTES and len(result['probes']) == 1024
        refs[engine] = {route:tensors(folder/route) for route in ROUTES}
        for route in ROUTES:
            values = refs[engine][route]
            assert set(values) == {'projection','stem'}
            assert all(v.dtype == np.float64 and v.shape == (1,74,1024) for v in values.values())
            assert np.array_equal(values['projection']+bias, values['stem'])
            probes = [p for p in result['probes'] if p['route'] == route]
            assert [p['index'] for p in probes] == coordinates()
            for p in probes:
                row,col = divmod(p['index'],1024); x = captured[route][RESHAPE]
                expected = math.fsum(float(x[0,row,k])*float(w[k,col]) for k in range(4096))
                actual = float(values['projection'].flat[p['index']]); error = abs(actual-expected)/max(1.,abs(expected))
                assert p['expected'] == expected and p['actual'] == actual and p['error'] == error <= REFERENCE_LIMIT
                dots += 1; max_dot_error = max(max_dot_error,error)
    for route in ROUTES:
        for name in ('projection','stem'):
            a,b = refs['numpy'][route][name], refs['torch'][route][name]
            m = metric(a,b,REFERENCE_LIMIT); scalar_metric_check(a,b,m,REFERENCE_LIMIT)
            assert m['failed'] == 0; agreements.append(dict(route=route,stage=name,**m))
    rows = []
    for engine in ('numpy','torch'):
        for route in ROUTES:
            kind = route.split('-')[1]; whole = spec['references'][engine+'-'+kind]
            target = array(ROOT/whole['stem']['file'], whole['stem'])
            convolution_target = array(ROOT/whole['reshape']['file'], whole['reshape'])
            actual = captured[route][STEM].astype(np.float64); own = refs[engine][route]['stem']
            values = [('convolution-input',captured[route][RESHAPE].astype(np.float64),convolution_target),
                      ('local-projection',actual,own), ('inherited-convolution',own,target), ('total',actual,target)]
            metrics = {}
            for name,a,b in values:
                m = metric(a,b); scalar_metric_check(a,b,m,ORIGINAL_LIMIT); metrics[name] = m
            local, inherited, total = actual-own, own-target, actual-target
            residual = float(np.abs(local+inherited-total).max()); assert residual <= 2e-14
            local_energy = float(np.mean(local*local)); inherited_energy = float(np.mean(inherited*inherited))
            interaction = float(2*np.mean(local*inherited)); total_energy = float(np.mean(total*total))
            assert math.isclose(local_energy+inherited_energy+interaction,total_energy,rel_tol=1e-12,abs_tol=1e-24)
            rows.append(dict(reference=engine,route=route,metrics=metrics,maximum_decomposition_residual=residual,
                             local_mean_square=local_energy,inherited_mean_square=inherited_energy,
                             interaction=interaction,total_mean_square=total_energy))
    summary = dict(qualified=True, manifest=pin(BASE/'manifest.json'), checks=checks, agreements=agreements,
        scalar_checks=dots, max_scalar_error=max_dot_error, rows=rows, samples=samples,
        peak_rss=max(r['peak_rss'] for r in state['runs']), identities=identities)
    write(BASE/'analysis.json',summary)
    files = dict(spec['files'])
    for path in BASE.rglob('*'):
        if path.is_file(): files[rel(path)] = pin(path)
    write(BASE/'closed.json',dict(qualified=True,files=files,identities=identities,analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(qualified=True,checks=len(checks),scalar_checks=dots,samples=samples,
                         rows=[dict(route=r['route'],local=r['metrics']['local-projection'],inherited=r['metrics']['inherited-convolution'])
                               for r in rows if r['reference']=='numpy'],closure=pin(BASE/'closed.json'))))


if __name__ == '__main__': main()
