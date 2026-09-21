import ctypes
import json
import math
from common import *


def metric(actual, expected):
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    assert a.shape == b.shape == (1, 74, 1024) and np.isfinite(a).all() and np.isfinite(b).all()
    delta = a - b
    scaled = np.abs(delta) / np.maximum(1., np.abs(b))
    result = dict(max_scaled=float(scaled.max()), failed=int(np.count_nonzero(scaled > 1e-4)),
                  rms=float(np.sqrt(np.mean(delta * delta))))
    # Separate scalar arithmetic checks the summarized metrics over every value.
    pairs = list(zip(a.flat, b.flat, strict=True))
    errors = [abs(float(x) - float(y)) / max(1., abs(float(y))) for x, y in pairs]
    assert max(errors) == result['max_scaled'] and sum(e > 1e-4 for e in errors) == result['failed']
    scalar_rms = math.sqrt(math.fsum((float(x) - float(y))**2 for x, y in pairs) / len(pairs))
    assert math.isclose(result['rms'], scalar_rms, rel_tol=1e-14)
    return result


def main():
    assert not (BASE / 'analysis.json').exists() and not (BASE / 'closed.json').exists()
    prepared = read(BASE / 'prepared.json')
    assert prepared['passed'] and prepared['blocks'] == BLOCKS
    verify(prepared['files'])
    identities = []
    resources = []
    for name in ('build-state.json', 'processes.json'):
        state = read(BASE / name)
        assert state['complete'] and state['code'] == 0
        identities.append(state['supervisor'])
        for run in state['runs']:
            assert run['complete'] and run['code'] == 0
            identities.extend(dict(pid=int(pid), birth=birth) for pid, birth in run['members'].items())
            samples = [json.loads(line) for line in (BASE / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
            assert len(samples) == run['samples'] > 0 and max(s['rss'] for s in samples) == run['peak_rss']
            assert run['preflight']['available'] >= 4 * 1024**3
            assert all(s['seconds'] < 600 and s['rss'] < 2 * 1024**3 and s['available'] >= 1024**3
                       and s['disk'] >= 20 * 1024**3 and s['output_bytes'] <= 1024**3
                       and all(m['affinity'] == [2] and run['members'][str(m['pid'])] == m['birth'] for m in s['members']) for s in samples)
            if name == 'processes.json':
                assert len(state['runs']) == 1 and run['name'] == 'arithmetic'
                assert all(len(s['members']) <= 1 for s in samples)
            resources.append(dict(name=run['name'], samples=len(samples), peak_rss=run['peak_rss']))
    for identity in identities:
        terminal(identity)
    result = read(BASE / 'output/result.json')
    assert result['passed'] and result['ownership'] and result['selftests'] == 160 and result['repeats'] == 16
    assert result['runtime'] == '10.0.12' and result['processor_count'] == 1 and result['affinity'] == 4 and result['fma']
    assert result['prepared_sha256'] == pin(BASE / 'prepared.json')['sha256']
    assert result['core_sha256'] == pin(PRODUCT / 'Lokad.Onnx.dll')['sha256']
    assert result['runner_sha256'] == pin(BASE / 'source/bin/Release/net10.0/Probe.dll')['sha256']
    assert [(r['route'], r['block']) for r in result['records']] == [(r, b) for r in ROUTES for b in [0, *BLOCKS]]
    values = {}
    for row in result['records']:
        values[row['route'], row['block']] = {}
        for kind in ('projection', 'stem'):
            spec = dict(row[kind], dtype='float32', shape=[1, 74, 1024])
            values[row['route'], row['block']][kind] = array(BASE / 'output' / row['route'] / spec['file'], spec)
    bias = np.fromfile(ROOT / prepared['bias'], dtype=np.float32)
    weight = np.fromfile(ROOT / prepared['weight'], dtype=np.float32).reshape(4096, 1024)
    crt = ctypes.CDLL('ucrtbase.dll')
    fma = crt.fmaf
    fma.argtypes = [ctypes.c_float] * 3
    fma.restype = ctypes.c_float
    fma_path = Path(os.environ['SystemRoot']) / 'System32/ucrtbase.dll'
    assert fma(2., 3., 4.) == 10.
    coords = [i * (74 * 1024 - 1) // 31 for i in range(32)]
    original = read(ORIGINAL / 'manifest.json')
    metrics = []
    dots = 0
    for route in ROUTES:
        captured = tensors(ORIGINAL / 'outputs' / route)
        x = captured['/pre_encode/Reshape_output_0'].reshape(74, 4096)
        for kind in ('projection', 'stem'):
            assert values[route, 0][kind].tobytes() == values[route, 4096][kind].tobytes()
        if route.startswith('managed-'):
            assert values[route, 0]['stem'].tobytes() == captured['/pre_encode/out/Add_output_0'].tobytes()
        for block in [0, *BLOCKS]:
            actual = values[route, block]
            assert (actual['projection'] + bias).tobytes() == actual['stem'].tobytes()
            if block:
                for index in coords:
                    row, col = divmod(index, 1024)
                    total = 0.
                    for begin in range(0, 4096, block):
                        subtotal = 0.
                        for k in range(begin, min(begin + block, 4096)):
                            subtotal = fma(float(weight[k, col]), float(x[row, k]), subtotal)
                        total = subtotal if begin == 0 else ctypes.c_float(total + subtotal).value
                    assert np.float32(total).tobytes() == actual['projection'].flat[index].tobytes(), (route, block, index)
                    dots += 1
            for engine in ('numpy', 'torch'):
                own = tensors(ORIGINAL / 'outputs' / engine / route)
                whole_spec = original['references'][engine + '-' + route.split('-')[1]]['stem']
                whole = array(ROOT / whole_spec['file'], whole_spec)
                metrics.append(dict(route=route, block=block, reference=engine,
                                    local=metric(actual['stem'], own['stem']),
                                    total=metric(actual['stem'], whole),
                                    native_original_local=metric(captured['/pre_encode/out/Add_output_0'], own['stem']) if route.startswith('native-') else None))
    decisions = {}
    for block in BLOCKS[:-1]:
        comparisons = []
        for route in ROUTES:
            for engine in ('numpy', 'torch'):
                baseline = next(r['local'] for r in metrics if (r['route'], r['block'], r['reference']) == (route, 0, engine))
                candidate = next(r['local'] for r in metrics if (r['route'], r['block'], r['reference']) == (route, block, engine))
                gain = baseline['rms'] / candidate['rms']
                comparisons.append(dict(route=route, reference=engine, rms_reduction=gain,
                                        passed=gain >= 2 and candidate['max_scaled'] <= baseline['max_scaled'] and candidate['failed'] <= baseline['failed']))
        decisions[str(block)] = dict(qualifies_for_full_model=all(c['passed'] for c in comparisons), comparisons=comparisons)
    eligible = [b for b in BLOCKS[:-1] if decisions[str(b)]['qualifies_for_full_model']]
    selected = 256 if 256 in eligible else min(eligible, key=lambda b: max(r['local']['rms'] for r in metrics if r['block'] == b), default=None)
    analysis = dict(passed=True, scalar_dots=dots, selftests=result['selftests'], metrics=metrics,
                    decisions=decisions, selected_for_full_model=selected, resources=resources, identities=identities,
                    scalar_library=dict(path=str(fma_path), **pin(fma_path)),
                    scope='Projection-only arithmetic; retained original failures still open; no native/model/timing promotion')
    save(BASE / 'analysis.json', analysis)
    files = dict(prepared['files'])
    for p in [*BASE.rglob('*'), *TOOLS.iterdir()]:
        if p.is_file():
            files[rel(p)] = pin(p)
    save(BASE / 'closed.json', dict(passed=True, files=files, scalar_library=analysis['scalar_library'],
                                  selected_for_full_model=selected, identities=identities))
    print(json.dumps(dict(passed=True, scalar_dots=dots, selected_for_full_model=selected,
                         native_feature_managed_input=[r for r in metrics if r['route'] == 'managed-native' and r['reference'] == 'numpy'],
                         resources=resources, closed=pin(BASE / 'closed.json'))))


if __name__ == '__main__':
    main()
