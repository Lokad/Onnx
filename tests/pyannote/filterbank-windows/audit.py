"""Audit all 32 windows, both reference routes and every retained host/setting."""
import argparse, re
from shared import *

def main():
    p = argparse.ArgumentParser(); p.add_argument('--artifact', required=True); a = p.parse_args()
    base = Path(a.artifact).resolve(); psutil_module().Process().cpu_affinity([0])
    spec = read(base / 'manifest.json'); state = read(base / 'run.json'); verify(spec['files'])
    assert len(spec['cases']) == 32 and spec['variants'] == list(VARIANTS) and spec['values'] == 2554880
    assert spec['limits'] == LIMITS and spec['reference_limit'] == REFERENCE_LIMIT and spec['original_limit'] == ORIGINAL_LIMIT
    assert spec['stages'] == list(STAGES) and spec['tests'] == pin(base / 'tests.json') and read(base / 'tests.json')['code'] == 0
    assert spec['interpreter'] == pin(sys.executable)
    for path, wanted in spec['numeric'].items(): assert pin(path) == wanted
    assert state['complete'] and state['code'] == 0 and not state.get('error') and absent(state['supervisor'])
    assert state['manifest'] == pin(base / 'manifest.json') and [r['engine'] for r in state['runs']] == ['numpy', 'torch']
    assert state['runs'][0]['ended'] <= state['runs'][1]['started']
    resources = []; results = {}; births = [state['supervisor']]
    for run in state['runs']:
        engine = run['engine']; assert absent(run['worker']); births.append(run['worker'])
        samples = [json.loads(v) for v in (base / (engine + '.samples.jsonl')).read_text().splitlines()]
        resources.append(resource_checks(run, samples, state['supervisor']))
        result = read(base / engine / 'result.json'); runtime = result['runtime']
        assert result['complete'] and result['engine'] == engine and result['manifest'] == state['manifest']
        assert [r['name'] for r in result['records']] == [c['name'] for c in spec['cases']]
        assert runtime['pid'] == run['worker']['pid'] and runtime['birth'] == run['worker']['birth'] and runtime['affinity'] == [0]
        assert runtime['numpy'] == '2.2.4' and runtime['torch'] == (None if engine == 'numpy' else '2.11.0+cpu')
        assert runtime['native_ort_loaded'] is False and runtime['blas_threads'] == 1
        if engine == 'torch':
            assert 'BLAS_INFO=mkl' in runtime['torch_config'] and any('torch_cpu.dll' in path for path in runtime['libraries'])
            for method in ['mkl_get_max_threads', 'at::get_num_threads', 'at::get_num_interop_threads']:
                assert re.search(re.escape(method) + r'\(\)\s*:\s*1\b', runtime['torch_parallel'])
        else:
            assert runtime['torch_config'] is None and not any('torch' in Path(path).parts for path in runtime['libraries'])
        for path, wanted in runtime['libraries'].items(): assert spec['numeric'][path] == wanted and pin(path) == wanted
        assert (base / (engine + '.stdout')).read_text().splitlines() == [c['name'] for c in spec['cases']]
        assert not (base / (engine + '.stderr')).read_text().strip()
        results[engine] = result
    expected_shapes = dict(windowed=[998, 512], real=[998, 257], imaginary=[998, 257], power=[998, 257],
                           energy=[998, 80], raw=[998, 80], features=[1, 998, 80])
    tables = direct_tables(); window = np.load(ROOT / spec['window'])
    stages = []; scalars = []; comparisons = []; originals = []; duplicates = []; seen = {}; total_bytes = 0
    for index, case in enumerate(spec['cases']):
        assert list(case['baselines']) == list(VARIANTS)
        pcm = np.load(ROOT / case['input'], allow_pickle=False)
        original = np.load(ROOT / case['original_pcm'], allow_pickle=False)
        start = case['window'] * 16000; stop = min(original.size, start + 160000)
        valid = stop - start; assert valid == case['valid_samples']
        derived = np.concatenate((original[start:stop], np.zeros(160000 - valid, np.float32)))
        assert pcm.dtype == np.float32 and pcm.shape == (160000,) and np.array_equal(pcm, derived)
        arrays = {}; stage_pins = {}
        for engine in ['numpy', 'torch']:
            record = results[engine]['records'][index]
            assert record['input_unchanged'] and record['coefficients_unchanged'] and list(record['stages']) == list(STAGES)
            directory = base / engine / case['name']; assert {p.name for p in directory.iterdir()} == {s + '.npy' for s in STAGES}
            arrays[engine] = {}; stage_pins[engine] = {}
            for stage in STAGES:
                path = directory / (stage + '.npy'); info = record['stages'][stage]
                assert pin(path) == info['pin'] and info['shape'] == expected_shapes[stage]
                value = np.load(path, allow_pickle=False)
                assert value.dtype == np.float64 and list(value.shape) == info['shape'] and np.isfinite(value).all()
                arrays[engine][stage] = value; stage_pins[engine][stage] = info['pin']; total_bytes += value.nbytes
        identity = pin(ROOT / case['input'])['sha256']
        if identity in seen:
            before = seen[identity]; assert before['stages'] == stage_pins
            duplicates.append(dict(first=before['name'], repeat=case['name'], complete_reference_bits_identical=True))
        else: seen[identity] = dict(name=case['name'], stages=stage_pins)
        for stage in STAGES:
            stages.append(dict(name=case['name'], stage=stage, **metric(arrays['numpy'][stage], arrays['torch'][stage], REFERENCE_LIMIT)))
        for frame in [0, 499, 997]:
            scalar = scalar_window(pcm, window, frame); real, imaginary = direct_fourier(scalar, tables)
            for engine in ['numpy', 'torch']:
                for stage, expected in [('windowed', scalar), ('real', real), ('imaginary', imaginary)]:
                    scalars.append(dict(name=case['name'], frame=frame, engine=engine, stage=stage,
                                        **metric(arrays[engine][stage][frame], expected, REFERENCE_LIMIT)))
        native = load_baseline(case['baselines']['native']); previous = {}
        for variant in VARIANTS:
            actual = load_baseline(case['baselines'][variant])
            if variant != 'native':
                old = metric(actual, native, ORIGINAL_LIMIT); assert old == case['original'][variant]
                originals.append(dict(name=case['name'], corpus=case['corpus'], variant=variant, **old))
                host, setting = variant.split('-')
                if setting == 'default': previous[host] = actual
                else: assert np.array_equal(actual.view(np.uint32), previous[host].view(np.uint32))
            for reference in ['numpy', 'torch']:
                comparisons.append(dict(name=case['name'], corpus=case['corpus'], variant=variant, reference=reference,
                                        **metric(actual, arrays[reference]['features'], ORIGINAL_LIMIT)))
    assert len(duplicates) == 3
    for corpus in CORPORA:
        assert sum(c['corpus'] == corpus['name'] for c in spec['cases']) == corpus['windows']
        for variant in VARIANTS[1:]:
            rows = [r for r in originals if r['corpus'] == corpus['name'] and r['variant'] == variant]
            assert sum(r['failed'] for r in rows) == corpus['failures'][variant.split('-')[0]]
            assert sum(r['values'] for r in rows) == corpus['windows'] * 79840
    answer = dict(structural_passed=True, reference_passed=all(r['failed'] == 0 for r in stages + scalars), cases=32, arrays=448, bytes=total_bytes,
                  manifest=pin(base / 'manifest.json'), births=births, resources=resources, stages=stages, scalars=scalars,
                  comparisons=comparisons, originals=originals, duplicates=duplicates,
                  files={path.relative_to(base).as_posix(): pin(path) for path in sorted(base.rglob('*')) if path.is_file()})
    write(base / 'audit.json', answer)
    print(json.dumps(dict(structural_passed=True, reference_passed=answer['reference_passed'], arrays=448, bytes=total_bytes,
                         max_stage=max(r['max_scaled'] for r in stages), max_scalar=max(r['max_scaled'] for r in scalars),
                         failed_fp32_arrays=sum(r['failed'] > 0 for r in comparisons))))

if __name__ == '__main__': main()
