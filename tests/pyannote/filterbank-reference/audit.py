"""Compare every saved stage/value and independent direct Fourier checks."""
import argparse, re
from common import *
from routes import scalar_window, direct_tables, direct_fourier

def resource_checks(run, samples, supervisor):
    assert run['complete'] and run['code'] == 0 and 0 < run['seconds'] < LIMITS['seconds']
    assert run['started'] <= run['ended'] and run['worker']['birth'] >= supervisor['birth']
    assert run['preflight_available'] >= LIMITS['preflight'] and run['preflight_disk'] >= LIMITS['disk']
    assert len(samples) == run['samples'] and samples
    for row in samples:
        assert row['pid'] == run['worker']['pid'] and row['birth'] == run['worker']['birth'] and row['affinity'] == [0]
        assert 0 <= row['seconds'] <= run['seconds'] and row['rss'] < LIMITS['rss'] and row['available'] >= LIMITS['available']
    gaps = [samples[0]['seconds']] + [b['seconds'] - a['seconds'] for a, b in zip(samples, samples[1:])] + [run['seconds'] - samples[-1]['seconds']]
    assert min(gaps) >= 0 and max(gaps) < 10
    return dict(engine=run['engine'], seconds=run['seconds'], samples=len(samples), peak_rss=max(r['rss'] for r in samples), min_available=min(r['available'] for r in samples))

def main():
    p = argparse.ArgumentParser(); p.add_argument('--artifact', required=True); a = p.parse_args()
    base = Path(a.artifact).resolve(); psutil_module().Process().cpu_affinity([0])
    spec = read(base / 'manifest.json'); state = read(base / 'run.json')
    verify(spec['files'])
    for path, wanted in spec['numeric'].items(): assert pin(path) == wanted
    assert spec['interpreter'] == pin(sys.executable)
    assert spec['limits'] == LIMITS and spec['reference_limit'] == REFERENCE_LIMIT and spec['original_limit'] == ORIGINAL_LIMIT
    assert spec['stages'] == list(STAGES) and spec['tests'] == pin(base / 'tests.json')
    assert state['complete'] and state['code'] == 0 and not state.get('error') and absent(state['supervisor'])
    assert state['manifest'] == pin(base / 'manifest.json') and [r['engine'] for r in state['runs']] == ['numpy', 'torch']
    assert state['runs'][0]['ended'] <= state['runs'][1]['started']
    resources = []; results = {}; births = [state['supervisor']]
    for run in state['runs']:
        name = run['engine']; assert absent(run['worker']); births.append(run['worker'])
        samples = [json.loads(v) for v in (base / (name + '.samples.jsonl')).read_text().splitlines()]
        resources.append(resource_checks(run, samples, state['supervisor']))
        result = read(base / name / 'result.json'); runtime = result['runtime']
        assert result['complete'] and result['engine'] == name and result['manifest'] == state['manifest']
        assert [r['name'] for r in result['records']] == [c['name'] for c in spec['cases']]
        assert runtime['pid'] == run['worker']['pid'] and runtime['birth'] == run['worker']['birth'] and runtime['affinity'] == [0]
        assert runtime['numpy'] == '2.2.4' and runtime['torch'] == (None if name == 'numpy' else '2.11.0+cpu')
        assert runtime['native_ort_loaded'] is False and runtime['blas_threads'] == 1
        if name == 'torch':
            assert 'BLAS_INFO=mkl' in runtime['torch_config']
            assert re.search(r'mkl_get_max_threads\(\)\s*:\s*1\b', runtime['torch_parallel'])
            assert any('torch_cpu.dll' in path for path in runtime['libraries'])
        else:
            assert runtime['torch_config'] is None and runtime['torch_parallel'] is None
            assert not any('torch' in Path(path).parts for path in runtime['libraries'])
        for path, wanted in runtime['libraries'].items(): assert spec['numeric'][path] == wanted and pin(path) == wanted
        assert (base / (name + '.stdout')).read_text().splitlines() == [c['name'] for c in spec['cases']]
        assert not (base / (name + '.stderr')).read_text().strip()
        results[name] = result
    tables = direct_tables(); window = np.load(ROOT / spec['window'])
    stages = []; scalars = []; fp32 = []; originals = []; total_bytes = 0
    for index, case in enumerate(spec['cases']):
        pcm = np.load(ROOT / case['input'], allow_pickle=False); frames = 1 + (pcm.size - 400) // 160
        expected_shapes = dict(windowed=[frames, 512], real=[frames, 257], imaginary=[frames, 257], power=[frames, 257],
                               energy=[frames, 80], raw=[frames, 80], features=[1, frames, 80])
        arrays = {}
        for engine in ['numpy', 'torch']:
            record = results[engine]['records'][index]
            assert record['input_unchanged'] and record['coefficients_unchanged'] and list(record['stages']) == list(STAGES)
            arrays[engine] = {}
            for stage in STAGES:
                path = base / engine / case['name'] / (stage + '.npy'); info = record['stages'][stage]
                assert info['pin'] == pin(path) and info['shape'] == expected_shapes[stage]
                value = np.load(path, allow_pickle=False)
                assert value.dtype == np.float64 and list(value.shape) == info['shape'] and np.isfinite(value).all()
                arrays[engine][stage] = value; total_bytes += value.nbytes
        for stage in STAGES:
            stages.append(dict(name=case['name'], stage=stage, **metric(arrays['numpy'][stage], arrays['torch'][stage], REFERENCE_LIMIT)))
        for frame in sorted({0, frames // 2, frames - 1}):
            scalar = scalar_window(pcm, window, frame); real, imaginary = direct_fourier(scalar, tables)
            for engine in ['numpy', 'torch']:
                for stage, expected in [('windowed', scalar), ('real', real), ('imaginary', imaginary)]:
                    scalars.append(dict(name=case['name'], frame=frame, engine=engine, stage=stage,
                                        **metric(arrays[engine][stage][frame], expected, REFERENCE_LIMIT)))
        native = np.load(ROOT / case['native'], allow_pickle=False)
        managed = np.fromfile(ROOT / case['managed'], dtype='<f4').reshape(case['shape'])
        old = metric(managed, native, ORIGINAL_LIMIT); assert old == case['original']
        originals.append(dict(name=case['name'], **old))
        for engine, actual in [('managed', managed), ('native', native)]:
            for reference in ['numpy', 'torch']:
                fp32.append(dict(name=case['name'], engine=engine, reference=reference,
                                 **metric(actual, arrays[reference]['features'], ORIGINAL_LIMIT)))
    assert sum(r['values'] for r in originals) == 711680 and sum(r['failed'] for r in originals) == 3
    answer = dict(structural_passed=True, reference_passed=all(r['failed'] == 0 for r in stages + scalars),
                  cases=len(spec['cases']), arrays=2 * len(spec['cases']) * len(STAGES), bytes=total_bytes,
                  manifest=pin(base / 'manifest.json'), births=births, resources=resources, stages=stages, scalars=scalars, fp32=fp32, originals=originals,
                  files={path.relative_to(base).as_posix(): pin(path) for path in sorted(base.rglob('*')) if path.is_file()})
    write(base / 'audit.json', answer)
    print(json.dumps(dict(structural_passed=True, reference_passed=answer['reference_passed'], arrays=answer['arrays'], bytes=total_bytes,
                         max_stage=max(r['max_scaled'] for r in stages), max_scalar=max(r['max_scaled'] for r in scalars),
                         fp32_failed_arrays=sum(r['failed'] > 0 for r in fp32))))

if __name__ == '__main__': main()
