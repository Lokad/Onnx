"""Independently sum every controlled energy and reconstruct all feature values."""
import argparse, re
from shared import *
from calculation import ideal_decimal, ideal_double, scalar_calculate

def main():
    p = argparse.ArgumentParser(); p.add_argument('--artifact', required=True); a = p.parse_args()
    base = Path(a.artifact).resolve(); psutil_module().Process().cpu_affinity([0])
    spec = read(base / 'manifest.json'); verify(spec['files']); state = read(base / 'run.json')
    assert spec['control_limit'] == CONTROL_LIMIT and spec['original_limit'] == ORIGINAL_LIMIT and spec['limits'] == LIMITS
    assert state['complete'] and state['code'] == 0 and not state.get('error') and state['manifest'] == pin(base / 'manifest.json')
    assert spec['tests'] == pin(base / 'tests.json') and read(base / 'tests.json')['code'] == 0
    assert [r['name'] for r in state['runs']] == ['build', 'five', 'dialogue', 'analysis']
    assert all(a['ended'] <= b['started'] for a, b in zip(state['runs'], state['runs'][1:]))
    births = [state['supervisor']]; resources = []
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0 and not run.get('error') and 0 < run['seconds'] < LIMITS['seconds']
        assert run['preflight_available'] >= LIMITS['preflight'] and run['preflight_disk'] >= LIMITS['disk']
        assert run['members'][str(run['child']['pid'])] == run['child']['birth']
        births += [dict(pid=int(pid), birth=birth) for pid, birth in run['members'].items()]
        samples = [json.loads(v) for v in (base / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == run['samples'] and samples
        previous = 0.
        for row in samples:
            assert previous <= row['seconds'] <= run['seconds'] and row['seconds'] - previous < 10; previous = row['seconds']
            assert row['available'] >= LIMITS['available'] and sum(v['rss'] for v in row['members']) < LIMITS['rss']
            for member in row['members']:
                assert member['affinity'] == [0] and run['members'][str(member['pid'])] == member['birth']
        assert run['seconds'] - previous < 10
        assert not (base / (run['name'] + '.stderr')).read_text().strip()
        resources.append(dict(stage=run['name'], seconds=run['seconds'], samples=len(samples),
                              peak_group_rss=max(sum(v['rss'] for v in row['members']) for row in samples)))
    assert all(absent(birth) for birth in births)
    captures = {}
    for name, wanted in spec['captures'].items():
        value = read(base / name / 'result.json'); runtime = value['runtime']
        assert value['complete'] and Path(value['data']['file']).resolve() == ROOT / wanted['directory'] / 'Lokad.Onnx.Data.dll'
        assert value['data']['sha256'] == wanted['data']['sha256'] and runtime['framework'] == '10.0.12'
        assert runtime['pid'] == next(r['child']['pid'] for r in state['runs'] if r['name'] == name)
        assert runtime['affinity'] == runtime['processor_count'] == 1 and runtime['native_ort_loaded'] is False
        assert [(r['name'], r['count']) for r in value['fields']] == [('Window', 400), ('MelWeights', 20480)]
        captures[name] = {}
        for row in value['fields']:
            assert row['read_only'] and pin(base / name / row['file'])['sha256'] == row['sha256']
            array = np.fromfile(base / name / row['file'], dtype='<f4'); assert array.size == row['count'] and np.isfinite(array).all()
            captures[name][row['name']] = array
        for row in value['loaded']:
            assert Path(row['file']).resolve().parent == ROOT / wanted['directory'] and spec['files'][rel(row['file'])]['sha256'] == row['sha256']
        assert (base / (name + '.stdout')).read_text().strip() == 'COEFFICIENT-CAPTURE-PASS window=400 mel=20480'
    for name, old_name in [('Window', 'window.f32'), ('MelWeights', 'mel.f32')]:
        assert np.array_equal(captures['five'][name].view(np.uint32), captures['dialogue'][name].view(np.uint32))
        assert np.array_equal(captures['five'][name].view(np.uint32), np.fromfile(ROOT / spec['diagnostic'] / old_name, dtype='<u4'))
    weights = dict(native=np.load(ROOT / spec['native_mel']).astype(np.float64), managed=captures['dialogue']['MelWeights'].reshape(80, 256).astype(np.float64), ideal=ideal_decimal())
    double = ideal_double(); assert np.max(np.abs(double - weights['ideal'])) <= CONTROL_LIMIT
    for name, expected in weights.items(): np.testing.assert_array_equal(np.load(base / 'analysis' / (name + '-weights.npy')), expected)
    np.testing.assert_array_equal(np.load(base / 'analysis/ideal-double-weights.npy'), double)
    result = read(base / 'analysis/result.json'); runtime = result['runtime']; run = state['runs'][-1]
    assert result['complete'] and result['manifest'] == state['manifest']
    assert runtime['pid'] == run['child']['pid'] and runtime['birth'] == run['child']['birth'] and runtime['affinity'] == [0]
    assert runtime['numpy'] == '2.2.4' and runtime['blas_threads'] == 1 and runtime['native_ort_loaded'] is False
    for path, wanted in runtime['libraries'].items(): assert spec['numeric'][path] == wanted and pin(path) == wanted
    assert [r['name'] for r in result['records']] == [c['name'] for c in spec['cases']]
    assert (base / 'analysis.stdout').read_text().splitlines() == [c['name'] for c in spec['cases']]
    comparisons = []; controls = []; scalars = []; decompositions = []; arrays = 0; data_bytes = 0
    for case, record in zip(spec['cases'], result['records']):
        power = np.load(WINDOWS / 'numpy' / case['name'] / 'power.npy'); saved = {}
        assert list(record['rows']) == list(weights)
        for setting, coefficients in weights.items():
            expected = scalar_calculate(power, coefficients); saved[setting] = {}
            for stage, row in record['rows'][setting].items():
                path = base / 'analysis' / case['name'] / (setting + '-' + stage + '.npy')
                assert pin(path) == row['pin']; actual = np.load(path)
                assert actual.dtype == np.float64 and list(actual.shape) == row['shape'] == list(expected[stage].shape)
                scalars.append(dict(name=case['name'], setting=setting, stage=stage, **metric(actual, expected[stage], CONTROL_LIMIT)))
                if setting == 'native':
                    prior = np.load(WINDOWS / 'numpy' / case['name'] / (stage + '.npy'))
                    controls.append(dict(name=case['name'], stage=stage, **metric(actual, prior, CONTROL_LIMIT)))
                saved[setting][stage] = actual; arrays += 1; data_bytes += actual.nbytes
        actual = baseline(case['baselines']['windows-default'])
        np.testing.assert_array_equal(actual.view(np.uint32), baseline(case['baselines']['windows-preferred']).view(np.uint32))
        for variant in ['windows-default', 'windows-preferred']:
            for setting in weights:
                comparisons.append(dict(name=case['name'], corpus=case['corpus'], variant=variant, setting=setting,
                                        **metric(actual, saved[setting]['features'], ORIGINAL_LIMIT)))
        control = saved['native']['features']; coefficient = saved['managed']['features'] - control
        total = actual.astype(np.float64) - control; remaining = actual - saved['managed']['features']
        assert np.max(np.abs(total - coefficient - remaining)) < 1e-13
        vectors = base / 'analysis' / case['name']
        for name, value in [('total', total), ('coefficient', coefficient), ('remaining', remaining)]:
            with (vectors / (name + '.npy')).open('xb') as stream: np.save(stream, value, allow_pickle=False)
        decompositions.append(dict(name=case['name'], total_l2=float(np.linalg.norm(total)), coefficient_l2=float(np.linalg.norm(coefficient)),
                                   remaining_l2=float(np.linalg.norm(remaining)), coefficient_band_max=np.abs(coefficient).max(axis=(0, 1)).tolist(),
                                   remaining_band_max=np.abs(remaining).max(axis=(0, 1)).tolist()))
    assert arrays == 288 and len(comparisons) == 192
    for corpus, failed in [('five', 1), ('dialogue', 36)]:
        assert sum(r['failed'] for r in comparisons if r['corpus'] == corpus and r['variant'] == 'windows-default' and r['setting'] == 'native') == failed
    log = (base / 'build.stdout').read_text(); assert re.search(r'0 Error\(s\)', log)
    warnings = sorted(set(re.findall(r'^.*warning [A-Z]+\d+.*$', log, re.MULTILINE)))
    answer = dict(structural_passed=True, controls_passed=all(r['failed'] == 0 for r in controls), scalar_passed=all(r['failed'] == 0 for r in scalars),
                  manifest=pin(base / 'manifest.json'), arrays=arrays, bytes=data_bytes, births=births, resources=resources, warnings=warnings,
                  formula_maximum=float(np.max(np.abs(weights['ideal'] - double))),
                  coefficients={name: metric(value, weights['ideal'], CONTROL_LIMIT) for name, value in weights.items()},
                  managed_native_weights=metric(weights['managed'], weights['native'], CONTROL_LIMIT),
                  controls=controls, scalars=scalars, comparisons=comparisons, decompositions=decompositions,
                  files={path.relative_to(base).as_posix(): pin(path) for path in sorted(base.rglob('*')) if path.is_file()})
    write(base / 'audit.json', answer)
    print(json.dumps({k: v for k, v in answer.items() if k in ['structural_passed', 'controls_passed', 'scalar_passed', 'arrays', 'bytes', 'warnings', 'formula_maximum']}))

if __name__ == '__main__': main()
