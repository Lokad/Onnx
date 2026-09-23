"""Verify current product prerequisites and every complete Parakeet timing request."""
import importlib.util
from fractions import Fraction
from protocol import pin, read


def records_protocol(base):
    spec = importlib.util.spec_from_file_location('original_audio_records', base/'runtime/protocol.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prereqs(base, spec):
    assert set(spec['prerequisites']) == {'application', 'root', 'parakeet'}
    reports = {}
    for name, wanted in spec['prerequisites'].items():
        folder = base/'evidence'/name
        assert pin(folder/'closed.json') == wanted['closed']
        proof = read(folder/'closed.json'); assert proof['passed']
        assert pin(folder/'analysis.json') == wanted['analysis'] == proof['analysis']
        reports[name] = read(folder/'analysis.json'); assert reports[name]['passed']
    current = spec['identities']['current']
    app = reports['application']; root = reports['root']; parakeet = reports['parakeet']
    assert app['identities']['candidate'] == root['measured'] == parakeet['identities']['candidate'] == current
    assert app['performance']['admitted'] and app['prerequisites']['passed']
    assert app['results']['native-parakeet']['passed'] and app['results']['native-parakeet']['requests'] == 20
    assert app['results']['meetings-run']['passed'] and app['results']['meetings-run']['calls'] == 3
    assert app['consumers']['AudioBenchmark'] == spec['consumers']['AudioBenchmark']
    assert root['root_source_verified'] and root['inventory'] == dict(passed=True, core_methods=3179, data_methods=697, public_surface_equal=True)
    assert root['suites']['backend']['passed'] == 3449 and root['suites']['backend']['skipped'] == 41
    assert root['suites']['tensors']['passed'] == 343 and root['suites']['tensors']['skipped'] == 0
    assert all(row['census_exact'] for row in root['suites'].values()) and root['consumer']['passed'] and root['package']['passed']
    native = parakeet['results']['candidate-native']['native']
    assert native['numeric_gate_passed'] and (native['arrays'], native['values']) == (784, 3090494)
    assert len(native['exact_selected_comparisons']) == 784 and all(r['bit_identical'] for r in native['exact_selected_comparisons'])
    public = parakeet['results']['candidate-public']
    assert public['passed'] and public['public_requests'] == 20 and public['complete_selected_results_exact']
    assert pin(base/'evidence/current-public.json') == public['result']
    retained = read(base/'evidence/current-public.json')
    assert retained['core_sha256'] == current['Lokad.Onnx.dll']['sha256'] and retained['data_sha256'] == current['Lokad.Onnx.Data.dll']['sha256']
    assert len(retained['records']) == 20 and retained['held_outputs_unchanged']
    return dict(passed=True, retained=spec['prerequisites'])


def validate(value, manifest, spec, native, original_protocol, reference, manifest_pin):
    original_protocol.validate_records(value, manifest, 'timing')
    assert manifest['family'] == 'parakeet' and len(manifest['cases']) == 20
    assert value['manifest_sha256'] == manifest_pin['sha256']
    assert value['engine'] == ('ort' if native else 'managed') and len(value['records']) == 80
    if native:
        assert value['python_binary'] == spec['interpreter']
        assert value['runner_sha256'] == spec['files']['runtime/native.py']['sha256']
        assert value['adapter_sha256'] == manifest['adapter']['sha256']
        assert value['versions'] == manifest['native_versions'] and value['native_binaries'] == manifest['native_binaries']
        assert value['native_settings'] == dict(provider='CPUExecutionProvider', intra_threads=1, inter_threads=1, sequential=True, graph_optimizations='all', spinning=False)
        assert value['numeric_libraries']
        for path, wanted in value['numeric_libraries'].items(): assert spec['external'][path] == wanted, path
        for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
            assert value['flags'][key] == '1'
    else:
        assert value['runtime'] == '.NET 10.0.8' and value['processor_count'] == 1 and not value['flags']
        assert value['runner_sha256'] == spec['consumers']['AudioBenchmark']['sha256']
        assert all(value[k] == manifest[k] for k in ['core_sha256', 'data_sha256'])
        expected = {r['name']:r['result'] for r in reference['records']}
        assert len(expected) == 20 and set(expected) == {c['name'] for c in manifest['cases']}
        assert all(row['result'] == expected[row['name']] for row in value['records'])
    return dict(passed=True, requests=80, complete_retained_results_exact=not native)


def qualify(base, name, spec):
    folder = base/name/'output'; value = read(folder/'result.json')
    manifest_path = base/'manifests/current-parakeet.json'
    result = validate(value, read(manifest_path), spec, name.endswith('-ort'), records_protocol(base),
                      read(base/'evidence/current-public.json'), pin(manifest_path))
    assert {p.name for p in folder.iterdir()} == {'result.json'} | {f'{i:03}.json' for i in range(80)}
    for index, row in enumerate(value['records']): assert row == read(folder/f'{index:03}.json')
    return dict(**result, result=pin(folder/'result.json'))


def evaluate(table):
    assert len(table) == 21 and len({r['name'] for r in table}) == 21
    assert sum(r['is_corpus'] for r in table) == 1
    corpus = next(r for r in table if r['is_corpus']); assert corpus['audio_seconds'] == 213.265
    def exact(value):
        fraction = Fraction(**value['exact_mean']); assert fraction > 0
        return fraction
    controls = []
    for row in table:
        for role in ['current', 'ort']:
            means = [exact(p) for p in row[role]['processes']]
            assert len(means) == 2 and exact(row[role]) == sum(means)/2
            ratio = max(means)/min(means); limit = Fraction(110 if row['is_corpus'] else 120, 100)
            controls.append(dict(name=row['name'], role=role, process_ratio=float(ratio), limit=float(limit), passed=ratio <= limit))
    assert len(controls) == 42
    stable = all(row['passed'] for row in controls)
    return dict(baseline_valid=stable, controls_passed=stable, controls=controls,
        parity_target_met=stable and exact(corpus['current'])/exact(corpus['ort']) <= Fraction(105,100),
        policy='Current/ORT fresh-process means: corpus max/min <=1.10, each clip <=1.20. All clocks retained; no unchanged retry. Baseline only, no product-change admission.')
