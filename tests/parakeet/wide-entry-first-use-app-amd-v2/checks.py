"""Verify current product prerequisites and every complete Parakeet timing request."""
import importlib.util
from fractions import Fraction
from protocol import pin, read


def records_protocol(base):
    spec = importlib.util.spec_from_file_location('original_audio_records', base/'runtime/protocol.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prereqs(base, spec):
    assert set(spec['prerequisites']) == {'baseline','models','build','screen','root','isolated-build','wide-build'}
    reports={}
    for name,wanted in spec['prerequisites'].items():
        folder=base/'evidence'/name;proof=read(folder/'closed.json')
        assert pin(folder/'closed.json')==wanted['closed'] and proof['passed']
        assert pin(folder/'analysis.json')==wanted['analysis']==proof['files']['analysis.json']
        reports[name]=read(folder/'analysis.json');assert reports[name]['passed']
    current=spec['identities']['current'];candidate=spec['identities']['candidate']
    baseline,models,build,screen,root=[reports[n] for n in ['baseline','models','build','screen','root']]
    isolated,wide=reports['isolated-build'],reports['wide-build']
    assert baseline['identities']['current']==root['measured']==models['identities']['selected']==current
    assert root['built']==isolated['measured']
    assert root['inventory']==dict(passed=True,core_methods=3179,data_methods=697,public_surface_equal=True)
    assert isolated['built']==wide['measured'] and wide['built']==build['measured']
    assert models['identities']['candidate']==build['built']==candidate
    assert screen['products']==spec['identities'] and screen['admitted']
    assert baseline['performance']['baseline_valid'] and root['root_source_verified']
    assert isolated['inventory']['original_general_body_exact'] and isolated['inventory']['existing_implementation_flags_equal']
    assert isolated['inventory']['exact_copy_bodies']==4 and isolated['inventory']['call_operands_changed']==4
    assert wide['inventory']['all_other_bodies_exact'] and wide['inventory']['all_flags_exact']
    assert wide['inventory']['removed_upper_row_guard']
    assert build['inventory']['entry_original_exact'] and build['inventory']['shared_flags_exact']
    assert build['inventory']['exact_cloned_bodies']==3 and build['inventory']['call_operands_changed']==4
    assert all(r['inventory']['public_surface_equal'] for r in [isolated,wide,build])
    assert models['consumers']['AudioBenchmark']==spec['consumers']['AudioBenchmark']==baseline['consumers']['AudioBenchmark']
    for role,original in [('current','selected'),('candidate','candidate')]:
        native=models['results'][original+'-native']['native']
        public=models['results'][original+'-public']
        assert native['numeric_gate_passed'] and (native['arrays'],native['values'])==(784,3090494)
        if role=='candidate':
            assert len(native['exact_selected_comparisons'])==784 and all(r['bit_identical'] for r in native['exact_selected_comparisons'])
            assert public['complete_selected_results_exact']
        assert public['passed'] and public['public_requests']==20
        assert pin(base/'evidence'/(role+'-public.json'))==public['result']
        reference=read(base/'evidence'/(role+'-public.json'))
        assert len(reference['records'])==20 and reference['held_outputs_unchanged']
        assert reference['core_sha256']==spec['identities'][role]['Lokad.Onnx.dll']['sha256']
        assert reference['data_sha256']==spec['identities'][role]['Lokad.Onnx.Data.dll']['sha256']
    return dict(passed=True,retained=spec['prerequisites'])


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
    role=name.split('-')[-1];assert role in ['current','candidate','ort']
    reference_role='current' if role=='ort' else role
    manifest_path=base/'manifests'/(reference_role+'-parakeet.json')
    result = validate(value, read(manifest_path), spec, name.endswith('-ort'), records_protocol(base),
                      read(base/'evidence'/(reference_role+'-public.json')), pin(manifest_path))
    assert {p.name for p in folder.iterdir()} == {'result.json'} | {f'{i:03}.json' for i in range(80)}
    for index, row in enumerate(value['records']): assert row == read(folder/f'{index:03}.json')
    return dict(**result, result=pin(folder/'result.json'))


def evaluate(table):
    assert len(table)==21 and len({r['name'] for r in table})==21 and sum(r['is_corpus'] for r in table)==1
    corpus=next(r for r in table if r['is_corpus']);assert corpus['audio_seconds']==213.265
    def exact(value):
        fraction=Fraction(**value['exact_mean']);assert fraction>0;return fraction
    controls=[];gates=[];ratios=[]
    for row in table:
        for role in ['current','candidate','ort']:
            means=[exact(p) for p in row[role]['processes']]
            assert len(means)==2 and exact(row[role])==sum(means)/2
            ratio=max(means)/min(means);limit=Fraction(110 if row['is_corpus'] else 120,100)
            controls.append(dict(name=row['name'],role=role,process_ratio=float(ratio),limit=float(limit),passed=ratio<=limit))
        ratio=exact(row['candidate'])/exact(row['current'])
        if not row['is_corpus']:
            gates.append(dict(name=row['name'],candidate_over_current=float(ratio),limit=1.05,passed=ratio<=Fraction(105,100)))
        ratios.append(dict(name=row['name'],candidate_over_current=float(ratio)))
    for role in ['current','candidate','ort']:
        assert exact(corpus[role])==sum(exact(r[role]) for r in table if not r['is_corpus'])
    ratio=exact(corpus['candidate'])/exact(corpus['current'])
    gates.append(dict(name='corpus-at-least-three-percent-gain',candidate_over_current=float(ratio),limit=.97,passed=ratio<=Fraction(97,100)))
    assert len(controls)==63 and len(gates)==21
    stable=all(r['passed'] for r in controls)
    return dict(admitted=stable and all(r['passed'] for r in gates),controls_passed=stable,controls=controls,gates=gates,ratios=ratios,
        parity_target_met=stable and exact(corpus['candidate'])/exact(corpus['ort'])<=Fraction(105,100),
        policy='Six fresh processes; corpus max/min <=1.10 and each clip <=1.20 for all three engines. Candidate corpus >=3% gain and no clip >5% slower. Every clock retained; no unchanged retry.')
