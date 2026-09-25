"""Verify current product prerequisites and every complete Parakeet timing request."""
import importlib.util
from fractions import Fraction
from protocol import pin, read


def records_protocol(base):
    spec = importlib.util.spec_from_file_location('original_audio_records', base/'runtime/protocol.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prereqs(base, spec):
    assert set(spec['prerequisites'])=={'baseline','models','contracts','profile','root','release_app'}
    reports={}
    for name,wanted in spec['prerequisites'].items():
        folder=base/'evidence'/name;proof=read(folder/'closed.json')
        assert pin(folder/'closed.json')==wanted['closed'] and proof['passed']
        assert pin(folder/'analysis.json')==wanted['analysis']==proof['analysis']
        reports[name]=read(folder/'analysis.json');assert reports[name]['passed']
    current=spec['identities']['current'];candidate=spec['identities']['candidate']
    baseline,models,contracts,root=[reports[n] for n in ['baseline','models','contracts','root']]
    assert root['measured']==models['identities']['selected']==current
    assert root['inventory']==dict(passed=True,core_methods=3253,data_methods=697,public_surface_equal=True,implementation_flags_equal=True)
    assert models['identities']['candidate']==candidate
    assert baseline['performance']['baseline_valid'] and root['root_source_verified']
    assert current['Lokad.Onnx.dll']['sha256']=='f95a13c58354bf07f3b7926b72903c18b1a560a56673297cb9fe001d3541b592'
    assert candidate['Lokad.Onnx.dll']['sha256']=='49c3a958850d3e57daa2b7e29e6bd15ff9fc4f27af098d8ce44d8f20b9e065e8'
    assert current['Lokad.Onnx.Data.dll']==candidate['Lokad.Onnx.Data.dll']
    assert current['Lokad.Onnx.Data.dll']['sha256']=='a893952f583f680ad9dcf677a32b9393541814396a35c6a4eb18a1e7325cbae1'
    source_path=base/'evidence/contracts/source-prepared.json';source=read(source_path)
    assert source['passed'] and len(source['source'])==428 and len(source['before'])==427
    assert source['changed']==['src/Lokad.Onnx/TensorSlice.cs','tests/Lokad.Onnx.Tensors.Tests/SliceDenseConversionTests.cs']
    assert source['root_release']==pin(base/'evidence/root/closed.json') and not source['component_comparison_admitted']
    build_path=base/'evidence/contracts/build-review.json';build=read(build_path)
    assert build['passed'] and contracts['compiled_review']==pin(build_path)
    scope=build['methods'];assert scope['original']==scope['unchanged']==3253
    assert scope['original_flags_equal'] and scope['effective_conversion_signature_equal']
    assert scope['added']=='Lokad.Onnx.TensorSlice`1[T]::ToDenseTensor::Lokad.Onnx.DenseTensor`1[T] ToDenseTensor()'
    assert len(scope['only_declared_addition'])==2 and not scope['declared_surface_equal']
    assert build['no_new_warning'] and not build['product_rebuilt']
    assert contracts['core']==candidate['Lokad.Onnx.dll'] and not contracts['product_rebuilt']
    assert [(s['mode'],s['passed'],s['skipped']) for s in contracts['suites']]==[('512',395,0),('256',395,0)]
    assert contracts['original_methods_unchanged']==3253 and contracts['new_copy_cases']==26
    assert contracts['corrected_test_review']==pin(base/'evidence/contracts/test-review.json')
    release=reports['release_app'];assert release['identities']['candidate']==current
    assert read(base/'evidence/release_app/closed.json')['admitted'] and release['performance']['admitted']
    controls=[r for r in release['performance']['controls'] if r['role']=='candidate']
    assert len(controls)==21 and all(r['passed'] for r in controls)
    profile=reports['profile'];folder=base/'evidence/profile';profile_spec=read(folder/'spec.json')
    assert profile_spec['core']==current['Lokad.Onnx.dll'] and profile_spec['candidate_core']==candidate['Lokad.Onnx.dll']
    assert profile_spec['model_closure']==pin(base/'evidence/models/closed.json')
    assert profile['requests']==160 and profile['clips']==20 and len(profile['frames'])==19
    assert profile['original_request_checks'] and profile['observer_reused_exactly'] and profile['attribution_only']
    assert not profile['application_gain_admitted'] and not profile['overhead_subtracted']
    first=read(folder/'closed.json')['initial']
    assert first==profile_spec['initial']==profile['initial_failure']
    assert profile['split_capture'] and not profile['completed_control_repeated']
    assert first['candidate_never_started'] and first['original_code']==1
    for key,name in [('spec','initial-spec.json'),('state','initial-state.json'),('collection','initial-collection.json')]:
        assert first[key]==pin(folder/name)
    assert read(folder/'initial-collection.json')['terminal'] and read(folder/'initial-collection.json')['code']==1
    assert profile_spec['require_complete_group_improves'] and profile_spec['require_all_24_kernels_improve']
    group=profile['positional'];assert group['passed'] and group['kernels']==group['improved_kernels']==24
    assert group['nodes']==34 and group['shared_ancestors_counted_once'] and group['complete_group_gain']>0
    assert all(r['candidate_seconds']<r['selected_seconds'] for r in group['rows'] if r['projection'])
    assert read(folder/'closed.json')['observer_review']==pin(folder/'observer-review.json')
    assert read(folder/'observer-review.json')['model_closure']==pin(base/'evidence/models/closed.json')
    assert profile['prospective_groups']==profile_spec['groups']==pin(folder/'groups.json')
    assert models['consumers']['AudioBenchmark']==spec['consumers']['AudioBenchmark']==baseline['consumers']['AudioBenchmark']
    assert set(models['results'])=={f'{role}-{mode}-{isa}' for role in ['selected','candidate'] for mode in ['native','public'] for isa in ['512','256']}
    for role,original in [('current','selected'),('candidate','candidate')]:
        for isa in ['512','256']:
            native_result=models['results'][original+'-native-'+isa]
            native=native_result['native'];public=models['results'][original+'-public-'+isa]
            assert native_result['passed'] and native['audit_consistent'] and native['application_passed']
            assert native['numeric_gate_passed'] and not native['failures'] and (native['arrays'],native['values'])==(784,3090494)
            if role=='candidate':
                assert len(native['exact_selected_comparisons'])==784 and all(r['bit_identical'] for r in native['exact_selected_comparisons'])
                assert public['complete_selected_results_exact']
            assert public['passed'] and public['public_requests']==20
        public=models['results'][original+'-public-512']
        assert pin(base/'evidence'/(role+'-public.json'))==public['result']
        reference=read(base/'evidence'/(role+'-public.json'))
        assert len(reference['records'])==20 and reference['held_outputs_unchanged']
        assert reference['core_sha256']==spec['identities'][role]['Lokad.Onnx.dll']['sha256']
        assert reference['data_sha256']==spec['identities'][role]['Lokad.Onnx.Data.dll']['sha256']
        assert reference['flags']=={} and reference['runner_sha256']==spec['consumers']['AudioBenchmark']['sha256']
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
