"""Verify current product prerequisites and every complete Parakeet timing request."""
import importlib.util
from fractions import Fraction
from protocol import pin, read


def records_protocol(base):
    spec = importlib.util.spec_from_file_location('original_audio_records', base/'runtime/protocol.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prereqs(base, spec):
    assert set(spec['prerequisites']) == {'baseline','models','build','contracts','residency','component','root'}
    reports={}
    for name,wanted in spec['prerequisites'].items():
        folder=base/'evidence'/name;proof=read(folder/'closed.json')
        assert pin(folder/'closed.json')==wanted['closed'] and proof['passed']
        assert pin(folder/'analysis.json')==wanted['analysis']==proof.get('analysis',proof['files'].get('analysis.json'))
        reports[name]=read(folder/'analysis.json');assert reports[name]['passed']
    current=spec['identities']['current'];candidate=spec['identities']['candidate']
    baseline,models,build,contracts,residency,component,root=[reports[n] for n in ['baseline','models','build','contracts','residency','component','root']]
    assert root['measured']==build['measured']==models['identities']['selected']==current
    assert root['inventory']==dict(passed=True,core_methods=3189,data_methods=697,public_surface_equal=True,implementation_flags_equal=True)
    assert models['identities']['candidate']==build['built']==candidate
    assert baseline['performance']['baseline_valid'] and root['root_source_verified']
    proof=read(base/'evidence/build/closed.json');refusal=read(base/'evidence/build/original-refusal.json')
    assert proof['original_refusal']==pin(base/'evidence/build/original-refusal.json') and not refusal['passed'] and refusal['build_jobs_passed']
    assert refusal['inference_calls']==refusal['performance_calls']==0
    inventory=build['inventory']
    assert inventory['passed'] and inventory['original_core_methods']==3189 and inventory['candidate_core_methods']==3250
    assert inventory['unchanged_core_methods']==3179 and inventory['data_methods']==697
    assert len(inventory['existing_method_changes'])==10 and len(inventory['added_methods'])==61
    assert all(inventory[k] for k in ['existing_flags_exact','public_surface_equal','ordered_projection_exact','existing_panels_exact'])
    assert all(v==0 for v in inventory['new_flags'].values())
    expected=dict(selected=current,candidate=candidate)
    def pairs(value):return {role:{k:p[k] for k in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']} for role,p in value.items()}
    assert contracts['identities']==models['identities']==expected
    assert pairs(residency['identities'])==pairs(component['identities'])==expected
    assert contracts['candidate_passes']==300 and contracts['required_selected_failure']
    for name in ['candidate-tests','candidate-tests-256']:
        row=contracts['contracts'][name]
        assert row['passed'] and (row['tests'],row['passed_tests'],row['expected_failures'],row['skipped'])==(150,150,0,0)
    assert residency['decoder_executions']==1520 and residency['exact_decoder_arrays']==6080
    assert residency['complete_calls']==3040 and residency['exact_component_arrays']==9120 and residency['max_native_error']<=1e-4
    assert len(residency['reviews'])==4 and len({r['output_digest'] for r in residency['reviews']})==1
    for row in residency['reviews']:
        assert row['prepared_routes_proven']==(3 if row['role']=='candidate' else 0)
        assert row['residencies'][0]['bytes']==(51461120 if row['role']=='candidate' else 25246720)
    assert component['complete_call_clocks']==30400 and component['exact_output_arrays']==91200
    performance=component['performance']
    assert not performance['controls_passed'] and not performance['admitted']
    assert len(performance['controls'])==84 and sum(not c['passed'] for c in performance['controls'])==40
    assert not read(base/'evidence/component/closed.json')['admitted']
    for report in [build,contracts,residency,component]:
        assert report['source_prepared']['sha256']=='b52a89c1043165de1c376b37fc5307cd003a7c8b76f0f52508c4cdefcc669eab'
        assert not report['root_product_changed']
    for report in [build,contracts,residency]:assert report['no_performance_measurement']
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
