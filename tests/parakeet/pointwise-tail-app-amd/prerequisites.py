"""Require qualified models and preserve every failed component verdict."""
from protocol import pin,read


def eligibility(reports,spec,compatible):
    baseline,models,control,qualified,screen,diagnosis = [reports[n] for n in
        ['baseline','models','control','qualified','screen','diagnosis']]
    current,candidate = spec['identities']['current'],spec['identities']['candidate']
    assert current['Lokad.Onnx.dll']['sha256'] == '47984318b082710c3a4f57a85b1500d49d7c1236c04b1234477d19e48d11207c'
    assert candidate['Lokad.Onnx.dll']['sha256'] == '7cac67880fa9a4d519ac18e5887f47f48f0f14903bdf74cc6561b45c851e4f27'
    assert current['Lokad.Onnx.Data.dll'] == candidate['Lokad.Onnx.Data.dll']
    assert current['Lokad.Onnx.Data.dll']['sha256'] == 'dd56902f44c640b9226e2e7ee05d1d3e8dc025f59471afb2261194715ca5e311'
    assert models['identities'] == dict(selected=current,candidate=candidate)
    assert qualified['built'] == control['products']['baseline'] == current
    assert control['products']['candidate'] == candidate and control['arithmetic_contract_passed']
    assert baseline['performance']['baseline_valid']
    assert qualified['root_source_verified'] and qualified['package']['passed'] and qualified['consumer']['passed']
    assert qualified['inventory']['method_bodies_equal'] and qualified['inventory']['implementation_flags_equal']
    for key in ['passed','original_public_bindings_preserved','all_data_methods_exact','all_original_method_flags_preserved','no_consumer_or_product_build']:
        assert compatible[key]
    assert compatible['selected'] == current and compatible['candidate'] == candidate
    assert [(r['unchanged'],len(r['changed']),len(r['added'])) for r in compatible['compiled_scope']] == [(3285,1,2),(697,0,0)]
    assert compatible['arithmetic_qualification'] == spec['prerequisites']['control']['closed']
    assert compatible['runtime_diagnosis'] == spec['prerequisites']['diagnosis']['closed']
    assert screen['identities'] == dict(baseline=current,candidate=candidate)
    performance = screen['performance']; assert not performance['component_admitted']
    failures = [r for r in performance['controls'] if not r['passed']]
    assert failures == spec['failed_component_controls'] == compatible['failed_component_controls']
    assert len(failures) == 38 and len(performance['controls']) == 246
    assert len(performance['gates']) == 43 and sum(not r['passed'] for r in performance['gates']) == 1
    assert diagnosis['diagnostic_only'] and not diagnosis['screen_rescored'] and diagnosis['lost'] == 0
    assert diagnosis['product'] == candidate and not spec['release_admitted']


def verify(base,spec):
    assert set(spec['prerequisites']) == {'baseline','models','control','qualified','screen','diagnosis'}
    reports = {}
    for name,wanted in spec['prerequisites'].items():
        folder = base/'evidence'/name; proof = read(folder/'closed.json')
        assert pin(folder/'closed.json') == wanted['closed']
        if name == 'control': assert proof['completed'] and proof['arithmetic_contract_passed']
        elif name == 'screen': assert proof['completed'] and not proof['component_admitted']
        else: assert proof['passed']
        assert pin(folder/'analysis.json') == wanted['analysis'] == proof['files']['analysis.json']
        reports[name] = read(folder/'analysis.json')
        if name not in ['control','screen']: assert reports[name]['passed']
    compatible_path = base/'evidence/models-compatibility.json'
    assert pin(compatible_path) == read(base/'evidence/models/closed.json')['files']['bundle/evidence/compatibility.json']
    eligibility(reports,spec,read(compatible_path))
    baseline,models = reports['baseline'],reports['models']
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
