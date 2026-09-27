"""Require complete correctness and preserve the diagnosed component failure."""
from protocol import pin, read


def eligibility(reports, spec):
    baseline, models, control, qualified, screen, diagnosis = [reports[n] for n in
        ['baseline', 'models', 'control', 'qualified', 'screen', 'diagnosis']]
    current, candidate = spec['identities']['current'], spec['identities']['candidate']
    assert current['Lokad.Onnx.dll']['sha256'] == '0d224bcff591563816d64c3a3cc51f7b7cd38f5a1d3c523b05c699a9b82b6b97'
    assert candidate['Lokad.Onnx.dll']['sha256'] == 'ad97b4ad632b3306ea3a39d14549ed98540bc8c2e85953a68d89b453d3ed2fdc'
    assert current['Lokad.Onnx.Data.dll']['sha256'] == 'a37453929e6103a187f9849d3040a663fa9d7dd85561e4f6f14d639fe38935d5'
    assert candidate['Lokad.Onnx.Data.dll']['sha256'] == '29b3f633eec461b1113d23d2626952bddff1a628bdba87bb865ef52b6da6dc3d'
    assert models['identities'] == dict(selected=current, candidate=candidate)
    assert qualified['built'] == control['baseline'] == current and control['candidate'] == candidate
    assert baseline['performance']['baseline_valid']
    assert qualified['root_source_verified'] and qualified['package']['passed'] and qualified['consumer']['passed']
    assert qualified['inventory']['method_bodies_equal'] and qualified['inventory']['implementation_flags_equal']
    assert not control['contract_regression_found'] and not control['original_campaign_passed']
    compiled = control['compiled']
    assert compiled['passed'] and compiled['public_surface_equal'] and compiled['flags_equal'] and compiled['attributes_equal']
    assert [(r['unchanged'], len(r['changed']), len(r['added'])) for r in compiled['methods']] == [(3282, 2, 2), (697, 0, 0)]
    assert control['projection_hashes_equal_across_modes']
    for role, expected in [('selected', current), ('candidate', candidate)]:
        assert {name:screen['identities'][role][name] for name in expected} == expected
        assert screen['identities'][role+'fallback'] == screen['identities'][role]
    performance = screen['performance']
    assert not performance['admitted'] and not performance['controls_passed']
    failures = [r for r in performance['controls'] if not r['passed']]
    assert len(failures) == 100 and failures == spec['failed_component_controls']
    assert len(performance['controls']) == 168 and len(performance['gates']) == 28
    assert all(r['passed'] for r in performance['gates'])
    assert diagnosis['diagnostic_only'] and not diagnosis['screen_rescored'] and diagnosis['lost'] == 0
    assert {name:diagnosis['product'][name] for name in current} == current
    assert not spec['release_admitted']


def verify(base, spec):
    assert set(spec['prerequisites']) == {'baseline', 'models', 'control', 'qualified', 'screen', 'diagnosis'}
    reports = {}
    for name, wanted in spec['prerequisites'].items():
        folder = base/'evidence'/name; proof = read(folder/'closed.json')
        assert pin(folder/'closed.json') == wanted['closed'] and proof['passed']
        assert pin(folder/'analysis.json') == wanted['analysis'] == proof['files']['analysis.json']
        reports[name] = read(folder/'analysis.json')
        if name != 'control': assert reports[name]['passed']
        if name == 'screen': assert not proof['admitted']
    eligibility(reports, spec)
    baseline, models = reports['baseline'], reports['models']
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
