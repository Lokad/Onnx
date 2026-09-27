"""Bind the exact prepared-row pair without reinterpreting either failed diagnostic."""
from protocol import pin, read


def eligibility(reports, spec):
    baseline, models, contracts, qualified, screen, diagnosis = [reports[n] for n in
        ['baseline', 'models', 'contracts', 'qualified', 'screen', 'diagnosis']]
    current, candidate = spec['identities']['current'], spec['identities']['candidate']
    assert current['Lokad.Onnx.dll']['sha256'] == '65f15a41764660af6166c9a965943b6f28cac0c9505117d0fd4beaa2687f9d03'
    assert candidate['Lokad.Onnx.dll']['sha256'] == 'af19b3b4429a07f7966b5e35ee04e8a31316f45991c300f3683a591caf5e9374'
    assert current['Lokad.Onnx.Data.dll'] == candidate['Lokad.Onnx.Data.dll']
    assert current['Lokad.Onnx.Data.dll']['sha256'] == 'da72ca547191de12a52f09bceeed25619a92a14b1bbff271a9e7d10515f40312'
    assert models['identities'] == dict(selected=current, candidate=candidate)
    assert qualified['built'] == current
    assert baseline['performance']['baseline_valid']
    assert qualified['root_source_verified'] and qualified['package']['passed'] and qualified['consumer']['passed']
    assert qualified['inventory']['method_bodies_equal'] and qualified['inventory']['implementation_flags_equal']
    products = {role: value['Lokad.Onnx.dll'] for role, value in spec['identities'].items()}
    assert contracts['products'] == screen['products'] == diagnosis['products'] == products
    assert contracts['passed'] and not contracts['performance_admitted'] and not contracts['root_product_changed']
    compiled = contracts['compiled']
    assert compiled['passed'] and compiled['public_surface_equal'] and compiled['flags_equal'] and compiled['assembly_metadata_equal']
    assert [(r['unchanged'], len(r['changed']), len(r['added'])) for r in compiled['assemblies']] == [(3281, 2, 1), (697, 0, 0)]
    assert not screen['admitted'] and not screen['release_admitted'] and screen['no_application_score']
    failures = [r for r in screen['controls'] if not r['passed']]
    assert len(failures) == 2 and failures == spec['failed_component_controls']
    assert len(screen['rows']) == 6 and len([r for r in screen['rows'] if not r['passed']]) == 2
    assert diagnosis['diagnostic_only'] and not diagnosis['admitted'] and not diagnosis['release_admitted']
    assert not diagnosis['first_call_explanation_supported'] and not diagnosis['previous_screen_rescored']
    assert diagnosis['individual_intervals'] == 21840 and not spec['release_admitted']


def verify(base, spec):
    assert set(spec['prerequisites']) == {'baseline', 'models', 'contracts', 'qualified', 'screen', 'diagnosis'}
    reports = {}
    for name, wanted in spec['prerequisites'].items():
        folder = base/'evidence'/name; proof = read(folder/'closed.json')
        assert pin(folder/'closed.json') == wanted['closed'] and proof['passed']
        assert pin(folder/'analysis.json') == wanted['analysis'] == proof['files']['analysis.json']
        reports[name] = read(folder/'analysis.json'); assert reports[name]['passed']
        if name in ['screen', 'diagnosis']: assert not proof['admitted']
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
