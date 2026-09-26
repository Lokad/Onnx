"""Require exact current products and retain the failed component qualification."""
from protocol import pin, read


def eligibility(reports, spec):
    baseline, models, contracts, qualified, screen, memory = [reports[n] for n in
        ['baseline','models','contracts','qualified','screen','memory']]
    current = spec['identities']['current']
    candidate = spec['identities']['candidate']
    assert current['Lokad.Onnx.dll']['sha256'] == 'f3992f40d889a932cd0d15e0323564db801a308f4b24d1848731af30ba3c19f6'
    assert current['Lokad.Onnx.Data.dll']['sha256'] == 'a8e0b583d6cd4f7e01315fd6cef146721cdf2b8ecb5a8d623a43602658bc42b5'
    assert candidate['Lokad.Onnx.dll']['sha256'] == 'a74acb17524f23be13e81ade871b2b2ffea2afde5bcdd12e339b75e0197edf10'
    assert candidate['Lokad.Onnx.Data.dll']['sha256'] == 'be954dc40376400336167f3153fcf5e388bf6bb0df14af595420a0aa30471b5f'
    assert models['identities'] == dict(selected=current,candidate=candidate)
    assert qualified['built'] == contracts['measured'] == current
    assert contracts['built'] == candidate
    assert screen['products'] == memory['products'] == spec['identities']
    assert baseline['performance']['baseline_valid']
    assert qualified['root_source_verified'] and qualified['package']['passed'] and qualified['consumer']['passed']
    inventory = contracts['inventory']
    assert inventory['passed'] and inventory['only_public_pad_changed'] and inventory['original_padcore_exact']
    assert inventory['helper_matches_reviewed_original'] and inventory['public_surface_equal']
    assert inventory['existing_implementation_flags_equal'] and inventory['generated_names_exact']
    assert (inventory['core_existing_methods'],inventory['core_unchanged_methods'],inventory['data_unchanged_methods']) == (3281,3280,697)
    assert inventory['public_pad']['call_targets_changed'] == 4
    assert [(v['passed'],v['skipped'],v['census_exact'],v['avx512_disabled']) for v in contracts['suites'].values()] == [(6,0,True,False),(6,0,True,True)]
    assert not screen['admitted'] and len(screen['rows']) == 12 and all(r['passed'] for r in screen['rows'])
    failures = [r for r in screen['controls'] if not r['passed']]
    assert len(failures) == 6 and failures == spec['failed_component_controls']
    assert [(r['role'],r['case']) for r in failures] == [('candidate',i) for i in [0,1,2,5,6,8]]
    assert memory['diagnostic_only'] and not memory['admitted'] and memory['memory']['passed']
    assert not contracts['root_product_changed'] and not spec['release_admitted']


def verify(base, spec):
    assert set(spec['prerequisites']) == {'baseline','models','contracts','qualified','screen','memory'}
    reports = {}
    for name, wanted in spec['prerequisites'].items():
        folder = base/'evidence'/name
        proof = read(folder/'closed.json')
        assert pin(folder/'closed.json') == wanted['closed'] and proof['passed']
        assert pin(folder/'analysis.json') == wanted['analysis'] == proof['files']['analysis.json']
        reports[name] = read(folder/'analysis.json')
        assert reports[name]['passed']
        if name in ['screen','memory']:
            assert not proof['admitted']
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
