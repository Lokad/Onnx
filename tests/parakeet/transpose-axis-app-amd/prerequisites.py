"""Require the exact qualified transpose route and complete unchanged model gates."""
from protocol import pin,read


def eligibility(reports,spec,compatible):
    baseline,models,control,qualified = [reports[n] for n in
        ['baseline','models','control','qualified']]
    current,candidate = spec['identities']['current'],spec['identities']['candidate']
    assert current['Lokad.Onnx.dll']['sha256'] == '4e97e2ae97e534b99281a361790ac35865b3d8c6171293f36109b24a354eec57'
    assert candidate['Lokad.Onnx.dll']['sha256'] == 'c471f5d1ead5889b00b141ce34f2a7689cfe179fbf513da4ce5db84ed01ce277'
    assert current['Lokad.Onnx.Data.dll'] == candidate['Lokad.Onnx.Data.dll']
    assert current['Lokad.Onnx.Data.dll']['sha256'] == 'b04aea50402006ccd67009f6d14efeeb1f4c265dc94a91ebd33d039a206e0fea'
    assert models['identities'] == dict(selected=current,candidate=candidate)
    assert qualified['built'] == current and control['product'] == candidate
    expected=[(mode,suite,38 if suite=='backend' else 83 if mode=='scalar' else 84,0)
        for mode in ['normal','256','scalar','faces-off'] for suite in ['backend','tensors']]
    assert [(r['mode'],r['suite'],r['passed'],r['skipped']) for r in control['suites']] == expected
    assert baseline['performance']['baseline_valid']
    assert qualified['root_source_verified'] and qualified['package']['passed'] and qualified['consumer']['passed']
    assert qualified['inventory']['method_bodies_equal'] and qualified['inventory']['implementation_flags_equal']
    for key in ['passed','original_public_bindings_preserved','all_data_methods_exact','all_original_method_flags_preserved','no_consumer_or_product_build']:
        assert compatible[key]
    assert compatible['selected'] == current and compatible['candidate'] == candidate
    assert compatible['underlying_methods_reconciled'] == 3985
    assert [(r['unchanged'],len(r['changed'])) for r in compatible['compiled_scope']] == [(3287,1),(697,0)]
    changed=compatible['compiled_scope'][0]['changed']
    assert {'::'.join(n.split('::')[:2]) for n in changed} == {'Lokad.Onnx.Tensor`1[T]::TransposeInto'}
    assert compatible['focused_contracts'] == spec['prerequisites']['control']['closed']
    assert spec['failed_component_controls'] == [] and not spec['release_admitted']


def verify(base,spec):
    assert set(spec['prerequisites']) == {'baseline','models','control','qualified'}
    reports = {}
    for name,wanted in spec['prerequisites'].items():
        folder = base/'evidence'/name; proof = read(folder/'closed.json')
        assert pin(folder/'closed.json') == wanted['closed'] and proof['passed']
        assert pin(folder/'analysis.json') == wanted['analysis'] == proof['files']['analysis.json']
        reports[name] = read(folder/'analysis.json')
        assert reports[name]['passed']
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
