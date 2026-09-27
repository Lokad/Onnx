"""Require the exact qualified ownership policy and complete unchanged model gates."""
from protocol import pin,read


def eligibility(reports,spec,compatible):
    baseline,models,control,qualified,census = [reports[n] for n in
        ['baseline','models','control','qualified','census']]
    current,candidate = spec['identities']['current'],spec['identities']['candidate']
    assert current['Lokad.Onnx.dll']['sha256'] == 'a6f7d9f9abf0dc3c10a1b566443a22352c8cc84f05a198766325fef2b09850a4'
    assert candidate['Lokad.Onnx.dll']['sha256'] == 'ee5218dbab0a970b0f20f273fc2a839bb9b96e9d5438791529eb11c731d66859'
    assert current['Lokad.Onnx.Data.dll'] == candidate['Lokad.Onnx.Data.dll']
    assert current['Lokad.Onnx.Data.dll']['sha256'] == '1ba343fd8b00fd85bddb33c57aaebf4217431955c40bbe45576448467e59c99f'
    assert models['identities'] == dict(selected=current,candidate=candidate)
    assert qualified['built'] == current and control['product'] == candidate
    assert [(r['mode'],r['passed'],r['skipped']) for r in control['suites']] == [('normal',93,0),('256',93,0),('scalar',26,0)]
    assert baseline['performance']['baseline_valid']
    assert qualified['root_source_verified'] and qualified['package']['passed'] and qualified['consumer']['passed']
    assert qualified['inventory']['method_bodies_equal'] and qualified['inventory']['implementation_flags_equal']
    for key in ['passed','original_public_bindings_preserved','all_data_methods_exact','all_original_method_flags_preserved','no_consumer_or_product_build']:
        assert compatible[key]
    assert compatible['selected'] == current and compatible['candidate'] == candidate
    assert compatible['underlying_methods_reconciled'] == 3985
    assert [(r['unchanged'],len(r['changed'])) for r in compatible['compiled_scope']] == [(3287,1),(697,0)]
    changed=compatible['compiled_scope'][0]['changed']
    assert {'::'.join(n.split('::')[:2]) for n in changed} == {'Lokad.Onnx.ComputationalGraph::PrepareOwnedMatMulWeights'}
    assert compatible['focused_contracts'] == spec['prerequisites']['control']['closed']
    assert compatible['actual_model_census'] == spec['prerequisites']['census']['closed']
    assert census['product'] == candidate and census['contracts'] == spec['prerequisites']['control']['closed']
    assert census['original_compiled_review'] == control['compiled_review']
    assert [(r['mode'],r['result']['owned_count'],r['result']['retained_maps']) for r in census['modes']] == [('512',179,37),('256',179,37)]
    assert census['expected_added_attention_weights'] == 92 and not census['application_scored']
    for mode in census['modes']:
        result=mode['result']
        assert result['passed'] and result['owned_bytes']==1845493760 and result['retained_clone_bytes']==268435456
        assert result['original_weight_count']==216 and result['logical_hashes_exact'] and result['identities_preserved']
    assert spec['failed_component_controls'] == [] and not spec['release_admitted']


def verify(base,spec):
    assert set(spec['prerequisites']) == {'baseline','models','control','qualified','census'}
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
