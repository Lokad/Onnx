"""Bind one direct-depthwise change to its numerical and observed mechanism evidence."""
from protocol import pin,read


def verify(base,spec):
    assert set(spec['prerequisites'])=={'baseline','models','contracts','mechanism','selected_app','graphs'}
    reports={}
    for name,wanted in spec['prerequisites'].items():
        folder=base/'evidence'/name;proof=read(folder/'closed.json')
        assert pin(folder/'closed.json')==wanted['closed'] and proof['passed']
        assert pin(folder/'analysis.json')==wanted['analysis']
        recorded=proof['files']['analysis.json'] if name=='graphs' else proof['analysis']
        assert wanted['analysis']==recorded
        reports[name]=read(folder/'analysis.json');assert reports[name]['passed']
    baseline,models,contracts,mechanism,selected,graphs=[reports[n] for n in
        ['baseline','models','contracts','mechanism','selected_app','graphs']]
    current=spec['identities']['current'];candidate=spec['identities']['candidate']
    assert current['Lokad.Onnx.dll']['sha256']=='49901366484570493b7a42028e5c5458f30fe2c1703a01335b68d9a5f9a1fea9'
    assert candidate['Lokad.Onnx.dll']['sha256']=='40260aef7fd93c5153601ec104a87843a2c017a2720fd64c9e24e3460d455749'
    assert current['Lokad.Onnx.Data.dll']==candidate['Lokad.Onnx.Data.dll']
    assert current['Lokad.Onnx.Data.dll']['sha256']=='01e9e7842f5e9861de3d6dc737db947c8a38f1a07038b403d5482ec676e810f1'
    assert models['identities']==dict(selected=current,candidate=candidate)
    assert baseline['performance']['baseline_valid']
    assert selected['identities']['candidate']==current and selected['performance']['admitted']
    assert read(base/'evidence/selected_app/closed.json')['admitted']
    assert len(selected['performance']['controls'])==63 and all(r['passed'] for r in selected['performance']['controls'])
    assert not read(base/'evidence/graphs/closed.json')['admitted']
    assert [r['key'] for r in graphs['performance'] if not r['regression_passed']]==['e5-8tok']
    assert spec['failed_graph_cases']==contracts['failed_graph_cases']
    assert spec['failed_graph_cases'] and not spec['release_admitted']
    assert contracts['product']==candidate and contracts['compiled_review']==pin(base/'evidence/contracts/build-review.json')
    build=read(base/'evidence/contracts/build-review.json')
    assert build['passed'] and build['product']==candidate and build['data_binary_unchanged'] and build['zero_added_warnings']
    assert [len(r['changed']) for r in build['methods']]==[1,0]
    assert len(build['methods'][0]['added'])==4 and not build['methods'][1]['added']
    assert [(s['mode'],s['passed'],s['skipped']) for s in contracts['suites']]==[('normal',8,0),('scalar',8,0)]
    assert all(s['geometry']['geometries']==59 and s['geometry']['checked_values']==57332736 for s in contracts['suites'])
    assert mechanism['diagnostic_only'] and mechanism['instrumented_times_not_scored'] and not mechanism['release_admitted']
    assert mechanism['exact_public_results'] and mechanism['public_requests']==80
    observed=mechanism['observed'];assert observed['every_geometry_exact'] and observed['zero_generic_work']
    assert observed['per_corpus']['direct_batches']==520
    observer_spec=read(base/'evidence/mechanism/spec.json')
    assert observer_spec['before_product']==candidate and observer_spec['source']==mechanism['source']
    source=read(base/'evidence/mechanism/source-prepared.json')
    assert pin(base/'evidence/mechanism/source-prepared.json')==mechanism['source'] and source['baseline']==build['source']
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
