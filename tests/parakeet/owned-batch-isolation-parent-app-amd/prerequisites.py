"""Bind parent retention to full Parakeet correctness and admitted graph cases."""
from protocol import pin, read


def verify(base, spec):
    assert set(spec['prerequisites']) == {'baseline', 'models', 'contracts', 'parent_app', 'graphs'}
    reports = {}
    for name, wanted in spec['prerequisites'].items():
        folder = base/'evidence'/name
        proof = read(folder/'closed.json')
        assert pin(folder/'closed.json') == wanted['closed'] and proof['passed']
        assert pin(folder/'analysis.json') == wanted['analysis']
        recorded = proof['files']['analysis.json'] if name == 'graphs' else proof['analysis']
        assert wanted['analysis'] == recorded
        reports[name] = read(folder/'analysis.json')
        assert reports[name]['passed']
    baseline, models, contracts, parent, graphs = [reports[n] for n in
        ['baseline', 'models', 'contracts', 'parent_app', 'graphs']]
    current = spec['identities']['current']
    candidate = spec['identities']['candidate']
    assert current['Lokad.Onnx.dll']['sha256'] == '40260aef7fd93c5153601ec104a87843a2c017a2720fd64c9e24e3460d455749'
    assert candidate['Lokad.Onnx.dll']['sha256'] == 'e07a45189b348fe55ce76300415c6c35ba6a2bc0d224f1fc13b0b92c303bccba'
    assert current['Lokad.Onnx.Data.dll'] == candidate['Lokad.Onnx.Data.dll']
    assert candidate['Lokad.Onnx.Data.dll']['sha256'] == '01e9e7842f5e9861de3d6dc737db947c8a38f1a07038b403d5482ec676e810f1'
    assert models['identities'] == dict(selected=current, candidate=candidate)
    assert baseline['performance']['baseline_valid']
    assert parent['identities']['candidate'] == current and parent['performance']['admitted']
    assert read(base/'evidence/parent_app/closed.json')['admitted']
    assert graphs['admitted'] and graphs['all_controls_passed'] and graphs['original_graph_failure_preserved']
    assert read(base/'evidence/graphs/closed.json')['admitted']
    assert len(graphs['performance']) == 8 and all(r['qualified'] for r in graphs['performance'])
    assert graphs['products']['candidate']['Lokad.Onnx.dll'] == candidate['Lokad.Onnx.dll']
    assert spec['failed_graph_cases'] == [] and not spec['release_admitted']
    assert contracts['product'] == candidate and contracts['compiled_review'] == pin(base/'evidence/contracts/build-review.json')
    build = read(base/'evidence/contracts/build-review.json')
    assert build['passed'] and build['product'] == candidate and build['release_dispatcher_restored']
    assert build['zero_added_warnings'] and [len(r['changed']) for r in build['methods']] == [3, 0]
    assert [s['passed'] for s in contracts['suites']] == [64, 25]
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
