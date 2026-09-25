"""Bind the actual release and relocation to complete retained correctness."""
from protocol import pin, read


def verify(base, spec):
    assert set(spec['prerequisites']) == {'baseline', 'models', 'release_models', 'contracts', 'graphs', 'parent_app', 'equality'}
    reports = {}
    for name, wanted in spec['prerequisites'].items():
        folder = base/'evidence'/name
        proof = read(folder/'closed.json')
        assert proof['passed'] and pin(folder/'closed.json') == wanted['closed']
        assert pin(folder/'analysis.json') == wanted['analysis']
        assert wanted['analysis'] == (proof['files']['analysis.json'] if name == 'graphs' else proof['analysis'])
        reports[name] = read(folder/'analysis.json')
        assert reports[name]['passed']
    baseline, models, release, contracts, graphs, parent, equality = [reports[n] for n in
        ['baseline', 'models', 'release_models', 'contracts', 'graphs', 'parent_app', 'equality']]
    current, candidate = spec['identities']['current'], spec['identities']['candidate']
    assert current == release['identities']['selected']
    assert candidate == models['identities']['candidate'] == contracts['product']
    assert current['Lokad.Onnx.dll']['sha256'] == 'f95a13c58354bf07f3b7926b72903c18b1a560a56673297cb9fe001d3541b592'
    assert current['Lokad.Onnx.Data.dll']['sha256'] == 'a893952f583f680ad9dcf677a32b9393541814396a35c6a4eb18a1e7325cbae1'
    assert candidate['Lokad.Onnx.dll']['sha256'] == 'e07a45189b348fe55ce76300415c6c35ba6a2bc0d224f1fc13b0b92c303bccba'
    assert candidate['Lokad.Onnx.Data.dll']['sha256'] == '01e9e7842f5e9861de3d6dc737db947c8a38f1a07038b403d5482ec676e810f1'
    assert baseline['performance']['baseline_valid']
    assert parent['identities'] == dict(current=models['identities']['selected'], candidate=candidate)
    assert parent['performance']['admitted'] and read(base/'evidence/parent_app/closed.json')['admitted']
    assert len(parent['performance']['controls']) == 63 and all(r['passed'] for r in parent['performance']['controls'])
    assert graphs['admitted'] and graphs['all_controls_passed'] and graphs['original_graph_failure_preserved']
    assert read(base/'evidence/graphs/closed.json')['admitted']
    assert graphs['products'] == {r: {'Lokad.Onnx.dll': identity['Lokad.Onnx.dll']} for r, identity in spec['identities'].items()}
    assert len(graphs['performance']) == 8 and all(r['qualified'] for r in graphs['performance'])
    assert equality['identities'] == spec['identities']
    assert equality['source_closures'] == dict(current=spec['prerequisites']['release_models']['closed'], candidate=spec['prerequisites']['models']['closed'])
    assert not equality['inference_performed'] and not equality['performance_scored']
    assert len(equality['native']) == len(equality['public']) == 2
    for result in equality['native']:
        assert (result['arrays'], result['values']) == (784, 3090494)
        assert len(result['comparisons']) == 784 and all(r['bit_identical'] for r in result['comparisons'])
    assert all(r['requests'] == 20 and r['complete_results_exact'] for r in equality['public'])
    assert spec['failed_graph_cases'] == [] and not spec['release_admitted']
    assert contracts['compiled_review'] == pin(base/'evidence/contracts/build-review.json')
    build = read(base/'evidence/contracts/build-review.json')
    assert build['passed'] and build['product'] == candidate and build['release_dispatcher_restored'] and build['zero_added_warnings']
    assert [s['passed'] for s in contracts['suites']] == [64, 25]
    for report in [models, release]:
        assert report['consumers']['AudioBenchmark'] == spec['consumers']['AudioBenchmark'] == baseline['consumers']['AudioBenchmark']
    for role, report, original in [('current', release, 'selected'), ('candidate', models, 'candidate')]:
        for isa in ['512', '256']:
            native_result = report['results'][original+'-native-'+isa]
            native = native_result['native']
            public = report['results'][original+'-public-'+isa]
            assert native_result['passed'] and native['audit_consistent'] and native['application_passed']
            assert native['numeric_gate_passed'] and not native['failures'] and (native['arrays'], native['values']) == (784, 3090494)
            assert public['passed'] and public['public_requests'] == 20
        public = report['results'][original+'-public-512']
        assert pin(base/'evidence'/(role+'-public.json')) == public['result']
        reference = read(base/'evidence'/(role+'-public.json'))
        assert len(reference['records']) == 20 and reference['held_outputs_unchanged']
        assert reference['core_sha256'] == spec['identities'][role]['Lokad.Onnx.dll']['sha256']
        assert reference['data_sha256'] == spec['identities'][role]['Lokad.Onnx.Data.dll']['sha256']
        assert reference['flags'] == {} and reference['runner_sha256'] == spec['consumers']['AudioBenchmark']['sha256']
    return dict(passed=True, retained=spec['prerequisites'])
