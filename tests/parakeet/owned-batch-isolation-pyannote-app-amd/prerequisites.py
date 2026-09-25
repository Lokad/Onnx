"""Require actual release/candidate correctness and direct application admission."""
from protocol import ROLES, pin, read
from graph_prerequisite import verify_bundle


def verify(base, spec):
    graph = verify_bundle(base, spec)
    assert set(spec['prerequisites']) == {'models', 'parakeet', 'shared', 'parakeet-release', 'equality', 'parakeet-app', 'baseline'}
    reports = {}
    for name, wanted in spec['prerequisites'].items():
        folder = base/'evidence'/name
        assert pin(folder/'closed.json') == wanted['closed']
        proof = read(folder/'closed.json')
        assert proof['passed'] and proof['analysis'] == wanted['analysis'] == pin(folder/'analysis.json')
        reports[name] = read(folder/'analysis.json')
        assert reports[name]['passed']
    products = spec['identities']
    selected, candidate = products['selected'], products['candidate']
    assert selected['Lokad.Onnx.dll']['sha256'] == 'f95a13c58354bf07f3b7926b72903c18b1a560a56673297cb9fe001d3541b592'
    assert selected['Lokad.Onnx.Data.dll']['sha256'] == 'a893952f583f680ad9dcf677a32b9393541814396a35c6a4eb18a1e7325cbae1'
    assert candidate['Lokad.Onnx.dll']['sha256'] == 'e07a45189b348fe55ce76300415c6c35ba6a2bc0d224f1fc13b0b92c303bccba'
    assert candidate['Lokad.Onnx.Data.dll']['sha256'] == '01e9e7842f5e9861de3d6dc737db947c8a38f1a07038b403d5482ec676e810f1'
    assert reports['parakeet-release']['identities']['selected'] == selected
    assert reports['parakeet']['identities']['candidate'] == candidate
    assert reports['baseline']['identities']['candidate'] == selected
    assert reports['baseline']['consumers'] == spec['consumers']
    for name in ['models', 'shared']:
        assert reports[name]['identities'] == products
    application = reports['parakeet-app']
    assert application['identities'] == dict(current=selected, candidate=candidate)
    assert read(base/'evidence/parakeet-app/closed.json')['admitted'] and application['performance']['admitted']
    assert (application['timing_requests'], application['warmup'], application['measured']) == (480, 120, 360)
    assert len(application['performance']['controls']) == 63 and all(row['passed'] for row in application['performance']['controls'])
    assert len(application['performance']['gates']) == 21 and all(row['passed'] for row in application['performance']['gates'])
    corpus_gate, = [row for row in application['performance']['gates'] if row['name'] == 'corpus-at-least-three-percent-gain']
    assert corpus_gate['limit'] == .97
    equality = reports['equality']
    assert spec['prerequisites']['equality']['closed']['sha256'] == '1e77ac84037a188869ccf89c8155545fd56f47fa033bab01b077b174535802bc'
    assert equality['identities'] == application['identities']
    assert equality['source_closures'] == dict(current=spec['prerequisites']['parakeet-release']['closed'], candidate=spec['prerequisites']['parakeet']['closed'])
    assert not equality['inference_performed'] and not equality['performance_scored']
    assert [row['isa'] for row in equality['native']] == [row['isa'] for row in equality['public']] == ['512', '256']
    for row in equality['native']:
        assert (row['arrays'], row['values'], len(row['comparisons'])) == (784, 3090494, 784)
        assert all(comparison['bit_identical'] for comparison in row['comparisons'])
    assert all(row['requests'] == 20 and row['complete_results_exact'] for row in equality['public'])
    for role in ROLES:
        pyannote = reports['models']['results'][role]
        assert pyannote['passed'] and (pyannote['arrays'], pyannote['values'], pyannote['public_calls']) == (18, 2917107, 16)
        if role == 'candidate':
            assert pyannote['complete_public_results_exact'] and pyannote['complete_public_semantics_exact']
            assert all(row['bit_identical'] for row in pyannote['comparisons'] if row['reference'] == 'production')
        report = reports['parakeet-release' if role == 'selected' else 'parakeet']
        for isa in ['512', '256']:
            native = report['results'][role+'-native-'+isa]
            public = report['results'][role+'-public-'+isa]
            assert native['passed'] and native['native']['numeric_gate_passed'] and native['native']['application_passed']
            assert (native['native']['arrays'], native['native']['values']) == (784, 3090494)
            assert not native['native']['failures'] and public['passed'] and public['public_requests'] == 20
        shared = [reports['shared']['results'][role+'-'+mode] for mode in ['shared', 'e5']]
        assert all(row['passed'] for row in shared)
        assert sum(row['arrays'] for row in shared) == 166 and sum(row['values'] for row in shared) == 5000814
        if role == 'candidate':
            assert all(row['exact_selected'] for result in shared for row in result['rows'])
    return dict(passed=True, retained=spec['prerequisites'], graph_qualification=graph)
