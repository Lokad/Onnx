"""Bind complete correctness and application admission to the current padding pair."""
from protocol import ROLES,pin,read
from graph_prerequisite import verify_bundle


def validate(reports,spec):
    assert set(reports)=={'models','parakeet','shared','parakeet-app','baseline'}
    assert all(r['passed'] for r in reports.values())
    products=spec['identities'];selected,candidate=products['selected'],products['candidate']
    assert selected['Lokad.Onnx.dll']['sha256']=='f3992f40d889a932cd0d15e0323564db801a308f4b24d1848731af30ba3c19f6'
    assert selected['Lokad.Onnx.Data.dll']['sha256']=='a8e0b583d6cd4f7e01315fd6cef146721cdf2b8ecb5a8d623a43602658bc42b5'
    assert candidate['Lokad.Onnx.dll']['sha256']=='a74acb17524f23be13e81ade871b2b2ffea2afde5bcdd12e339b75e0197edf10'
    assert candidate['Lokad.Onnx.Data.dll']['sha256']=='be954dc40376400336167f3153fcf5e388bf6bb0df14af595420a0aa30471b5f'
    for name in ['models','parakeet','shared']:assert reports[name]['identities']==products
    assert reports['baseline']['consumers']==spec['consumers']
    assert reports['parakeet']['consumers']['AudioBenchmark']==spec['consumers']['AudioBenchmark']
    assert spec['consumers']['AudioBenchmark']['sha256']=='7eca033a1b986a4cb90621392639d230c95097cb703dd25274fd72d66c5ba4f1'
    assert spec['consumers']['NaturalMeetings']['sha256']=='79e3e7990ba6aa29e42da788277aad41b774ff3b8c3966b18ab1101944d0c0f1'
    application=reports['parakeet-app']
    assert application['identities']==dict(current=selected,candidate=candidate)
    assert application['performance']['admitted']
    assert (application['timing_requests'],application['warmup'],application['measured'])==(480,120,360)
    for key,count in [('controls',63),('gates',21)]:
        assert len(application['performance'][key])==count and all(r['passed'] for r in application['performance'][key])
    corpus,=[r for r in application['performance']['gates'] if r['name']=='corpus-at-least-three-percent-gain']
    assert corpus['limit']==.97
    assert reports['models']['identity_guards']['passed'] and reports['models']['identity_guards']['probes']==4
    for role in ROLES:
        pyannote=reports['models']['results'][role]
        assert pyannote['passed'] and (pyannote['arrays'],pyannote['values'],pyannote['public_calls'])==(18,2917107,16)
        if role=='candidate':
            assert pyannote['complete_public_results_exact'] and pyannote['complete_public_semantics_exact']
            production=[r for r in pyannote['comparisons'] if r['reference']=='production']
            assert len(production)==18 and all(r['bit_identical'] for r in production)
        for isa in ['512','256']:
            native=reports['parakeet']['results'][role+'-native-'+isa]
            public=reports['parakeet']['results'][role+'-public-'+isa]
            assert native['passed'] and native['native']['numeric_gate_passed'] and native['native']['application_passed']
            assert (native['native']['arrays'],native['native']['values'])==(784,3090494)
            assert not native['native']['failures'] and public['passed'] and public['public_requests']==20
            if role=='candidate':
                comparisons=native['native']['exact_selected_comparisons']
                assert len(comparisons)==784 and all(r['bit_identical'] for r in comparisons)
                assert public['complete_selected_results_exact']
        shared=[reports['shared']['results'][role+'-'+mode] for mode in ['shared','e5']]
        assert all(r['passed'] for r in shared)
        assert sum(r['arrays'] for r in shared)==166 and sum(r['values'] for r in shared)==5000814
        if role=='candidate':assert all(r['exact_selected'] for result in shared for r in result['rows'])
    return True


def verify(base,spec):
    graph=verify_bundle(base,spec);reports={}
    for name,wanted in spec['prerequisites'].items():
        folder=base/'evidence'/name
        assert pin(folder/'closed.json')==wanted['closed']
        proof=read(folder/'closed.json')
        assert proof['passed'] and proof['analysis']==wanted['analysis']==pin(folder/'analysis.json')
        if name=='parakeet-app':assert proof['admitted']
        reports[name]=read(folder/'analysis.json')
    validate(reports,spec)
    return dict(passed=True,retained=spec['prerequisites'],graph_qualification=graph)
