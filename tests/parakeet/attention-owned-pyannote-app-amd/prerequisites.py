"""Bind closed attention ownership qualification without changing public application gates."""
import math
from protocol import ROLES,pin,read
from graph_prerequisite import verify_bundle


def verify_comparisons(rows):
    assert len(rows) == 784 and sum(r['values'] for r in rows) == 3090494
    assert len({(r['case'], r['label'], r['output']) for r in rows}) == 784
    for row in rows:
        assert row['bit_identical'] and row['values'] > 0
        assert len(row['sha256']) == 64 and all(c in '0123456789abcdef' for c in row['sha256'])


def validate(reports,spec,compatible,qualified):
    assert set(reports)=={'models','parakeet','shared','parakeet-app','baseline'}
    assert all(r['passed'] for r in reports.values())
    products=spec['identities'];selected,candidate=products['selected'],products['candidate']
    assert selected['Lokad.Onnx.dll']['sha256']=='a6f7d9f9abf0dc3c10a1b566443a22352c8cc84f05a198766325fef2b09850a4'
    assert selected['Lokad.Onnx.Data.dll']['sha256']=='1ba343fd8b00fd85bddb33c57aaebf4217431955c40bbe45576448467e59c99f'
    assert candidate['Lokad.Onnx.dll']['sha256']=='ee5218dbab0a970b0f20f273fc2a839bb9b96e9d5438791529eb11c731d66859'
    assert candidate['Lokad.Onnx.Data.dll']['sha256']=='1ba343fd8b00fd85bddb33c57aaebf4217431955c40bbe45576448467e59c99f'
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
    corpus,=[r for r in application['performance']['gates'] if r['name']=='corpus-at-least-one-percent-gain']
    assert corpus['limit']==.99
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
            assert math.isfinite(native['native']['maximum']) and 0 <= native['native']['maximum'] <= 1e-4
            assert not native['native']['failures'] and public['passed'] and public['public_requests']==20
            if role=='candidate':
                verify_comparisons(native['native']['exact_selected_comparisons'])
                assert public['complete_selected_results_exact']
        shared=[reports['shared']['results'][role+'-'+mode] for mode in ['shared','e5']]
        assert all(r['passed'] for r in shared)
        assert sum(r['arrays'] for r in shared)==166 and sum(r['values'] for r in shared)==5000814
        if role=='candidate':assert all(r['exact_selected'] for result in shared for r in result['rows'])
    assert compatible['passed'] and compatible['original_public_bindings_preserved']
    assert compatible['all_original_method_flags_preserved'] and compatible['all_data_methods_exact']
    assert compatible['underlying_methods_reconciled']==3985
    core,data=compatible['compiled_scope']
    assert core['assembly']=='Lokad.Onnx.dll' and core['unchanged']==3287
    assert core['original']==3288 and len(core['changed'])==1
    assert {tuple(name.split('::')[:2]) for name in core['changed']}=={
        ('Lokad.Onnx.ComputationalGraph','PrepareOwnedMatMulWeights')}
    assert data==dict(assembly='Lokad.Onnx.Data.dll',original=697,unchanged=697,changed=[])
    assert compatible['no_consumer_or_product_build']
    assert compatible['selected']==selected and compatible['candidate']==candidate
    assert compatible['qualified_model_product']==qualified['identities']['candidate']
    assert qualified['passed'] and qualified['performance']['admitted']
    assert qualified['consumers']==spec['consumers']
    assert compatible['focused_contracts']['sha256']=='a293403a5dc3a7bdabb7fdc448470feea2b54dbf139d98b7b3f9ed0eef39854f'
    assert compatible['actual_model_census']['sha256']=='ef4e41eca49d33f6cac4ae7d510922053874002787709cc3d83bc98c3437d3d7'
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
    qualified=base/'evidence/consumer-qualification'
    closure=read(qualified/'closed.json')
    assert pin(qualified/'closed.json')['sha256']=='7e0b08fc21b487d9e945e00133ff8394e9b3ab825a8cac3f6213e61b9406d4c4'
    assert closure['passed'] and closure['admitted']
    assert pin(qualified/'analysis.json')==closure['files']['analysis.json']
    validate(reports,spec,read(base/'evidence/product-compatibility.json'),read(qualified/'analysis.json'))
    return dict(passed=True,retained=spec['prerequisites'],graph_qualification=graph)
