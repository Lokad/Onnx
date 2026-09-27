"""Bind closed prepared-row qualification without changing public application gates."""
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
    assert selected['Lokad.Onnx.dll']['sha256']=='65f15a41764660af6166c9a965943b6f28cac0c9505117d0fd4beaa2687f9d03'
    assert selected['Lokad.Onnx.Data.dll']['sha256']=='da72ca547191de12a52f09bceeed25619a92a14b1bbff271a9e7d10515f40312'
    assert candidate['Lokad.Onnx.dll']['sha256']=='af19b3b4429a07f7966b5e35ee04e8a31316f45991c300f3683a591caf5e9374'
    assert candidate['Lokad.Onnx.Data.dll']['sha256']=='da72ca547191de12a52f09bceeed25619a92a14b1bbff271a9e7d10515f40312'
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
    assert compatible['underlying_methods_reconciled']==3980
    compiled=compatible['compiled']
    assert compiled['passed'] and compiled['public_surface_equal'] and compiled['flags_equal'] and compiled['assembly_metadata_equal']
    core,data=compiled['assemblies']
    assert core['assembly']=='Lokad.Onnx.dll' and core['unchanged']==3281
    assert [name.split('::')[1] for name in core['changed']]==['ResolvePackedKernel','RunPreparedPackedRows']
    assert len(core['added'])==1 and core['added'][0].startswith('Lokad.Onnx.PreparedSingleRowKernel::Multiply::')
    assert data==dict(assembly='Lokad.Onnx.Data.dll',unchanged=697,changed=[],added=[])
    assert compatible['no_consumer_or_product_build'] and compatible['failed_first_call_prediction']
    assert compatible['selected']==selected and compatible['candidate']==candidate
    assert compatible['qualified_model_product']==qualified['identities']['candidate']
    assert qualified['passed'] and qualified['performance']['admitted']
    assert qualified['consumers']==spec['consumers']
    assert not compatible['component_screen_admitted']
    assert len(compatible['failed_component_controls'])==len(compatible['failed_component_cases'])==2
    assert all(not r['passed'] for r in compatible['failed_component_controls']+compatible['failed_component_cases'])
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
    assert pin(qualified/'closed.json')['sha256']=='f1c4e873f19f5562b2da8a31a59ec5b54f47959b8f4c1c996ec6c996f5e2218d'
    assert closure['passed'] and closure['admitted']
    assert pin(qualified/'analysis.json')==closure['files']['analysis.json']
    validate(reports,spec,read(base/'evidence/product-compatibility.json'),read(qualified/'analysis.json'))
    return dict(passed=True,retained=spec['prerequisites'],graph_qualification=graph)
