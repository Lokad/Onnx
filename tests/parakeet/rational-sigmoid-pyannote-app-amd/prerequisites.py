"""Bind closed rational arithmetic qualification without changing public application gates."""
import math
from protocol import ROLES,pin,read
from graph_prerequisite import verify_bundle


def verify_comparisons(rows):
    assert len(rows) == 784 and sum(r['values'] for r in rows) == 3090494
    assert len({(r['case'], r['label'], r['output']) for r in rows}) == 784
    for row in rows:
        assert row['dtype'] in ['Float','Int32','Int64']
        assert row['values'] == math.prod(row['shape'])
        assert math.isfinite(row['maximum_scaled_error']) and 0 <= row['maximum_scaled_error'] <= 1e-4
        if row['dtype'] != 'Float':
            assert row['bit_identical'] and row['maximum_scaled_error'] == 0


def validate(reports,spec,compatible,qualified):
    assert set(reports)=={'models','parakeet','shared','parakeet-app','baseline'}
    assert all(r['passed'] for r in reports.values())
    products=spec['identities'];selected,candidate=products['selected'],products['candidate']
    assert selected['Lokad.Onnx.dll']['sha256']=='8bb22038d0b4c09b56b2cdae06c49c165b8e646bc73ca28ad400f4ace0bfc659'
    assert selected['Lokad.Onnx.Data.dll']['sha256']=='d02dbf550d7a6ea0ddf24985ffff7b86db135035ce7fab31f4bab063e0090620'
    assert candidate['Lokad.Onnx.dll']['sha256']=='946ddfb66492c48a0fc6078ecbe1957ac494ff5d9ff0be42259d70e66e3b1f24'
    assert candidate['Lokad.Onnx.Data.dll']['sha256']=='dbe959361209bbc20db9bd566f037f58e01b9cc40d807c5745dbe2f4e1c1aca6'
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
                verify_comparisons(native['native']['selected_comparisons'])
                assert public['complete_selected_results_exact']
        shared=[reports['shared']['results'][role+'-'+mode] for mode in ['shared','e5']]
        assert all(r['passed'] for r in shared)
        assert sum(r['arrays'] for r in shared)==166 and sum(r['values'] for r in shared)==5000814
        if role=='candidate':assert all(r['exact_selected'] for result in shared for r in result['rows'])
    assert compatible['passed'] and compatible['original_public_bindings_preserved']
    assert compatible['all_original_method_flags_preserved'] and compatible['all_data_methods_exact']
    assert compatible['underlying_methods_reconciled']==3979
    assert compatible['changed_core_methods']==['Sigmoid'] and compatible['added_private_methods']==['SigmoidRationalVector']
    assert compatible['selected']==selected and compatible['candidate']==candidate
    assert compatible['qualified_model_product']==qualified['identities']['candidate']
    assert qualified['passed'] and qualified['performance']['admitted']
    assert qualified['consumers']==spec['consumers']
    assert not compatible['component_screen_admitted']
    assert len(compatible['failed_component_controls'])==13 and len(compatible['failed_component_cases'])==4
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
    assert pin(qualified/'closed.json')['sha256']=='60ea42f647c8c55b5ea97c289a69ccefd04e8b6cdb645c12daf996611e3fb7e8'
    assert closure['passed'] and closure['admitted']
    assert pin(qualified/'analysis.json')==closure['files']['analysis.json']
    validate(reports,spec,read(base/'evidence/product-compatibility.json'),read(qualified/'analysis.json'))
    return dict(passed=True,retained=spec['prerequisites'],graph_qualification=graph)
