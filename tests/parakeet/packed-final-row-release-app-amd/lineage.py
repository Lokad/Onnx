"""Bind release -> M73 -> M78 through complete identical model/public outputs."""
from protocol import pin,read


def qualify_lineage(base,reports,spec):
    before=reports['parakeet-release'];after=reports['parakeet']
    assert reports['product']['identities']==after['identities']
    assert before['identities']['selected']==spec['identities']['selected']
    assert before['identities']['candidate']==after['identities']['selected']
    assert after['identities']['candidate']==spec['identities']['candidate']
    assert before['identities']['selected']['Lokad.Onnx.dll']['sha256']=='f95a13c58354bf07f3b7926b72903c18b1a560a56673297cb9fe001d3541b592'
    assert before['identities']['candidate']['Lokad.Onnx.dll']['sha256']=='49c3a958850d3e57daa2b7e29e6bd15ff9fc4f27af098d8ce44d8f20b9e065e8'
    assert after['identities']['candidate']['Lokad.Onnx.dll']['sha256']=='49901366484570493b7a42028e5c5458f30fe2c1703a01335b68d9a5f9a1fea9'
    results={}
    for isa in ['512','256']:
        expected_public=None
        for generation,report in [('release',before),('m78',after)]:
            assert report['passed']
            for role in ['selected','candidate']:
                native=report['results'][role+'-native-'+isa]
                assert native['passed'] and native['native']['numeric_gate_passed']
                assert (native['native']['arrays'],native['native']['values'])==(784,3090494)
                public=report['results'][role+'-public-'+isa]
                assert public['passed'] and public['public_requests']==20
                path=base/'evidence/public-lineage'/generation/(role+'-'+isa+'.json')
                assert pin(path)==public['result']
                value=read(path);identity=report['identities'][role]
                assert value['core_sha256']==identity['Lokad.Onnx.dll']['sha256']
                assert value['data_sha256']==identity['Lokad.Onnx.Data.dll']['sha256']
                assert value['held_outputs_unchanged'] and value['family']=='parakeet' and value['engine']=='managed'
                assert len(value['records'])==20 and all(r['ownership'] for r in value['records'])
                complete=[dict(name=r['name'],input_sha256=r['input_sha256'],result=r['result']) for r in value['records']]
                assert len({r['name'] for r in complete})==20
                if expected_public is None:expected_public=complete
                assert complete==expected_public,('Complete public lineage mismatch',generation,role,isa)
                if role=='candidate':
                    assert public['complete_selected_results_exact']
                    comparisons=native['native']['exact_selected_comparisons']
                    assert len(comparisons)==784 and all(r['bit_identical'] for r in comparisons)
        old=before['results']['candidate-native-'+isa]['native']['exact_selected_comparisons']
        new=after['results']['candidate-native-'+isa]['native']['exact_selected_comparisons']
        assert old==new,('Intermediate tensor identity mismatch',isa)
        for kind in ['native','public']:
            results['selected-'+kind+'-'+isa]=before['results']['selected-'+kind+'-'+isa]
            results['candidate-'+kind+'-'+isa]=after['results']['candidate-'+kind+'-'+isa]
    return dict(passed=True,identities=spec['identities'],results=results)
