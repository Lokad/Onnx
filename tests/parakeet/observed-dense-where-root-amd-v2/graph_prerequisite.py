"""Require every case and control in the fresh complete graph comparison."""
from protocol import pin,read

CASES=['e5-8tok','e5-30tok','e5-30pad128','e5-128tok','e5-512tok','dinov3','resnet50','gpt2']


def verify_bundle(base,spec):
    folder=base/'evidence/graph-qualification';wanted=spec['graph_qualification']
    assert pin(folder/'closed.json')==wanted['closed'] and pin(folder/'analysis.json')==wanted['analysis']
    proof=read(folder/'closed.json');value=read(folder/'analysis.json')
    assert proof['passed'] and proof['admitted'] and proof['all_controls_passed']
    assert proof['files']['analysis.json']==wanted['analysis']
    assert value['passed'] and not value['root_product_changed']
    assert (value['clocks'],value['measured'],len(value['resources']))==(41112,8640,72)
    assert pin(base/'evidence/graphs/closed.json')==wanted['closed']
    assert pin(base/'evidence/graphs/analysis.json')==wanted['analysis']
    payload=read(base/'evidence/graphs/payload.json')
    assert proof['files']['payload.json']==pin(base/'evidence/graphs/payload.json')
    assert value['consumer']['consumer']==payload['consumer']
    assert value['e5_consumer']['consumer']==payload['e5_consumer']
    assert value['e5_consumer']['previous_consumer']==payload['consumer']
    for key in ['consumer','e5_consumer']:
        assert value[key]['branches_locals_exceptions_equal'] and value[key]['implementation_flags_equal']
    assert [(r['before'],r['after']) for r in value['e5_consumer']['changes']]==[(780,1380),(600,1200)]
    assert [r['key'] for r in value['performance']]==CASES
    for row in value['performance']:
        assert row['qualified'] and row['regression_passed'] and row['candidate_over_current']<=1.05
        assert [c['role'] for c in row['controls']]==['current','candidate','ort']
        assert all(c['passed'] and c['ratio']<=1.10 for c in row['controls'])
    if 'identities' in spec:
        assert all(payload['products'][role]['Lokad.Onnx.dll']==spec['identities'][label]['Lokad.Onnx.dll']
            for role,label in [('current','selected'),('candidate','candidate')])
    if 'measured' in spec:assert payload['products']['candidate']['Lokad.Onnx.dll']==spec['measured']['Lokad.Onnx.dll']
    return dict(passed=True,closed=wanted['closed'],source_closures=dict(graphs=wanted['closed']))
