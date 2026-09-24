"""Verify an explicitly sourced eight-case qualification inside a frozen bundle."""
from protocol import pin,read


def verify_bundle(base,spec):
    folder=base/'evidence/graph-qualification';wanted=spec['graph_qualification']
    assert pin(folder/'closed.json')==wanted['closed'] and pin(folder/'analysis.json')==wanted['analysis']
    proof=read(folder/'closed.json');value=read(folder/'analysis.json')
    assert proof['passed'] and proof['admitted'] and proof['all_controls_passed']
    assert proof['files']['analysis.json']==wanted['analysis']
    assert value['passed'] and value['admitted'] and value['all_controls_passed']
    assert value['original_graph_failure_preserved'] and not value['root_product_changed']
    assert (value['clocks'],value['measured'],value['source_calls_retained'],len(value['setups']))==(41112,8640,45801,72)
    old=read(base/'evidence/graphs/analysis.json');new=read(base/'evidence/e5/analysis.json')
    assert value['source_closures']==proof['source_closures']==dict(
        original_graphs=pin(base/'evidence/graphs/closed.json'),e5_successor=pin(base/'evidence/e5/closed.json'))
    old_proof=read(base/'evidence/graphs/closed.json');new_proof=read(base/'evidence/e5/closed.json')
    assert old_proof['passed'] and not old_proof['admitted']
    assert new_proof['passed'] and new_proof['admitted'] and new_proof['all_controls_passed']
    assert old_proof['files']['analysis.json']==pin(base/'evidence/graphs/analysis.json')
    assert new_proof['files']['analysis.json']==pin(base/'evidence/e5/analysis.json')
    assert old['clocks']==37512 and old['measured']==8640 and new['clocks']==8289 and new['measured']==1080
    assert value['products']==read(base/'evidence/graphs/payload.json')['products']==read(base/'evidence/e5/payload.json')['products']
    assert value['products']==new['products']
    for analysis in [old,new]:
        assert analysis['consumer']['branches_locals_exceptions_equal'] and analysis['consumer']['implementation_flags_equal']
    assert [(r['before'],r['after']) for r in new['consumer']['changes']]==[(780,1380),(600,1200)]
    expected=[]
    for original in old['performance']:
        changed=original['key']=='e5-30tok';row=new['performance'] if changed else original
        assert row['key']==original['key'] and row['qualified'] and row['regression_passed']
        assert len(row['controls'])==3 and all(c['passed'] and c['ratio']<=1.10 for c in row['controls'])
        assert row['candidate_over_current']<=1.05
        expected.append(dict(row,source='e5-successor' if changed else 'original-graphs',
            warmups=1200 if changed else 600,measured_per_process=180))
    assert value['performance']==expected and len(expected)==8 and len({r['key'] for r in expected})==8
    if 'identities' in spec:
        assert all(value['products'][role]['Lokad.Onnx.dll']==spec['identities'][label]['Lokad.Onnx.dll']
            for role,label in [('current','selected'),('candidate','candidate')])
    if 'measured' in spec:assert value['products']['candidate']['Lokad.Onnx.dll']==spec['measured']['Lokad.Onnx.dll']
    return dict(passed=True,closed=wanted['closed'],source_closures=value['source_closures'])
