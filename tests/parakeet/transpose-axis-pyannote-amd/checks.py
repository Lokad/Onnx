"""Require complete native arrays, exact same-platform outputs and public results."""
from qualify_outputs import pyannote
from protocol import pin,read


def inventory(value,spec,built):
    from identity_scope import inventory as compare
    return compare(value,spec['reference_consumer'],built['consumer'])


def semantic(value):
    assert set(value)=={'Intervals','ExclusiveIntervals','Speakers','Status','AudioDuration','Windows'}
    for speaker in value['Speakers']:assert set(speaker)=={'Speaker','Centroid','HasEmbedding'}
    return dict(value,Speakers=[{k:v for k,v in speaker.items() if k!='Centroid'} for speaker in value['Speakers']])


def semantic_agreement(actual,expected):
    assert semantic(actual)==semantic(expected),'Changed public speaker assignment, timeline or status'
    return True


def model(base,role):
    folder=base/role/'output';baseline=base/'selected/output' if role=='candidate' else None
    result=pyannote(base,folder,role,baseline)
    assert result['passed'] and result['arrays']==18 and result['values']==2917107 and result['public_calls']==16
    value=read(folder/'result.json');complete_exact=None;semantics=None
    if role=='candidate':
        production=[c for c in result['comparisons'] if c['reference']=='production']
        assert len(production)==18
        assert all(c['bit_identical'] for c in production)
        selected=read(baseline/'result.json')
        assert len(value['applications'])==len(selected['applications'])==16
        for a,b in zip(value['applications'],selected['applications'],strict=True):semantic_agreement(a['result'],b['result'])
        complete_exact=[r['result'] for r in value['applications']]==[r['result'] for r in selected['applications']]
        assert complete_exact, 'Changed complete public result including centroid values'
        semantics=True
    result.update(complete_public_results_exact=complete_exact,complete_public_semantics_exact=semantics,
        numerical_policy='Original native 1e-4 bounds; every selected tensor and complete public result exact.',
        no_performance_measurement=True)
    return result
