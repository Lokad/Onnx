"""Require complete native arrays, exact same-platform outputs and public results."""
from qualify_outputs import pyannote
from protocol import pin,read


def inventory(value,spec,built):
    assert value['inventory_complete'] and len(value['observations'])==1
    row=value['observations'][0]
    assert row['assembly']=='GraphQualification.dll' and row['methods']==96 and row['unchanged_methods']==95
    assert row['public_surface_equal'] and row['compiler_rename'] is None and not row['removed'] and not row['added']
    assert row['before_sha256']==spec['consumers']['selected']['sha256'] and row['after_sha256']==built['consumer']['sha256']
    key,=row['differences'];assert key.startswith('Program::<Main>$::')
    old=row['normalized_methods'][key];new=row['candidate_methods'][key]
    assert old.count(spec['old_data'])==1 and old.replace(spec['old_data'],spec['new_data'])==new
    return dict(passed=True,methods=96,unchanged=95,literal_only=True)


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
        assert all(c['bit_identical'] for c in production if c['model']=='segmentation')
        selected=read(baseline/'result.json')
        assert len(value['applications'])==len(selected['applications'])==16
        for a,b in zip(value['applications'],selected['applications'],strict=True):semantic_agreement(a['result'],b['result'])
        complete_exact=[r['result'] for r in value['applications']]==[r['result'] for r in selected['applications']]
        semantics=True
    result.update(complete_public_results_exact=complete_exact,complete_public_semantics_exact=semantics,
        numerical_policy='Original native 1e-4 bounds; old-product float differences retained diagnostically; segmentation, semantic public results and owned repeats exact.',
        no_performance_measurement=True)
    return result
