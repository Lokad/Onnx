"""Require complete native arrays, bounded arithmetic differences and exact public semantics."""
import numpy as np
from qualify_outputs import pyannote, scaled
from protocol import pin,read



def semantic(value):
    assert set(value)=={'Intervals','ExclusiveIntervals','Speakers','Status','AudioDuration','Windows'}
    for speaker in value['Speakers']:assert set(speaker)=={'Speaker','Centroid','HasEmbedding'}
    return dict(value,Speakers=[{k:v for k,v in speaker.items() if k!='Centroid'} for speaker in value['Speakers']])


def semantic_agreement(actual,expected):
    assert semantic(actual)==semantic(expected),'Changed public speaker assignment, timeline or status'
    return True


def centroid_agreement(actual,expected):
    semantic_agreement(actual,expected)
    rows=[]
    for a,b in zip(actual['Speakers'],expected['Speakers'],strict=True):
        left,right=np.asarray(a['Centroid']),np.asarray(b['Centroid'])
        assert left.shape==right.shape==(256,)
        value=scaled(left,right);assert value['failed_values']==0
        rows.append(dict(speaker=a['Speaker'],**value))
    return rows


def model(base,role):
    folder=base/role/'output';baseline=base/'selected/output' if role=='candidate' else None
    result=pyannote(base,folder,role,baseline)
    assert result['passed'] and result['arrays']==18 and result['values']==2917107 and result['public_calls']==16
    value=read(folder/'result.json');complete_exact=None;semantics=None;centroids=[]
    if role=='candidate':
        production=[c for c in result['comparisons'] if c['reference']=='production']
        assert len(production)==18
        assert all(c['failed_values']==0 for c in production)
        selected=read(baseline/'result.json')
        assert len(value['applications'])==len(selected['applications'])==16
        for a,b in zip(value['applications'],selected['applications'],strict=True):
            assert (a['name'],a['pass'],a['phase'])==(b['name'],b['pass'],b['phase'])
            centroids.append(dict(name=a['name'],pass_index=a['pass'],speakers=centroid_agreement(a['result'],b['result'])))
        complete_exact=[r['result'] for r in value['applications']]==[r['result'] for r in selected['applications']]
        semantics=True
    result.update(complete_public_results_exact=complete_exact,complete_public_semantics_exact=semantics,
        public_centroid_comparisons=centroids,
        numerical_policy='Original native 1e-4 bounds; selected tensors/centroids bounded at 1e-4; public semantics exact.',
        no_performance_measurement=True)
    return result
