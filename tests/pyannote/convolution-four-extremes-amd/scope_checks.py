"""Constrain the changed driver and preserve every original numerical case."""
from copy import deepcopy

def inventory(value,previous,current):
    assert value['inventory_complete']
    row,=value['observations']
    assert row['assembly']=='ConvExtremes.dll' and row['public_surface_equal']
    assert not row['added'] and not row['removed'] and row.get('compiler_rename') is None
    assert row['before_sha256']==previous['sha256'] and row['after_sha256']==current['sha256']
    assert row['methods']==len(row['normalized_methods'])>1 and row['unchanged_methods']==row['methods']-1
    key,=row['differences'];assert key=='Extremes::Main::Int32 Main(System.String[])'
    assert set(row['candidate_methods'])=={key} and row['candidate_methods'][key]!=row['normalized_methods'][key]
    return dict(passed=True,methods=row['methods'],unchanged=row['unchanged_methods'],changed=[key],public_surface_equal=True)

def original_subset(current,previous):
    assert previous['passed'] and previous['cases']==1280
    assert current['cases']==3840 and current['lanes']==previous['lanes']
    for field in ['observations','graph_cases']:
        subset=[deepcopy(row) for row in current[field] if row['m'] in [32,48]]
        assert len(subset)==1280
        for index,row in enumerate(subset):row['index']=index
        assert subset==previous[field],field
