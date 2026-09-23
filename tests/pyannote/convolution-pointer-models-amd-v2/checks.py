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


def model(base,role):
    folder=base/role/'output';baseline=base/'selected/output' if role=='candidate' else None
    result=pyannote(base,folder,role,baseline)
    assert result['passed'] and result['arrays']==18 and result['values']==2917107 and result['public_calls']==16
    value=read(folder/'result.json')
    if role=='candidate':
        assert all(c['bit_identical'] for c in result['comparisons'] if c['reference']=='production')
        selected=read(baseline/'result.json')
        assert [r['result'] for r in value['applications']]==[r['result'] for r in selected['applications']]
    result.update(complete_public_results_exact=True if role=='candidate' else None,no_performance_measurement=True)
    return result
