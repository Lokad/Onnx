"""Identity, complete caller coverage, preparation and fixed sample checks."""
from protocol import read
from score import call_totals, iteration_manifest


def check_result(result, role, spec, base, lanes, benchmark):
    job=spec['job_details'][role]
    assert result['passed'] and result['role']==role and result['mode']==('benchmark' if benchmark else 'validate')
    assert result['core']==job['core']['sha256'] and result['probe']==job['probe']['sha256'] and result['executable']==spec['driver']['sha256']
    assert result['flags']==[] and result['lanes']==lanes and result['no_performance_measurement']==(not benchmark)
    assert result['read_only_operands'] and result['graph_dispatches']==result['calls']==len(result['observations'])
    assert result['graph_retained_bytes']==63258624 and result['prepared_bytes']==21086208 and result['cases']==108
    fixture=read(base/'fixtures/result.json');reference=read(base/'reference.json');calls=fixture['calls']
    expected={(r['name'],r['index']):r for r in reference['observations']}
    assert iteration_manifest(calls)==read(base/'iterations.json')
    constants=[c for c in calls[:36] if c['eligible']];assert len(constants)==32
    assert len(result['preparation'])==(128 if benchmark else 32)
    for index,row in enumerate(result['preparation']):
        call=constants[index%32]; repeat=index//32
        assert row['kind']=='preparation' and row['role']==role and row['pass']==repeat and row['warmup']==(repeat==0)
        assert row['index']==call['index'] and row['node']==call['node'] and row['bytes']==call['weights']['bytes']
        assert type(row['frequency']) is type(row['ticks']) is int and row['frequency']>0
        assert row['ticks']>0 if benchmark else row['ticks']==0
        assert len(row['sha256'])==64
    if benchmark:
        assert lanes==16 and result['runtime']=='10.0.8'
        return call_totals(result,calls,expected,role)
    assert result['calls']==result['warmups']==108 and result['measured']==0
    for call,row in zip(calls,result['observations'],strict=True):
        item=expected[(call['case'],call['index'])]
        assert row['kind']=='call' and row['role']==role and row['pass']==0 and row['warmup'] and row['ticks']==0
        assert (row['name'],row['index'],row['form'],row['eligible'])==(call['case'],call['index'],call['form'],call['eligible'])
        assert row['iteration']==0 and row['iterations']==1 and row['values']==item['values'] and row['sha256']==item['production'] and row['exact']
    return dict(passed=True,calls=108,preparations=32,no_performance_measurement=True)
