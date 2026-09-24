"""Qualify every complete-call clock, and judge only the frozen exact fractions."""
from fractions import Fraction
from protocol import TIMING_JOBS, pin, read


def fraction(value):return dict(numerator=value.numerator,denominator=value.denominator)


def qualify(base,name,payload,built,worker):
    role,duplicate,mode=name.split('-');result=read(base/name/'output/result.json')
    assert result['role']==role and result['mode']==mode and result['worker']==name and result['pid']==worker['child']['pid']
    assert result['affinity']==4 and result['processor_count']==1 and result['runtime']=='10.0.8' and result['avx512']==(mode=='512')
    assert result['flags']==({} if mode=='512' else {'DOTNET_EnableAVX512':'0'})
    for field,dll in [('core','Lokad.Onnx.dll'),('data','Lokad.Onnx.Data.dll')]:assert result[field]==payload['identities'][role][dll]['sha256']
    assert result['consumer']==built['consumer']['sha256'] and type(result['frequency']) is int and result['frequency']>0
    assert result['spec_sha256']==pin(base/'spec.json')['sha256'] and result['capture_sha256']==pin(base/'fixtures/result.json')['sha256']
    assert all(result[k] is True for k in ['passed','inputs_unchanged','held_outputs_unchanged','exact_selected_outputs','complete_calls',
        'setup_includes_graph_preparation_and_context','per_call_includes_reset_validation_allocation_and_execution'])
    calls=read(base/'fixtures/result.json')['calls'];assert len(calls)==380
    assert len(result['setup'])==2
    for index,row in enumerate(result['setup']):
        assert row['index']==index and type(row['ticks']) is int and row['ticks']>0
        assert type(row['allocated_bytes']) is int and row['allocated_bytes']>=0
        assert row['retained_bytes']==(13107200 if role=='candidate' else 0)
    expected=[(phase,repeat,c) for phase in ['warmup','measured'] for repeat in range(5) for c in calls]
    assert len(result['rows'])==len(expected)==3800
    for row,(phase,repeat,call) in zip(result['rows'],expected,strict=True):
        assert (row['phase'],row['repeat'],row['name'],row['step'],row['index'])==(phase,repeat,call['name'],call['step'],call['index'])
        assert type(row['ticks']) is int and row['ticks']>0 and type(row['allocated_bytes']) is int and row['allocated_bytes']>=0
        assert row['output_sha256']==[out['sha256'] for out in call['outputs']]
    total=sum(r['ticks'] for r in result['rows'])+sum(r['ticks'] for r in result['setup'])
    # after() runs before the final supervisor seconds field is available; compare during audit.
    if 'seconds' in worker:assert total/result['frequency']<=worker['seconds']
    return dict(passed=True,result=pin(base/name/'output/result.json'),calls=3800,warmup=1900,measured=1900,
        exact_output_arrays=11400,frequency=result['frequency'],setup=result['setup'],total_timed_seconds=fraction(Fraction(total,result['frequency'])))


def evaluate(workers,cases):
    assert set(workers)==set(TIMING_JOBS) and len(cases)==6 and len(set(cases))==6
    groups=['corpus',*cases];controls=[];gates=[];tables=[]
    for mode in ['512','256']:
        summaries={}
        for name in [n for n in TIMING_JOBS if n.endswith('-'+mode)]:
            result=workers[name];frequency=result['frequency'];summaries[name]={}
            for group in groups:
                times=[Fraction(sum(r['ticks'] for r in result['rows'] if r['phase']=='measured' and r['repeat']==repeat and (group=='corpus' or r['name']==group)),frequency) for repeat in range(5)]
                assert all(t>0 for t in times)
                ratio=max(times)/min(times);limit=Fraction(110 if group=='corpus' else 120,100)
                controls.append(dict(mode=mode,name=name,group=group,kind='within-process',ratio=fraction(ratio),limit=fraction(limit),passed=ratio<=limit))
                summaries[name][group]=sum(times)/5
        for group in groups:
            means={}
            for role in ['selected','candidate']:
                values=[summaries[f'{role}-{duplicate}-{mode}'][group] for duplicate in [0,1]]
                ratio=max(values)/min(values);limit=Fraction(110 if group=='corpus' else 120,100)
                controls.append(dict(mode=mode,name=role,group=group,kind='between-processes',ratio=fraction(ratio),limit=fraction(limit),passed=ratio<=limit))
                means[role]=sum(values)/2
            ratio=means['candidate']/means['selected'];limit=Fraction(90 if group=='corpus' else 105,100)
            gates.append(dict(mode=mode,group=group,ratio=fraction(ratio),limit=fraction(limit),passed=ratio<=limit))
            tables.append(dict(mode=mode,group=group,selected=fraction(means['selected']),candidate=fraction(means['candidate']),candidate_over_selected=fraction(ratio)))
    assert len(controls)==84 and len(gates)==14
    stable=all(c['passed'] for c in controls)
    return dict(controls_passed=stable,admitted=stable and all(g['passed'] for g in gates),controls=controls,gates=gates,table=tables,
        policy='Five fixed warmups and measured passes; both modes; all 84 controls; corpus gain>=10%, no case>5% slower; all raw clocks kept; no unchanged retry.')
