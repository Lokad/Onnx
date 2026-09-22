"""Reconcile every clock/output, then apply fixed complete-call selection gates."""
import math
import statistics
from protocol import read,pin


def schedule(capture):
    cases=[]
    for ordinal,call in enumerate(capture['calls']):
        shape=call['inputs'][0]['shape'];index=call['index']
        assert index==ordinal%4 and shape==[589,1,60 if index==0 else 256]
        macs=2*589*512*(shape[2]+128);repeats=(2**31+macs-1)//macs
        assert repeats==(19 if index==0 else 10)
        cases.append(dict(ordinal=ordinal,index=index,name=call['name'],shape=shape,macs=macs,repeats=repeats,tail=1))
    assert len(cases)==12 and sum(c['repeats'] for c in cases)==147
    return dict(cases=cases,passes=[-1,0,1,2],process_order=['selected','candidate','candidate','selected'],
        clocks_per_process=588,warmup_clocks=588,measured_clocks=1764,total_clocks=2352,preparations=48,
        boundary='Reset plus ordinary graph Execute; validation and graph preparation outside timer',
        controls=dict(aggregate=1.10,node=1.20),speed=dict(aggregate=.90,node=1.05))


def check_events(events,mode,role,capture,frequency):
    assert frequency>0 and len(events)==12+2*(588 if mode=='time' else 24)
    previous=0
    for ordinal,event in enumerate(events[:12]):
        assert event['kind']=='prepare' and event['ordinal']==ordinal
        assert event['frequency']==frequency and previous<event['start']<event['end'];previous=event['end']
    cases=schedule(capture)['cases'];position=12;total_values=0;maximum=0.;times=[[] for _ in cases]
    for pass_index in ([-1,0,1,2] if mode=='time' else [0]):
        for c,source in zip(cases,capture['calls']):
            for repeat in range(c['repeats'] if mode=='time' else 2):
                clock,checked=events[position:position+2];position+=2
                for value,kind in [(clock,'call'),(checked,'verified')]:
                    assert value['kind']==kind and (value['ordinal'],value['pass'],value['repeat'])==(c['ordinal'],pass_index,repeat)
                assert clock['ok'] and clock['frequency']==frequency and previous<clock['start']<clock['end'];previous=clock['end']
                assert checked['hashes']==[v['sha256'] for v in source['outputs']]
                assert checked['values']==[v['values'] for v in source['outputs']]
                assert len(checked['errors'])==3 and all(math.isfinite(v) and 0<=v<=1e-4 for v in checked['errors'])
                assert checked['readonly_operands'] and checked['held_outputs_unchanged']
                assert checked['scratch']==source['inputs'][1]['bytes']+source['inputs'][2]['bytes']+8192
                total_values+=sum(checked['values']);maximum=max(maximum,*checked['errors'])
                if mode=='time' and pass_index>=0:times[c['ordinal']].append((clock['end']-clock['start'])/frequency)
    means=[statistics.mean(v) for v in times] if mode=='time' else None
    return dict(passed=True,values=total_values,maximum=maximum,means=means,
        clocks=(position-12)//2,warmups=147 if mode=='time' else 0,preparations=12)


def check_result(result,name,spec,folder,capture):
    role,number=name.split('-');mode='qualify' if number=='qualify' else 'time'
    assert result['passed'] and result['failure'] is None and result['role']==role and result['mode']==mode
    assert result['core']==spec['cores'][role]['sha256'] and result['executable']==spec['consumer']['sha256']
    assert result['runtime']=='10.0.8' and result['vector_count']==8 and result['avx512'] and result['flags']==[]
    assert result['events']==pin(folder/'events.jsonl')['sha256']
    events=[__import__('json').loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]
    value=check_events(events,mode,role,capture,result['frequency'])
    assert result['preparations']==12 and result['clocks']==result['verified']==value['clocks']
    assert result['held']==(36 if mode=='time' else 72) and result['values']==value['values'] and result['maximum']==value['maximum']
    return value


def score(results):
    order=['selected-0','candidate-1','candidate-2','selected-3'];assert set(results)==set(order)
    process={}
    for name in order:
        means=results[name]['means'];assert len(means)==12 and all(math.isfinite(v) and v>0 for v in means)
        process[name]=dict(aggregate=sum(means),nodes=[sum(means[i::4])/3 for i in range(4)],cases=means)
    controls=[];roles={}
    for role,names in [('selected',['selected-0','selected-3']),('candidate',['candidate-1','candidate-2'])]:
        rows=[process[n] for n in names]
        for node in [None,0,1,2,3]:
            values=[r['aggregate'] if node is None else r['nodes'][node] for r in rows]
            ratio=max(values)/min(values);limit=1.10 if node is None else 1.20
            controls.append(dict(role=role,node=node,ratio=ratio,limit=limit,passed=ratio<=limit))
        roles[role]=dict(aggregate=statistics.mean(r['aggregate'] for r in rows),nodes=[statistics.mean(r['nodes'][i] for r in rows) for i in range(4)])
    gates=[]
    for node in [None,0,1,2,3]:
        values={role:r['aggregate'] if node is None else r['nodes'][node] for role,r in roles.items()}
        ratio=values['candidate']/values['selected'];limit=.90 if node is None else 1.05
        gates.append(dict(node=node,**values,ratio=ratio,limit=limit,passed=ratio<=limit))
    return dict(admitted=all(r['passed'] for r in controls+gates),controls=controls,gates=gates,roles=roles,processes=process)


def consumer_inventory(value,previous,built):
    assert value['inventory_complete'] and len(value['observations'])==1
    row=value['observations'][0]
    assert row['assembly']=='LstmScreen.dll' and row['before_sha256']==previous['sha256'] and row['after_sha256']==built['sha256']
    assert row['public_surface_equal'] and not row['added'] and not row['removed'] and row.get('compiler_rename') is None
    key,=row['differences'];assert key.startswith('Screen::Main::')
    assert set(row['candidate_methods'])=={key} and row['normalized_methods'][key]!=row['candidate_methods'][key]
    assert len(row['normalized_methods'])==row['methods'] and row['unchanged_methods']==row['methods']-1
    return dict(passed=True,methods=row['methods'],unchanged=row['unchanged_methods'],changed=key,public_surface_equal=True)

