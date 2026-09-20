"""Independent full-array, configuration, timing and owned-process evidence checks."""
from pathlib import Path
import argparse,hashlib,itertools,json,math,statistics,struct,sys
import numpy as np
from protocol import CASES,POLICIES,ROLES,ORDERS,CORE,NATIVE,PROTOCOL,LIMITS,CRITERIA,schedule,specification,cycles,commands

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)

def input_hash(fixture):
    value=bytearray(b'LOKAD-CAMPAIGN-INPUTS-1\0')+struct.pack('<i',3)
    assert sorted(fixture['inputs'])==['attention_mask','input_ids','token_type_ids']
    for name,items in sorted(fixture['inputs'].items()):
        assert len(items)==fixture['shape'][1]
        encoded=name.encode();value+=struct.pack('<i',len(encoded))+encoded+struct.pack('<iiiiq',7,2,1,len(items),len(items))
        value+=struct.pack('<'+'q'*len(items),*items)
    return hashlib.sha256(value).hexdigest()

def rows(v):
    s=v['specification'];freq=v['frequency'];assert type(freq) is int and freq>0
    assert len(v['measured'])==s['blocks']*s['calls']
    assert [(r['block'],r['call']) for r in v['measured']]==list(itertools.product(range(s['blocks']),range(s['calls'])))
    assert (v['first']['block'],v['first']['call'])==(-1,0)
    warm=v['conditioning'];assert s['conditioning_minimum']<=len(warm)<=20000
    assert [(r['block'],r['call']) for r in warm]==[(-1,i) for i in range(len(warm))]
    total=0.
    for index,r in enumerate(warm):
        assert total<s['conditioning_seconds'] or index<s['conditioning_minimum']
        total+=r['execute']/freq
    assert total==v['conditioned'] and total>=s['conditioning_seconds']
    assert type(v['conditioning_wall_ticks']) is int and sum(r['request'] for r in warm)<=v['conditioning_wall_ticks']<=181*freq
    assert v['solo_stage'] in [-1,0,1]
    assert len(v['solo'])==(0 if v['solo_stage']==-1 else s['solo_calls'])
    assert [(r['block'],r['call']) for r in v['solo']]==[(-2-v['solo_stage'],i) for i in range(len(v['solo']))]
    for r in [v['first']]+warm+v['measured']+v['solo']:
        for k in ['block','call','execute','request','bytes','g0','g1','g2']:assert type(r[k]) is int
        assert 0<r['execute']<=r['request']<=600*freq
        assert all(r[k]>=0 for k in ['call','bytes','g0','g1','g2'])

def worker(folder,inputs,model,probe,ort_managed,native_pin,job,phase,smoke=False):
    v=read(folder/'result.json');assert v['specification']==specification(job,phase,smoke);rows(v)
    s=v['specification'];native=job['role']=='N';enabled=phase=='compare' and job['role']=='C'
    assert v['solo_stage']==(0 if job['role']==job['creation'][0] else 1 if job['role']==job['creation'][-1] else -1)
    assert v['passed'] is True and v['enabled'] is enabled and v['wide_enabled'] is enabled and v['core_sha256']==CORE
    assert v['probe_sha256']==probe['sha256'] and v['ort_managed_sha256']==ort_managed['sha256']
    fixture_path=inputs/(job['case']+'.json');fixture=read(fixture_path)
    assert fixture['name']==job['case'] and v['fixture_sha256']==pin(fixture_path)['sha256']
    assert v['model_sha256']==fixture['model_sha256']==model['sha256']
    assert v['input_sha256']==fixture['input_sha256']==input_hash(fixture)
    assert v['reference_sha256']==fixture['reference_sha256']
    assert v['shape']==fixture['shape']==[1,[8,30,128,128,512][job['case_index']],384]
    assert v['affinity']==4 and v['processor_count']==1
    assert v['flags']==({'LOKAD_ONNX_FINGERPRINT_STRINGS':'1','LOKAD_ONNX_LAYERNORM_WIDE_OUTPUT':'1'} if enabled else {})
    assert v['optimization']==('Memory' if job['policy']=='memory' else 'Speed')
    if not smoke:assert v['runtime']=='10.0.8' and v['avx2'] is True and v['avx512'] is True
    else:assert v['runtime']=='10.0.12'
    for k in ['unchanged_cache','unchanged_inputs','unchanged_held_output']:assert v[k] is True
    assert type(v['load_ticks']) is int and v['load_ticks']>0
    if native:
        n=v['native_identity'];assert n['sha256']==native_pin['sha256'] and n['version']=='1.23.2'
        assert n['threads']==1 and n['sequential'] is True and n['all_optimizations'] is True and n['spinning'] is False
        assert v['fingerprint'] is None
    else:assert v['native_identity'] is None and type(v['fingerprint']) is int
    if enabled:assert v['cache_entries']==2330 and isinstance(v['cache_sha256'],str) and len(v['cache_sha256'])==64
    else:assert v['cache_entries']==0 and v['cache_sha256'] is None
    reference=inputs/fixture['reference_file'];assert pin(reference)['sha256']==fixture['reference_sha256']
    want=np.fromfile(reference,dtype='<f4');assert want.size==math.prod(fixture['shape']) and np.isfinite(want).all()
    for stage in ['before','after']:
        path=folder/(stage+'.f32');actual=np.fromfile(path,dtype='<f4')
        assert actual.size==want.size and np.isfinite(actual).all()
        error=float((np.abs(actual.astype(np.float64)-want.astype(np.float64))/np.maximum(1,np.abs(want.astype(np.float64)))).max(initial=0))
        assert error<=1e-4 and abs(error-v[stage+'_error'])<=1e-15
        assert pin(path)['sha256']==v['output_sha256']
    assert {p.name for p in folder.iterdir()}=={'result.json','before.f32','after.f32'}
    return v

def evaluate(values,phase):
    assert len(values)==160
    result=[];passed=True
    for index,case in enumerate(CASES):
        for policy in POLICIES:
            group=[v for v in values if v['specification']['case_index']==index and v['specification']['policy']==policy]
            assert len(group)==16
            jobs={j['visit']:j for j in schedule() if j['case']==case and j['policy']==policy}
            def samples(role,boundary,visit=None,position=None):
                data=[]
                for v in group:
                    s=v['specification']
                    if s['role']!=role or visit is not None and s['visit']!=visit:continue
                    orders=cycles(jobs[s['visit']])
                    data.extend(r[boundary]/v['frequency'] for r in v['measured'] if position is None or orders[r['block']][position]==role)
                assert data;return data
            def mean(role,boundary,visit=None,position=None):return statistics.fmean(samples(role,boundary,visit,position))
            boundaries={}
            for boundary in ['execute','request']:
                controls=[]
                for numerator,denominator in ([('B','A'),('C','A'),('C','B')] if phase=='aa' else [('B','A')]):
                    ratio=mean(numerator,boundary)/mean(denominator,boundary)
                    visits=[mean(numerator,boundary,v)/mean(denominator,boundary,v) for v in range(4)]
                    positions=[mean(numerator,boundary,position=p)/mean(denominator,boundary,position=p) for p in range(4)]
                    contrast=max(positions)/min(positions)
                    gates=dict(aggregate=.995<=ratio<=1.005,visits=all(.99<=x<=1.01 for x in visits),positions=all(.99<=x<=1.01 for x in positions),position_contrast=contrast<=1.01)
                    controls.append(dict(numerator=numerator,denominator=denominator,ratio=ratio,visits=visits,positions=positions,position_contrast=contrast,gates=gates))
                bridges=[]
                for v in group:
                    if not v['solo']:continue
                    own=statistics.fmean(r[boundary]/v['frequency'] for r in v['measured']);solo=statistics.fmean(r[boundary]/v['frequency'] for r in v['solo'])
                    bridges.append(dict(role=v['specification']['role'],visit=v['specification']['visit'],stage=v['solo_stage'],resident_seconds=own,solo_seconds=solo,ratio=own/solo,passed=.95<=own/solo<=1.05))
                assert len(bridges)==8 and all(sum(b['role']==r for b in bridges)==2 for r in ROLES)
                means={r:mean(r,boundary) for r in ROLES};candidate=means['C']/statistics.fmean([means['A'],means['B']])
                visits=[mean('C',boundary,v)/statistics.fmean([mean(r,boundary,v) for r in ['A','B']]) for v in range(4)]
                candidate_gates=dict(aggregate=candidate<=CRITERIA['candidate_aggregate'][index],visits=all(x<=1.02 for x in visits)) if phase=='compare' else {}
                controls_pass=all(all(c['gates'].values()) for c in controls);bridges_pass=all(b['passed'] for b in bridges)
                passed=passed and controls_pass and bridges_pass and all(candidate_gates.values())
                distribution={r:dict(mean=means[r],median=statistics.median(samples(r,boundary)),minimum=min(samples(r,boundary)),maximum=max(samples(r,boundary)),
                    allocations=statistics.fmean(row['bytes'] for v in group if v['specification']['role']==r for row in v['measured']),
                    gc_counts=[sum(row[g] for v in group if v['specification']['role']==r for row in v['measured']) for g in ['g0','g1','g2']]) for r in ROLES}
                boundaries[boundary]=dict(role_mean_seconds=means,controls=controls,controls_passed=controls_pass,bridges=bridges,bridges_passed=bridges_pass,
                    candidate_ratio=candidate,candidate_visits=visits,candidate_gates=candidate_gates,distribution=distribution,
                    native_ratios={r:means[r]/means['N'] for r in ['A','B','C']})
            result.append(dict(case=case,policy=policy,boundaries=boundaries))
    return dict(passed=bool(passed),phase=phase,cases=result,interpretation='Four fresh resident-process cohorts; all samples retained; no independent per-call or confidence claim')

def telemetry(state,records,jobs,smoke=False):
    assert state['complete'] is True and state['code']==0 and state['limits']==LIMITS
    assert [r['job'] for r in state['runs']]==jobs
    previous=state['started'];births=[state['supervisor']];summary=[]
    for run in state['runs']:
        job=run['job'];assert run['complete'] is True and run['code']==0
        assert previous<=run['started']<run['ended']<=state['ended'];previous=run['ended']
        assert 0<run['seconds']<LIMITS['seconds'] and list(run['workers'])==job['creation']
        assert [(e['role'],e['op'],e['index']) for e in run['events']]==commands(job,smoke)
        last=0.
        for event in run['events']:
            assert last<=event['started']<event['ended']<=run['seconds'];last=event['ended']
            assert event['ack']==[event['op'],event['index']]
        for role,worker in run['workers'].items():
            assert worker['code']==0 and worker['birth']>=run['started']-.05
            assert run['started']<=worker['started']<worker['ended']<=run['ended']
            births.append(dict(pid=worker['pid'],birth=worker['birth']))
        seen=set();paused={};last=-1.;peak=0;minimum=2**64
        samples=records[job['name']];assert len(samples)==run['samples']>0
        for row in samples:
            assert last<=row['seconds']<=run['seconds'];last=row['seconds']
            assert row['available']>=LIMITS['available'];minimum=min(minimum,row['available'])
            assert len({m['role'] for m in row['members']})==len(row['members'])
            assert row['active'] is None or row['active'] in ROLES
            for member in row['members']:
                role=member['role'];worker=run['workers'][role];seen.add(role)
                assert member['pid']==worker['pid'] and member['birth']==worker['birth'] and member['affinity']==[2]
                assert member['rss']>=0 and math.isfinite(member['cpu_seconds']) and member['cpu_seconds']>=0
                if role!=row['active']:
                    assert member['suspended'] is True
                    if not smoke:assert member['status']=='stopped'
                    if role not in paused:paused[role]=member['cpu_seconds']
                    assert -.000001<=member['cpu_seconds']-paused[role]<=.02
                else:paused.pop(role,None)
            rss=sum(m['rss'] for m in row['members']);assert rss<LIMITS['rss'];peak=max(peak,rss)
        assert seen==set(ROLES) and peak==run['peak_rss']
        summary.append(dict(job=job['name'],seconds=run['seconds'],samples=len(samples),peak_rss=peak,minimum_available=minimum))
    assert len({(b['pid'],b['birth']) for b in births})==len(births)
    return dict(births=births,cohorts=summary)

def audit(base,phase):
    meta=read(base/'frozen.json');assert meta['protocol']==PROTOCOL and meta['criteria']==CRITERIA and meta['schedule']==schedule()
    for name,wanted in meta['files'].items():assert pin(base/name)==wanted,name
    folder=base/('result-'+phase);state=read(folder/'identity.json')
    assert state['phase']==phase and state['frozen']==pin(base/'frozen.json')
    samples={j['name']:[json.loads(line) for line in (folder/j['name']/'samples.jsonl').read_text().splitlines()] for j in schedule()}
    resources=telemetry(state,samples,schedule())
    sys.path.insert(0,str(base));import campaign_processes as accounting
    for run,summary in zip(state['runs'],resources['cohorts'],strict=True):
        directory=folder/run['job']['name']
        foreign=accounting.foreign_fraction(read(directory/'pre.json'),read(directory/'post.json'),state['supervisor']['pid'])
        assert foreign==run['accounting'] and foreign['foreign_cpu_fraction']<=.02
        def cpu(name):return [int(v) for v in (directory/name).read_text().splitlines()[0].split()[1:]]
        delta=[b-a for a,b in zip(cpu('cpu-before.txt'),cpu('cpu-after.txt'),strict=True)]
        assert len(delta)>=8 and all(d>=0 for d in delta) and sum(delta[:8])>0
        steal=delta[7]/sum(delta[:8]);assert steal<=.005;summary.update(foreign=foreign,steal=steal)
    values=[]
    for job in schedule():
        for role in ROLES:
            values.append(worker(folder/job['name']/role/'output',base/'inputs',meta['model'],meta['files']['bin/InterleavedProcesses.dll'],
                meta['files']['bin/Microsoft.ML.OnnxRuntime.dll'],meta['native'],job|dict(role=role),phase))
    for index in range(5):
        for native in [False,True]:
            group=[v for v in values if v['specification']['case_index']==index and (v['specification']['role']=='N')==native]
            assert len({v['output_sha256'] for v in group})==1
            if not native:assert len({v['fingerprint'] for v in group})==1
    return dict(passed=True,phase=phase,frozen=pin(base/'frozen.json'),identity=pin(folder/'identity.json'),resources=resources,timing=evaluate(values,phase),
        measured_calls=sum(len(v['measured']) for v in values),conditioning_calls=sum(len(v['conditioning']) for v in values),
        solo_calls=sum(len(v['solo']) for v in values),maximum_native_error=max(max(v['before_error'],v['after_error']) for v in values))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--payload',type=Path,required=True);p.add_argument('--phase',choices=['aa','compare'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();result=audit(a.payload.resolve(),a.phase);write(a.output,result)
    print(json.dumps(dict(passed=True,timing_passed=result['timing']['passed'],measured_calls=result['measured_calls'],solo_calls=result['solo_calls'])))
