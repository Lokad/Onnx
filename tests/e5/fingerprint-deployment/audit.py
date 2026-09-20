"""Independent full-array, configuration, timing and owned-process evidence checks."""
from pathlib import Path
import argparse,hashlib,itertools,json,math,statistics,struct,sys
import numpy as np
from protocol import CASES,POLICIES,ROLES,ORDERS,CORE,NATIVE,PROTOCOL,LIMITS,CRITERIA,schedule,specification

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
    assert 0<len(v['conditioning'])<=20000
    assert [(r['block'],r['call']) for r in v['conditioning']]==[(-2,i) for i in range(len(v['conditioning']))]
    total=0.
    for r in v['conditioning']:
        assert total<s['conditioning_seconds'];total+=r['execute']/freq
    assert total==v['conditioned'] and total>=s['conditioning_seconds']
    assert type(v['conditioning_wall_ticks']) is int and sum(r['request'] for r in v['conditioning'])<=v['conditioning_wall_ticks']<=91*freq
    for r in [v['first']]+v['conditioning']+v['measured']:
        for k in ['block','call','execute','request','bytes','g0','g1','g2']:assert type(r[k]) is int
        assert 0<r['execute']<=r['request']<=300*freq
        assert all(r[k]>=0 for k in ['call','bytes','g0','g1','g2'])

def worker(folder,inputs,model,probe,ort_managed,native_pin,job,phase,smoke=False):
    v=read(folder/'result.json');assert v['specification']==specification(job,phase,smoke);rows(v)
    s=v['specification'];native=job['role']=='N';enabled=phase=='compare' and job['role']=='C'
    assert v['passed'] is True and v['enabled'] is enabled and v['core_sha256']==CORE
    assert v['probe_sha256']==probe['sha256'] and v['ort_managed_sha256']==ort_managed['sha256']
    fixture_path=inputs/(job['case']+'.json');fixture=read(fixture_path)
    assert fixture['name']==job['case'] and v['fixture_sha256']==pin(fixture_path)['sha256']
    assert v['model_sha256']==fixture['model_sha256']==model['sha256']
    assert v['input_sha256']==fixture['input_sha256']==input_hash(fixture)
    assert v['reference_sha256']==fixture['reference_sha256']
    assert v['shape']==fixture['shape']==[1,[8,30,128,128,512][job['case_index']],384]
    assert v['affinity']==4 and v['processor_count']==1
    assert v['flags']==({'LOKAD_ONNX_FINGERPRINT_STRINGS':'1'} if enabled else {})
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
    jobs=schedule();assert len(values)==len(jobs)==160
    for v,j in zip(values,jobs,strict=True):assert v['specification']==specification(j,phase);rows(v)
    result=[];passed=True
    for index,name in enumerate(CASES):
        for policy in POLICIES:
            group=[(j,v) for j,v in zip(jobs,values,strict=True) if j['case_index']==index and j['policy']==policy]
            def samples(role,boundary,visit=None,position=None):
                return [r[boundary]/v['frequency'] for j,v in group if j['role']==role and (visit is None or j['visit']==visit) and (position is None or j['position']==position) for r in v['measured']]
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
                means={r:mean(r,boundary) for r in ROLES};ratio=means['C']/statistics.fmean([means['A'],means['B']])
                visits=[mean('C',boundary,v)/statistics.fmean([mean(r,boundary,v) for r in ['A','B']]) for v in range(4)]
                controls_passed=all(all(c['gates'].values()) for c in controls)
                candidate_gates=dict(aggregate=ratio<=[.98,.99,1.01,1.01,1.01][index],visits=all(x<=1.02 for x in visits)) if phase=='compare' else {}
                passed &= controls_passed and all(candidate_gates.values())
                distribution={}
                for role in ROLES:
                    data=samples(role,boundary);raw=[r for j,v in group if j['role']==role for r in v['measured']]
                    distribution[role]=dict(calls=len(data),median=statistics.median(data),minimum=min(data),maximum=max(data),
                        allocated_bytes_mean=statistics.fmean(r['bytes'] for r in raw),gc_calls=sum(any(r[k] for k in ['g0','g1','g2']) for r in raw))
                boundaries[boundary]=dict(role_mean_seconds=means,controls=controls,controls_passed=controls_passed,candidate_ratio=ratio,
                    candidate_visits=visits,candidate_gates=candidate_gates,distribution=distribution,
                    native_ratios={r:means[r]/means['N'] for r in ['A','B','C']},
                    native_visit_ratios={r:[mean(r,boundary,v)/mean('N',boundary,v) for v in range(4)] for r in ['A','B','C']})
            result.append(dict(name=name,policy=policy,boundaries=boundaries))
    return dict(passed=bool(passed),phase=phase,cases=result,interpretation='Fixed empirical screens on fresh sequential processes; descriptive native ratios, no calibrated confidence claim')

def telemetry(state,samples,jobs,limits=LIMITS):
    assert state['complete'] is True and state['code']==0 and state['limits']==limits
    assert [r['job'] for r in state['runs']]==jobs
    previous=state['started'];births=[state['supervisor']];summary=[]
    for run in state['runs']:
        assert all(math.isfinite(run[k]) for k in ['started','ended','seconds'])
        assert previous<=run['started']<run['ended']<=state['ended'];previous=run['ended']
        assert run['code']==0 and 0<run['seconds']<limits['seconds']
        assert run['members'][str(run['child']['pid'])]==run['child']['birth']
        rows=samples[run['job']['name']];assert len(rows)==run['samples']>0;last=-1.;peak=0;seen={}
        for row in rows:
            assert math.isfinite(row['seconds']) and last<=row['seconds']<=run['seconds'];last=row['seconds']
            assert type(row['available']) is int and row['available']>=limits['available']
            assert len({m['pid'] for m in row['members']})==len(row['members'])
            for m in row['members']:
                assert type(m['pid']) is int and m['pid']>0 and math.isfinite(m['birth']) and m['birth']>=run['child']['birth']
                assert type(m['rss']) is int and m['rss']>=0 and m['affinity']==[2]
                assert seen.get(str(m['pid']),m['birth'])==m['birth'];seen[str(m['pid'])]=m['birth']
            rss=sum(m['rss'] for m in row['members']);assert rss<limits['rss'];peak=max(peak,rss)
        assert peak==run['peak_rss'] and seen==run['members']
        births.extend(dict(pid=int(pid),birth=birth) for pid,birth in seen.items())
        summary.append(dict(job=run['job']['name'],seconds=run['seconds'],samples=len(rows),peak_rss=peak,minimum_available=min(r['available'] for r in rows)))
    return dict(births=births,workers=summary)

def resources(folder,meta,phase):
    state=read(folder/'identity.json');assert state['phase']==phase and state['frozen']==pin(folder.parent/'frozen.json')
    jobs=schedule();assert meta['schedule']==jobs and meta['limits']==LIMITS
    samples={j['name']:[json.loads(line) for line in (folder/j['name']/'samples.jsonl').read_text().splitlines()] for j in jobs}
    result=telemetry(state,samples,jobs)
    sys.path.insert(0,str(folder.parent));import campaign_processes as accounting
    for run,row in zip(state['runs'],result['workers'],strict=True):
        directory=folder/run['job']['name'];foreign=accounting.foreign_fraction(read(directory/'pre.json'),read(directory/'post.json'),state['supervisor']['pid'])
        assert foreign==run['accounting'] and foreign['foreign_cpu_fraction']<=.02
        def cpu(p):return [int(v) for v in p.read_text().splitlines()[0].split()[1:]]
        delta=[b-a for a,b in zip(cpu(directory/'cpu-before.txt'),cpu(directory/'cpu-after.txt'),strict=True)]
        assert len(delta)>=8 and all(v>=0 for v in delta) and sum(delta[:8])>0
        steal=delta[7]/sum(delta[:8]);assert steal<=.005
        row.update(foreign=foreign,steal=steal)
    return result

def audit(base,phase):
    meta=read(base/'frozen.json');assert meta['protocol']==PROTOCOL and meta['criteria']==CRITERIA and meta['schedule']==schedule()
    for name,want in meta['files'].items():assert pin(base/name)==want,name
    folder=base/('result-'+phase);r=resources(folder,meta,phase)
    values=[worker(folder/j['name']/'output',base/'inputs',meta['model'],meta['files']['bin/FingerprintDeployment.dll'],meta['files']['bin/Microsoft.ML.OnnxRuntime.dll'],meta['native'],j,phase) for j in schedule()]
    for index in range(5):
        for native in [False,True]:
            group=[v for v in values if v['specification']['case_index']==index and (v['specification']['role']=='N')==native]
            assert len({v['output_sha256'] for v in group})==1
            if not native:assert len({v['fingerprint'] for v in group})==1
    return dict(passed=True,phase=phase,frozen=pin(base/'frozen.json'),identity=pin(folder/'identity.json'),resources=r,timing=evaluate(values,phase),
                measured_calls=sum(len(v['measured']) for v in values),conditioning_calls=sum(len(v['conditioning']) for v in values),maximum_native_error=max(max(v['before_error'],v['after_error']) for v in values))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--payload',type=Path,required=True);p.add_argument('--phase',choices=['aa','compare'],required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not args.output.exists();value=audit(args.payload.resolve(),args.phase);write(args.output,value)
    print(json.dumps(dict(passed=True,timing_passed=value['timing']['passed'],measured_calls=value['measured_calls'])))
