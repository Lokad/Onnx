"""Independent schedule, full-array and all-sample analysis of one prepared graph."""
from pathlib import Path
import argparse,hashlib,itertools,json,math,statistics,struct,sys
import numpy as np

CASES=['e5-8tok','e5-30tok','e5-30pad128','e5-128tok','e5-512tok']
CORE='48ca1d62ee2586d81072b8e347a671013d00abf8fe637c89eff65314e13cc710'
PERMUTATIONS=[list(p) for p in itertools.permutations(range(3))]

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)

def schedule(phase,visit,index):
    assert phase in ['smoke','aa','compare'] and 0<=visit<4 and 0<=index<5
    smoke=phase=='smoke';cycles=6 if smoke else 24;order=[i%6 for i in range(cycles)];state=20260920+100*visit+index
    for i in range(cycles-1,0,-1):
        state=(state*1664525+1013904223)&0xffffffff;j=state%(i+1);order[i],order[j]=order[j],order[i]
    return dict(protocol='single-graph-fingerprint-v1',phase=phase,visit=visit,case_index=index,cycles=cycles,calls=2 if smoke else [32,16,4,4,2][index],
                conditioning_seconds_per_setting=.1 if smoke else 15,order=order,permutations=PERMUTATIONS)

def rows(value):
    s=value['schedule'];assert s==schedule(s['phase'],s['visit'],s['case_index'])
    assert type(value['frequency']) is int and value['frequency']>0
    expected=[(role,'measured',cycle,position,call,s['phase']!='aa' and role==2)
              for cycle,p in enumerate(s['order']) for position,role in enumerate(PERMUTATIONS[p]) for call in range(s['calls'])]
    def key(row):return tuple(row[k] for k in ['role','stage','cycle','position','call','enabled'])
    assert [key(r) for r in value['measured']]==expected
    assert [key(r) for r in value['settling']]==[(2,'settling',i,0,0,True) for i in range(4)]
    conditioned=[0.,0.];cursor=0;cycle=0
    while any(v<s['conditioning_seconds_per_setting'] for v in conditioned):
        for position in range(2):
            setting=(cycle+position+s['visit'])%2
            if conditioned[setting]>=s['conditioning_seconds_per_setting']:continue
            row=value['conditioning'][cursor];cursor+=1
            assert key(row)==(setting,'conditioning',cycle,position,0,bool(setting))
            assert type(row['execute']) is int and row['execute']>0
            conditioned[setting]+=row['execute']/value['frequency']
        cycle+=1
    assert cursor==len(value['conditioning']) and conditioned==value['conditioned']
    for row in value['settling']+value['conditioning']+value['measured']:
        assert type(row['enabled']) is bool
        for k in ['role','cycle','position','call','execute','request','bytes','g0','g1','g2']:assert type(row[k]) is int and row[k]>=0
        assert 0<row['execute']<=row['request']<=300*value['frequency']

def worker(directory,inputs,model_pin,probe_pin,phase,visit,index):
    value=read(directory/'result.json');rows(value)
    assert value['schedule']==schedule(phase,visit,index)==read(directory/'schedule.json')
    fixture=read(inputs/(CASES[index]+'.json'))
    assert value['passed'] is True and value['core_sha256']==CORE and value['probe_sha256']==probe_pin['sha256']
    assert value['model_sha256']==fixture['model_sha256']==model_pin['sha256'] and value['case_sha256']==pin(inputs/(CASES[index]+'.json'))['sha256']
    assert value['input_sha256']==fixture['input_sha256'] and value['reference_sha256']==fixture['reference_sha256']
    assert value['affinity']==4 and value['processor_count']==1
    assert value['flags']=={'LOKAD_ONNX_FINGERPRINT_STRINGS':'1'}
    if phase!='smoke':assert value['runtime']=='10.0.8' and value['avx2'] is True and value['avx512'] is True
    for key in ['unchanged_cache','unchanged_inputs','unchanged_held_output','single_graph']:assert value[key] is True
    assert type(value['fingerprint']) is int and value['cache_entries']>0 and len(value['cache_sha256'])==64
    for key in ['load_ticks','preparation_ticks','preparation_bytes']:assert type(value[key]) is int and value[key]>=0
    reference=inputs/fixture['reference_file'];assert pin(reference)['sha256']==fixture['reference_sha256']
    want=np.fromfile(reference,dtype='<f4');assert np.isfinite(want).all()
    assert value['shape']==fixture['shape'] and want.size==math.prod(value['shape'])
    for name in ['before','after']:
        path=directory/(name+'.f32');actual=np.fromfile(path,dtype='<f4')
        assert actual.size==want.size and np.isfinite(actual).all()
        error=float((np.abs(actual.astype(np.float64)-want.astype(np.float64))/np.maximum(1,np.abs(want.astype(np.float64)))).max(initial=0))
        assert error<=1e-4 and abs(error-value[name+'_error'])<=1e-15
        assert pin(path)['sha256']==value['output_sha256']
    assert {p.name for p in directory.iterdir()}=={'result.json','schedule.json','before.f32','after.f32'}
    return value

def mean(values,role,boundary,position=None):
    samples=[r[boundary]/v['frequency'] for v in values for r in v['measured'] if r['role']==role and (position is None or r['position']==position)]
    assert samples;return statistics.fmean(samples)

def evaluate(values,phase):
    assert phase in ['aa','compare']
    assert sorted((v['schedule']['case_index'],v['schedule']['visit']) for v in values)==list(itertools.product(range(5),range(4)))
    result=[];passed=True
    for index,name in enumerate(CASES):
        group=sorted([v for v in values if v['schedule']['case_index']==index],key=lambda v:v['schedule']['visit']);boundaries={}
        for boundary in ['execute','request']:
            controls=[]
            for numerator,denominator in ([(1,0),(2,0),(2,1)] if phase=='aa' else [(1,0)]):
                aggregate=mean(group,numerator,boundary)/mean(group,denominator,boundary)
                visits=[mean([v],numerator,boundary)/mean([v],denominator,boundary) for v in group]
                positions=[mean(group,numerator,boundary,p)/mean(group,denominator,boundary,p) for p in range(3)]
                contrast=max(positions)/min(positions)
                gates=dict(aggregate=.995<=aggregate<=1.005,workers=all(.99<=r<=1.01 for r in visits),
                           positions=all(.99<=r<=1.01 for r in positions),position_contrast=contrast<=1.01)
                controls.append(dict(numerator=numerator,denominator=denominator,ratio=aggregate,workers=visits,positions=positions,position_contrast=contrast,gates=gates))
            controls_passed=all(all(c['gates'].values()) for c in controls)
            role_means=[mean(group,r,boundary) for r in range(3)]
            ratio=role_means[2]/statistics.fmean(role_means[:2]);visits=[mean([v],2,boundary)/statistics.fmean([mean([v],r,boundary) for r in [0,1]]) for v in group]
            limit=[.98,.99,1.01,1.01,1.01][index]
            candidate_gates=dict(aggregate=ratio<=limit,workers=all(r<=1.02 for r in visits)) if phase=='compare' else {}
            passed &= controls_passed and all(candidate_gates.values())
            distribution={str(role):dict(calls=len(samples:=[r[boundary]/v['frequency'] for v in group for r in v['measured'] if r['role']==role]),
                median=statistics.median(samples),maximum=max(samples),minimum=min(samples),
                allocated_bytes_mean=statistics.fmean(r['bytes'] for v in group for r in v['measured'] if r['role']==role),
                gc_calls=sum(any(r[k] for k in ['g0','g1','g2']) for v in group for r in v['measured'] if r['role']==role)) for role in range(3)}
            boundaries[boundary]=dict(role_mean_seconds=role_means,controls=controls,controls_passed=controls_passed,candidate_ratio=ratio,
                candidate_workers=visits,candidate_limit=limit if phase=='compare' else None,candidate_gates=candidate_gates,distribution=distribution)
        result.append(dict(name=name,boundaries=boundaries))
    return dict(passed=bool(passed),phase=phase,cases=result,interpretation='Descriptive empirical screens; common prepared state; no confidence or ORT parity claim')

def resources(folder,meta,phase):
    state=read(folder/'identity.json');assert state['complete'] is True and state['code']==0 and state['phase']==phase
    assert state['limits']==meta['limits'] and [r['job'] for r in state['runs']]==meta['schedule']
    sys.path.insert(0,str(folder.parent));import campaign_processes as accounting
    births=[state['supervisor']];summaries=[]
    for run in state['runs']:
        directory=folder/run['job']['name'];assert run['code']==0 and 0<run['seconds']<300
        assert run['members'][str(run['child']['pid'])]==run['child']['birth']
        births.extend(dict(pid=int(pid),birth=birth) for pid,birth in run['members'].items())
        samples=[json.loads(line) for line in (directory/'samples.jsonl').read_text().splitlines()]
        assert len(samples)==run['samples']>0;previous=-1.;peak=0;minimum=math.inf
        for sample in samples:
            assert previous<=sample['seconds']<300;previous=sample['seconds']
            assert sample['available']>=1024**3;minimum=min(minimum,sample['available'])
            for item in sample['members']:assert run['members'][str(item['pid'])]==item['birth'] and item['affinity']==[2] and item['rss']>=0
            rss=sum(item['rss'] for item in sample['members']);assert rss<6*1024**3;peak=max(peak,rss)
        assert peak==run['peak_rss']
        foreign=accounting.foreign_fraction(read(directory/'pre.json'),read(directory/'post.json'),state['supervisor']['pid'])
        assert foreign==run['accounting'] and foreign['foreign_cpu_fraction']<=.02
        def cpu(p):return [int(v) for v in p.read_text().splitlines()[0].split()[1:]]
        before,after=cpu(directory/'cpu-before.txt'),cpu(directory/'cpu-after.txt')
        delta=[b-a for a,b in zip(before,after,strict=True)];assert all(v>=0 for v in delta)
        steal=delta[7]/sum(delta[:8]);assert steal<=.005
        summaries.append(dict(job=run['job']['name'],seconds=run['seconds'],samples=len(samples),peak_rss=peak,minimum_available=minimum,foreign=foreign,steal=steal))
    return dict(births=births,workers=summaries)

def audit(base,phase):
    meta=read(base/'frozen.json')
    for name,wanted in meta['files'].items():assert pin(base/name)==wanted,name
    folder=base/('result-'+phase);state=read(folder/'identity.json')
    assert state['frozen']==pin(base/'frozen.json')
    r=resources(folder,meta,phase)
    values=[worker(folder/j['name']/'output',base/'inputs',meta['model'],meta['files']['bin/FingerprintModel.dll'],phase,j['visit'],j['case_index']) for j in meta['schedule']]
    for index in range(5):
        cases=[v for v in values if v['schedule']['case_index']==index]
        assert len({v['output_sha256'] for v in cases})==1 and len({v['fingerprint'] for v in cases})==1 and len({v['cache_sha256'] for v in cases})==1
    return dict(passed=True,phase=phase,frozen=pin(base/'frozen.json'),identity=pin(folder/'identity.json'),resources=r,timing=evaluate(values,phase),
        measured_calls=sum(len(v['measured']) for v in values),conditioning_calls=sum(len(v['conditioning']) for v in values),
        maximum_native_error=max(max(v['before_error'],v['after_error']) for v in values))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--payload',type=Path,required=True);p.add_argument('--phase',choices=['aa','compare'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();value=audit(a.payload.resolve(),a.phase);write(a.output,value)
    print(json.dumps(dict(passed=value['passed'],timing_passed=value['timing']['passed'],measured_calls=value['measured_calls'])))
