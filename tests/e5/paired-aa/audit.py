"""Independent complete-output, schedule and all-sample timing checks for paired A/A."""
from pathlib import Path
import argparse,collections,json,math,statistics
import numpy as np
from prepare_inputs import CASES,CORE,sha,read,pin,input_hash,write_new

def schedule(visit,index,smoke=False):
    pairs,solo,seconds=(4,4,1) if smoke else (64,32,30)
    order=[i%2 for i in range(pairs)];state=20260920+100*visit+index
    for i in range(pairs-1,0,-1):
        state=(state*1664525+1013904223)&0xffffffff;j=state%(i+1);order[i],order[j]=order[j],order[i]
    creation=[0,1] if visit%2==0 else [1,0]
    return dict(schema=1,protocol='paired-managed-aa-v1',visit=visit,case_index=index,smoke=smoke,creation=creation,
        pair_first=order,solo_order=creation,solo_before_pairs=visit%2!=0,conditioning_seconds=seconds,pairs=pairs,solo=solo)

def measured_order(s):
    solo=[(a,'solo',i,p) for p,a in enumerate(s['solo_order']) for i in range(s['solo'])]
    paired=[(a,'paired',i,p) for i,first in enumerate(s['pair_first']) for p,a in enumerate((first,1-first))]
    return solo+paired if s['solo_before_pairs'] else paired+solo

def rows_valid(value,smoke):
    s=value['schedule'];assert s==schedule(s['visit'],s['case_index'],smoke)
    assert type(value['frequency']) is int and value['frequency']>0
    assert [(r['arm'],r['phase'],r['group'],r['position']) for r in value['measured']]==measured_order(s)
    conditioned=[0.,0.];position=0;round_index=0
    while any(v<s['conditioning_seconds'] for v in conditioned):
        for p,a in enumerate(s['creation']):
            if conditioned[a]>=s['conditioning_seconds']:continue
            row=value['conditioning'][position];position+=1
            assert (row['arm'],row['phase'],row['group'],row['position'])==(a,'conditioning',round_index,p)
            conditioned[a]+=row['execute']/value['frequency']
        round_index+=1
    assert position==len(value['conditioning']) and conditioned==value['conditioned']
    for row in value['conditioning']+value['measured']:
        for key in ('execute','request','bytes','g0','g1','g2'):assert type(row[key]) is int and row[key]>=0
        assert 0<row['execute']<=row['request']<600*value['frequency']
    return s

def outputs(base,inputs,expected_binaries,smoke=False):
    value=read(base/'result.json');assert value['schema']==1 and value['protocol']=='paired-managed-aa-v1' and value['passed'] is True
    s=rows_valid(value,smoke);assert read(base/'schedule.json')==s
    name=CASES[s['case_index']];case_path=inputs/(name+'.json');case=read(case_path)
    assert value['case_sha256']==sha(case_path) and case['name']==name and input_hash(case['inputs'])==case['input_sha256']
    assert value['affinity']==4 and value['settings']==[] and value['private_cores'] is True and value['static_isolation'] is True
    assert value['avx2'] is True and (smoke or (value['avx512'] is True and value['runtime']=='.NET 10.0.8'))
    assert value['host_sha256']==expected_binaries['PairedHost.dll']['sha256']
    assert len(value['flags'])==len(value['initial'])==len(value['final'])==2 and value['flags'][0]==value['flags'][1]
    assert value['flags'][0]['EnableBiasGeluInterleaved'] is False and value['flags'][0]['LegacySoftmaxUsed'] is False
    for key in ('EnablePackedAvx512Rows','EnableDeferredReleaseCache','EnableReleasedBufferCache','EnableFusedTempRelease',
        'EnableSoftmaxExpPrune','EnableSoftmaxExpInline','EnableSoftmaxNonpositive','EnableBiasGeluInline','EnableVectorTransposeFaces'):
        assert value['flags'][0][key] is True,key
    reference=inputs/case['reference_file'];assert sha(reference)==case['reference_sha256']
    native=np.fromfile(reference,dtype='<f4');assert native.size==math.prod(case['shape']) and np.isfinite(native).all()
    maximum=0.;hashes=[]
    for arm in range(2):
        initial,final=value['initial'][arm],value['final'][arm]
        assert initial['name']==final['name']==name and initial['options']=='Memory'
        assert initial['inputs']==case['inputs'] and initial['input_sha256']==case['input_sha256']
        assert initial['shape']==case['shape'] and initial['reference_sha256']==case['reference_sha256']
        assert initial['core_sha256']==CORE and initial['bridge_sha256']==expected_binaries['PairedBridge.dll']['sha256']
        assert final['inputs_unchanged'] is True and final['held_outputs_unchanged'] is True
        for stage in ('before','after'):
            path=base/('arm'+str(arm))/(stage+'.f32');a=np.fromfile(path,dtype='<f4')
            assert a.shape==native.shape and np.isfinite(a).all()
            error=float(np.max(np.abs(a.astype(np.float64)-native)/np.maximum(1,np.abs(native.astype(np.float64)))))
            assert error<=1e-4 and error==(initial['before_error'] if stage=='before' else final['after_error'])
            assert sha(path)==final['before_sha256' if stage=='before' else 'output_sha256']
            maximum=max(maximum,error);hashes.append(sha(path))
    assert len(set(hashes))==1,'Complete output bits differ'
    return value,maximum

def mean(records,arm,boundary,phase='paired',first=None):
    samples=[r[boundary]/v['frequency'] for v in records for r in v['measured']
        if r['arm']==arm and r['phase']==phase and (first is None or v['schedule']['pair_first'][r['group']]==first)]
    assert samples and all(math.isfinite(x) and x>0 for x in samples)
    return statistics.mean(samples)

def timing(records):
    assert len(records)==20
    assert collections.Counter((v['schedule']['visit'],v['schedule']['case_index']) for v in records)==collections.Counter((v,i) for v in range(4) for i in range(5))
    observations=[];passed=True
    for index,name in enumerate(CASES):
        group=[v for v in records if v['schedule']['case_index']==index]
        row=dict(name=name,boundaries={})
        for boundary in ('execute','request'):
            a,b=mean(group,0,boundary),mean(group,1,boundary);ratio=b/a
            visits=[mean([v],1,boundary)/mean([v],0,boundary) for v in group]
            order_ratios=[mean(group,1,boundary,first=first)/mean(group,0,boundary,first=first) for first in (0,1)]
            contrast=order_ratios[0]/order_ratios[1]
            paired_solo=[mean(group,arm,boundary)/mean(group,arm,boundary,'solo') for arm in (0,1)]
            gates=dict(aggregate=.995<=ratio<=1.005,visits=all(.99<=v<=1.01 for v in visits),
                order=.99<=contrast<=1.01,paired_solo=all(.95<=v<=1.05 for v in paired_solo))
            passed&=all(gates.values())
            row['boundaries'][boundary]=dict(a_seconds=a,b_seconds=b,ratio=ratio,visits=visits,
                order_ratios=order_ratios,order_contrast=contrast,paired_solo=paired_solo,gates=gates)
        observations.append(row)
    return dict(passed=passed,observations=observations,scope='Prospective managed A/A eligibility screens, not confidence intervals or ORT parity')

def resources(identity,collection,collected):
    assert identity['complete'] is True and 'error' not in identity and identity['supervisor']['affinity']=='0'
    assert identity['limits']==dict(rss=8*1024**3,seconds=600,available_memory=1024**3)
    assert collection['complete'] is True and collection['code']==0 and collection['checkout']=='172181fc5ab4eb2bdc2eb7f37e80d25e482a0887'
    expected=[(v,i) for v in range(4) for i in (range(5) if v%2==0 else reversed(range(5)))]
    assert len(identity['runs'])==len(expected)
    all_births={(identity['supervisor']['pid'],identity['supervisor']['start'])}
    peak=0;minimum=math.inf;count=0;previous_end=identity['started']
    for row,(visit,index) in zip(identity['runs'],expected,strict=True):
        assert row['name']==f'v{visit}-{CASES[index]}' and row['code']==0 and 0<row['seconds']<600
        assert previous_end<=row['started']<=row['ended']<=identity['ended'];previous_end=row['ended']
        samples=[json.loads(s) for s in (collected/'result'/(row['name']+'-samples.jsonl')).read_text(encoding='utf-8').splitlines()]
        assert len(samples)==row['samples'] and len(samples)>1
        births={};last=-1;worker_peak=0;cpu={}
        for sample in samples:
            assert last<=sample['seconds']<row['seconds'];last=sample['seconds']
            assert sample['available_memory']>=1024**3;minimum=min(minimum,sample['available_memory'])
            seen=set()
            for member in sample['members']:
                pid,birth=member['pid'],member['start']
                assert pid not in seen and member['group']==row['pid'] and member['affinity']=='2' and member['state']!='Z'
                assert birth>=row['start'] and births.get(str(pid),birth)==birth and member['rss']>=0
                assert member['cpu_seconds']>=cpu.get(pid,0);cpu[pid]=member['cpu_seconds']
                if pid==row['pid']:assert birth==row['start']
                births[str(pid)]=birth;seen.add(pid)
            rss=sum(m['rss'] for m in sample['members']);assert rss<8*1024**3;worker_peak=max(worker_peak,rss)
        assert births==row['members'] and births[str(row['pid'])]==row['start'] and worker_peak==row['peak_rss']
        assert row['accounting']['valid'] and math.isfinite(row['accounting']['foreign_cpu_fraction'])
        all_births.update((int(pid),birth) for pid,birth in births.items());peak=max(peak,worker_peak);count+=len(samples)
    assert all_births=={(v['pid'],v['start']) for v in collection['terminal_processes']}
    return dict(peak_rss=peak,minimum_available_memory=minimum,samples=count,
        terminal_processes=[dict(pid=pid,start=birth) for pid,birth in sorted(all_births)],
        maximum_foreign_cpu_fraction=max(r['accounting']['foreign_cpu_fraction'] for r in identity['runs']))

def campaign(base):
    payload=base/'payload';collected=base/'collected';bundle=read(payload/'bundle.json');collection=read(collected/'collection.json')
    assert sha(payload/'bundle.json')==sha(collected/'bundle.json')==read(base/'preparation.json')['bundle_sha256']
    assert sha(collected/'collection.json')==read(base/'download.json')['collection_sha256']
    assert {p.relative_to(collected).as_posix() for p in collected.rglob('*') if p.is_file()}==set(collection['files'])|{'collection.json'}
    for name,wanted in bundle['files'].items():
        assert pin(payload/name)==wanted,name
        if name in collection['files']:assert pin(collected/name)==wanted,name
    for name,wanted in collection['files'].items():assert pin(collected/name)==wanted,name
    binaries={name[4:]:wanted for name,wanted in bundle['files'].items() if name.startswith('bin/')}
    identity=read(collected/'result/identity.json');assert identity['bundle_sha256']==sha(payload/'bundle.json')
    limits=resources(identity,collection,collected);values=[];maximum=0
    for row in identity['runs']:
        value,error=outputs(collected/'result'/row['name'],payload/'inputs',binaries)
        assert row['name']==f"v{value['schedule']['visit']}-{CASES[value['schedule']['case_index']]}"
        values.append(value);maximum=max(maximum,error)
    conclusion=timing(values)
    return dict(schema=1,execution_passed=True,timing=conclusion,maximum_error=maximum,resources=limits,
        measured_calls=sum(len(v['measured']) for v in values),conditioning_calls=sum(len(v['conditioning']) for v in values),
        bundle_sha256=sha(payload/'bundle.json'),collection_sha256=sha(collected/'collection.json'),auditor_sha256=sha(Path(__file__)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();base=a.artifact.resolve();assert not a.output.exists()
    if a.smoke:
        process=read(base/'smoke-process.json');assert process['complete'] and process['code']==0
        for name,wanted in process['binaries'].items():assert pin(base/'bin'/name)==wanted
        for name,wanted in process['source'].items():assert pin(base/'smoke-source'/name)==wanted
        value,error=outputs(base/'smoke',base/'inputs',process['binaries'],True)
        assert process['samples'] and all(s['affinity']==[2] and s['birth']==process['birth'] and s['rss']<8*1024**3 and s['available']>=1024**3 and s['seconds']<300 for s in process['samples'])
        write_new(a.output,dict(passed=True,maximum_error=error,measured_calls=len(value['measured']),
            process_sha256=sha(base/'smoke-process.json'),auditor_sha256=sha(Path(__file__)),scope='Windows functional smoke only'))
        print('Independent smoke audit passed; maximum error',error)
    else:
        result=campaign(base);write_new(a.output,result)
        print(json.dumps(result,indent=2))
