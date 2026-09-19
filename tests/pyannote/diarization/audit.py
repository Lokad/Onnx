"""Audit every raw pipeline value; exit 1 means numerical failure, not incomplete evidence."""
from pathlib import Path
import argparse,json,math
import numpy as np
from evidence import PINS,load,local,sha,require,detail_keys

def public_case(row,c):
    value=row['result']
    require(value['Status']==['Completed','NoSpeech','NoUsableEmbeddings'].index(c['status']),'Public status')
    require(value['AudioDuration']==c['seconds'] and value['Windows']==len(c['windows']),'Public geometry')
    for field,key in [('Intervals','intervals'),('ExclusiveIntervals','exclusive_intervals')]:
        require(len(value[field])==len(c[key]),'Interval coverage')
        for a,(s,e,k) in zip(value[field],c[key]):
            require(math.isfinite(a['Start']) and math.isfinite(a['End']) and abs(a['Start']-s)<=1e-12 and abs(a['End']-e)<=1e-12 and a['Speaker']==k,'Interval value')
            require(0<=a['Start']<a['End']<=c['seconds'],'Interval bounds')
    expected=c.get('speakers',[]);require(len(value['Speakers'])==len(expected),'Speaker coverage');error=0.
    for a,e in zip(value['Speakers'],expected):
        require(a['Speaker']==e['speaker'] and a['HasEmbedding']==e['has_embedding'],'Speaker identity')
        x=np.asarray(a['Centroid']);y=np.asarray(e['centroid'])
        require(x.shape==y.shape==(256,) and np.isfinite(x).all() and np.isfinite(y).all(),'Centroid layout')
        error=max(error,float(np.max(np.abs(x-y)/np.maximum(1,np.abs(y)))))
    require(error<=PINS['tolerance'],'Centroid gate')
    return error

def audit(reference,public,detail):
    m,arrays=load(reference);r=json.loads(public.read_text(encoding='utf-8'));d=json.loads((detail/'result.json').read_text(encoding='utf-8'))
    digest=sha(reference/'manifest.json')
    require(r['reference_sha256']==d['reference_sha256']==digest,'Result reference identity')
    for assembly in ('Lokad.Onnx','Lokad.Onnx.Data'):
        require(r['assemblies'][assembly]==d['assemblies'][assembly],'Replay assembly mismatch')
    cases={c['name']:c for c in m['cases']};seen=set();first={};maximum=0.
    expected={(c['name'],i) for c in m['cases'] for i in range(2)}|{(m['cases'][0]['name'],2)}
    for row in r['reports']:
        key=(row['name'],row['repeat']);require(key in expected and key not in seen,'Public coverage');seen.add(key)
        error=public_case(row,cases[row['name']]);maximum=max(maximum,error)
        require(error==row['error'] and row['passed'] and row['exact_timeline'],'Public summary')
        if row['repeat']==0:first[row['name']]=row['result']
        else:require(first[row['name']]==row['result'],'Repeated output')
    require(seen==expected and r['maximum']==maximum and r['passed'] and r['held_outputs_unchanged'],'Public completion')
    require(r['refusals']==['sample-rate','nonfinite','range','canceled'],'Request refusals')
    require(r['empty']==dict(Intervals=[],ExclusiveIntervals=[],Speakers=[],Status=1,AudioDuration=0,Windows=0),'Empty request')
    require(len(r['concurrent'])==2 and all(v==first[m['cases'][0]['name']] for v in r['concurrent']),'Concurrent requests')
    expected=set(detail_keys(m));seen=set();rows=[];bad=values=0;maximum_numeric=0.;raw_files=set()
    require(d['execution_complete'],'Detail execution')
    for row in d['reports']:
        key=(row['name'],row['stage'],row['reference']);require(key in expected and key not in seen,'Detail coverage');seen.add(key)
        path=local(detail,row['file']);require(row['file'] not in raw_files,'Reused detail file');raw_files.add(row['file'])
        require(sha(path)==row['sha256'] and path.stat().st_size==row['length']*4,'Detail identity')
        actual=np.fromfile(path,'<f4').astype(np.float64);native=arrays[row['reference']].astype(np.float64).reshape(-1)
        require(actual.shape==native.shape and np.isfinite(actual).all(),'Detail shape or finite values')
        error=np.abs(actual-native)/np.maximum(1,np.abs(native))
        gate=0 if row['stage'] in ('activity','masks','count','labels') else PINS['tolerance']
        n=int(np.count_nonzero(error>gate));b=float(error.max(initial=0));bad+=n;values+=len(actual);maximum_numeric=max(maximum_numeric,b)
        rows.append(dict(name=row['name'],stage=row['stage'],values=len(actual),bad=n,maximum=b))
    require(seen==expected,'Incomplete detail result')
    require({p.name for p in detail.glob('*.f32')}==raw_files,'Unreported detail file')
    ties=[]
    for c in m['cases']:
        if 'labels' not in c:continue
        for kind in ('ordinary','exclusive'):
            a=arrays[c[kind+'_native_frames']];b=arrays[c[kind+'_frames']]
            ties.append(dict(name=c['name'],kind=kind,changed=int(np.count_nonzero(a!=b))))
    return dict(execution_complete=True,application_gate_passed=True,numerical_gate_passed=bad==0,public_requests=2*len(m['cases'])+3,
        reference_sha256=digest,public_sha256=sha(public),detail_sha256=sha(detail/'result.json'),assemblies=r['assemblies'],
        maximum_centroid_error=maximum,numerical=dict(arrays=len(rows),values=values,bad=bad,maximum=maximum_numeric,rows=rows),native_tie_differences=ties)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','public','detail','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'Choose a fresh audit output')
    result=audit(a.reference,a.public,a.detail)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('numerical','assemblies','native_tie_differences')},indent=2))
    print('Numerical:',{k:v for k,v in result['numerical'].items() if k!='rows'})
    raise SystemExit(0 if result['numerical_gate_passed'] else 1)
