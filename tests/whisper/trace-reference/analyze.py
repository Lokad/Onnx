"""Read-only all-boundary comparison of selected FP32 traces and both closed FP64 references."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import hashlib,json,math,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil
BASE=ROOT/'artifacts/whisper-trace-reference-20260920'
TRACE=ROOT/'artifacts/whisper-trace-selected-20260920'
REF=ROOT/'artifacts/whisper-full-reference-20260920'
LIMIT=1e-4


def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def write(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def compare(a,b):
    maximum=absolute=0.;worst=absolute_worst=0;failed=0;squares=[]
    assert a.size==b.size
    for offset in range(0,a.size,131072):
        x=np.asarray(a[offset:offset+131072],dtype=np.float64);y=np.asarray(b[offset:offset+131072],dtype=np.float64)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        delta=x-y;d=np.abs(delta);scaled=d/np.maximum(1,np.abs(y))
        i=int(scaled.argmax());j=int(d.argmax())
        if scaled[i]>maximum:maximum=float(scaled[i]);worst=offset+i
        if d[j]>absolute:absolute=float(d[j]);absolute_worst=offset+j
        failed+=int(np.count_nonzero(scaled>LIMIT));squares.append(float(np.sum(delta*delta)))
    return dict(values=int(a.size),max_scaled=maximum,scaled_flat_index=worst,max_absolute=absolute,absolute_flat_index=absolute_worst,failed_values=failed,rms=math.sqrt(math.fsum(squares)/a.size))


def main():
    assert not BASE.exists();BASE.mkdir();started=time.monotonic();process=psutil.Process()
    tc=read(TRACE/'closed.json');rc=read(REF/'closed.json');assert tc['passed'] and rc['structural_passed'] and rc['reference_passed']
    assert pin(REF/'closed.json')['sha256']=='24b132085567d6f84cec2582ef3910cac892533191a353d4bc25af6e5769ff0e'
    assert pin(TRACE/'manifest.json')['sha256']=='eddcb6c3bc31d93e6d991724e7e78589272e6c56c7538adc20bab4b98f1b9c6f'
    inputs={};checked={}
    def closed_file(base,closed,name):
        p=base/name;relative=p.relative_to(ROOT).as_posix()
        if relative not in checked:
            assert pin(p)==closed['files'][name],relative
            checked[relative]=closed['files'][name]
        return p
    for base in [TRACE,REF]:inputs[(base/'closed.json').relative_to(ROOT).as_posix()]=pin(base/'closed.json')
    tm=read(closed_file(TRACE,tc,'manifest.json'));rm=read(closed_file(REF,rc,'manifest.json'))
    assert tm['outputs']==rm['outputs'] and len(tm['outputs'])==41 and rm['original_limit']==LIMIT
    assert tm['scaled_error_limit']==LIMIT
    refs={(j['request'],j['features'],j['engine']):j for j in rm['jobs']}
    rows=[];paths=[];repeat={};resources=[]
    with (BASE/'comparisons.jsonl').open('x') as stream:
        for job in tm['schedule']:
            request=tm['requests'][job['request']];assert rm['requests'][request['original_request']]['name']==request['name']
            result=read(closed_file(TRACE,tc,'outputs/'+job['id']+'/result.json'));assert result['complete']
            for call in result['records']:
                kind=call['kind'];feature='managed' if kind[1]=='M' else 'native'
                assert call['input_sha256']==request[feature+'_features']['raw_sha256']
                ref_results={}
                for engine in ['numpy','ort']:
                    rjob=refs[request['original_request'],feature,engine];assert rjob['input']==request[feature+'_features']
                    rr=read(closed_file(REF,rc,'outputs/'+rjob['id']+'/result.json'));assert rr['complete'] and rr['input_unchanged'] and rr['input_sha256']==call['input_sha256']
                    ref_results[engine]=(rjob,rr)
                path_rows=[]
                for index,desc in enumerate(tm['outputs']):
                    item=call['outputs'][index];assert item['index']==index and item['name']==desc['name'] and item['shape']==desc['shape']
                    actual=closed_file(TRACE,tc,'outputs/'+job['id']+'/'+item['file'])
                    assert checked[actual.relative_to(ROOT).as_posix()]==dict(bytes=math.prod(desc['shape'])*4,sha256=item['sha256'])
                    if job['request']==0:repeat[kind,index]=item['sha256']
                    if job['request']==3:assert repeat[kind,index]==item['sha256']
                    a=np.memmap(actual,dtype='<f4',mode='r')
                    for engine,(rjob,rr) in ref_results.items():
                        ri=rr['outputs'][index];assert ri['index']==index and ri['name']==desc['name'] and ri['shape']==desc['shape']
                        reference=closed_file(REF,rc,'outputs/'+rjob['id']+'/'+ri['file'])
                        assert checked[reference.relative_to(ROOT).as_posix()]==ri['pin'] and ri['pin']['bytes']==math.prod(desc['shape'])*8
                        b=np.memmap(reference,dtype='<f8',mode='r');metric=compare(a,b);del b
                        row=dict(request=job['request'],original_request=request['original_request'],name=request['name'],kind=kind,reference=engine,index=index,output=desc['name'],shape=desc['shape'],actual=actual.relative_to(ROOT).as_posix(),expected=reference.relative_to(ROOT).as_posix(),**metric)
                        rows.append(row);path_rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                    del a
                    resource=dict(seconds=time.monotonic()-started,rss=process.memory_info().rss,available=psutil.virtual_memory().available)
                    assert resource['seconds']<900 and resource['rss']<2*1024**3 and resource['available']>=1024**3
                    resources.append(resource)
                paths.append(dict(request=job['request'],name=request['name'],kind=kind,instrumentation=call['instrumentation'],
                    references={engine:dict(first_failure=next((r['index'] for r in path_rows if r['reference']==engine and r['failed_values']),None),
                    failed_boundaries=sum(r['failed_values']>0 for r in path_rows if r['reference']==engine),final=next(r for r in path_rows if r['reference']==engine and r['index']==40)) for engine in ['numpy','ort']}))
                print(json.dumps(dict(request=job['request'],kind=kind,first={k:v['first_failure'] for k,v in paths[-1]['references'].items()})),flush=True)
    assert len(rows)==1312 and len(paths)==16 and len(resources)==656
    inputs.update(checked);write(BASE/'inputs.json',inputs);write(BASE/'resources.json',resources)
    write(BASE/'observations.json',dict(structural_passed=True,limit=LIMIT,comparisons=len(rows),paths=paths,inputs=len(inputs),input_bytes=sum(v['bytes'] for v in inputs.values()),resource_samples=len(resources),maximum_rss=max(r['rss'] for r in resources),seconds=time.monotonic()-started,source=subprocess_revision(),numpy=np.__version__))


def subprocess_revision():
    import subprocess
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


if __name__=='__main__':main()
