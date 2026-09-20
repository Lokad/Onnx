"""Independently audit complete saved LayerNorm arrays, synthetic inputs and ownership."""
from pathlib import Path
import argparse,hashlib,json,math,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
CORE='48ca1d62ee2586d81072b8e347a671013d00abf8fe637c89eff65314e13cc710'
CASES=['e5-8tok','e5-30tok','e5-30pad128','e5-128tok','e5-512tok']
WIDTHS=[1,2,3,4,7,8,9,15,16,17,23,24,25,31,32,33,63,64,65,383,384,385,1024,1536]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def sha(values):return hashlib.sha256(np.asarray(values,dtype='<f4').tobytes()).hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
def inventory(folder,meta,own):
    assert {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}==set(meta['files'])|{own}
    for name,wanted in meta['files'].items():assert pin(folder/name)==wanted,name
def identity(value,probe):
    assert value['core_sha256']==CORE and value['probe_sha256']==probe['sha256']
    assert value['runtime']=='10.0.12' and value['affinity']==4 and value['vector_width']==8 and value['settings']=={}
    assert value['vector512_hardware'] is False and value['avx512'] is False,'Local software-fallback proof scope'
def array(p,expected):
    v=np.fromfile(p,dtype='<f4');assert v.size==expected and np.isfinite(v).all(),p
    return v
def reference(x,scale,bias,epsilon):
    # Deliberately scalar accumulation, unlike the product's lane accumulators.
    result=np.empty(x.shape,dtype='<f4');eps=float(np.float32(epsilon))
    for row in range(x.shape[0]):
        mean=sum(float(v) for v in x[row])/x.shape[1]
        variance=sum((float(v)-mean)**2 for v in x[row])/x.shape[1]
        inv=1/math.sqrt(variance+eps)
        for i in range(x.shape[1]):result[row,i]=((float(x[row,i])-mean)*inv)*float(scale[i])+(0 if bias is None else float(bias[i]))
    return result
def scaled(actual,want):
    assert actual.shape==want.shape and np.isfinite(actual).all() and np.isfinite(want).all()
    return float((np.abs(actual.astype(np.float64)-want.astype(np.float64))/np.maximum(1,np.abs(want.astype(np.float64)))).max(initial=0))

def synthetic(records):
    state=20260920;cursor=0
    def bits():
        nonlocal state
        state=(state*1664525+1013904223)&0xffffffff;return state
    def finite(n):return np.array([(bits()>>8)/1048576-8 for _ in range(n)],dtype='<f4')
    def check(kind,x,s,b,width,rows,epsilon):
        nonlocal cursor
        r=records[cursor];cursor+=1
        assert r['name'].startswith(kind+':') and r['block']==width and r['outer']==rows and r['values']==width*rows
        assert np.float32(r['epsilon'])==np.float32(epsilon) and r['has_bias']==(b is not None)
        assert r['input_sha256']==sha(x) and r['scale_sha256']==sha(s) and r['bias_sha256']==(None if b is None else sha(b))
    for w in WIDTHS:
        for rows in [0,1,2,7]:
            for eps in [0,1e-12,1e-5]:
                for biased in [False,True]:check('shape',finite(w*rows),finite(w),finite(w) if biased else None,w,rows,eps)
    # The proof's constant-array initializer quiets its two literal signaling
    # NaNs before the recorded case name/hash. Runtime-generated random bits
    # below remain unmodified and include signaling NaNs.
    raw=[0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0xffc00000,0x7fe12345,0xffe54321,0x7f7fffff,0xff7fffff]
    exceptional=np.array(raw,dtype='<u4').view('<f4')
    for value in exceptional:
        for w in [17,32,65]:
            check('uniform-exceptional',np.full(2*w,value,dtype='<f4'),np.ones(w,dtype='<f4'),None,w,2,1e-5)
            check('bias-exceptional',finite(2*w),np.ones(w,dtype='<f4'),np.full(w,value,dtype='<f4'),w,2,1e-5)
    for center in np.array([0,1e-20,1,1e6,1e20],dtype='<f4'):
        for w in [16,17,384,385]:
            x=np.full(3*w,center,dtype='<f4');x[::3]=np.nextafter(center,np.float32(np.inf))
            check('nearly-constant',x,np.ones(w,dtype='<f4'),np.full(w,-0.,dtype='<f4'),w,3,1e-12)
    for index in range(128):
        w=WIDTHS[index%len(WIDTHS)];x=np.array([bits() for _ in range(2*w)],dtype='<u4').view('<f4')
        check('random-bits',x,finite(w),None if index%2==0 else finite(w),w,2,1e-5)
    assert cursor==790
    return cursor

def process(folder,base):
    state=read(folder/'identity.json');assert state['complete'] is True and state['code']==0
    assert state['limits']==dict(seconds=180,rss=6*1024**3,available=2*1024**3)
    assert state['members'][str(state['child']['pid'])]==state['child']['birth']
    for name,wanted in state['binaries'].items():assert pin(base/'bin-final'/name)==wanted
    for name,wanted in state['source'].items():assert pin(Path(__file__).parent/name)==wanted
    for name,wanted in state['generated'].items():assert pin(base/'generated'/name)==wanted
    rows=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()];assert len(rows)==state['samples']>0
    previous=-1.;peak=0;observed={}
    for row in rows:
        assert math.isfinite(row['seconds']) and previous<=row['seconds']<180;previous=row['seconds']
        assert row['available']>=2*1024**3 and len({x['pid'] for x in row['members']})==len(row['members'])
        for item in row['members']:
            assert item['affinity']==[2] and item['rss']>=0 and item['birth']>=state['child']['birth']
            assert state['members'][str(item['pid'])]==item['birth'];observed[str(item['pid'])]=item['birth']
        peak=max(peak,sum(x['rss'] for x in row['members']));assert peak<6*1024**3
    assert peak==state['peak_rss'] and observed==state['members']
    return dict(samples=len(rows),peak_rss=peak,minimum_available=min(r['available'] for r in rows),
        births=[state['supervisor']]+[dict(pid=int(p),birth=b) for p,b in state['members'].items()])

def audit(base):
    probe=pin(base/'bin-final/LayerNormOutput.dll');assert pin(base/'bin-final/Lokad.Onnx.dll')['sha256']==CORE
    capture=read(base/'capture/capture.json');proof=read(base/'proof/proof.json')
    assert capture['passed'] is True and proof['passed'] is True
    inventory(base/'capture',capture,'capture.json');inventory(base/'proof',proof,'proof.json')
    identity(capture['identity'],probe);identity(proof['identity'],probe)
    assert proof['capture_sha256']==pin(base/'capture/capture.json')['sha256']
    assert len(proof['records'])==proof['cases']==915 and proof['captured']==125
    assert len({r['name'] for r in proof['records']})==915
    assert proof['comparisons']==sum(r['values']*4 for r in proof['records'])==32182096
    for r in proof['records']:
        for key in ['exact','guards','inputs_preserved','inplace']:assert r[key] is True
        assert math.isfinite(r['scalar_error']) and 0<=r['scalar_error']<=1e-5
    assert proof['maximum_scalar_error']==max(r['scalar_error'] for r in proof['records'])
    assert synthetic(proof['records'])==790
    fixtures=ROOT/'artifacts/e5-fingerprint-product-v2-20260920/payload/inputs'
    model=ROOT/'models/multilingual-e5-small/model.onnx';assert capture['model_sha256']==pin(model)['sha256']=='ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665'
    assert [c['name'] for c in capture['cases']]==CASES
    result=[];cursor=790;compared=0
    for case in capture['cases']:
        name=case['name'];fixture=read(fixtures/(name+'.json'));folder=base/'capture'/name
        assert case==read(folder/'capture.json') and case['fixture_sha256']==pin(fixtures/(name+'.json'))['sha256']
        assert case['input_sha256']==fixture['input_sha256'] and case['reference_sha256']==fixture['reference_sha256']
        assert case['output_shape']==fixture['shape'] and case['inputs_unchanged'] is True
        native=fixtures/fixture['reference_file'];assert pin(native)['sha256']==fixture['reference_sha256']
        count=math.prod(fixture['shape']);want=array(native,count);actual=array(folder/'output.f32',count)
        error=scaled(actual,want);assert error<=1e-4 and error==case['native_error']
        assert len(case['nodes'])==25 and [n['index'] for n in case['nodes']]==list(range(25))
        norms=[]
        for node in case['nodes']:
            prefix=f"{node['index']:02}";rows=node['outer'];width=node['block'];count=rows*width
            assert width==384 and rows==fixture['shape'][1] and node['shape']==fixture['shape'] and node['axis'] in [-1,2]
            assert node['product_replay_bitwise'] is True and math.isfinite(node['epsilon']) and node['epsilon']>=0
            x=array(folder/(prefix+'-x.f32'),count).reshape(rows,width);s=array(folder/(prefix+'-scale.f32'),width)
            b=array(folder/(prefix+'-bias.f32'),width) if node['has_bias'] else None
            y=array(folder/(prefix+'-y.f32'),count).reshape(rows,width)
            record=proof['records'][cursor];cursor+=1
            assert record['name']==name+':'+str(node['index']) and record['prefix']==name+'-'+prefix
            assert record['block']==width and record['outer']==rows and record['values']==count and record['has_bias']==node['has_bias']
            assert np.float32(record['epsilon'])==np.float32(node['epsilon'])
            assert record['input_sha256']==sha(x) and record['scale_sha256']==sha(s) and record['bias_sha256']==(None if b is None else sha(b))
            for variant in ['copy','wide']:
                path=base/'proof'/(record['prefix']+'-'+variant+'.f32');assert pin(path)==pin(folder/(prefix+'-y.f32'))
                assert pin(path)['sha256']==record['product_sha256'];compared+=count
            scalar=reference(x,s,b,node['epsilon']);scalar_error=scaled(y,scalar);assert scalar_error<=1e-5
            norms.append(dict(index=node['index'],values=count,scalar_error=scalar_error))
        result.append(dict(name=name,model_native_error=error,layers=norms))
    assert cursor==915
    resources={mode:process(base/(mode+'-process'),base) for mode in ['capture','proof']}
    return dict(passed=True,cases=915,synthetic_inputs_reconstructed=790,captured_nodes=125,independent_real_values_compared=compared,
        maximum_model_native_error=max(r['model_native_error'] for r in result),maximum_independent_scalar_error=max(n['scalar_error'] for r in result for n in r['layers']),
        resources=resources,results=result,scope='Local256-bit product versus portable512-bit final transform; no AMD code or timing qualification')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();value=audit(a.artifact.resolve());write(a.output,value);print(json.dumps({k:v for k,v in value.items() if k not in ['resources','results']}))
