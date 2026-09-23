"""Matched CPU ORT forward calls, owning every returned output array."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import onnxruntime.capi.onnxruntime_pybind11_state as binding

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    manifest,key,target,mode=sys.argv[1:];root=Path(manifest).parent;out=Path(target)
    assert mode in ['verify','timing'] and not out.exists() and os.sched_getaffinity(0)=={2}
    out.mkdir();spec=json.loads(Path(manifest).read_text());item=next(c for c in spec['cases'] if c['key']==key)
    assert ort.__version__=='1.29.0'
    flags={k:v for k,v in os.environ.items() if k.lower().startswith(('lokad_','dotnet_','complus_'))};assert not flags
    feed={i['name']:np.array(i['values'],dtype='<i8').reshape(i['shape']) if 'values' in i else np.load(root/i['file'],allow_pickle=False) for i in item['inputs']}
    before={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in feed.items()}
    refs=[np.load(root/i['file'],allow_pickle=False) if i['file'].endswith('.npy') else np.fromfile(root/i['file'],dtype='<f4').reshape(i['shape']) for i in item['outputs']]
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    start=time.perf_counter_ns();session=ort.InferenceSession(item['model'],sess_options=options,providers=['CPUExecutionProvider']);setup=(time.perf_counter_ns()-start)/1e9
    assert session.get_providers()==['CPUExecutionProvider']
    names=[i['name'] for i in item['outputs']];assert sorted(names)==sorted(x.name for x in session.get_outputs())
    first=None;hashes=None;errors=[0.0]*len(names);clocks=[];calls=3 if mode=='verify' else 120
    for index in range(calls):
        start=time.perf_counter_ns();values=session.run(names,feed);end=time.perf_counter_ns()
        for j,(value,reference) in enumerate(zip(values,refs,strict=True)):
            assert value.dtype==np.float32 and value.shape==reference.shape and np.isfinite(value).all() and np.isfinite(reference).all()
            error=float(np.max(np.abs(value.astype(np.float64)-reference)/np.maximum(1,np.abs(reference.astype(np.float64)))))
            assert error<=1e-4;errors[j]=max(errors[j],error)
            if hashes is not None:assert hashlib.sha256(value.tobytes()).hexdigest()==hashes[j]==hashlib.sha256(first[j].tobytes()).hexdigest()
        assert {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in feed.items()}==before
        if first is None:first=values;hashes=[hashlib.sha256(v.tobytes()).hexdigest() for v in values]
        clocks.append(dict(index=index,warmup=mode=='verify' or index<60,ticks=end-start,frequency=1000000000))
    arrays=[]
    for j,value in enumerate(first):
        file=str(j)+'.f32';value.tofile(out/file);assert digest(out/file)==hashes[j]
        arrays.append(dict(name=names[j],shape=list(value.shape),file=file,sha256=hashes[j],values=value.size,max_scaled_error=errors[j]))
    result=dict(passed=True,role='ort',key=key,mode=mode,pid=os.getpid(),runtime=ort.__version__,consumer=digest(__file__),flags=flags,setup_seconds=setup,calls=calls,clocks=clocks,arrays=arrays,inputs_unchanged=True,held_outputs_unchanged=True,
        native=digest(binding.__file__),numpy=np.__version__)
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
