"""Single-process ORT graph worker with complete copied outputs and frozen inputs."""
from pathlib import Path
import hashlib,json,os,sys,time


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    import numpy as np
    import onnxruntime as ort
    manifest,key,destination,mode=sys.argv[1:]
    assert mode in ('verify','timing')
    manifest=Path(manifest).resolve();destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=False)
    spec=json.loads(manifest.read_text(encoding='utf-8-sig'))
    case=next(c for c in spec['cases'] if c['key']==key)
    def path(name):return manifest.parent/name
    def pin(array):return hashlib.sha256(array.tobytes(order='C')).hexdigest()
    feed={item['name']:np.asarray(item['values'],dtype=np.int64).reshape(item['shape']) if 'values' in item
          else np.load(path(item['file']),allow_pickle=False) for item in case['inputs']}
    assert all(list(feed[item['name']].shape)==item['shape'] for item in case['inputs'])
    inputs={name:pin(value) for name,value in feed.items()}
    references=[np.load(path(item['file']),allow_pickle=False) if item['file'].endswith('.npy')
                else np.fromfile(path(item['file']),dtype='<f4').reshape(item['shape']) for item in case['outputs']]
    names=[item['name'] for item in case['outputs']]
    setup=time.perf_counter()
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session=ort.InferenceSession(case['model'],sess_options=options,providers=['CPUExecutionProvider'])
    setup=time.perf_counter()-setup
    assert sorted(item.name for item in session.get_outputs())==sorted(names)
    warmups=1200 if key=='e5-30tok' else 600
    calls=3 if mode=='verify' else warmups+180
    clocks=[];first=None;hashes=None;errors=[0.]*len(names)
    for index in range(calls):
        start=time.perf_counter_ns()
        outputs=[value.copy() for value in session.run(names,feed)]
        end=time.perf_counter_ns()
        for j,(value,reference,item) in enumerate(zip(outputs,references,case['outputs'],strict=True)):
            assert value.dtype==np.float32 and list(value.shape)==item['shape'] and value.shape==reference.shape
            assert np.isfinite(value).all() and np.isfinite(reference).all()
            error=float(np.max(np.abs(value.astype(np.float64)-reference)/np.maximum(1,np.abs(reference.astype(np.float64)))))
            assert error<=1e-4,(key,item['name'],error)
            errors[j]=max(errors[j],error)
            if hashes is not None:assert pin(value)==hashes[j] and pin(first[j])==hashes[j]
        assert all(pin(value)==inputs[name] for name,value in feed.items())
        if first is None:first=outputs;hashes=[pin(value) for value in outputs]
        clocks.append(dict(index=index,warmup=mode=='verify' or index<warmups,ticks=end-start,frequency=1_000_000_000))
    arrays=[]
    for j,(item,value) in enumerate(zip(case['outputs'],first,strict=True)):
        filename=f'{j}.f32';value.astype('<f4',copy=False).tofile(destination/filename)
        arrays.append(dict(name=item['name'],shape=item['shape'],file=filename,sha256=digest(destination/filename),
                           values=value.size,max_scaled_error=errors[j]))
    native=Path(sys.modules['onnxruntime.capi.onnxruntime_pybind11_state'].__file__)
    result=dict(passed=True,key=key,mode=mode,pid=os.getpid(),runtime='Python '+sys.version.split()[0],
        native_versions=dict(numpy=np.__version__,onnxruntime=ort.__version__),native_module=dict(path=str(native),sha256=digest(native)),
        configuration_sha256=digest(manifest),setup_seconds=setup,calls=calls,clocks=clocks,arrays=arrays,
        inputs_unchanged=True,held_outputs_unchanged=True)
    with (destination/'result.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
