"""Run application-level ORT timings on already decoded PCM with validation outside timing."""
from pathlib import Path
import argparse,ctypes,hashlib,importlib.metadata,inspect,json,math,os,sys,time

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def agree(actual,expected,path=''):
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and actual.keys()==expected.keys(),path
        return max([agree(actual[k],v,path+'/'+k) for k,v in expected.items()]+[0.])
    if isinstance(expected,list):
        assert isinstance(actual,list) and len(actual)==len(expected),path
        return max([agree(a,e,path+'/'+str(i)) for i,(a,e) in enumerate(zip(actual,expected))]+[0.])
    if isinstance(expected,(int,float)) and not isinstance(expected,bool):
        assert isinstance(actual,(int,float)) and math.isfinite(actual) and math.isfinite(expected),path
        if '/centroid/' in path:
            error=abs(actual-expected)/max(1,abs(expected));assert error<=1e-4,path;return error
        tolerance=1e-12 if path.startswith(('/intervals/','/exclusive_intervals/')) else 0
        assert abs(actual-expected)<=tolerance,(path,actual,expected)
    else:assert type(actual)==type(expected) and actual==expected,(path,actual,expected)
    return 0.

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);p.add_argument('mode',choices=['conformance','timing']);a=p.parse_args();root=a.root.resolve()
    assert os.name=='nt','This version records Windows process affinity'
    process_mask=ctypes.c_size_t();system_mask=ctypes.c_size_t();kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p;kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    assert kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(),ctypes.byref(process_mask),ctypes.byref(system_mask)) and process_mask.value==4
    manifest=read(a.manifest);assert manifest['schema']==1
    from native_adapters import Parakeet,Pyannote,np,ort
    assert np.__version__=='2.2.4' and ort.__version__=='1.29.0'
    def verify(spec):
        path=root/spec['path'];assert path.stat().st_size==spec['bytes'] and sha(path)==spec['sha256'],path;return path
    for spec in manifest['models'].values():verify(spec)
    verify(manifest['reference'])
    if manifest['family']=='parakeet':verify(manifest['upstream'])
    else:
        for spec in list(manifest['upstream'].values())+list(manifest['native_assets'].values()):verify(spec)
        for name,version in manifest['pins']['versions'].items():assert importlib.metadata.version(name)==version,name
        from torchaudio.compliance import kaldi
        assert hashlib.sha256(Path(inspect.getfile(kaldi)).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==manifest['pins']['kaldi_lf_sha256']
    cases=[]
    for c in manifest['cases']:
        pcm=np.load(verify(c['pcm']),allow_pickle=False);assert pcm.dtype==np.float32 and pcm.shape==(c['samples'],) and np.isfinite(pcm).all()
        cases.append((c,pcm,pcm.tobytes()))
    a.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();model=(Parakeet if manifest['family']=='parakeet' else Pyannote)(root,manifest);setup=time.perf_counter()-start
    native_binaries={str(path):sha(path) for path in sorted((Path(ort.__file__).parent/'capi').iterdir()) if path.suffix in ('.dll','.pyd')}
    assert native_binaries,'Missing ORT binary identity'
    records=[];held=[];first={};warmup=1 if a.mode=='conformance' else manifest['warmup_passes'];measured=0 if a.mode=='conformance' else manifest['measured_passes']
    for iteration in range(warmup+measured):
        for c,pcm,before in cases:
            assert pcm.tobytes()==before
            for result,snapshot in held:assert json.dumps(result,sort_keys=True,allow_nan=False)==snapshot,'Held result mutation'
            start=time.perf_counter_ns();actual=model(pcm);end=time.perf_counter_ns()
            error=agree(actual,c['expected']);assert pcm.tobytes()==before
            snapshot=json.dumps(actual,sort_keys=True,allow_nan=False)
            if c['name'] in first:assert snapshot==first[c['name']],'Repeated output changed'
            else:first[c['name']]=snapshot
            held.append((actual,snapshot))
            row=dict(name=c['name'],**{'pass':iteration},phase='warmup' if iteration<warmup else 'measured',seconds=(end-start)/1e9,start_ticks=start,end_ticks=end,frequency=1000000000,
                result=actual,maximum_centroid_error=error,input_sha256=hashlib.sha256(before).hexdigest(),ownership=True)
            records.append(row)
            with (a.output/f'{len(records)-1:03d}.json').open('x',encoding='utf-8') as f:json.dump(row,f,indent=2,allow_nan=False)
            print(manifest['family'],row['phase'],iteration,c['name'],row['seconds'],flush=True)
    assert all(pcm.tobytes()==before for _,pcm,before in cases)
    assert all(json.dumps(result,sort_keys=True,allow_nan=False)==snapshot for result,snapshot in held)
    result=dict(schema=1,family=manifest['family'],engine='ort',conformance=a.mode=='conformance',records=records,setup_seconds=setup,manifest_sha256=sha(a.manifest),
        runner_sha256=sha(Path(__file__)),adapter_sha256=sha(Path(__file__).with_name('native_adapters.py')),onnxruntime=ort.__version__,numpy=np.__version__,affinity=process_mask.value,
        native_settings=dict(provider='CPUExecutionProvider',intra_threads=1,inter_threads=1,execution='sequential',optimization='all',spinning=False),held_outputs_unchanged=True,
        native_binaries=native_binaries,python=sys.version,python_binary_sha256=sha(Path(sys.executable)),
        flags={k:v for k,v in os.environ.items() if k.startswith(('LOKAD_','DOTNET_','COMPlus_','OMP_','MKL_','OPENBLAS_'))})
    with (a.output/'result.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)

if __name__=='__main__':main()
