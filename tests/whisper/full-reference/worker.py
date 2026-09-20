"""One complete encoder reference, retaining every declared boundary."""
import argparse,ctypes,os,platform,time
import onnx
from common import *

def runtime(engine):
    packages();import psutil
    p=psutil.Process();assert p.cpu_affinity()==[2]
    assert all(os.environ.get(k)==v for k,v in THREADS.items())
    libs=list((Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*.dll'));assert len(libs)==1
    dll=ctypes.CDLL(str(libs[0]));threads=dll.scipy_openblas_get_num_threads64_;threads.restype=ctypes.c_int;assert threads()==1
    loaded={m.path:pin(m.path) for m in p.memory_maps() if m.path.lower().endswith(('.dll','.pyd')) and any(n in m.path.lower() for n in ['numpy','scipy','onnxruntime'])}
    native_loaded=any('onnxruntime' in n.lower() for n in loaded);assert native_loaded is (engine=='ort')
    return dict(pid=p.pid,birth=p.create_time(),affinity=p.cpu_affinity(),blas_threads=threads(),native_loaded=native_loaded,loaded=loaded,python=sys.version,platform=platform.platform())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',required=True);parser.add_argument('--job',required=True);args=parser.parse_args()
    base=Path(args.artifact).resolve();spec=read(base/'manifest.json');job=next(j for j in spec['jobs'] if j['id']==args.job)
    folder=base/'outputs'/job['id'];folder.mkdir(parents=True);features=source(job['input']);before=raw(features);x=features.astype(np.float64)
    wanted={o['name']:(i,o) for i,o in enumerate(spec['outputs'])};saved={};started=time.monotonic()
    def capture(name,value):
        if name not in wanted:return
        assert name not in saved,name
        index,desc=wanted[name];assert value.dtype==np.float64 and list(value.shape)==desc['shape'] and np.isfinite(value).all(),name
        path=folder/f'{index:02}.f64'
        with path.open('xb') as stream:np.ascontiguousarray(value,dtype='<f8').tofile(stream)
        saved[name]=dict(index=index,name=name,shape=list(value.shape),file=path.name,pin=pin(path))
        print(json.dumps(dict(saved=index,name=name,seconds=time.monotonic()-started)),flush=True)
    if job['engine']=='numpy':
        from interpreter import run
        records=run(onnx.load(ROOT/spec['trace_model'],load_external_data=False),MODEL.parent,x,capture)
        assert len(records)==1559;settings=dict(engine='numpy',erf='scipy.special.erf double',numpy=np.__version__)
    else:
        from native import run,ort
        assert ort.__version__=='1.29.0';directory=ROOT/spec['promoted']
        records=run(directory,read(directory/'stages.json'),'input_features',x,capture)
        assert len(records)==69;settings=dict(engine='ort',erf='Python math.erf double',native=ort.__version__,threads=1,sequential=True,optimizations='disabled',spinning=False)
    assert set(saved)==set(wanted) and raw(features)==before and raw(x)==raw(features.astype(np.float64))
    result=dict(complete=True,job=job,manifest=pin(base/'manifest.json'),input_sha256=before,input_unchanged=True,settings=settings,runtime=runtime(job['engine']),
        outputs=[saved[o['name']] for o in spec['outputs']],records=records,seconds=time.monotonic()-started)
    write(folder/'result.json',result)

if __name__=='__main__':main()
