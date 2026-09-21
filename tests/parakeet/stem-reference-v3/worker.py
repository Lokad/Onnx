"""One independent float64 stem request with complete arrays and scalar probes."""
from common import *
from routes import numpy_route,torch_route,scalar_conv
import math
import re


def main():
    spec=read(BASE/'manifest.json');verify(spec)
    job=next(j for j in JOBS if j['id']==sys.argv[1]);folder=BASE/'outputs'/job['id'];folder.mkdir(parents=True,exist_ok=False)
    process=psutil.Process();assert process.cpu_affinity()==[2]
    assert not any(k.lower().startswith(('lokad_','dotnet_','complus_')) for k in os.environ)
    assert np.__version__==spec['numpy']
    torch_config=None
    if job['engine']=='torch':
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert torch.__version__==spec['torch'] and torch.get_num_threads()==torch.get_num_interop_threads()==1
        torch_config=dict(parallel=torch.__config__.parallel_info(),build=torch.__config__.show())
        assert re.search(r'mkl_get_max_threads\(\)\s*:\s*1\b',torch_config['parallel']) and 'BLAS_INFO=mkl' in torch_config['build']
    features=np.load(ROOT/spec['inputs'][job['input']],allow_pickle=False)
    weights={key:np.load(ROOT/value['file'],allow_pickle=False) for key,value in spec['weights'].items()}
    assert features.dtype==np.float32 and features.shape==(1,128,586)
    before={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in dict(features=features,**weights).items()}
    for k,v in weights.items():assert v.dtype==np.float32 and list(v.shape)==spec['weights'][k]['shape'] and before[k]==spec['weights'][k]['raw_sha256']
    records=[];probes=[];held=[]
    result=dict(complete=False,job=job,manifest=pin(BASE/'manifest.json'),outputs=records,probes=probes,
                process=dict(pid=process.pid,birth=process.create_time(),affinity=process.cpu_affinity()))
    save(folder/'result.json',result)
    def capture(name,value,x=None,w=None,bias=None,geometry=None):
        assert name==STAGES[len(records)] and value.dtype==np.float64 and list(value.shape)==SHAPES[name] and np.isfinite(value).all()
        path=folder/(name+'.f64')
        with path.open('xb') as f:f.write(value.tobytes(order='C'))
        records.append(dict(name=name,file=path.name,shape=list(value.shape),dtype='float64',**pin(path)))
        held.append((value,hashlib.sha256(value.tobytes()).hexdigest()))
        if x is not None:
            for index in coordinates(value.shape,name):
                coordinate=tuple(int(i) for i in np.unravel_index(index,value.shape))
                if name=='projection':
                    n,row,col=coordinate;expected=math.fsum(float(x[n,row,k])*float(w[k,col]) for k in range(w.shape[0]))
                else:expected=scalar_conv(x,w,bias,coordinate,*geometry)
                actual=float(value[coordinate]);error=abs(actual-expected)/max(1.,abs(expected))
                probes.append(dict(stage=name,index=index,coordinate=coordinate,actual=actual,expected=expected,error=error))
        save(folder/'result.json',result)
        assert all(p['error']<=REF_LIMIT for p in probes)
    (numpy_route if job['engine']=='numpy' else torch_route)(features,weights,capture)
    assert len(records)==11 and len(probes)==1536
    assert all(hashlib.sha256(v.tobytes()).hexdigest()==before[k] for k,v in dict(features=features,**weights).items())
    assert all(hashlib.sha256(v.tobytes()).hexdigest()==h for v,h in held)
    helpers=reference_helpers();assert helpers.openblas_threads()==1
    libraries=helpers.libraries(process)
    for path,expected in libraries.items():assert spec['numerical_libraries'].get(path)==expected,path
    assert not any('onnxruntime' in row.path.lower() for row in process.memory_maps())
    result.update(complete=True,inputs_unchanged=True,weights_unchanged=True,held_outputs_unchanged=True,
                  libraries=libraries,openblas_threads=1,torch_config=torch_config,numpy=np.__version__,native_ort_loaded=False)
    save(folder/'result.json',result)
    print(json.dumps(dict(job=job['id'],arrays=len(records),scalar_checks=len(probes),max_scalar_error=max(p['error'] for p in probes))))


if __name__=='__main__':main()
