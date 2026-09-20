"""Compute one bounded float64 affine reference at a time."""
import argparse,ctypes,os,platform,time
from common import *

def numerical_runtime(spec):
    assert np.__version__==spec['numpy']
    for name,wanted in spec['numerical_files'].items():assert pin(Path(name))==wanted,name
    assert all(os.environ.get(k)==v for k,v in THREAD_ENV.items())
    dlls=[Path(p) for p in spec['numerical_files'] if 'openblas' in p.lower() and p.endswith('.dll')];assert len(dlls)==1
    lib=ctypes.CDLL(str(dlls[0]));get_threads=lib.scipy_openblas_get_num_threads64_;get_threads.restype=ctypes.c_int
    get_config=lib.scipy_openblas_get_config64_;get_config.restype=ctypes.c_char_p
    assert get_threads()==1
    ps=psutil_module();process=ps.Process();assert process.cpu_affinity()==[2]
    loaded={m.path:pin(m.path) for m in process.memory_maps() if m.path.lower().endswith(('.dll','.pyd')) and ('numpy' in m.path.lower() or 'openblas' in m.path.lower())}
    assert any(Path(n).name==dlls[0].name for n in loaded)
    return dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,blas_threads=get_threads(),blas_config=get_config().decode(),
        affinity=process.cpu_affinity(),pid=process.pid,birth=process.create_time(),thread_count=process.num_threads(),loaded=loaded)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',required=True);args=parser.parse_args()
    base=Path(args.artifact).resolve();spec=read(base/'manifest.json');verify(spec['files'])
    runtime=numerical_runtime(spec);out=base/'references';out.mkdir();rows=[];started=time.monotonic()
    for job in spec['jobs']:
        begin=time.monotonic();x=load(job['input']);actual=load(job['actual']);w,b=weights(job['projection'])
        y,bound=reference(x,w,b);assert y.shape==actual.shape
        error=metrics(actual-y,np.maximum(1,np.abs(y)));denominator=np.maximum(1,np.abs(y))
        coords=coordinates(y.shape,[error['scaled_index'],error['absolute_index']])
        checks=coordinate_checks(x,w,b,y,coords)
        path=out/(job['id']+'.f64')
        with path.open('xb') as stream:y.astype('<f8',copy=False).tofile(stream)
        rows.append(dict(id=job['id'],file=path.relative_to(base).as_posix(),pin=pin(path),shape=list(y.shape),
            local_error=error,reference_error_bound_max=float(bound.max()),reference_scaled_bound_max=float((bound/denominator).max()),
            checks=checks,seconds=time.monotonic()-begin))
        write(out/(job['id']+'.json'),rows[-1]);print(json.dumps(dict(completed=len(rows),id=job['id'],seconds=rows[-1]['seconds'])),flush=True)
        del x,actual,w,b,y,bound,denominator
    verify(spec['files']);runtime_after=numerical_runtime(spec)
    assert runtime_after['pid']==runtime['pid'] and runtime_after['birth']==runtime['birth']
    write(base/'result.json',dict(complete=True,manifest=pin(base/'manifest.json'),runtime=runtime,runtime_after=runtime_after,rows=rows,seconds=time.monotonic()-started))

if __name__=='__main__':main()
