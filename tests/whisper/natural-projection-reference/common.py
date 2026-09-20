"""Frozen natural-input affine diagnostic; no model execution."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
PRIOR=ROOT/'artifacts/whisper-layer20-cross-20260920'
RECEIPT='13478ae21b6a545639cac4ad27f7465aafa46a010aa9d56f9d802745a40ebe32'
MODEL=ROOT/'models/whisper-large-v3-turbo/onnx/encoder_layer20_20260918.onnx'
WEIGHTS=MODEL.parent/'encoder_model.onnx_data'
MODEL_SHA='bd03b8953354ca3d9fa5dda33c4f9c76e15bbd56af9ae0e92559633b9e9ef6fd'
WEIGHTS_SHA='c87f74d17ea85478cd563a614215868eff35a8a7ec4ff692d1edd49413f56f86'
CELLS=['MM','MN','NM','NN']
LIMITS=dict(seconds=900,rss=4*1024**3,available=1024**3,preflight_available=6*1024**3,disk=10*1024**3)
THREAD_ENV={k:'1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']}
PROJECTIONS={
    'k':dict(input=0,output=2,weight='onnx::MatMul_4090',shape=[1280,1280],offset=1609973760,bytes=6553600,
             node='/layers.20/self_attn/k_proj/MatMul',bias=None),
    'fc1':dict(input=7,output=8,weight='onnx::MatMul_4100',shape=[1280,5120],offset=1629660160,bytes=26214400,
               node='/layers.20/fc1/MatMul',bias=dict(name='layers.20.fc1.bias',shape=[5120],offset=1655874560,bytes=20480))}

def psutil_module():
    # NumPy must resolve from the global installation before this retained venv.
    sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
    import psutil
    return psutil

def pin(path):
    with Path(path).open('rb') as stream:
        return dict(bytes=Path(path).stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def verify(files):
    for name,wanted in files.items():assert pin(ROOT/name)==wanted,name
def absent(identity):
    ps=psutil_module()
    try:return ps.Process(identity['pid']).create_time()!=identity['birth']
    except ps.NoSuchProcess:return True

def load(desc):
    path=ROOT/desc['file'];assert pin(path)==desc['pin'],str(path)
    value=np.fromfile(path,dtype='<f4').reshape(desc['shape'])
    assert np.isfinite(value).all()
    return value.reshape(-1,desc['shape'][-1]).astype(np.float64)

def weights(name):
    p=PROJECTIONS[name]
    w=np.memmap(WEIGHTS,dtype='<f4',mode='r',offset=p['offset'],shape=tuple(p['shape'])).astype(np.float64)
    b=p['bias'];bias=None if b is None else np.memmap(WEIGHTS,dtype='<f4',mode='r',offset=b['offset'],shape=tuple(b['shape'])).astype(np.float64)
    assert np.isfinite(w).all() and (bias is None or np.isfinite(bias).all())
    return w,bias

def gamma(n):
    v=(2*n+2)*2.**-53
    return v/(1-v)

def reference(x,w,bias=None):
    # Inputs are exact promotions of finite float32: each product fits float64.
    assert x.ndim==w.ndim==2 and x.shape[1]==w.shape[0]
    assert np.isfinite(x).all() and np.isfinite(w).all()
    y=x@w
    total=np.abs(x)@np.abs(w)
    if bias is not None:y+=bias;total+=np.abs(bias)
    g=gamma(x.shape[1])
    upper=np.nextafter(total/(1-g),np.inf)
    bound=np.nextafter(g*upper,np.inf)
    assert np.isfinite(y).all() and np.isfinite(bound).all()
    return y,bound

def metrics(delta,denominator):
    assert delta.shape==denominator.shape and np.isfinite(delta).all() and np.all(denominator>=1)
    absolute=np.abs(delta);scaled=absolute/denominator;i=int(np.argmax(scaled));j=int(np.argmax(absolute))
    squares=float(np.sum(delta*delta))
    return dict(values=delta.size,max_absolute=float(absolute.flat[j]),absolute_index=list(map(int,np.unravel_index(j,delta.shape))),
        max_scaled=float(scaled.flat[i]),scaled_index=list(map(int,np.unravel_index(i,delta.shape))),
        failed_values=int(np.count_nonzero(scaled>1e-4)),rms=math.sqrt(squares/delta.size),l2=math.sqrt(squares))

def coordinates(shape,extra=()):
    rows,columns=shape
    return sorted({(r,c) for r in [0,rows//4,rows//2,3*rows//4,rows-1] for c in [0,1,columns//3,columns//2,columns-1]}|{tuple(x) for x in extra})

def coordinate_checks(x,w,bias,y,coords):
    result=[];g=gamma(x.shape[1])
    for row,col in coords:
        terms=[float(a)*float(b) for a,b in zip(x[row],w[:,col])]
        if bias is not None:terms.append(float(bias[col]))
        exact=math.fsum(terms);magnitude=math.fsum(map(abs,terms));observed=float(y[row,col])
        # Extra ulps cover fsum final rounding and scalar construction of bound.
        bound=math.nextafter(g*magnitude/(1-g)+4*math.ulp(exact),math.inf)
        assert abs(observed-exact)<=bound,(row,col,observed,exact,bound)
        result.append(dict(index=[row,col],reference=observed,fsum=exact,error=abs(observed-exact),bound=bound))
    return result

def decompose(actual,refs):
    assert set(actual)==set(refs)==set(CELLS)
    denominator=np.maximum(1,np.abs(actual['NN']));answer={}
    for a,b in [('MM','NN'),('MM','NM'),('MN','NN'),('MM','MN'),('NM','NN')]:
        delta=actual[a]-actual[b];propagated=refs[a]-refs[b]
        local=(actual[a]-refs[a])-(actual[b]-refs[b]);closure=delta-propagated-local
        bound=32*2.**-53*(np.abs(actual[a])+np.abs(actual[b])+np.abs(refs[a])+np.abs(refs[b])+1)
        assert np.all(np.abs(closure)<=bound)
        answer[a+'-'+b]=dict(actual=metrics(delta,denominator),propagated=metrics(propagated,denominator),
            local_residual_difference=metrics(local,denominator),closure_max=float(np.abs(closure).max()),
            closure_bound_max=float(bound.max()),pairwise_actual=metrics(delta,np.maximum(1,np.abs(actual[b]))))
    return answer
