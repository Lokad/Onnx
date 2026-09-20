"""Shared identities and IO; reference engines use separate computation paths."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
MODEL=ROOT/'models/whisper-large-v3-turbo/onnx/encoder_model.onnx'
DATA=MODEL.with_name('encoder_model.onnx_data')
ORIGINAL='67241dccb6b4023e3e56b87e89a44077577f84c4f6fe3f629e3447dd27eabb59'
CROSS_RECEIPT='513f355ef7a36959c6c6454fbb65da33d3feb20585104267bf500113b1da6054'
WEIGHTS='c87f74d17ea85478cd563a614215868eff35a8a7ec4ff692d1edd49413f56f86'
CROSS=ROOT/'artifacts/whisper-input-cross-isolated-20260920'
TRACE=ROOT/'artifacts/whisper-trace-selected-20260920'
SCIPY=ROOT/'artifacts/pyannote-clustering-20260919/python'
PACKAGES=ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'
OPS={'Add','MatMul','Mul','ReduceMean','Transpose','Reshape','Div','Sub','Pow','Sqrt','Erf','Softmax','Conv'}
LIMITS=dict(seconds=900,rss=4*1024**3,available=1024**3,preflight_available=6*1024**3,disk=100*1024**3)
THREADS={n:'1' for n in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']}

def packages():
    for path in [SCIPY,PACKAGES]:
        if str(path) not in sys.path:sys.path.append(str(path))

def pin(path):
    path=Path(path)
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
def raw(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
def verify(files):
    for path,wanted in files.items():assert pin(ROOT/path)==wanted,path
def source(desc):
    path=ROOT/desc['file'];value=np.load(path,allow_pickle=False) if desc['format']=='npy' else np.fromfile(path,dtype='<f4').reshape(desc['shape'])
    assert value.dtype==np.float32 and list(value.shape)==desc['shape'] and raw(value)==desc['raw_sha256'] and np.isfinite(value).all()
    return value
def absent(identity):
    packages();import psutil
    try:return psutil.Process(identity['pid']).create_time()!=identity['birth']
    except psutil.NoSuchProcess:return True

def tensor_array(tensor,directory):
    import onnx
    if tensor.data_location==onnx.TensorProto.EXTERNAL:
        ext={v.key:v.value for v in tensor.external_data};path=Path(directory)/ext['location'];dtype=onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
        expected=int(np.prod(tensor.dims))*np.dtype(dtype).itemsize
        assert expected==int(ext['length']) and path.is_file()
        value=np.memmap(path,dtype=dtype,mode='r',offset=int(ext['offset']),shape=tuple(tensor.dims)).copy()
    else:value=onnx.numpy_helper.to_array(tensor).copy()
    assert value.dtype in [np.float32,np.float64,np.int64] and np.isfinite(value).all()
    return value

def metric(actual,reference,limit):
    assert actual.shape==reference.shape and np.isfinite(actual).all() and np.isfinite(reference).all()
    delta=actual.astype(np.float64)-reference.astype(np.float64);absolute=np.abs(delta);scaled=absolute/np.maximum(1,np.abs(reference))
    i=int(scaled.argmax());j=int(absolute.argmax());l2=float(np.sqrt(np.sum(delta*delta)))
    return dict(values=delta.size,max_absolute=float(absolute.flat[j]),absolute_index=list(map(int,np.unravel_index(j,delta.shape))),
        max_scaled=float(scaled.flat[i]),scaled_index=list(map(int,np.unravel_index(i,delta.shape))),failed_values=int(np.count_nonzero(scaled>limit)),
        rms=l2/np.sqrt(delta.size),l2=l2,limit=limit)
