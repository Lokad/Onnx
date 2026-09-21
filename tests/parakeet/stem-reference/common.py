"""Fixed scope and evidence helpers for independent stem references."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import psutil

BASE=ROOT/'artifacts/parakeet-stem-reference-v2-20260921'
TRACE=ROOT/'artifacts/parakeet-layer-trace-v2-20260921'
DYNAMIC=ROOT/'artifacts/parakeet-layer-trace-dynamic-20260921'
PRIOR=ROOT/'artifacts/parakeet-layer-trace-final-20260921/verified.json'
PROTOCOL='parakeet-stem-reference-v1'
STAGES=('conv0','relu0','conv2','conv3','relu3','conv5','conv6','relu6','reshape','projection','stem')
CONVS=(0,2,3,5,6)
GEOMETRY={0:(2,1,1),2:(2,1,256),3:(1,0,1),5:(2,1,256),6:(1,0,1)}
LIMITS=dict(seconds=900,rss=8*1024**3,available=1024**3,preflight_available=10*1024**3,disk=20*1024**3)
JOBS=[dict(id=engine+'-'+kind,engine=engine,input=kind.removesuffix('-repeat'))
      for kind in ('native','managed','native-repeat') for engine in ('numpy','torch')]
REF_LIMIT=1e-9
SHAPES={name:([1,256,293,64] if name in ('conv0','relu0') else
              [1,256,147,32] if name in ('conv2','conv3','relu3') else
              [1,256,74,16] if name in ('conv5','conv6','relu6') else
              [1,74,4096] if name=='reshape' else [1,74,1024]) for name in STAGES}


def pin(path):
    path=Path(path)
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def rel(path):
    p=Path(path).resolve()
    try:return p.relative_to(ROOT).as_posix()
    except ValueError:return p.as_posix()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)


def save(path,value):
    p=Path(path);t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8');t.replace(p)


def verify(spec):
    assert spec['protocol']==PROTOCOL and spec['limits']==LIMITS and spec['jobs']==JOBS
    for name,expected in spec['files'].items():assert pin(ROOT/name)==expected,name


def absent(identity):
    try:return psutil.Process(identity['pid']).create_time()!=identity['birth']
    except psutil.NoSuchProcess:return True


def load_array(path,record):
    assert pin(path)=={k:record[k] for k in ('bytes','sha256')}
    a=np.fromfile(path,dtype=record['dtype']).reshape(record['shape']);assert np.isfinite(a).all();return a


def reference_helpers():
    path=ROOT/'tests/pyannote/filterbank-reference/common.py'
    spec=importlib.util.spec_from_file_location('retained_reference_helpers',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def coordinates(shape,stage):
    count=int(np.prod(shape));assert count>=256
    chosen={0,count-1,count//2}
    rng=np.random.default_rng(20260921+STAGES.index(stage))
    if len(shape)==4:
        for c in (0,1,shape[1]//2,shape[1]-1):
            for h in (0,1,shape[2]-1):
                for w in (0,1,shape[3]-1):chosen.add(int(np.ravel_multi_index((0,c,h,w),shape)))
    while len(chosen)<256:chosen.add(int(rng.integers(0,count)))
    return sorted(chosen)
