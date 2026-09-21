"""Local actual-product profiler-allocation experiment."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/e5-profiler-allocation-20260921'
TOOLS=Path(__file__).parent
sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil

SOURCE='d875d99'
CASES=['e5-8tok','e5-30tok','e5-30pad128','e5-128tok','e5-512tok']
LIMITS=dict(seconds=1800,rss=8*1024**3,available=1024**3,preflight_available=10*1024**3,disk=20*1024**3)
JOBS=['baseline','candidate']


def pin(path):
    path=Path(path)
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)


def save(path,value):
    p=Path(path);t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8');t.replace(p)


def clean_env():return {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}


def absent(identity):
    try:return psutil.Process(identity['pid']).create_time()!=identity['birth']
    except psutil.NoSuchProcess:return True


def verify(spec):
    assert spec['protocol']=='e5-profiler-allocation-v1' and spec['jobs']==JOBS and spec['limits']==LIMITS
    assert spec['cases']==CASES and spec['warmup']==spec['observed']==16
    for name,expected in spec['files'].items():assert pin(ROOT/name)==expected,name
