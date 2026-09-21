import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-wide-matmul-v3-20260921'
TRACE = ROOT/'artifacts/parakeet-performance-profile-v2-20260921'
SITE = ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'
sys.path.insert(0,str(SITE))
import psutil


def read(path): return json.loads(path.read_text(encoding='utf8'))
def pin(path):
    with path.open('rb') as f: return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def verify(files):
    for name,wanted in files.items(): assert pin(ROOT/name)==wanted,name
def save(path,value):
    for attempt in range(20):
        try:
            p=path.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8');p.replace(path);return
        except PermissionError:
            if attempt==19: raise
            time.sleep(.05)
def terminal(identity):
    try: assert psutil.Process(identity['pid']).create_time()!=identity['birth']
    except psutil.NoSuchProcess: pass
