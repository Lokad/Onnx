"""Local orchestration identities for the separate storage-corrected campaign."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
SITE = ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'
sys.path.insert(0, str(SITE))
sys.path.append(str(ROOT/'tests/audio/amd-comparison'))
from protocol import pin, read, write, LIMITS, validate_records

BASE = ROOT/'artifacts/audio-whisper-storage-20260921'
REMOTE = '/dev/shm/lokad-whisper-storage-20260921'
OLD = ROOT/'artifacts/audio-whisper-amd-20260921'
OLD_REMOTE = '/home/vermorel/Onnx/artifacts/audio-whisper-amd-20260921'
NATIVE = ROOT/'artifacts/audio-amd-comparison-v2-20260920'
E5 = ROOT/'artifacts/e5-randomized-processes-20260921'
E5_REMOTE = '/dev/shm/lokad-e5-independent-20260921'
E5_CONTROL = ROOT/'artifacts/e5-randomized-processes-finish-v2-20260921'
TOOLS = Path(__file__).parent
KEY = 'C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem'
HOST = 'vermorel@74.178.91.76'
SSH = ['ssh', '-i', KEY, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST]
PROTOCOL = 'whisper-amd-storage-v1'
FAILURE = '123487c72b74bede3d2ad5b04e3b9178f2ef14e3eab35cfb901660f289edfd99'


def ssh(script, timeout=300):
    compile(script, 'checked-whisper-storage-remote', 'exec')
    return subprocess.check_output(SSH+['python3 -B -'], input=script, text=True,
        encoding='utf8', timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW)


def original_audit():
    spec = importlib.util.spec_from_file_location('original_audio_audit', ROOT/'tests/audio/amd-comparison/audit.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def absent(identity):
    import psutil
    try:
        return psutil.Process(identity['pid']).create_time() != identity['birth']
    except psutil.NoSuchProcess:
        return True


PRELUDE = '''from pathlib import Path
import hashlib,json,os,sys,subprocess,time,shutil
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path(%r);old=Path(%r);e5=Path(%r)
def pin(p):
 p=Path(p)
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def terminal(births):
 for b in births:
  try:assert psutil.Process(b['pid']).create_time()!=b['birth'],('live',b)
  except psutil.NoSuchProcess:pass
os.sched_setaffinity(0,{0})
''' % (REMOTE, OLD_REMOTE, E5_REMOTE)
