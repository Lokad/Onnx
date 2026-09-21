"""Paths and checked transport for the finite AMD public-contract replay."""
from pathlib import Path
import hashlib,json,subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/whisper-memory-contracts-amd-20260921'
LOCAL=ROOT/'artifacts/whisper-memory-contracts-v2-20260921'
PRODUCT=ROOT/'artifacts/whisper-weight-sharing-20260920'
SHARING=ROOT/'artifacts/whisper-weight-sharing-v2-20260920'
PRIOR=ROOT/'artifacts/whisper-recording-v2-20260919'
REMOTE='/home/vermorel/Onnx/artifacts/whisper-memory-contracts-amd-20260921'
REMOTE_SHARING='/home/vermorel/Onnx/artifacts/whisper-weight-sharing-v2-20260920'
REMOTE_RECORDING='/home/vermorel/Onnx/artifacts/whisper-recording-amd-v2-20260919'
KEY='C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem'
HOST='vermorel@74.178.91.76'
LIMITS=dict(seconds=1800,rss=14*1024**3,available=1024**3,preflight=13*1024**3,disk=32*1024**2,preflight_disk=64*1024**2)


def pin(path):
    path=Path(path)
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def ssh(script):
    compile(script,'checked-remote-contract-script','exec')
    return subprocess.check_output(['ssh','-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'python3 -B -'],input=script,text=True,encoding='utf-8')


PRELUDE='''from pathlib import Path
import hashlib,json,os,subprocess,sys,tarfile
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path(%r);sharing=Path(%r)
def pin(path):
 path=Path(path)
 with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def terminal(births):
 for b in births:
  try:assert psutil.Process(b['pid']).create_time()!=b['birth'],('live',b)
  except psutil.NoSuchProcess:pass
'''%(REMOTE,REMOTE_SHARING)
