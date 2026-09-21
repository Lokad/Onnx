"""Fixed paths for the integrated Whisper comparison; original protocol is unchanged."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/audio/amd-comparison'))
from protocol import pin,read,write,LIMITS,validate_records
from deploy import ssh,KEY,HOST
BASE=ROOT/'artifacts/audio-whisper-amd-20260921'
REMOTE='/home/vermorel/Onnx/artifacts/audio-whisper-amd-20260921'
PRIOR=ROOT/'artifacts/audio-amd-comparison-v2-20260920'
PRIOR_REMOTE='/home/vermorel/Onnx/artifacts/audio-amd-comparison-v2-20260920'
PRODUCT=ROOT/'artifacts/whisper-memory-product-v2-20260921'
BIN=PRODUCT/'source/src/Lokad.Onnx.CLI/bin/Release/net10.0'
CONTRACTS=ROOT/'artifacts/whisper-memory-contracts-amd-20260921'
INTEGRATION=ROOT/'artifacts/whisper-memory-integration-20260921'
LOCAL_SITE=ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'
PRELUDE='''from pathlib import Path
import hashlib,json,os,sys,subprocess,time
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path(%r);old=Path(%r)
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
'''%(REMOTE,PRIOR_REMOTE)
