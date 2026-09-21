"""Transport helpers; no implicit model launch or reporting."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/e5-randomized-processes-20260921'
REMOTE = '/dev/shm/lokad-e5-independent-20260921'
HOST = 'vermorel@74.178.91.76'
KEY = 'C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem'
SSH = ['ssh', '-i', KEY, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', HOST]
SITE = ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'
WHISPER = ROOT/'artifacts/audio-whisper-amd-20260921'
MONITOR = ROOT/'artifacts/audio-whisper-amd-finish-20260921'
PRELUDE = '''from pathlib import Path
import hashlib,json,os,sys,subprocess,time,shutil
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path(%r)
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
''' % REMOTE


def ssh(script, timeout=120):
    compile(script, 'checked-e5-remote-script', 'exec')
    result = subprocess.run(SSH+['python3 -B -'], input=script, capture_output=True,
                            text=True, encoding='utf8', timeout=timeout,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError(f'Remote exit{result.returncode}: {result.stderr[-6000:]}')
    return result.stdout


def copy_to(local, remote):
    subprocess.run(['scp', '-i', KEY, '-o', 'BatchMode=yes', str(local), HOST+':'+remote],
                   check=True, timeout=300, creationflags=subprocess.CREATE_NO_WINDOW)


def copy_from(remote, local):
    assert not Path(local).exists()
    subprocess.run(['scp', '-i', KEY, '-o', 'BatchMode=yes', HOST+':'+remote, str(local)],
                   check=True, timeout=300, creationflags=subprocess.CREATE_NO_WINDOW)
