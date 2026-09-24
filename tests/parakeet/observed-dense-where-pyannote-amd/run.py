"""Prepare, stage, observe and collect actual normal-product graph qualification."""
import ast
import base64
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from protocol import JOBS, LIMITS, pin, read, save
from prepare import previous_closed, prepare, monitor

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-observed-dense-where-pyannote-amd-20260924'
REMOTE = '/dev/shm/lokad-parakeet-observed-dense-where-pyannote-20260924'
SITE = '/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python'
SSH = ['ssh','-i','C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem','-o','BatchMode=yes','-o','ConnectTimeout=20','vermorel@74.178.91.76']


def prepared():
    value = read(BASE/'prepared.json'); assert value['passed']
    for name,wanted in value['files'].items(): assert pin(ROOT/name) == wanted,name
    assert pin(BASE/'payload.tar.gz')==value['archive'] and pin(BASE/'bundle/stage.json')==value['stage']
    if (BASE/'staged.json').exists():
        value['payload']=read(BASE/'staged.json')['payload'];assert pin(BASE/'payload.json')==value['payload']
    previous_closed()
    return value


PRELUDE = f'''from pathlib import Path
import os,sys,json,subprocess,tarfile,time
sys.path.insert(0,{SITE!r})
import psutil
base=Path({REMOTE!r})
os.sched_setaffinity(0,{{0}})
sys.path.insert(0,str(base/'tools'))
'''


def ssh(script, timeout=180):
    compile(script,'native-layout-transport','exec')
    result = subprocess.run(SSH+['python3 -B -'],input=script,text=True,encoding='utf8',capture_output=True,
        timeout=timeout,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0,(result.returncode,result.stderr[-5000:])
    return result.stdout


def stage():
    spec = prepared(); previous_closed(); assert not (BASE/'staged.json').exists()
    first = json.loads(ssh(PRELUDE+'''
assert not base.exists()
assert psutil.boot_time()==1789634288.0
assert psutil.virtual_memory().available>=11*1024**3 and psutil.disk_usage('/dev/shm').free>=3*1024**3
base.mkdir()
print(json.dumps(dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free)))
'''))
    save(BASE/'stage-started.json',first)
    subprocess.run(['scp',*SSH[1:-1],str(BASE/'payload.tar.gz'),SSH[-1]+':'+REMOTE+'/transfer.tar.gz'],
                   check=True,timeout=300,creationflags=subprocess.CREATE_NO_WINDOW)
    result = json.loads(ssh(PRELUDE+f'''
import hashlib
archive=base/'transfer.tar.gz'
with archive.open('rb') as f: actual=dict(bytes=archive.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
assert actual=={spec['archive']!r}
with tarfile.open(archive) as tar:
 members=tar.getmembers()
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
 assert len({{m.name for m in members}})==len(members)
 tar.extractall(base,filter='data')
import importlib,base64
importlib.invalidate_caches()
from protocol import pin,save,verify
import remote
remote.idle()
assert pin(base/'stage.json')=={spec['stage']!r}
archive.unlink()
env=dict(os.environ,PYTHONPATH={SITE!r},PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
p=subprocess.run([sys.executable,'-B',str(base/'tools/remote_prepare.py')],cwd=base,env=env,text=True,capture_output=True,timeout=180)
assert p.returncode==0,(p.returncode,p.stdout[-3000:],p.stderr[-4000:])
value=verify(base);assert not remote.live(value['previous_owner'])
receipt=dict(passed=True,payload=pin(base/'payload.json'),files=len(value['files']),external=len(value['external']))
save(base/'staged.json',receipt)
print(json.dumps(dict(**receipt,payload_base64=base64.b64encode((base/'payload.json').read_bytes()).decode('ascii'))))
''',300))
    encoded=result.pop('payload_base64');(BASE/'payload.json').write_bytes(base64.b64decode(encoded));assert pin(BASE/'payload.json')==result['payload']
    save(BASE/'staged.json',result);print(json.dumps(result))


def launch():
    spec = prepared(); assert not (BASE/'deployment.json').exists()
    result = json.loads(ssh(PRELUDE+f'''
from protocol import read,pin,save,verify
import remote
assert read(base/'staged.json')['passed'] and not (base/'deployment.json').exists() and not (base/'identity.json').exists()
assert pin(base/'payload.json')=={spec['payload']!r}
remote.idle();verify(base)
env=dict(os.environ,PYTHONPATH={SITE!r},PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
with (base/'supervisor.stdout').open('x') as out,(base/'supervisor.stderr').open('x') as err:
 p=subprocess.Popen([sys.executable,'-B',str(base/'tools/remote.py')],cwd=base,env=env,
  stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 receipt=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time())
save(base/'deployment.json',receipt);print(json.dumps(receipt))
''',300))
    save(BASE/'deployment.json',result); print(json.dumps(result))


def observe():
    value = json.loads(ssh(PRELUDE+'''
from protocol import read
from remote import live
deployment=read(base/'deployment.json');state=read(base/'identity.json') if (base/'identity.json').exists() else None
ids=[deployment]
if state:
 for r in state['runs']: ids.extend(dict(pid=int(p),birth=b) for p,b in r['members'].items())
print(json.dumps(dict(live=[i for i in ids if live(i)],complete=False if state is None else state['complete'],
 code=None if state is None else state['code'],latest=None if not state or not state['runs'] else {k:state['runs'][-1].get(k) for k in ['name','samples','complete','code']},
 stderr=(base/'supervisor.stderr').read_text()[-4000:])))
'''))
    with (BASE/'observations.jsonl').open('a') as log: log.write(json.dumps(value)+'\n')
    print(json.dumps(value)); return value


def collect():
    prepared(); assert not (BASE/'collected').exists() and not (BASE/'results.tar.gz').exists()
    script = PRELUDE+'''
from protocol import JOBS,pin,read,save,verify
from remote import live
deployment=read(base/'deployment.json');state=read(base/'identity.json')
ids=[deployment]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
assert state['complete'] and not any(live(i) for i in ids)
error=None
try:verify(base)
except BaseException as e:error=repr(e)
paths=[p for folder in [*JOBS,'logs','built','evidence','manifests','graph-reference'] for p in (base/folder).rglob('*') if p.is_file()]
paths += [p for p in base.iterdir() if p.is_file() and p.name!='transfer.tar.gz']
files={p.relative_to(base).as_posix():pin(p) for p in sorted(paths)}
save(base/'collection.json',dict(terminal=True,identities=ids,code=state['code'],input_error=error,files=files,payload=pin(base/'payload.json')))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',dereference=True) as tar:
 for name in [*files,'collection.json']:tar.add(base/name,arcname=name,recursive=False)
'''
    compile(script,'native-layout-collect','exec')
    with (BASE/'results.tar.gz').open('xb') as out,(BASE/'collection.stderr').open('x') as err:
        result = subprocess.run(SSH+['python3 -B -'],input=script.encode(),stdout=out,stderr=err,timeout=600,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0,'Preserve partial collection; do not relaunch'
    target = BASE/'collected'; target.mkdir()
    with tarfile.open(BASE/'results.tar.gz') as tar:
        members = tar.getmembers()
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert len({m.name for m in members}) == len(members)
        tar.extractall(target,filter='data')
    receipt = read(target/'collection.json')
    for name,wanted in receipt['files'].items(): assert pin(target/name) == wanted,name
    assert receipt['payload']==pin(BASE/'payload.json')
    save(BASE/'collection-transfer.json',dict(passed=True,archive=pin(BASE/'results.tar.gz'),receipt=pin(target/'collection.json')))
    print(json.dumps(dict(code=receipt['code'],terminal=receipt['terminal'],files=len(receipt['files']))))


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare','stage','launch','observe','collect']
    globals()[sys.argv[1]]()
