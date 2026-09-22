"""Prepare, stage, observe and collect one native-layout diagnostic."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from protocol import LIMITS, pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-native-layout-amd-20260922'
AMD = ROOT/'artifacts/parakeet-single-panel-amd-execution-v2-20260922'
OLD = ROOT/'artifacts/pyannote-single-panel-amd-execution-v2-20260922'
PAYLOAD = ROOT/'artifacts/parakeet-single-panel-amd-payload-v2-20260922/payload'
REMOTE = '/dev/shm/lokad-pyannote-native-layout-20260922'
SITE = '/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python'
SSH = ['ssh','-i','C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem','-o','BatchMode=yes','-o','ConnectTimeout=20','vermorel@74.178.91.76']


def previous_closed():
    # This diagnostic is permitted after either verdict, but never before the
    # actual preceding worker and its collecting controller have terminated.
    sys.path.insert(0,str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
    import psutil
    proof = read(AMD/'closed.json'); assert proof['passed']
    for name,wanted in proof['files'].items(): assert pin(AMD/name) == wanted,name
    state = read(AMD/'controller/state.json'); assert state['complete'] and state['code'] == 0
    for identity in [state['supervisor']]+[r['child'] for r in state['stages']]:
        try: assert psutil.Process(identity['pid']).create_time() != identity['birth']
        except psutil.NoSuchProcess: pass
    receipt = read(AMD/'collected/collection.json'); assert receipt['terminal'] and receipt['input_error'] is None
    return read(AMD/'collected/campaign/identity.json')['supervisor']


def prepare():
    assert not BASE.exists(); owner = previous_closed()
    proof = read(OLD/'closed.json'); assert proof['passed']
    assert pin(OLD/'closed.json')['sha256'] == '2649716b4523325d46d416894b161cb209c51bd2e77fb083ac20d0ae95532c5b'
    for name,wanted in proof['files'].items(): assert pin(OLD/name) == wanted,name
    payload = BASE/'payload'; payload.mkdir(parents=True); (payload/'tools').mkdir(); (payload/'inputs').mkdir()
    files = {}
    for p in TOOLS.glob('*'):
        if p.is_file():
            if p.suffix == '.py': ast.parse(p.read_text(),str(p))
            files[p.relative_to(ROOT).as_posix()] = pin(p)
    # Parse collected graphs locally. The target only needs its existing ORT
    # and NumPy; do not add a new ONNX parser dependency to the benchmark VM.
    for name in ['onnx','numpy']:
        origin = Path(importlib.util.find_spec(name).origin).resolve()
        for p in origin.parent.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts: files[p.as_posix()] = pin(p)
    files[Path(sys.executable).as_posix()] = pin(Path(sys.executable))
    for name in ['protocol.py','worker.py','remote.py']: shutil.copy2(TOOLS/name,payload/'tools'/name)
    previous = OLD/'collected/campaign/portable-pyannote-output'
    records = read(previous/'result.json')['rows']; cases = []
    references = {(r['name'],r['model']):r for r in read(PAYLOAD/'graph-reference.json')}
    for r in records:
        if r['pass'] != 0: continue
        source = previous/r['input']['file']; wanted = dict(bytes=r['input']['values']*4,sha256=r['input']['sha256'])
        assert pin(source) == wanted
        target = payload/'inputs'/source.name; shutil.copy2(source,target)
        reference = references[(r['name'],r['model'])]; native = PAYLOAD/reference['path']
        assert pin(native) == {k:reference[k] for k in ['bytes','sha256']}
        shutil.copy2(native,payload/'inputs'/native.name)
        cases.append(dict(name=r['name'],model=r['model'],input='inputs/'+source.name,shape=r['input']['shape'],
            input_pin=wanted,reference='inputs/'+native.name,reference_pin=pin(native)))
    assert len(cases) == 6 and len({(r['name'],r['model']) for r in cases}) == 6
    manifest = read(PAYLOAD/'manifests/production-pyannote.json'); original = read(PAYLOAD/'payload.json')
    execution = read(AMD/'execution/execution.json')
    external = dict(execution['external'])
    # The benchmark inventory is the prior source of native/interpreter pins.
    # Model paths are absolute and must already be in that frozen inventory.
    models = {name:manifest['models'][key]['path'] for name,key in [('embedding','encoder'),('segmentation','segmentation')]}
    for name,key in [('embedding','encoder'),('segmentation','segmentation')]:
        external[models[name]] = original['external'][models[name]]
        assert external[models[name]] == {k:manifest['models'][key][k] for k in ['bytes','sha256']}
    for p in [ROOT/'models/pyannote-embedding/embedding_encoder.onnx', ROOT/'models/pyannote-segmentation/segmentation/model.onnx']:
        files[p.relative_to(ROOT).as_posix()] = pin(p)
    for p in [AMD/'closed.json',OLD/'closed.json',previous/'result.json',PAYLOAD/'graph-reference.json',
              ROOT/'artifacts/pyannote-nchwc-windows-v2-20260922/closed.json']:
        files[p.relative_to(ROOT).as_posix()] = pin(p)
    spec = dict(passed=True,limits=LIMITS,previous_owner=owner,boot_time=1789634288.0,
        interpreter=original['interpreter'],models=models,cases=cases,external=external,
        files={p.relative_to(payload).as_posix():pin(p) for p in payload.rglob('*') if p.is_file()},
        previous_closure=pin(AMD/'closed.json'),scope='Native graph construction and profiled node execution only; no speed selection')
    save(payload/'payload.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(payload.rglob('*')):
            if p.is_file(): archive.add(p,arcname=p.relative_to(payload).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=files,payload=pin(payload/'payload.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),cases=len(cases),external=len(external))))


def prepared():
    value = read(BASE/'prepared.json'); assert value['passed']
    for name,wanted in value['files'].items(): assert pin(ROOT/name) == wanted,name
    assert pin(BASE/'payload.tar.gz') == value['archive'] and pin(BASE/'payload/payload.json') == value['payload']
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
assert psutil.virtual_memory().available>=12*1024**3 and psutil.disk_usage('/dev/shm').free>=3*1024**3
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
from protocol import pin,save,verify
import remote
remote.idle()
assert pin(base/'payload.json')=={spec['payload']!r}
value=verify(base);assert not remote.live(value['previous_owner'])
receipt=dict(passed=True,payload=pin(base/'payload.json'),files=len(value['files']),external=len(value['external']))
save(base/'staged.json',receipt);print(json.dumps(receipt))
''',300))
    save(BASE/'staged.json',result); print(json.dumps(result))


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
from protocol import pin,read,save,verify
from remote import live
deployment=read(base/'deployment.json');state=read(base/'identity.json')
ids=[deployment]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
assert state['complete'] and not any(live(i) for i in ids)
error=None
try:verify(base)
except BaseException as e:error=repr(e)
paths=[p for folder in ['metadata','profile','logs'] for p in (base/folder).rglob('*') if p.is_file()]
paths += [p for p in base.iterdir() if p.is_file() and p.name!='transfer.tar.gz']
files={p.relative_to(base).as_posix():pin(p) for p in sorted(paths)}
save(base/'collection.json',dict(terminal=True,identities=ids,code=state['code'],input_error=error,files=files,payload=pin(base/'payload.json')))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
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
    assert receipt['payload'] == pin(BASE/'payload/payload.json')
    save(BASE/'collection-transfer.json',dict(passed=True,archive=pin(BASE/'results.tar.gz'),receipt=pin(target/'collection.json')))
    print(json.dumps(dict(code=receipt['code'],terminal=receipt['terminal'],files=len(receipt['files']))))


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare','stage','launch','observe','collect']
    globals()[sys.argv[1]]()
