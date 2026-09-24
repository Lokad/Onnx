"""Stage, launch, observe and collect the original-workload ORT diagnosis."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-ort-diagnosis-amd-20260924'
APP = ROOT/'artifacts/parakeet-prepared-recurrence-app-amd-20260924'
REMOTE = '/dev/shm/lokad-parakeet-ort-diagnosis-20260924'
REMOTE_APP = '/dev/shm/lokad-parakeet-prepared-recurrence-app-20260924'
SITE = '/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python'
SSH = ['ssh', '-i', 'C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=20', 'vermorel@74.178.91.76']
PRELUDE = f'''import os,sys,json,subprocess,time,hashlib,tarfile
from pathlib import Path
sys.path.insert(0,{SITE!r})
import psutil
os.sched_setaffinity(0,{{0}})
base=Path({REMOTE!r})
'''


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def ssh(script):
    compile(script, 'ort-diagnosis-transport', 'exec')
    result = subprocess.run(SSH+['python3', '-B', '-'], input=script, text=True, encoding='utf8',
                            capture_output=True, timeout=120, creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0, result.stderr[-5000:]
    return json.loads(result.stdout)


def stage():
    assert not BASE.exists()
    proof = read(APP/'closed.json')
    assert proof['passed']
    for name, wanted in proof['files'].items():
        assert pin(APP/name) == wanted, name
    receipt = read(APP/'collected/collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    from observer import instrument
    native = APP/'collected/runtime/native.py'
    instrument(native.read_text(encoding='utf8'))
    files = {name: (TOOLS/name).read_bytes() for name in ['observer.py', 'remote.py', 'README.md']}
    for name, content in files.items():
        if name.endswith('.py'):
            ast.parse(content, name)
    spec = dict(app=REMOTE_APP, boot_time=1789634288.0, previous_closure=pin(APP/'closed.json'),
        payload=pin(APP/'payload.json'), collection=pin(APP/'collected/collection.json'),
        native_consumer=pin(native), manifest=pin(APP/'collected/manifests/current-parakeet.json'),
        tools={name: pin(TOOLS/name) for name in files}, modes=['control', 'profile'],
        limits=dict(preflight_available=12*1024**3, preflight_tmpfs=3*1024**3, rss=12*1024**3,
                    available=1024**3, tmpfs=1024**3, output=1024**3, seconds_per_worker=900))
    BASE.mkdir(); write(BASE/'spec.json', spec)
    for name, content in files.items():
        (BASE/name).write_bytes(content)
    files['spec.json'] = (BASE/'spec.json').read_bytes()
    encoded = {name: base64.b64encode(content).decode('ascii') for name, content in files.items()}
    result = ssh(PRELUDE+f'''
import base64
app=Path({REMOTE_APP!r})
sys.path.insert(0,str(app/'tools'))
from remote import live,idle
state=json.loads((app/'identity.json').read_text())
assert state['complete'] and state['code']==0
receipt=json.loads((app/'collection.json').read_text())
assert all(not live(i) for i in receipt['identities'])
idle()
assert not base.exists()
assert psutil.virtual_memory().available>=12*1024**3 and psutil.disk_usage('/dev/shm').free>=3*1024**3
base.mkdir()
for name,data in {encoded!r}.items():
 with (base/name).open('xb') as stream:stream.write(base64.b64decode(data))
print(json.dumps(dict(passed=True,spec_sha256=hashlib.sha256((base/'spec.json').read_bytes()).hexdigest())))
''')
    assert result['spec_sha256'] == pin(BASE/'spec.json')['sha256']
    write(BASE/'staged.json', result); print(json.dumps(result))


def launch():
    assert read(BASE/'staged.json')['passed'] and not (BASE/'deployment.json').exists()
    result = ssh(PRELUDE+f'''
assert not (base/'deployment.json').exists() and not (base/'state.json').exists()
assert hashlib.sha256((base/'spec.json').read_bytes()).hexdigest()=={pin(BASE/'spec.json')['sha256']!r}
env=dict(os.environ,PYTHONPATH={SITE!r},PYTHONDONTWRITEBYTECODE='1')
env.pop('PYTHONOPTIMIZE',None)
with (base/'supervisor.stdout').open('x') as out,(base/'supervisor.stderr').open('x') as err:
 p=subprocess.Popen([sys.executable,'-B',str(base/'remote.py')],cwd=base,env=env,
  stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
value=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time())
(base/'deployment.json').write_text(json.dumps(value))
print(json.dumps(value))
''')
    write(BASE/'deployment.json', result); print(json.dumps(result))


def observe():
    assert not (BASE/'closed.json').exists()
    result = ssh(PRELUDE+'''
sys.path.insert(0,str(base))
from remote import live,read
state=read(base/'state.json') if (base/'state.json').exists() else None
owners=[read(base/'deployment.json')]+([] if state is None else [r['owner'] for r in state['runs'] if 'owner' in r])
print(json.dumps(dict(live=[i for i in owners if live(i)],complete=bool(state and state['complete']),
 code=None if state is None else state['code'],latest=None if not state or not state['runs'] else
 {k:state['runs'][-1].get(k) for k in ['mode','samples','complete','code']},
 stderr=(base/'supervisor.stderr').read_text()[-5000:])))
''')
    with (BASE/'observations.jsonl').open('a') as stream:
        stream.write(json.dumps(result)+'\n')
    print(json.dumps(result))


def collect():
    assert not (BASE/'collected').exists() and not (BASE/'results.tar.gz').exists()
    script = PRELUDE+'''
sys.path.insert(0,str(base))
from remote import live,read,pin,save
state=read(base/'state.json');owners=[read(base/'deployment.json')]+[r['owner'] for r in state['runs'] if 'owner' in r]
assert state['complete'] and not any(live(i) for i in owners)
files={p.relative_to(base).as_posix():pin(p) for p in base.rglob('*') if p.is_file()}
save(base/'collection.json',dict(terminal=True,identities=owners,code=state['code'],files=files))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',dereference=True) as archive:
 for name in [*files,'collection.json']:archive.add(base/name,arcname=name,recursive=False)
'''
    compile(script, 'ort-diagnosis-collection', 'exec')
    with (BASE/'results.tar.gz').open('xb') as out, (BASE/'collection.stderr').open('x') as err:
        result = subprocess.run(SSH+['python3', '-B', '-'], input=script.encode(), stdout=out, stderr=err,
                                timeout=300, creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0, 'Retain partial collection; do not repeat inference'
    target = BASE/'collected'; target.mkdir()
    with tarfile.open(BASE/'results.tar.gz') as archive:
        members = archive.getmembers()
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert len({m.name for m in members}) == len(members)
        archive.extractall(target, filter='data')
    receipt = read(target/'collection.json')
    for name, wanted in receipt['files'].items():
        assert pin(target/name) == wanted, name
    write(BASE/'transfer.json', dict(passed=True, archive=pin(BASE/'results.tar.gz'), collection=pin(target/'collection.json')))
    print(json.dumps(dict(terminal=True, code=receipt['code'], files=len(receipt['files']))))


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['stage', 'launch', 'observe', 'collect']
    globals()[sys.argv[1]]()
