"""Prepare an isolated Data observer; keep the qualified Core and application flow."""
import ast
import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-packed-final-row-profile-amd-20260925'
REMOTE = '/dev/shm/lokad-parakeet-packed-final-row-profile-20260925'
PRIOR = ROOT/'artifacts/parakeet-managed-phase-amd-20260924'
APP = ROOT/'artifacts/parakeet-packed-final-row-release-app-amd-20260925'
REMOTE_PRIOR = REMOTE+'/runtime-base'
REMOTE_APP = '/dev/shm/lokad-parakeet-packed-final-row-release-app-20260925'
spec = importlib.util.spec_from_file_location('native_transport', TOOLS.parent/'ort-diagnosis-amd/run.py')
transport = importlib.util.module_from_spec(spec); spec.loader.exec_module(transport)
pin, read, write, ssh, SSH = transport.pin, transport.read, transport.write, transport.ssh, transport.SSH
PRELUDE = transport.PRELUDE+f"\nbase=Path({REMOTE!r})\nsys.path.insert(0,str(base))\n"


def prepare():
    from prepare import prepare as create_bundle
    create_bundle()


def prepared():
    value=read(BASE/'prepared.json')
    assert value['passed'] and pin(BASE/'payload.tar.gz')==value['archive']
    assert pin(BASE/'bundle/spec.json')==value['spec']
    for name,wanted in value['inputs'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in read(BASE/'bundle/spec.json')['files'].items():assert pin(BASE/'bundle'/name)==wanted,name
    return value


def stage():
    prepared()
    from prepare import release_gates
    release_gates()
    receipt=read(BASE/'prepared.json');assert not (BASE/'staged.json').exists()
    result=ssh(PRELUDE+'''
assert not base.exists()
assert psutil.boot_time()==1789634288.0
assert psutil.virtual_memory().available>=2*1024**3 and psutil.disk_usage('/dev/shm').free>=1024**3
base.mkdir()
print(json.dumps(dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free)))
''')
    write(BASE/'stage-started.json',result)
    subprocess.run(['scp',*SSH[1:-1],str(BASE/'payload.tar.gz'),SSH[-1]+':'+REMOTE+'/transfer.tar.gz'],check=True,timeout=90,creationflags=subprocess.CREATE_NO_WINDOW)
    result=ssh(PRELUDE+f'''
with (base/'transfer.tar.gz').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=={receipt['archive']['sha256']!r}
with tarfile.open(base/'transfer.tar.gz') as tar:
 members=tar.getmembers()
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
 assert len({{m.name for m in members}})==len(members)
 tar.extractall(base,filter='data')
from remote import verify,idle,pin
verify();idle()
assert pin(base/'spec.json')=={receipt['spec']!r}
print(json.dumps(dict(passed=True,spec=pin(base/'spec.json'))))
''')
    write(BASE/'staged.json',result);print(json.dumps(result))


def launch(kind):
    prepared()
    assert kind in ['build','capture'] and not (BASE/(kind+'-deployment.json')).exists()
    result=ssh(PRELUDE+f'''
from remote import verify,idle,pin
verify();idle();kind={kind!r}
assert not (base/(kind+'-state.json')).exists()
env=dict(os.environ,PYTHONPATH={transport.SITE!r},PYTHONDONTWRITEBYTECODE='1')
env.pop('PYTHONOPTIMIZE',None)
with (base/(kind+'-supervisor.stdout')).open('x') as out,(base/(kind+'-supervisor.stderr')).open('x') as err:
 p=subprocess.Popen([sys.executable,'-B',str(base/'remote.py'),kind],cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 value=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time())
print(json.dumps(value))
''')
    write(BASE/(kind+'-deployment.json'),result);print(json.dumps(result))


def observe(kind):
    assert not (BASE/'closed.json').exists()
    result=ssh(PRELUDE+f'''
from remote import read,live
path=base/({kind!r}+'-state.json');state=read(path) if path.exists() else None
ids=[{read(BASE/(kind+'-deployment.json'))!r}]
if state:ids += [dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
print(json.dumps(dict(state=state,live=[i for i in ids if live(i)],stderr=(base/({kind!r}+'-supervisor.stderr')).read_text()[-5000:])))
''')
    with (BASE/(kind+'-observations.jsonl')).open('a') as stream:stream.write(json.dumps(result)+'\n')
    state=result['state'];print(json.dumps(dict(live=result['live'],complete=state and state['complete'],code=state and state['code'],latest=state and state['runs'][-1],error=state and state.get('error'),stderr=result['stderr'])))


def collect(kind):
    prepared()
    target=BASE/(kind+'-collected');assert not target.exists()
    script=PRELUDE+f'''
from remote import read,live,pin,verify
kind={kind!r};state=read(base/(kind+'-state.json'))
assert state['complete'] and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
verify()
paths=[p for p in base.rglob('*') if p.is_file() and ('logs' in p.parts or p.parent==base or (kind=='build' and p.parent.name in ['inventory','runtime-control','runtime-observed']) or (kind=='capture' and p.parent.name in ['control','phase','wall'])) and p.name!='transfer.tar.gz']
files={{p.relative_to(base).as_posix():pin(p) for p in paths}}
(base/(kind+'-collection.json')).write_text(json.dumps(dict(files=files,state=pin(base/(kind+'-state.json')),terminal=True,code=state['code'])))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name in [*files,kind+'-collection.json']:tar.add(base/name,arcname=name,recursive=False)
'''
    archive=BASE/(kind+'-results.tar.gz')
    with archive.open('xb') as out,(BASE/(kind+'-collection.stderr')).open('x') as err:
        result=subprocess.run(SSH+['python3','-B','-'],input=script.encode(),stdout=out,stderr=err,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode==0
    target.mkdir()
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert len({m.name for m in members})==len(members)
        tar.extractall(target,filter='data')
    value=read(target/(kind+'-collection.json'))
    for name,wanted in value['files'].items():assert pin(target/name)==wanted,name
    write(BASE/(kind+'-transfer.json'),dict(passed=True,archive=pin(archive),collection=pin(target/(kind+'-collection.json'))))
    print(json.dumps(dict(code=value['code'],files=len(value['files']),archive=pin(archive))))


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='stage':stage()
    else:dict(launch=launch,observe=observe,collect=collect)[action](sys.argv[2])
