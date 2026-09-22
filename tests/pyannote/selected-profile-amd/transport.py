"""Stage once, observe exact owners, and stream evidence only after termination."""
import subprocess
import sys
import tarfile
from common import *


def ssh(script, timeout=120):
    compile(script, 'amd-profile-transport', 'exec')
    result = subprocess.run(SSH+['python3 -B -'], input=script, text=True, encoding='utf8', capture_output=True,
        timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0, (result.returncode, result.stderr[-6000:])
    return result.stdout


PRELUDE = f'''from pathlib import Path
import os,sys,json,subprocess,tarfile,hashlib,shutil,time
sys.path.insert(0,{REMOTE_SITE!r})
import psutil
base=Path({REMOTE!r})
os.sched_setaffinity(0,{{0}})
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p)
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def live(i):
 try:
  p=psutil.Process(i['pid']);return p.create_time()==i['birth'] and p.status()!=psutil.STATUS_ZOMBIE
 except psutil.NoSuchProcess:return False
'''


def stage():
    spec = prepared(); assert not (BASE / 'staged.json').exists()
    receipt = json.loads(ssh(PRELUDE+'''
assert not base.exists()
active=[p.info for p in psutil.process_iter(['pid','name','create_time','cmdline']) if p.info['name'] in ('dotnet','perf')]
assert not active,active
assert psutil.boot_time()==1789634288.0
assert psutil.virtual_memory().available>=10*1024**3 and shutil.disk_usage('/dev/shm').free>=3*1024**3
base.mkdir();write(base/'stage-started.json',dict(time=time.time(),boot_time=psutil.boot_time(),active=active))
print(json.dumps(dict(available=psutil.virtual_memory().available,tmpfs_free=shutil.disk_usage(base).free)))
'''))
    save(BASE / 'stage-started.json', receipt)
    subprocess.run(['scp', *SSH[1:-1], str(BASE / 'payload.tar.gz'), SSH[-1]+':'+REMOTE+'/transfer.tar.gz'],
        check=True, timeout=300, creationflags=subprocess.CREATE_NO_WINDOW)
    receipt = json.loads(ssh(PRELUDE+f'''
assert pin(base/'transfer.tar.gz')=={spec['archive']!r}
with tarfile.open(base/'transfer.tar.gz') as tar:
 members=tar.getmembers()
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
 assert len({{m.name for m in members}})==len(members)
 tar.extractall(base,filter='data')
assert pin(base/'payload.json')=={spec['payload']!r}
sys.path.insert(0,str(base/'tools'))
import remote
remote.idle(); value=remote.verify()
assert remote.artifact_size()<=remote.LIMITS['artifacts']
receipt=dict(passed=True,payload=pin(base/'payload.json'),files=len(value['files']),external=len(value['external']))
write(base/'staged.json',receipt);print(json.dumps(receipt))
''', 300))
    save(BASE / 'staged.json', receipt); print(json.dumps(receipt))


def launch():
    spec = prepared(); assert not (BASE / 'deployment.json').exists()
    receipt = json.loads(ssh(PRELUDE+f'''
assert read(base/'staged.json')['passed'] and not (base/'deployment.json').exists() and not (base/'identity.json').exists()
assert pin(base/'payload.json')=={spec['payload']!r}
sys.path.insert(0,str(base/'tools'));import remote
remote.idle();remote.verify()
with (base/'supervisor.stdout').open('x') as out,(base/'supervisor.stderr').open('x') as err:
 p=subprocess.Popen([sys.executable,'-B',str(base/'tools/remote.py')],cwd=base,env=remote.clean_env(),
  stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 receipt=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),payload=pin(base/'payload.json'))
write(base/'deployment.json',receipt);print(json.dumps(receipt))
''', 300))
    save(BASE / 'deployment.json', receipt); print(json.dumps(receipt))


def observe():
    value = json.loads(ssh(PRELUDE+'''
deployment=read(base/'deployment.json')
state=read(base/'identity.json') if (base/'identity.json').exists() else None
ids=[deployment]
if state:
 for row in state['runs']:ids.extend(row['processes'].values())
last=None if not state or not state['runs'] else state['runs'][-1]
print(json.dumps(dict(supervisor_live=live(deployment),live=[i for i in ids if live(i)],complete=False if not state else state['complete'],
 code=None if not state else state['code'],latest=None if last is None else {k:last.get(k) for k in ['name','complete','code','samples','peak_rss','processes']},
 stderr=(base/'supervisor.stderr').read_text()[-5000:],stdout=(base/'supervisor.stdout').read_text()[-2000:])))
'''))
    with (BASE / 'observations.jsonl').open('a', encoding='utf8') as out: out.write(json.dumps(value)+'\n')
    print(json.dumps(value))


def collect():
    prepared(); assert not (BASE / 'collected').exists() and not (BASE / 'results.tar.gz').exists()
    script = PRELUDE+'''
deployment=read(base/'deployment.json');assert not live(deployment)
state=read(base/'identity.json');assert state['complete']
ids=[deployment]
for row in state['runs']:
 assert row['complete'];ids.extend(row['processes'].values())
assert not any(live(i) for i in ids)
sys.path.insert(0,str(base/'tools'));import remote
error=None
try:remote.verify()
except BaseException as e:error=repr(e)
paths=[p for folder in ['control','sampled-a','sampled-b','logs'] for p in (base/folder).rglob('*') if p.is_file()]
paths += [p for p in base.iterdir() if p.is_file() and p.name!='transfer.tar.gz']
files={p.relative_to(base).as_posix():pin(p) for p in sorted(paths)}
receipt=dict(terminal=True,identities=ids,code=state['code'],input_error=error,files=files,payload=pin(base/'payload.json'))
write(base/'collection.json',receipt)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name in [*files,'collection.json']:tar.add(base/name,arcname=name,recursive=False)
'''
    compile(script, 'stream-profile-evidence', 'exec')
    with (BASE / 'results.tar.gz').open('xb') as out, (BASE / 'collection.stderr').open('x') as err:
        result = subprocess.run(SSH+['python3 -B -'], input=script.encode(), stdout=out, stderr=err,
            timeout=600, creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0, 'Retain partial archive; collection failed'
    target = BASE / 'collected'; target.mkdir()
    with tarfile.open(BASE / 'results.tar.gz') as tar:
        members = tar.getmembers()
        assert all(m.isfile() and (target / m.name).resolve().is_relative_to(target) for m in members)
        assert len({m.name for m in members}) == len(members)
        tar.extractall(target, filter='data')
    receipt = read(target / 'collection.json')
    for name, wanted in receipt['files'].items(): assert pin(target / name) == wanted, name
    assert receipt['payload'] == pin(BASE / 'payload/payload.json')
    assert {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()} == set(receipt['files']) | {'collection.json'}
    save(BASE / 'collection-transfer.json', dict(archive=pin(BASE / 'results.tar.gz'), receipt=pin(target / 'collection.json'),
        files=len(receipt['files']), terminal=True, code=receipt['code'], input_error=receipt['input_error']))
    print(json.dumps(read(BASE / 'collection-transfer.json')))


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['stage','launch','observe','collect']
    globals()[sys.argv[1]]()
