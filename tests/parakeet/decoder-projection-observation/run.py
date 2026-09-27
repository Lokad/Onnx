"""Stage once, reuse the original launcher/observer, and retain complete results."""
import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from protocol import LIMITS, pin, read, save
from prepare import ROOT, TOOLS, PARENT, BASE, QUALIFIED, prepare, previous_closed

loader = importlib.util.spec_from_file_location('decoder_observation_transport', PARENT/'run.py')
transport = importlib.util.module_from_spec(loader); loader.loader.exec_module(transport)
REMOTE = '/dev/shm/lokad-parakeet-decoder-projection-observation-20260927'
transport.PRELUDE = transport.PRELUDE.replace(transport.REMOTE, REMOTE)
transport.BASE, transport.REMOTE, transport.TOOLS = BASE, REMOTE, TOOLS
SSH, PRELUDE, ssh = transport.SSH, transport.PRELUDE, transport.ssh
prepared, launch, observe = transport.prepared, transport.launch, transport.observe


def stage():
    spec = prepared()
    assert not (BASE/'staged.json').exists()
    owners = read(QUALIFIED/'collected/collection.json')['identities']
    first = json.loads(ssh(PRELUDE+f'''
assert not base.exists() and psutil.boot_time()==1789634288.0
for identity in {owners!r}:
 try:
  process=psutil.Process(identity['pid'])
  assert process.create_time()!=identity['birth'] or process.status()==psutil.STATUS_ZOMBIE
 except psutil.NoSuchProcess:pass
own=psutil.Process();ancestors={{own.pid,*[p.pid for p in own.parents()]}}
for p in psutil.process_iter(['pid','name','cmdline']):
 if p.pid in ancestors:continue
 command=' '.join(p.info['cmdline'] or [])
 assert p.info['name'] not in ['dotnet','perf'],('Runtime already active',p.pid)
 assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in command),('Campaign already active',p.pid)
assert psutil.virtual_memory().available>={LIMITS['preflight_available']}
assert psutil.disk_usage('/dev/shm').free>={LIMITS['preflight_tmpfs']}
base.mkdir()
print(json.dumps(dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free)))
'''))
    save(BASE/'stage-started.json', first)
    subprocess.run(['scp', *SSH[1:-1], str(BASE/'payload.tar.gz'), SSH[-1]+':'+REMOTE+'/transfer.tar.gz'],
                   check=True, timeout=180, creationflags=subprocess.CREATE_NO_WINDOW)
    extracted = json.loads(ssh(PRELUDE+f'''
import hashlib
archive=base/'transfer.tar.gz'
with archive.open('rb') as stream:actual=dict(bytes=archive.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
assert actual=={spec['archive']!r}
with tarfile.open(archive) as tar:
 members=tar.getmembers()
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
 assert len({{m.name for m in members}})==len(members)
 tar.extractall(base,filter='data')
from protocol import pin,read,save
assert pin(base/'stage.json')=={spec['stage']!r}
for name,wanted in read(base/'stage.json')['files'].items():assert pin(base/name)==wanted,name
archive.unlink()
print(json.dumps(dict(passed=True,archive_retired=True)))
'''))
    save(BASE/'extracted.json', extracted)
    staged = json.loads(ssh(PRELUDE+f'''
import base64
from protocol import pin,save,verify
from remote import idle,live
idle()
env=dict(os.environ,PYTHONPATH={transport.SITE!r},PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
process=subprocess.run([sys.executable,'-B',str(base/'tools/remote_prepare.py')],cwd=base,env=env,text=True,capture_output=True,timeout=180)
with (base/'stage-prepare.stdout').open('x') as stream:stream.write(process.stdout)
with (base/'stage-prepare.stderr').open('x') as stream:stream.write(process.stderr)
assert process.returncode==0,(process.returncode,process.stderr[-4000:])
value=verify(base);assert not live(value['previous_owner'])
receipt=dict(passed=True,payload=pin(base/'payload.json'),files=len(value['files']),external=len(value['external']))
save(base/'staged.json',receipt)
print(json.dumps(dict(**receipt,payload_base64=base64.b64encode((base/'payload.json').read_bytes()).decode('ascii'))))
''', 240))
    raw = staged.pop('payload_base64')
    (BASE/'payload.json').write_bytes(base64.b64decode(raw))
    assert pin(BASE/'payload.json') == staged['payload']
    save(BASE/'staged.json', staged)
    print(json.dumps(staged))


def collect():
    prepared()
    assert not (BASE/'collected').exists() and not (BASE/'results.tar.gz').exists()
    script = PRELUDE+'''
from protocol import pin,read,save,verify
from remote import live
state=read(base/'identity.json');deployment=read(base/'deployment.json')
identities=[deployment]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
assert state['complete'] and not any(live(i) for i in identities)
error=None
try:verify(base)
except BaseException as e:error=repr(e)
files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
assert 'collection.json' not in files
save(base/'collection.json',dict(terminal=True,identities=identities,code=state['code'],input_error=error,files=files,payload=pin(base/'payload.json')))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',dereference=True) as tar:
 for name in [*files,'collection.json']:tar.add(base/name,arcname=name,recursive=False)
'''
    compile(script, 'decoder-collection', 'exec')
    with (BASE/'results.tar.gz').open('xb') as out, (BASE/'collection.stderr').open('x') as err:
        process = subprocess.run(SSH+['python3 -B -'], input=script.encode(), stdout=out, stderr=err,
                                 timeout=300, creationflags=subprocess.CREATE_NO_WINDOW)
    assert process.returncode == 0, 'Inspect the same terminal run; preserve partial collection'
    target = BASE/'collected'; target.mkdir()
    with tarfile.open(BASE/'results.tar.gz') as archive:
        members = archive.getmembers()
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert len({m.name for m in members}) == len(members)
        archive.extractall(target, filter='data')
    receipt = read(target/'collection.json')
    for name, wanted in receipt['files'].items(): assert pin(target/name) == wanted, name
    assert receipt['payload'] == pin(BASE/'payload.json')
    save(BASE/'collection-transfer.json', dict(passed=True, archive=pin(BASE/'results.tar.gz'), receipt=pin(target/'collection.json')))
    print(json.dumps(dict(terminal=True, code=receipt['code'], files=len(receipt['files']))))


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare', 'stage', 'launch', 'observe', 'collect']
    globals()[sys.argv[1]]()
