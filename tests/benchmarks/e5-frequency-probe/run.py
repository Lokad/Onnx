"""Freeze, launch, observe and collect one short counter proof without models."""
import base64
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'tests/parakeet/packed-final-row-pyannote-amd'))
from protocol import pin, read, save
spec = importlib.util.spec_from_file_location('qualified_transport', ROOT/'tests/parakeet/packed-final-row-pyannote-amd/run.py')
transport = importlib.util.module_from_spec(spec); spec.loader.exec_module(transport)
transport_ssh = transport.ssh

BASE = ROOT/'artifacts/e5-frequency-proof-amd-20260925'
REMOTE = '/dev/shm/lokad-e5-frequency-proof-20260925'
SITE = '/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python'
PRELUDE = f'''import os,sys,json,subprocess,time,base64
from pathlib import Path
sys.path.insert(0,{SITE!r})
import psutil
os.sched_setaffinity(0,{{0}})
base=Path({REMOTE!r})
sys.path.insert(0,'/dev/shm/lokad-parakeet-packed-final-row-pyannote-20260925/tools')
from remote import live,idle
'''


def launch():
    assert not BASE.exists()
    for name in ['e5-repeatability-diagnostic-amd-20260925', 'e5-frequency-counter-capabilities-20260925',
                 'parakeet-packed-final-row-pyannote-amd-20260925']:
        assert read(ROOT/'artifacts'/name/'closed.json')['passed']
    files = {p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()}
    source = {name:base64.b64encode((TOOLS/name).read_bytes()).decode() for name in ['counter.py', 'proof.py']}
    BASE.mkdir(); save(BASE/'prepared.json', dict(files=files, inference_calls=0))
    value = json.loads(transport_ssh(PRELUDE+f'''
idle()
assert psutil.boot_time()==1789634288.0 and not base.exists()
base.mkdir()
for name,encoded in {source!r}.items(): (base/name).write_bytes(base64.b64decode(encoded))
env=dict(os.environ,PYTHONPATH={SITE!r},PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
env.pop('PYTHONOPTIMIZE',None)
with (base/'supervisor.stdout').open('x') as out,(base/'supervisor.stderr').open('x') as err:
 p=subprocess.Popen([sys.executable,'-B',str(base/'proof.py')],cwd=base,env=env,
  stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
value=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time())
(base/'deployment.json').write_text(json.dumps(value))
print(json.dumps(value))
'''))
    save(BASE/'deployment.json', value); print(json.dumps(value))


def observe():
    print(transport_ssh(PRELUDE+'''
owner=json.loads((base/'deployment.json').read_text())
state=json.loads((base/'state.json').read_text()) if (base/'state.json').exists() else None
print(json.dumps(dict(live=live(owner),complete=state and state['complete'],code=state and state['code'],
 error=None if state is None else state.get('error'),stderr=(base/'supervisor.stderr').read_text())))
'''))


def collect():
    assert not (BASE/'collected').exists()
    value = json.loads(transport_ssh(PRELUDE+'''
owner=json.loads((base/'deployment.json').read_text());state=json.loads((base/'state.json').read_text())
assert not live(owner) and state['complete'] and state['terminal']
assert all(not live(dict(pid=int(pid),birth=birth)) for pid,birth in state['identities'].items())
files={p.relative_to(base).as_posix():base64.b64encode(p.read_bytes()).decode() for p in base.rglob('*') if p.is_file()}
print(json.dumps(dict(terminal=True,owner=owner,files=files)))
'''))
    target = BASE/'collected'; target.mkdir()
    for name, encoded in value['files'].items():
        p = target/name; assert p.resolve().is_relative_to(target.resolve())
        p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(base64.b64decode(encoded))
    receipt = dict(terminal=True, owner=value['owner'], files={name:pin(target/name) for name in value['files']})
    save(BASE/'collected.json', receipt)
    print(json.dumps(dict(terminal=True, files=len(receipt['files']), code=read(target/'state.json')['code'])))


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['launch', 'observe', 'collect']
    globals()[sys.argv[1]]()
