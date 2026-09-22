"""Identity-checked AMD staging and terminal streaming collection; no retries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from candidate_protocol import pin, read, write, verified_files

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
PREPARED = ROOT/'artifacts/pyannote-single-panel-amd-payload-20260922'
BASE = ROOT/'artifacts/pyannote-single-panel-amd-execution-20260922'
REMOTE = '/dev/shm/lokad-pyannote-single-panel-app-20260922'
E5_CONTROL = ROOT/'artifacts/e5-randomized-processes-finish-v2-20260921/state.json'
SITE = ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'
SSH = ['ssh', '-i', 'C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'vermorel@74.178.91.76']
REMOTE_SITE = '/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python'


def local_e5_terminal():
    sys.path.insert(0, str(SITE)); import psutil
    state = read(E5_CONTROL)
    assert state['complete'] is True, 'Existing e5 controller still owns both VM phases'
    identities = [state['supervisor']]+[s['child'] for s in state['stages'] if 'child' in s]
    for identity in identities:
        try:
            assert psutil.Process(identity['pid']).create_time() != identity['birth'], ('Live e5 controller/collector', identity)
        except psutil.NoSuchProcess:
            pass
    prior = ROOT/'artifacts/pyannote-combined-amd-execution-v2-20260922/controller/state.json'
    value = read(prior)
    assert value['complete'] and value['code'] == 0
    for identity in [value['supervisor']] + [r['child'] for r in value['stages'] if 'child' in r]:
        try:
            assert psutil.Process(identity['pid']).create_time() != identity['birth'], ('Live prior audio owner', identity)
        except psutil.NoSuchProcess:
            pass
    return dict(state=pin(E5_CONTROL), code=state['code'], identities=identities)


def ssh(script, timeout=120):
    compile(script, 'audio-candidates-transport', 'exec')
    result = subprocess.run(SSH+['python3 -B -'], input=script, capture_output=True, text=True, encoding='utf8',
                            timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0, (result.returncode, result.stderr[-8000:])
    return result.stdout


PRELUDE = '''from pathlib import Path
import json,os,sys,subprocess,tarfile,time,shutil,hashlib
sys.path.insert(0,%r)
import psutil
base=Path(%r)
os.sched_setaffinity(0,{0})
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p)
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def live(identity):
 try:
  p=psutil.Process(identity['pid'])
  return p.create_time()==identity['birth'] and p.status()!=psutil.STATUS_ZOMBIE
 except psutil.NoSuchProcess:return False
''' % (REMOTE_SITE, REMOTE)


def checked_local():
    prepared = read(PREPARED/'prepared.json'); bundle = read(BASE/'prepared.json')
    assert pin(PREPARED/'payload.tar.gz') == prepared['archive'] == bundle['payload_archive']
    assert pin(PREPARED/'payload/payload.json') == prepared['payload']
    assert pin(BASE/'execution.tar.gz') == bundle['archive']
    execution = read(BASE/'execution/execution.json')
    assert pin(BASE/'execution/execution.json') == bundle['execution']
    assert execution['payload'] == prepared['payload']
    verified_files(BASE/'execution', execution['files'])
    for name, wanted in execution['local_tools'].items():
        assert pin(ROOT/name) == wanted, name
    return prepared, bundle, execution


def stage():
    e5 = local_e5_terminal(); prepared, bundle, execution = checked_local()
    assert not (BASE/'staged.json').exists()
    # Check remote e5 identities before the first remote directory/file write.
    guard = (TOOLS/'supervise.py').read_text().split('def e5_terminal():', 1)[1].split('\n\ndef native_result', 1)[0]
    guard = 'def e5_terminal():'+guard
    script = PRELUDE+'\ndef absent(identity):return not live(identity)\n'+guard+'''
e5=e5_terminal()
assert not base.exists()
assert psutil.virtual_memory().available >= 12*1024**3 and shutil.disk_usage('/dev/shm').free >= 3*1024**3
base.mkdir()
write(base/'stage-started.json',dict(e5_remote=e5,e5_local=%r,started=time.time()))
print(json.dumps(dict(e5_remote=e5,available=psutil.virtual_memory().available,tmpfs_free=shutil.disk_usage(base).free)))
''' % e5
    receipt = json.loads(ssh(script)); write(BASE/'stage-started.json', receipt)
    for source, name in [(PREPARED/'payload.tar.gz', 'transfer-payload.tar.gz'), (BASE/'execution.tar.gz', 'transfer-execution.tar.gz')]:
        subprocess.run(['scp', *SSH[1:-1], str(source), SSH[-1]+':'+REMOTE+'/'+name], check=True,
                       timeout=300, creationflags=subprocess.CREATE_NO_WINDOW)
    script = PRELUDE+'''
assert pin(base/'transfer-payload.tar.gz')==%r
assert pin(base/'transfer-execution.tar.gz')==%r
for name,destination in [('transfer-payload.tar.gz',base),('transfer-execution.tar.gz',base/'execution')]:
 with tarfile.open(base/name) as archive:
  members=archive.getmembers()
  assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
  assert len({m.name for m in members})==len(members)
  archive.extractall(destination,filter='data')
sys.path.insert(0,str(base/'execution'))
from candidate_protocol import verify
spec,execution=verify(base)
assert pin(Path(sys.executable))==spec['interpreter']
receipt=dict(passed=True,payload=pin(base/'payload.json'),execution=pin(base/'execution/execution.json'),
             available=psutil.virtual_memory().available,tmpfs_free=shutil.disk_usage(base).free)
write(base/'staged.json',receipt)
print(json.dumps(receipt))
''' % (prepared['archive'], bundle['archive'])
    receipt = json.loads(ssh(script, 600)); write(BASE/'staged.json', receipt)
    print(json.dumps(receipt))


def launch():
    local_e5_terminal(); _, bundle, _ = checked_local()
    assert not (BASE/'deployment.json').exists()
    receipt = json.loads(ssh(PRELUDE+'''
sys.path.insert(0,str(base/'execution'))
from candidate_protocol import verify
from supervise import e5_terminal
e5_terminal();spec,execution=verify(base)
assert pin(base/'execution/execution.json')==%r
assert read(base/'staged.json')['passed'] is True and not (base/'deployment.json').exists() and not (base/'campaign').exists()
env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
env.pop('PYTHONOPTIMIZE',None)
env.update(PYTHONPATH=os.pathsep.join(spec['python_paths']),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
env.update({k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
with (base/'supervisor.stdout').open('x') as out,(base/'supervisor.stderr').open('x') as err:
 p=subprocess.Popen([sys.executable,'-B',str(base/'execution/supervise.py'),str(base)],cwd=base,env=env,
                    stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 identity=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),payload=pin(base/'payload.json'),execution=pin(base/'execution/execution.json'))
write(base/'deployment.json',identity);print(json.dumps(identity))
''' % bundle['execution'], 600))
    write(BASE/'deployment.json', receipt); print(json.dumps(receipt))


def observe():
    return json.loads(ssh(PRELUDE+'''
deployment=read(base/'deployment.json');path=base/'campaign/identity.json'
state=read(path) if path.exists() else None
identities=[deployment]
if state is not None:
 for run in state['runs']:
  identities.extend(dict(pid=int(pid),birth=birth) for pid,birth in run['members'].items())
live_identities=[i for i in identities if live(i)]
print(json.dumps(dict(supervisor_live=live(deployment),live=live_identities,complete=False if state is None else state['complete'],
 code=None if state is None else state['code'],error=None if state is None else state.get('error'),
 latest=None if state is None or not state['runs'] else state['runs'][-1],stderr=(base/'supervisor.stderr').read_text()[-4000:])))
'''))


def collect():
    assert not (BASE/'results.tar.gz').exists() and not (BASE/'collected').exists()
    # Transfer only evidence and frozen inventories. Source, archives and package
    # caches are already preserved locally; do not manufacture a remote archive.
    script = PRELUDE+'''
deployment=read(base/'deployment.json');assert not live(deployment)
state_path=base/'campaign/identity.json';state=read(state_path) if state_path.exists() else None
identities=[deployment]
if state is not None:
 assert state['complete'] is True
 for run in state['runs']:
  assert run['complete'] is True
  identities.extend(dict(pid=int(pid),birth=birth) for pid,birth in run['members'].items())
assert not any(live(i) for i in identities)
sys.path.insert(0,str(base/'execution'))
from candidate_protocol import verify
input_error=None
try:verify(base)
except BaseException as error:input_error=repr(error)
paths=[p for directory in ['campaign','execution'] for p in (base/directory).rglob('*') if p.is_file()]
paths.extend(p for p in base.iterdir() if p.is_file() and not p.name.startswith('transfer-'))
if (base/'campaign/built-files.json').exists():
 paths.extend(base/name for name in read(base/'campaign/built-files.json'))
files={p.relative_to(base).as_posix():pin(p) for p in sorted(paths)}
assert 'collection.json' not in files
receipt=dict(terminal=True,identities=identities,input_error=input_error,files=files,
             code=None if state is None else state['code'],payload=pin(base/'payload.json'),execution=pin(base/'execution/execution.json'))
write(base/'collection.json',receipt)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name in [*files,'collection.json']:archive.add(base/name,arcname=name,recursive=False)
'''
    compile(script, 'stream-audio-evidence', 'exec')
    with (BASE/'results.tar.gz').open('xb') as out, (BASE/'collection.stderr').open('x') as err:
        process = subprocess.run(SSH+['python3 -B -'], input=script.encode(), stdout=out, stderr=err,
                                 timeout=600, creationflags=subprocess.CREATE_NO_WINDOW)
    assert process.returncode == 0, ('Collection failed; retain partial archive', process.returncode)
    target = BASE/'collected'; target.mkdir()
    with tarfile.open(BASE/'results.tar.gz') as archive:
        archive.extractall(target, filter='data')
    receipt = read(target/'collection.json'); verified_files(target, receipt['files'])
    assert {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()} == set(receipt['files']) | {'collection.json'}
    assert receipt['payload'] == pin(PREPARED/'payload/payload.json')
    assert receipt['execution'] == pin(BASE/'execution/execution.json')
    transfer = dict(archive=pin(BASE/'results.tar.gz'), receipt=pin(target/'collection.json'),
                    files=len(receipt['files']), terminal=True, code=receipt['code'], input_error=receipt['input_error'])
    write(BASE/'collection-transfer.json', transfer); print(json.dumps(transfer))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'launch', 'observe', 'collect'])
    args = parser.parse_args()
    value = globals()[args.action]()
    if value is not None:
        print(json.dumps(value))
