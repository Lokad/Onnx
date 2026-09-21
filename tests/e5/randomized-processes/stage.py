"""Bind and launch A/A once, only after the existing Whisper campaign is closed."""
import json
import subprocess

from contract import pin, read, write
from remote import ROOT, BASE, REMOTE, WHISPER, MONITOR, PRELUDE, SITE, ssh, copy_to, copy_from, sys


def main():
    assert not (BASE/'stage.json').exists() and not (BASE/'deployment-aa.json').exists()
    monitor = read(MONITOR/'state.json')
    assert monitor['complete'] is True and monitor['code'] == 0, 'Existing Whisper completion monitor has not finished successfully'
    assert read(WHISPER/'final-verification.json')['passed'] is True
    sys.path.append(str(SITE)); import psutil
    try:
        assert psutil.Process(monitor['supervisor']['pid']).create_time() != monitor['supervisor']['birth']
    except psutil.NoSuchProcess:
        pass
    prepared = read(BASE/'prepared.json'); assert prepared['passed'] is True
    assert pin(BASE/'full-template.json') == prepared['template']
    assert pin(BASE/'payload.tar.gz') == prepared['archive']
    meta = read(BASE/'full-template.json')
    for name, identity in meta['files'].items():
        assert pin(BASE/'payload'/name) == identity
    predecessor = WHISPER/'collected/campaign/identity.json'
    assert predecessor.exists()
    state = read(predecessor); assert state['complete'] is True and state['code'] == 0 and len(state['runs']) == 5
    births = [state['supervisor']]+[dict(pid=int(pid), birth=birth) for run in state['runs'] for pid, birth in run['members'].items()]
    meta['predecessor_births'] = births
    meta['predecessor_identity'] = dict(path='/home/vermorel/Onnx/artifacts/audio-whisper-amd-20260921/campaign/identity.json', **pin(predecessor))
    script = PRELUDE+'''
os.sched_setaffinity(0,{0});terminal(%r)
assert not base.exists()
assert shutil.disk_usage('/dev/shm').free>=4*1024**3 and psutil.virtual_memory().available>=10*1024**3
assert pin(Path(%r))==%r
meta=%r
for item in [meta['model'],meta['native'],meta['dotnet']]+meta['runtime_files']:
 assert pin(item['path'])=={k:item[k] for k in ['bytes','sha256']},item['path']
for name,expected in meta['python_files'].items():assert pin(name)==expected,name
assert pin(Path(sys.executable).resolve())==meta['interpreter']
base.mkdir();print(json.dumps(dict(created=True,available=psutil.virtual_memory().available,free=shutil.disk_usage(base).free,python=str(Path(sys.executable).resolve()))))
''' % (births, meta['predecessor_identity']['path'], pin(predecessor), meta)
    staged = json.loads(ssh(script)); write(BASE/'stage.json', staged)
    copy_to(BASE/'payload.tar.gz', REMOTE+'/payload.tar.gz')
    meta['python_executable'] = staged['python']
    script = PRELUDE+'''
import tarfile
os.sched_setaffinity(0,{0});terminal(%r)
assert pin(base/'payload.tar.gz')==%r
meta=%r
with tarfile.open(base/'payload.tar.gz','r:gz') as archive:
 members=archive.getmembers()
 assert len(members)==len(meta['files']) and {m.name for m in members}==set(meta['files'])
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
 archive.extractall(base,filter='data')
for name,expected in meta['files'].items():assert pin(base/name)==expected,name
assert {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}==set(meta['files'])|{'payload.tar.gz'}
assert meta['predecessor_births'] and not (base/'frozen.json').exists()
write(base/'frozen.json',meta)
print(json.dumps(dict(frozen=pin(base/'frozen.json'),files=len(meta['files']),free=shutil.disk_usage(base).free)))
''' % (births, prepared['archive'], meta)
    frozen = json.loads(ssh(script)); write(BASE/'freeze-receipt.json', frozen)
    # Fetch actual bytes: Windows text-mode JSON serialization uses different
    # newlines and must not stand in for the frozen Linux record.
    copy_from(REMOTE+'/frozen.json', BASE/'frozen.json')
    assert pin(BASE/'frozen.json') == frozen['frozen'] and read(BASE/'frozen.json') == meta
    script = PRELUDE+'''
os.sched_setaffinity(0,{0});terminal(%r)
assert pin(base/'frozen.json')==%r
assert not (base/'deployment-aa.json').exists() and not (base/'result-aa').exists()
frozen=read(base/'frozen.json')
env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
env.update(PYTHONPATH=os.pathsep.join(frozen['python_paths']),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
with (base/'supervisor-aa.stdout').open('x') as out,(base/'supervisor-aa.stderr').open('x') as err:
 p=subprocess.Popen([frozen['python_executable'],'-B',str(base/'tests/e5/randomized-processes/run.py'),'--payload',str(base),'--phase','aa'],cwd=base,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)
deployment=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),frozen=pin(base/'frozen.json'))
write(base/'deployment-aa.json',deployment);print(json.dumps(deployment))
''' % (births, frozen['frozen'])
    deployment = json.loads(ssh(script)); write(BASE/'deployment-aa.json', deployment)
    print(json.dumps(deployment))


if __name__ == '__main__':
    main()
