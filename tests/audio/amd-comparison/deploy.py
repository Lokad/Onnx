"""Stage, freeze, then explicitly launch one bounded AMD audio campaign."""
from pathlib import Path
import argparse, json, subprocess, tarfile
from protocol import pin,read,write

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/audio-amd-comparison-v2-20260920'
DEPS=ROOT/'artifacts/audio-amd-dependencies-20260920'
REMOTE='/home/vermorel/Onnx/artifacts/audio-amd-comparison-v2-20260920'
DEPENDENCIES='/home/vermorel/Onnx/artifacts/audio-amd-native-20260920'
KEY='C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem'
HOST='vermorel@74.178.91.76'
RAM='/dev/shm/audio-amd-staging-20260920'


def ssh(script):
    return subprocess.check_output(['ssh','-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'python3 -B -'],input=script,text=True,encoding='utf-8')


def copy(source,destination):
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',str(source),HOST+':'+destination],check=True)


PRELUDE='''from pathlib import Path
import hashlib,json,os,subprocess,sys,shutil,time,tarfile
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path(%r);deps=Path(%r);ram=Path(%r)
def pin(p):
 with Path(p).open('rb') as f:return dict(bytes=Path(p).stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2)
def absent(pid,birth):
 try:return psutil.Process(pid).create_time()!=birth
 except psutil.NoSuchProcess:return True
meeting=Path('/home/vermorel/Onnx/artifacts/pyannote-frame-meetings-20260920')
state=json.loads((meeting/'process-managed-run/identity.json').read_text())
assert state['complete'] and state['code']==0
for b in [state['supervisor'],state['child']]:assert absent(b['pid'],b['birth'])
for pid,birth in state['members'].items():assert absent(int(pid),birth)
env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
env.update(PYTHONPATH=os.pathsep.join([str(deps/'python'),'/home/vermorel/Onnx/artifacts/asr-native-linux-20260920/python','/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python']),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
env.update({k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
os.sched_setaffinity(0,{0})
'''%(REMOTE,DEPENDENCIES,RAM)


def stage():
    # The full connected application audit, not merely process exit, precedes this lane.
    meeting=ROOT/'artifacts/pyannote-frame-meetings-20260920'
    audit=read(meeting/'audit.json');assert audit['execution_passed'] and audit['public_passed'] and audit['accuracy_unchanged']
    assert (meeting/'closed.json').exists()
    transfer=read(BASE/'transfer.json');assert pin(BASE/'payload.tar.gz')==transfer['archive']
    candidates=read(DEPS/'cleanup-candidates.json')['candidates']
    # The just-collected replay also has a complete verified local archive.
    replay_archive=meeting/'results.tar.gz'
    assert pin(replay_archive)==read(meeting/'collection-transfer.json')['archive']
    candidates.append(dict(remote='/home/vermorel/Onnx/artifacts/pyannote-frame-meetings-20260920-results.tar.gz',backup=str(replay_archive),**pin(replay_archive)))
    for item in candidates:assert pin(item['backup'])=={k:item[k] for k in ['bytes','sha256']}
    wheels=read(DEPS/'manifest.json')
    for name,item in wheels['files'].items():assert pin(DEPS/name)=={k:item[k] for k in ['bytes','sha256']}
    archive=DEPS/'wheels.tar'
    if not archive.exists():
        with tarfile.open(archive,'w') as tar:
            for name in ['manifest.json',*wheels['files']]:tar.add(DEPS/name,arcname=name,recursive=False)
    script=PRELUDE+'''
assert not base.exists() and not deps.exists() and not ram.exists()
artifact_root=base.parent.resolve();removed=[];before=shutil.disk_usage(artifact_root).free
for item in %r:
 path=Path(item['remote']);assert path.resolve().parent==artifact_root and path.is_file() and not path.is_symlink()
 assert pin(path)=={k:item[k] for k in ['bytes','sha256']},str(path)
for item in %r:
 Path(item['remote']).unlink();removed.append(item)
base.mkdir();deps.mkdir();ram.mkdir()
receipt=dict(before=before,after=shutil.disk_usage(artifact_root).free,removed=removed,meeting_terminal=True)
write(base/'archive-cleanup.json',receipt);print(json.dumps(receipt))
'''%(candidates,candidates)
    result=json.loads(ssh(script));write(BASE/'cleanup.json',result)
    copy(BASE/'payload.tar.gz',RAM+'/payload.tar.gz');copy(archive,RAM+'/wheels.tar')
    script=PRELUDE+'''
assert pin(ram/'payload.tar.gz')==%r and pin(ram/'wheels.tar')==%r
with tarfile.open(ram/'payload.tar.gz') as tar:tar.extractall(base,filter='data')
with tarfile.open(ram/'wheels.tar') as tar:tar.extractall(ram/'wheels',filter='data')
prepared=json.loads((base/'preparation.json').read_text())
for name,wanted in prepared['files'].items():assert pin(base/name)==wanted,name
for name,wanted in prepared['remote_models'].items():assert pin(Path(name))==wanted,name
wheels=json.loads((ram/'wheels/manifest.json').read_text())
for name,item in wheels['files'].items():assert pin(ram/'wheels'/name)=={k:item[k] for k in ['bytes','sha256']}
assert psutil.virtual_memory().available>13*1024**3
assert shutil.disk_usage(base).free>sum(i['uncompressed'] for i in wheels['files'].values())+64*1024**2
(ram/'tmp').mkdir();env['TMPDIR']=str(ram/'tmp')
result=subprocess.run(['python3','-B','-m','pip','install','--disable-pip-version-check','--no-index','--no-deps','--no-compile','--no-cache-dir','--target',str(deps/'python')]+[str(ram/'wheels'/name) for name in wheels['files']],env=env,capture_output=True,text=True,timeout=300)
receipt=dict(code=result.returncode,stdout=result.stdout,stderr=result.stderr,wheels=wheels,free_disk=shutil.disk_usage(base).free,meeting_terminal=True)
write(base/'dependency-install.json',receipt);assert result.returncode==0,result.stderr
# Delete only the verified staging files; their complete originals remain local.
for name in wheels['files']:(ram/'wheels'/name).unlink()
(ram/'wheels/manifest.json').unlink();(ram/'wheels').rmdir()
(ram/'wheels.tar').unlink();(ram/'payload.tar.gz').unlink();(ram/'tmp').rmdir();ram.rmdir()
print(json.dumps(receipt))
'''%(transfer['archive'],pin(archive))
    result=json.loads(ssh(script));write(BASE/'installation.json',result);print(json.dumps(result))


def freeze():
    script=PRELUDE+'''
assert not (base/'frozen.json').exists()
result=subprocess.run(['python3','-B',str(base/'runtime/freeze_linux.py'),str(base)],env=env,cwd=base,capture_output=True,text=True,timeout=600)
receipt=dict(code=result.returncode,stdout=result.stdout,stderr=result.stderr)
write(base/'freeze-attempt.json',receipt);assert result.returncode==0,receipt
receipt['frozen']=pin(base/'frozen.json');print(json.dumps(receipt))
'''
    result=json.loads(ssh(script));write(BASE/'freeze.json',result)
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',HOST+':'+REMOTE+'/frozen.json',str(BASE/'frozen.json')],check=True)
    assert pin(BASE/'frozen.json')==result['frozen'];print(json.dumps(result))


def launch():
    wanted=read(BASE/'freeze.json')['frozen'];assert pin(BASE/'frozen.json')==wanted
    script=PRELUDE+'''
assert pin(base/'frozen.json')==%r
assert not (base/'campaign').exists() and not (base/'deployment.json').exists()
assert psutil.virtual_memory().available>13*1024**3 and shutil.disk_usage(base).free>64*1024**2
with (base/'supervisor.stdout').open('x') as out,(base/'supervisor.stderr').open('x') as err:
 process=subprocess.Popen(['python3','-B',str(base/'runtime/supervise.py'),str(base)],env=env,cwd=base,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 identity=dict(pid=process.pid,birth=psutil.Process(process.pid).create_time(),frozen=pin(base/'frozen.json'))
write(base/'deployment.json',identity);print(json.dumps(identity))
'''%wanted
    result=json.loads(ssh(script));write(BASE/'deployment.json',result);print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['stage','freeze','launch']);args=parser.parse_args()
    globals()[args.phase]()
