"""Freeze and launch one collection diagnostic after the completed AMD benchmark."""
from pathlib import Path
import importlib.util,json,subprocess,sys

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('audio_deployment',ROOT/'tests/audio/amd-comparison/deploy.py')
sys.path.insert(0,str(ROOT/'tests/audio/amd-comparison'))
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
ssh=original.ssh;KEY=original.KEY;HOST=original.HOST
from protocol import pin,read,write
BASE=ROOT/'artifacts/whisper-memory-collection-20260920'
REMOTE='/home/vermorel/Onnx/artifacts/whisper-memory-collection-20260920'
PRIOR=ROOT/'artifacts/audio-amd-comparison-v2-20260920/collected'
OLD='/home/vermorel/Onnx/artifacts/audio-amd-comparison-v2-20260920'
BENCH=ROOT/'artifacts/audio-amd-two-family-20260920'
BENCH_REMOTE='/home/vermorel/Onnx/artifacts/audio-amd-two-family-20260920'


def main():
    assert not (BASE/'deployment.json').exists() and not (BASE/'frozen.json').exists()
    prepared=read(BASE/'prepared.json');assert prepared['prepared'] and prepared['calls']==20 and prepared['collect_after_calls']==[8,16,20]
    for name,wanted in prepared['files'].items():assert pin(BASE/name)==wanted,name
    assert read(BENCH/'final-verification.json')['passed'] and read(BENCH/'closed.json')['passed']
    assert pin(BENCH/'closed.json')==read(BENCH/'final-verification.json')['closure']
    folder=Path(__file__).resolve().parent;assert pin(folder/'Program.cs')==prepared['files']['source/Program.cs']
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    uploads={}
    for name in ['WhisperMemoryCollection.dll','WhisperMemoryCollection.deps.json','WhisperMemoryCollection.runtimeconfig.json']:uploads['bin/'+name]=BASE/'bin'/name
    for name in ['supervise.py','memory_protocol.py']:uploads['runtime/'+name]=folder/name
    uploads.update({'source/Program.cs':BASE/'source/Program.cs','source/NpySupport.cs':BASE/'source/NpySupport.cs',
        'source/WhisperMemoryCollection.csproj':BASE/'source/WhisperMemoryCollection.csproj','prepared.json':BASE/'prepared.json',
        'prospective-plan.md':ROOT/'.agent/m4-whisper-memory-collection-20260920.md','prior-baseline-closed.json':BENCH/'closed.json',
        'prior-failure-closed.json':ROOT/'artifacts/audio-amd-comparison-v2-20260920/failure-closed.json'})
    upload_pins={name:pin(p) for name,p in uploads.items()}
    script='''from pathlib import Path
import os,sys,json,hashlib
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
old=Path(%r);base=Path(%r);bench=Path(%r)
def pin(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
assert pin(bench/'collection.json')==%r
state=json.loads((bench/'campaign/identity.json').read_text());assert state['complete'] and state['code']==0
for b in %r:
 try:assert psutil.Process(b['pid']).create_time()!=b['birth']
 except psutil.NoSuchProcess:pass
frozen=json.loads((old/'frozen.json').read_text());assert pin(old/'frozen.json')==%r
for name,wanted in frozen['files'].items():assert pin(old/name)==wanted,name
for name,wanted in frozen['external'].items():assert pin(Path(name))==wanted,name
assert not base.exists();base.mkdir()
for name in frozen['files']:
 if name in ['runtime/supervise.py','prospective-plan.md']:continue
 target=base/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(old/name,target)
(base/'source').mkdir()
print(json.dumps(dict(prior_terminal=True,available=psutil.virtual_memory().available,disk=psutil.disk_usage(str(base)).free)))
'''%(OLD,REMOTE,BENCH_REMOTE,pin(BENCH/'collected/collection.json'),read(BENCH/'closed.json')['births'],pin(PRIOR/'frozen.json'))
    write(BASE/'stage-check.json',json.loads(ssh(script)))
    for name,path in uploads.items():subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',str(path),HOST+':'+REMOTE+'/'+name],check=True)
    script='''from pathlib import Path
import json,hashlib,shutil,subprocess
old=Path(%r);base=Path(%r);bench=Path(%r)
def pin(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
for name,wanted in %r.items():assert pin(base/name)==wanted,name
frozen=json.loads((old/'frozen.json').read_text());frozen.update(source=%r,scope='whisper-memory-collection',calls=20,collect_after_calls=[8,16,20],prior_frozen=pin(old/'frozen.json'))
frozen['external'][str(bench/'collection.json')]=pin(bench/'collection.json')
dotnet=Path(shutil.which('dotnet')).resolve();runtime=dotnet.parent/'shared/Microsoft.NETCore.App/10.0.8'
assert runtime.is_dir()
managed={str(dotnet):pin(dotnet)}
for root in [runtime,dotnet.parent/'host/fxr']:
 for p in root.rglob('*'):
  if p.is_file():managed[str(p)]=pin(p)
frozen['external'].update(managed)
frozen['managed_runtime']=dict(host=str(dotnet),files=managed,list_runtimes=subprocess.check_output([str(dotnet),'--list-runtimes'],text=True))
frozen['files']={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
with (base/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2)
print(json.dumps(dict(frozen=pin(base/'frozen.json'),files=len(frozen['files']),external=len(frozen['external']))))
'''%(OLD,REMOTE,BENCH_REMOTE,upload_pins,source)
    frozen=json.loads(ssh(script));write(BASE/'freeze-receipt.json',frozen)
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',HOST+':'+REMOTE+'/frozen.json',str(BASE/'frozen.json')],check=True)
    assert pin(BASE/'frozen.json')==frozen['frozen']
    script='''from pathlib import Path
import os,sys,json,subprocess
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path(%r);assert not (base/'campaign').exists() and not (base/'deployment.json').exists()
frozen=json.loads((base/'frozen.json').read_text())
env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
env.update(PYTHONPATH=os.pathsep.join(frozen['python_paths']),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
env.update({k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
with (base/'supervisor.stdout').open('x') as out,(base/'supervisor.stderr').open('x') as err:
 p=subprocess.Popen(['python3','-B',str(base/'runtime/supervise.py'),str(base)],cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
value=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),frozen=%r)
with (base/'deployment.json').open('x') as f:json.dump(value,f,indent=2)
print(json.dumps(value))
'''%(REMOTE,frozen['frozen'])
    deployed=json.loads(ssh(script));write(BASE/'deployment.json',deployed);print(json.dumps(deployed))


if __name__=='__main__':main()
