"""Freeze an isolated private prototype, then launch the finite conditional schedule once."""
from pathlib import Path
import importlib.util,json,subprocess,sys,tarfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT/'tests/audio/amd-comparison'))
spec=importlib.util.spec_from_file_location('original_audio_deployment',ROOT/'tests/audio/amd-comparison/deploy.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
ssh=old.ssh;KEY=old.KEY;HOST=old.HOST
from protocol import pin,read,write
BASE=ROOT/'artifacts/whisper-buffer-reuse-20260920'
REMOTE='/home/vermorel/Onnx/artifacts/whisper-buffer-reuse-20260920'
PREVIOUS=ROOT/'artifacts/whisper-memory-collection-20260920'
PREVIOUS_REMOTE='/home/vermorel/Onnx/artifacts/whisper-memory-collection-20260920'


def main():
    assert not (BASE/'deployment.json').exists() and not (BASE/'frozen.json').exists()
    prepared=read(BASE/'prepared.json');built=read(BASE/'built.json')
    assert prepared['prepared'] and not prepared['explicit_gc'] and built['tests_passed']
    for receipt in [prepared,built]:
        for name,wanted in receipt['files'].items():assert pin(BASE/name)==wanted,name
    prior=read(PREVIOUS/'closed.json');assert prior['passed'] and read(PREVIOUS/'final-verification.json')['passed']
    for name,wanted in prior['files'].items():assert pin(ROOT/name)==wanted,name
    source=read(BASE/'source.json');supplement=read(BASE/'cli-source.json')
    for receipt in [source,supplement]:
        for name,wanted in receipt['files'].items():assert pin(BASE/'source'/name)==wanted,name
    folder=Path(__file__).parent
    uploads={'bin/'+p.name:p for p in (BASE/'bin').iterdir() if p.suffix in ['.dll','.json']}
    uploads.update({'runtime/'+n:folder/n for n in ['supervise.py','reuse_protocol.py']})
    uploads['runtime/memory_protocol.py']=ROOT/'tests/whisper/memory-collection/memory_protocol.py'
    for name in ['source.json','cli-source.json','original-source.tar','cli-source.tar','built.json','prepared.json','build.log','tests.log','cli-build.log','complete-tests.log','consumer-build.log','prospective-plan.md','ComputationalGraph.patch','WhisperTranscriber.patch','ReleasedBufferCacheTests.patch']:
        uploads['evidence/'+name]=BASE/name
    for path in ['test-results/backend.trx','complete-test-results/backend.trx']:
        uploads['evidence/'+path]=BASE/path
    for name in ['Program.cs','NpySupport.cs','WhisperBufferReuse.csproj']:uploads['consumer/'+name]=BASE/'consumer'/name
    for name in source['changes']:uploads['prototype-source/'+name]=BASE/'source'/name
    for name in ['prepare.py','complete_build.py','prepare_consumer.py','deploy.py','reuse_protocol.py','supervise.py']:
        uploads['tools/'+name]=folder/name
    uploads['prospective-plan.md']=ROOT/'.agent/m4-whisper-buffer-reuse-prototype-20260920.md'
    uploads['prior-closed.json']=PREVIOUS/'closed.json'
    manifest=read(PREVIOUS/'collected/manifests/whisper.json')
    manifest['core_sha256']=pin(BASE/'bin/Lokad.Onnx.dll')['sha256'];manifest['data_sha256']=pin(BASE/'bin/Lokad.Onnx.Data.dll')['sha256']
    write(BASE/'prototype-manifest.json',manifest);uploads['manifests/whisper.json']=BASE/'prototype-manifest.json'
    prefix=ROOT/'artifacts/audio-amd-comparison-v2-20260920/collected/campaign/conformance/05-whisper-managed/worker'
    for index in range(16):uploads[f'original-prefix/{index:03}.json']=prefix/f'{index:03}.json'
    archive=BASE/'deployment.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name,path in uploads.items():tar.add(path,arcname=name,recursive=False)
    upload_pins={name:pin(p) for name,p in uploads.items()}
    prelude='''from pathlib import Path
import os,sys,json,hashlib,tarfile,subprocess
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path(%r);prior=Path(%r)
def pin(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)
'''%(REMOTE,PREVIOUS_REMOTE)
    script=prelude+'''assert not base.exists()
assert pin(prior/'collection.json')==%r
for b in %r:
 try:assert psutil.Process(b['pid']).create_time()!=b['birth']
 except psutil.NoSuchProcess:pass
frozen=json.loads((prior/'frozen.json').read_text());assert pin(prior/'frozen.json')==%r
for name,wanted in frozen['files'].items():assert pin(prior/name)==wanted,name
for name,wanted in frozen['external'].items():assert pin(Path(name))==wanted,name
base.mkdir()
for name in frozen['files']:
 if name.startswith('assets/') or name in ['runtime/protocol.py','runtime/campaign_processes.py']:
  target=base/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(prior/name,target)
print(json.dumps(dict(prior_terminal=True,available=psutil.virtual_memory().available,disk=psutil.disk_usage(str(base)).free)))
'''%(pin(PREVIOUS/'collected/collection.json'),prior['births'],pin(PREVIOUS/'frozen.json'))
    write(BASE/'stage-check.json',json.loads(ssh(script)))
    ram='/dev/shm/whisper-buffer-reuse-20260920-deployment.tar.gz'
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',str(archive),HOST+':'+ram],check=True)
    script=prelude+'''archive=Path(%r);assert pin(archive)==%r
with tarfile.open(archive) as tar:
 for item in tar.getmembers():assert not (base/item.name).exists(),item.name
 tar.extractall(base,filter='data')
for name,wanted in %r.items():assert pin(base/name)==wanted,name
frozen=json.loads((prior/'frozen.json').read_text())
for k in ['calls','collect_after_calls']:frozen.pop(k,None)
frozen.update(source=%r,product_source=%r,scope='whisper-buffer-reuse',conformance_calls=20,endurance_calls=80,explicit_gc=False,prior_frozen=pin(prior/'frozen.json'))
frozen['files']={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
write(base/'frozen.json',frozen)
print(json.dumps(dict(frozen=pin(base/'frozen.json'),files=len(frozen['files']),external=len(frozen['external']))))
'''%(ram,pin(archive),upload_pins,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dict(revision=source['revision'],prototype_source=pin(BASE/'source.json'),cli_source=pin(BASE/'cli-source.json')))
    frozen=json.loads(ssh(script));write(BASE/'freeze-receipt.json',frozen)
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',HOST+':'+REMOTE+'/frozen.json',str(BASE/'frozen.json')],check=True)
    assert pin(BASE/'frozen.json')==frozen['frozen']
    script=prelude+'''assert not (base/'campaign').exists() and not (base/'deployment.json').exists()
frozen=json.loads((base/'frozen.json').read_text())
env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
env.update(PYTHONPATH=os.pathsep.join(frozen['python_paths']),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
env.update({k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
with (base/'supervisor.stdout').open('x') as out,(base/'supervisor.stderr').open('x') as err:
 p=subprocess.Popen(['python3','-B',str(base/'runtime/supervise.py'),str(base)],cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
value=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),frozen=pin(base/'frozen.json'))
write(base/'deployment.json',value);print(json.dumps(value))
'''
    value=json.loads(ssh(script));write(BASE/'deployment.json',value);print(json.dumps(value))


if __name__=='__main__':main()
