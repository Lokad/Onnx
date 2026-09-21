"""Freeze the tested private sharing candidate and launch its finite AMD protocol once."""
from pathlib import Path
import importlib.util,json,subprocess,sys,tarfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT/'tests/audio/amd-comparison'))
spec=importlib.util.spec_from_file_location('original_audio_deployment',ROOT/'tests/audio/amd-comparison/deploy.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
ssh=old.ssh;KEY=old.KEY;HOST=old.HOST
from protocol import pin,read,write
BASE=ROOT/'artifacts/whisper-weight-sharing-v2-20260920'
REMOTE='/home/vermorel/Onnx/artifacts/whisper-weight-sharing-v2-20260920'
PRODUCT=ROOT/'artifacts/whisper-weight-sharing-20260920'
PREVIOUS=PRODUCT
DIAGNOSTIC=ROOT/'artifacts/whisper-weight-metadata-20260920'
PREVIOUS_REMOTE='/home/vermorel/Onnx/artifacts/whisper-weight-sharing-20260920'

def checked_ssh(script):
    compile(script,'frozen-deployment-script','exec')
    return ssh(script)


def main():
    assert not (BASE/'deployment.json').exists() and not (BASE/'frozen.json').exists()
    prepared=read(BASE/'prepared.json');built=read(PRODUCT/'built.json')
    assert prepared['product_built']==pin(PRODUCT/'built.json') and prepared['product_source']==pin(PRODUCT/'source.json')
    assert prepared['prepared'] and not prepared['explicit_gc'] and built['tests_passed']
    for root,receipt in [(BASE,prepared),(PRODUCT,built)]:
        for name,wanted in receipt['files'].items():assert pin(root/name)==wanted,name
    prior=read(PREVIOUS/'failure-closed.json');local=read(PRODUCT/'local-closed.json');diagnostic=read(DIAGNOSTIC/'closed.json')
    assert prior['closure_passed'] and not prior['campaign_passed'] and local['passed'] and diagnostic['passed']
    assert read(BASE/'preparation-verification.json')['passed'] and read(PRODUCT/'local-final-verification.json')['passed']
    for receipt in [prior,local,diagnostic]:
        for name,wanted in receipt['files'].items():assert pin(ROOT/name)==wanted,name
    source=read(PRODUCT/'source.json')
    for name,wanted in source['files'].items():assert pin(PRODUCT/'source'/name)==wanted,name
    folder=Path(__file__).parent
    uploads={'bin/'+p.name:p for p in (BASE/'bin').iterdir() if p.suffix in ['.dll','.json']}
    uploads.update({'runtime/'+n:folder/n for n in ['supervise.py','sharing_protocol.py','weight_transition.py']})
    uploads['runtime/reuse_protocol.py']=ROOT/'tests/whisper/buffer-reuse/reuse_protocol.py'
    uploads['runtime/memory_protocol.py']=ROOT/'tests/whisper/memory-collection/memory_protocol.py'
    for name in ['source.json','built.json','cli-build.log','test-build.log','tests.log','WhisperTranscriber.patch',
                 'local-closed.json','local-final-verification.json','weight-census.json','test-results/backend.trx']:
        uploads['evidence/'+name]=PRODUCT/name
    for name in ['prepared.json','consumer-build.log','prospective-plan.md','transition-tests.json','preparation-verification.json']:
        uploads['evidence/'+name]=BASE/name
    uploads['evidence/metadata-closed.json']=DIAGNOSTIC/'closed.json'
    for name in ['Program.cs','NpySupport.cs','WhisperWeightSharingV2.csproj','WeightSnapshot.cs']:uploads['consumer/'+name]=BASE/'consumer'/name
    for name in source['files']:
        if source['files'][name]!=source['inherited'].get(name):uploads['prototype-source/'+name]=PRODUCT/'source'/name
    for name in ['prepare.py','check_transition.py','deploy.py','sharing_protocol.py','weight_transition.py','WeightSnapshot.cs','supervise.py']:
        uploads['tools/'+name]=folder/name
    uploads['prospective-plan.md']=ROOT/'.agent/m4-whisper-sharing-v2-20260920.md'
    uploads['prior-closed.json']=PREVIOUS/'failure-closed.json'
    manifest=read(PREVIOUS/'collected/manifests/whisper.json')
    manifest['core_sha256']=pin(BASE/'bin/Lokad.Onnx.dll')['sha256'];manifest['data_sha256']=pin(BASE/'bin/Lokad.Onnx.Data.dll')['sha256']
    write(BASE/'prototype-manifest.json',manifest);uploads['manifests/whisper.json']=BASE/'prototype-manifest.json'
    previous_frozen=read(PREVIOUS/'frozen.json')
    links={}
    for name,wanted in previous_frozen['files'].items():
        if name.startswith(('assets/','original-prefix/')) or name in ['runtime/protocol.py','runtime/campaign_processes.py']:
            links[name]=wanted
    for name,path in list(uploads.items()):
        if previous_frozen['files'].get(name)==pin(path):links[name]=pin(path);del uploads[name]
    assert not set(links)&set(uploads)
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
for name,wanted in %r.items():
 assert pin(prior/name)==wanted,name
 target=base/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(prior/name,target)
print(json.dumps(dict(prior_terminal=True,available=psutil.virtual_memory().available,disk=psutil.disk_usage(str(base)).free)))
'''%(pin(PREVIOUS/'collected/collection.json'),prior['births']+next(h['births'] for h in diagnostic['births'] if h['host']=='amd'),pin(PREVIOUS/'frozen.json'),links)
    write(BASE/'stage-check.json',json.loads(checked_ssh(script)))
    ram='/dev/shm/whisper-weight-sharing-v2-20260920-deployment.tar.gz'
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',str(archive),HOST+':'+ram],check=True)
    script=prelude+'''archive=Path(%r);assert pin(archive)==%r
with tarfile.open(archive) as tar:
 for item in tar.getmembers():assert not (base/item.name).exists(),item.name
 tar.extractall(base,filter='data')
for name,wanted in %r.items():assert pin(base/name)==wanted,name
frozen=json.loads((prior/'frozen.json').read_text())
frozen.update(source=%r,product_source=%r,scope='whisper-weight-sharing-v2',conformance_calls=20,endurance_calls=80,explicit_gc=False,prior_frozen=pin(prior/'frozen.json'),logical_shared_bytes=635187200)
frozen['files']={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
write(base/'frozen.json',frozen)
print(json.dumps(dict(frozen=pin(base/'frozen.json'),files=len(frozen['files']),external=len(frozen['external']))))
'''%(ram,pin(archive),upload_pins,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dict(parent_revision=source['parent_revision'],prototype_source=pin(PRODUCT/'source.json'),local_closure=pin(PRODUCT/'local-closed.json')))
    frozen=json.loads(checked_ssh(script));write(BASE/'freeze-receipt.json',frozen)
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
    value=json.loads(checked_ssh(script));write(BASE/'deployment.json',value);print(json.dumps(value))


if __name__=='__main__':main()
