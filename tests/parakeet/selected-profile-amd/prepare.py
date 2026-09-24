"""Freeze unchanged product, current full corpus and the diagnostic-only consumer."""
import ast,hashlib,shutil,tarfile
import numpy as np
from common import *

PREVIOUS=ROOT/'artifacts/pyannote-winograd-profile-amd-20260923'
BUILD=ROOT/'artifacts/parakeet-selected-profile-build-amd-20260924'
BASELINE=ROOT/'artifacts/parakeet-wide-entry-first-use-app-amd-v2-20260923'
ASSETS=ROOT/'artifacts/pyannote-blocked-spatial-app-amd-payload-20260922/payload/assets'

def previous_closed():
    for folder,digest,root_relative in [
        (PREVIOUS,'8d960999ebda0a6f82f3548c89d4b01ec302ce2802bb9454b3fa371f2a5de126',True),
        (BUILD,'b5364b30098c9fa7b375d4d8b404292d7085a315336100e725b451d395cb09b2',False),
        (BASELINE,'ed90f6bab75fc1b6aae2b322fa2d2dec36458ba83364f62ea7588b1df841c0de',False)]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin((ROOT if root_relative else folder)/name)==wanted,name
    assert read(BUILD/'analysis.json')['inventory']['passed']
    assert read(BUILD/'analysis.json')['consumer']==dict(bytes=49664,sha256='a196f652296db675c47947e2de969711b19356ddf505687c9df2b0ceb48fbbd5')
    for name,wanted in read(BUILD/'bundle/evidence/selected-source.json')['source'].items():
        assert pin(ROOT/name)==wanted,name
    for name in ['remote.py','remote_export.py','export_audit.py','selected_stacks.py']:
        assert (TOOLS/name).read_text()==(ROOT/'tests/parakeet/current-profile-amd'/name).read_text(),name

def main():
    assert not BASE.exists();previous_closed();BASE.mkdir();payload=BASE/'payload';payload.mkdir();files={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);files[rel(source)]=pin(source)
    old=read(PREVIOUS/'payload/payload.json')
    for name,wanted in old['files'].items():
        if name.startswith('tracer/') or name in ['tools/public_audit.py','tools/stacks.py','tools/stacks_v2.py']:
            source=PREVIOUS/'payload'/name;assert pin(source)==wanted;copy(source,payload/name)
    source=PREVIOUS/'payload/tools/pair.py.txt';before=source.read_text(encoding='utf8')
    assert before.count("sample['rss'] < 8*1024**3")==1
    after=before.replace("sample['rss'] < 8*1024**3","sample['rss'] < 12*1024**3")
    (payload/'tools/pair.py.txt').write_text(after,encoding='utf8');files[rel(source)]=pin(source)
    for source in (BUILD/'collected/runtime').iterdir():
        if source.is_file():copy(source,payload/'runtime'/source.name)
    built=read(BUILD/'analysis.json')
    assert pin(payload/'runtime/Lokad.Onnx.dll')['sha256']==CORE
    assert pin(payload/'runtime/Lokad.Onnx.Data.dll')['sha256']==DATA
    assert pin(payload/'runtime/SampledAudio.dll')==built['consumer']
    copy(BASELINE/'collected/manifests/candidate-parakeet.json',payload/'manifest.json')
    manifest=read(payload/'manifest.json')
    assert manifest['family']=='parakeet' and len(manifest['cases'])==20
    assert manifest['core_sha256']==CORE and manifest['data_sha256']==DATA
    assert manifest['warmup_passes']==1 and manifest['measured_passes']==3
    assert sum(c['samples'] for c in manifest['cases'])==3412240
    raw={}
    for item in [manifest['reference'],manifest['upstream'],*[c['pcm'] for c in manifest['cases']]]:
        source=ASSETS/item['path'];assert pin(source)=={k:item[k] for k in ['bytes','sha256']};copy(source,payload/item['path'])
    for c in manifest['cases']:
        pcm=np.load(payload/c['pcm']['path'],allow_pickle=False)
        assert pcm.dtype==np.float32 and pcm.shape==(c['samples'],) and np.isfinite(pcm).all()
        raw[c['name']]=hashlib.sha256(pcm.tobytes()).hexdigest()
    expected=BASELINE/'collected/timing-01-candidate/output/result.json';copy(expected,payload/'prior-amd-result.json')
    public=module('parakeet_profile_public_audit',payload/'tools/public_audit.py')
    checked=json.loads(json.dumps(manifest))
    for case in checked['cases']:case['raw_sha256']=raw[case['name']]
    public.validate_worker(read(expected),checked,'timing');assert len(read(expected)['records'])==80
    for name in ['remote.py','remote_export.py']:copy(TOOLS/name,payload/'tools'/name)
    for source in [BUILD/'closed.json',BUILD/'analysis.json',BASELINE/'closed.json']:
        copy(source,payload/'evidence'/(( 'baseline-' if source.parent==BASELINE else 'build-')+source.name))
    copy(TOOLS/'README.md',payload/'prospective-plan.md')
    external={name:wanted for name,wanted in old['external'].items() if '/.dotnet/' in name or '/psutil/' in name or name=='/usr/bin/python3'}
    baseline=read(BASELINE/'payload.json')
    for item in manifest['models'].values():
        name=item['path'];wanted={k:item[k] for k in ['bytes','sha256']}
        assert baseline['external'][name]==wanted;external[name]=wanted
    specification=dict(core=CORE,data=DATA,consumer=built['consumer'],raw_pcm=raw,external=external,
        previous_identities=read(BUILD/'collected/collection.json')['identities']+read(BASELINE/'collected/collection.json')['identities'],
        boot_time=1789634288.0,limits=dict(seconds=900,preflight_available=12*1024**3,preflight_tmpfs=3*1024**3,
            rss=12*1024**3,available=1024**3,tmpfs=1024**3,output=1024**3,artifacts=2*1024**3),
        jobs=['control','sampled-a','sampled-b'],files={p.relative_to(payload).as_posix():pin(p) for p in payload.rglob('*') if p.is_file()})
    save(payload/'payload.json',specification)
    for p in [*TOOLS.iterdir(),MONITOR,BUILD/'closed.json',BASELINE/'closed.json',ROOT/'artifacts/pyannote-selected-profile-amd-20260922/payload/tools/stacks.py']:
        if p.is_file():
            files[rel(p)]=pin(p)
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
    for p in payload.rglob('*'):
        if p.is_file():files[rel(p)]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(payload.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(payload).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=files,archive=pin(BASE/'payload.tar.gz'),payload=pin(payload/'payload.json'),build=pin(BUILD/'closed.json')))
    print(json.dumps(dict(payload=pin(payload/'payload.json'),archive=pin(BASE/'payload.tar.gz'),files=len(specification['files']),external=len(external))))

if __name__=='__main__':main()
