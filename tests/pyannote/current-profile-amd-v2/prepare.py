"""Bind the unchanged profile protocol to the exact selected M22 consumer and outputs."""
import ast
import shutil
import tarfile
from common import *

PREVIOUS=ROOT/'artifacts/pyannote-prepared-profile-amd-v2-20260922'
BUILD=ROOT/'artifacts/pyannote-current-profile-build-amd-v2-20260922'
APP=ROOT/'artifacts/pyannote-lstm-input-app-amd-20260922'


def previous_closed():
    failure=ROOT/'artifacts/pyannote-current-profile-amd-20260922/closed.json'
    assert pin(failure)=={'bytes': 25722, 'sha256': '28f8e0d223e69f0c4ab2ade81719e4f1a68714a84725d7bd27e14d2be2dea7ce'}
    for name,wanted in read(failure)['files'].items():assert pin(ROOT/name)==wanted,name
    for folder,digest,root_relative in [
        (PREVIOUS,'87446536bf2ca97f398ec2407c57269e9bc72caf883aaff2a573f520a95a99a4',True),
        (BUILD,'f2b4960a2ffd29987455a93a3db95ca2a7b3b1fba19005dced5afbe8e9cba6c3',False),
        (APP,'73a4897a4db8e1bd729cb3c5486bcb11814d4ff669c9b9a472003572b08c64d0',False)]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin((ROOT if root_relative else folder)/name)==wanted,name


def main():
    assert not BASE.exists();previous_closed()
    BASE.mkdir();payload=BASE/'payload';payload.mkdir();files={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);files[rel(source)]=pin(source)
    original=read(PREVIOUS/'payload/payload.json')
    for name,wanted in original['files'].items():
        source=PREVIOUS/'payload'/name;assert pin(source)==wanted,name;copy(source,payload/name)
    assert pin(TOOLS/'remote.py')==pin(payload/'tools/remote.py')
    built=read(BUILD/'analysis.json')
    for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']:
        source=BUILD/'collected/runtime'/name;assert pin(source)==built['product'][name]
        copy(source,payload/'runtime'/name)
    assert built['product']['Lokad.Onnx.dll']['sha256']==CORE and built['product']['Lokad.Onnx.Data.dll']['sha256']==DATA
    for suffix in ['dll','deps.json','runtimeconfig.json']:copy(BUILD/'collected/runtime'/('SampledAudio.'+suffix),payload/'runtime'/('SampledAudio.'+suffix))
    assert pin(payload/'runtime/SampledAudio.dll')==built['consumer']
    current=APP/'collected/manifests/candidate-pyannote.json';manifest=read(current);old=read(payload/'manifest.json')
    assert manifest['core_sha256']==CORE and manifest['data_sha256']==DATA
    metadata={'core_sha256','data_sha256','product_source'}
    assert {k:v for k,v in old.items() if k not in metadata}=={k:v for k,v in manifest.items() if k not in metadata}
    copy(current,payload/'manifest.json')
    expected=APP/'collected/timing-01-candidate/output/result.json';value=read(expected)
    assert value['core_sha256']==CORE and value['data_sha256']==DATA
    public=module('current_profile_reference_audit',payload/'tools/public_audit.py')
    checked=json.loads(json.dumps(manifest))
    for case in checked['cases']:case['raw_sha256']=original['raw_pcm'][case['name']]
    public.validate_worker(value,checked,'timing')
    assert len(value['records'])==16
    copy(expected,payload/'prior-amd-result.json')
    for name in ['remote_export.py']:copy(TOOLS/name,payload/'tools'/name)
    copy(BUILD/'closed.json',payload/'evidence/build-closed.json')
    copy(BUILD/'analysis.json',payload/'evidence/build-analysis.json')
    copy(APP/'closed.json',payload/'evidence/application-closed.json')
    copy(TOOLS/'README.md',payload/'prospective-current-plan.md')
    specification=dict(original,core=CORE,data=DATA,consumer=built['consumer'],previous_identities=read(BUILD/'collected/collection.json')['identities'],
        files={p.relative_to(payload).as_posix():pin(p) for p in payload.rglob('*') if p.is_file()})
    save(payload/'payload.json',specification)
    for p in [*TOOLS.iterdir(),MONITOR,PREVIOUS/'closed.json',BUILD/'closed.json',APP/'closed.json',ROOT/'artifacts/pyannote-selected-profile-amd-20260922/payload/tools/stacks.py']:
        if p.is_file():
            files[rel(p)]=pin(p)
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
    for p in payload.rglob('*'):
        if p.is_file():files[rel(p)]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(payload.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(payload).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=files,archive=pin(BASE/'payload.tar.gz'),payload=pin(payload/'payload.json'),
        build=pin(BUILD/'closed.json'),existing_methods=160,unchanged_methods=159))
    print(json.dumps(dict(payload=pin(payload/'payload.json'),archive=pin(BASE/'payload.tar.gz'),consumer=built['consumer'])))


if __name__=='__main__':main()
