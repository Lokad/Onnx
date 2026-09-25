"""Freeze the isolated candidate build, complete IL review and focused graph tests."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
loader=importlib.util.spec_from_file_location('owned_transport',TOOLS.parent/'weight-ownership-probe/run.py')
prior=importlib.util.module_from_spec(loader);loader.loader.exec_module(prior)
pin,read,write,ssh=prior.pin,prior.read,prior.write,prior.ssh
BASE=ROOT/'artifacts/parakeet-owned-packed-weight-amd-20260925'
REMOTE='/dev/shm/lokad-parakeet-owned-packed-weight-20260925'
PRELUDE=prior.PRELUDE.replace(prior.REMOTE,REMOTE)
prior.BASE,prior.REMOTE,prior.PRELUDE=BASE,REMOTE,PRELUDE
transport=prior.transport;transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE
SOURCE=ROOT/'artifacts/parakeet-owned-packed-weight-source-20260925'
RUNTIME=ROOT/'artifacts/parakeet-weight-ownership-amd-v2-20260925/capture-collected/runtime'
REMOTE_RUNTIME='/dev/shm/lokad-parakeet-weight-ownership-v2-20260925/runtime'


def source_verified():
    value=read(SOURCE/'prepared.json');isolated,costs,closure=prior.references()
    assert pin(SOURCE/'prepared.json')['sha256']=='3b95b79b2a8ccc653f6173ef2ff439e19a0e874202ee7114ac3f2dff8054f452'
    assert value['passed'] and not value['built'] and not value['root_product_changed'] and not value['release_admitted']
    assert value['product']==isolated['product'] and value['inventory']==isolated['inventory']
    assert value['before']==isolated['source_files'] and value['isolated_evidence']==isolated['evidence']
    assert value['failed_release_controls']==isolated['failed_release_controls']
    for name,wanted in value['source'].items():assert pin(SOURCE/'source'/name)==wanted,name
    for name,wanted in value['templates'].items():assert pin(TOOLS.parent/'owned-packed-weight-source'/name)==wanted,name
    assert value['plan']==pin(SOURCE/'prospective-plan.md') and value['patch']==pin(SOURCE/'candidate.patch')
    return value


def prepare():
    assert not BASE.exists();source=source_verified();assert (TOOLS/'review.py').exists()
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(encoding='utf8'),str(p))
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,content):
        p=bundle/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as stream:stream.write(content if isinstance(content,bytes) else content.encode())
    for name in source['source']:put('source/'+name,(SOURCE/'source'/name).read_bytes())
    put('source/tests/Lokad.Onnx.Backend.Tests/OwnedPackedRuntimeIdentityTests.cs',(TOOLS/'RuntimeIdentityTests.cs.txt').read_bytes())
    put('bridge-source/Program.cs',(TOOLS/'Bridge.cs.txt').read_bytes())
    put('bridge-source/Bridge.csproj',(TOOLS.parent/'selected-profile-build-amd/Bridge.csproj').read_bytes())
    put('bridge-source/global.json',(ROOT/'global.json').read_bytes())
    put('common.py',(TOOLS.parent/'managed-phase-amd/remote.py').read_bytes())
    put('remote.py',(TOOLS/'vm.py').read_bytes())
    put('plan.md',(ROOT/'.agent/m76-parakeet-owned-packed-weights-20260925.md').read_bytes())
    put('baseline.json',json.dumps(source['inventory'],indent=2))
    runtime={p.name:pin(p) for p in RUNTIME.glob('*.dll') if p.name!='WeightOwnershipProbe.dll'}
    assert len(runtime)==6 and all(runtime[n]==v for n,v in source['product'].items())
    spec=dict(boot=1789634288.0,prior=REMOTE_RUNTIME,before_product=source['product'],
        external={REMOTE_RUNTIME+'/'+name:value for name,value in runtime.items()},
        source_prepared=pin(SOURCE/'prepared.json'),failed_release_controls=source['failed_release_controls'],
        release_admitted=False,candidate_selected=True,source_files=len(source['source']),
        core_changed_methods=source['core_changed_methods'],data_changed_methods=source['data_changed_methods'],
        feed='/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed',
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=180),
        capture_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=300),
        minimum_free=1024**3,output_limit=512*1024**2,expected_tests={'512':26,'256':26,'scalar':2},
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in bundle.rglob('*'):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    helpers=['weight-ownership-probe/run.py','managed-phase-amd/run.py','managed-phase-amd/remote.py','ort-diagnosis-amd/run.py',
        'feed-forward-cost-diagnostic/isolated_baseline.py','slice-dense-conversion-build-amd/Bridge.cs.txt','selected-profile-build-amd/Bridge.csproj']
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},
        helpers={str((TOOLS.parent/n).relative_to(ROOT)).replace('\\','/'):pin(TOOLS.parent/n) for n in helpers}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),source_files=spec['source_files'])))


def prepared():
    source_verified();value=read(BASE/'prepared.json');spec=read(BASE/'bundle/spec.json')
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json')
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in value['helpers'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in spec['files'].items():assert pin(BASE/'bundle'/name)==wanted,name


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    else:
        prepared()
        if action=='stage':transport.stage()
        else:
            kind=sys.argv[2];assert kind in ['build','capture']
            if action=='launch':
                if kind=='capture':assert read(BASE/'build-review-transferred.json')['passed']
                transport.launch(kind)
            else:{'observe':prior.observe,'collect':prior.collect}[action](kind)
