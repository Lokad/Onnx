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
BASE=ROOT/'artifacts/parakeet-packed-final-row-build-amd-20260925'
REMOTE='/dev/shm/lokad-parakeet-packed-final-row-build-20260925'
PRELUDE=prior.PRELUDE.replace(prior.REMOTE,REMOTE)
prior.BASE,prior.REMOTE,prior.PRELUDE=BASE,REMOTE,PRELUDE
transport=prior.transport;transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE
SOURCE=ROOT/'artifacts/parakeet-packed-final-row-source-20260925'
PROOF=ROOT/'artifacts/parakeet-packed-final-row-proof-amd-20260925'
RUNTIME=PROOF/'capture-collected/runtime'
REMOTE_RUNTIME='/dev/shm/lokad-parakeet-packed-final-row-proof-20260925/runtime'


def source_verified():
    assert pin(SOURCE/'prepared.json')['sha256']=='e3ca64b50ea4a5dd276d90a19779d602f8ed78b6275b9e5448fe3b021bec5e1b'
    value=read(SOURCE/'prepared.json')
    assert value['passed'] and not value['built'] and not value['root_product_changed'] and not value['release_admitted']
    assert value['preparer']==pin(TOOLS.parent/'packed-final-row-source/prepare.py')
    for name,wanted in value['source'].items():assert pin(SOURCE/'source'/name)==wanted,name
    assert value['plan']==pin(SOURCE/'prospective-plan.md') and value['patch']==pin(SOURCE/'candidate.patch')
    assert value['proof']==pin(PROOF/'closed.json') and read(PROOF/'closed.json')['proof_passed']
    assert read(PROOF/'closed.json')['analysis']==pin(PROOF/'analysis.json')
    proof=read(PROOF/'analysis.json');assert proof['proof_passed'] and proof['cases']==98
    assert proof['product']==value['product'] and proof['consumer']==value['proof_consumer']==pin(RUNTIME/'PackedFinalRowProbe.dll')
    assert value['proof_helper']==pin(SOURCE/'source/src/Lokad.Onnx/PackedFinalRowKernel.cs')==proof['helper']
    assert [len(r['normalized_methods']) for r in value['inventory']['observations']]==[3277,697]
    assert not value['prior_quantitative_attribution']
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
    put('plan.md',(ROOT/'.agent/m78-parakeet-packed-final-row-20260925.md').read_bytes())
    put('baseline.json',json.dumps(source['inventory'],indent=2))
    runtime={n+'.dll':pin(RUNTIME/(n+'.dll')) for n in ['Lokad.Onnx','Lokad.Onnx.Data','Google.Protobuf','FastBertTokenizer','Lokad.Tokenizers','SixLabors.ImageSharp']}
    assert len(runtime)==6 and all(runtime[n]==v for n,v in source['product'].items())
    spec=dict(boot=1789634288.0,prior=REMOTE_RUNTIME,before_product=source['product'],
        external={**{REMOTE_RUNTIME+'/'+name:value for name,value in runtime.items()},REMOTE_RUNTIME+'/PackedFinalRowProbe.dll':source['proof_consumer']},
        proof_runtime=REMOTE_RUNTIME,proof_consumer=source['proof_consumer'],proof=source['proof'],
        source_prepared=pin(SOURCE/'prepared.json'),failed_release_controls=source['failed_release_controls'],
        release_admitted=False,candidate_selected=True,source_files=len(source['source']),
        core_changed_methods=source['core_changed_methods'],data_changed_methods=source['data_changed_methods'],
        feed='/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed',
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=180),
        capture_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=300),
        minimum_free=1024**3,output_limit=512*1024**2,expected_tests=source['expected_tests'],
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in bundle.rglob('*'):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    helpers=['weight-ownership-probe/run.py','managed-phase-amd/run.py','managed-phase-amd/remote.py','ort-diagnosis-amd/run.py',
        'feed-forward-cost-diagnostic/isolated_baseline.py','slice-dense-conversion-build-amd/Bridge.cs.txt','selected-profile-build-amd/Bridge.csproj',
        'owned-packed-weight-scope-build/Bridge.cs.txt','packed-final-row-source/prepare.py']
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
