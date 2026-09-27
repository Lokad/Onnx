"""Build the one-method attention preparation candidate and focused contracts."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-attention-owned-build-amd-20260928'
REMOTE='/dev/shm/lokad-attention-owned-build-20260928'
SOURCE=ROOT/'artifacts/parakeet-attention-owned-source-20260928'
QUALIFIED=ROOT/'artifacts/parakeet-pointwise-tail-root-amd-20260927'
PROFILE=ROOT/'artifacts/parakeet-pointwise-tail-profile-amd-20260927'
REMOTE_RUNTIME='/dev/shm/lokad-pwt-profile-20260927/runtime-control'


def load(name,path):
    loader=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(loader);loader.loader.exec_module(value)
    return value


prior=load('attention_owned_transport',TOOLS.parent/'weight-ownership-probe/run.py')
pin,read,write,ssh=prior.pin,prior.read,prior.write,prior.ssh
PRELUDE=prior.PRELUDE.replace(prior.REMOTE,REMOTE)
prior.BASE,prior.REMOTE,prior.PRELUDE=BASE,REMOTE,PRELUDE
transport=prior.transport
transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE
source_tool=load('attention_owned_source',TOOLS.parent/'attention-owned-source/prepare.py')


def references():
    original=source_tool.references()
    value=read(SOURCE/'prepared.json')
    assert value['passed'] and not value['release_admitted'] and not value['root_product_changed']
    assert value['source_before']==original and len(value['source'])==446
    assert value['changed_product_files']==[source_tool.TARGET] and value['changed_methods']==['PrepareOwnedMatMulWeights']
    assert value['added_product_files']==value['added_methods']==[] and value['added_tests']==[source_tool.TEST]
    assert value['source_reversible'] and value['arithmetic_leaves_unchanged']
    assert value['plan']==pin(SOURCE/'prospective-plan.md') and value['patch']==pin(SOURCE/'candidate.patch')
    for name,wanted in value['source'].items(): assert pin(SOURCE/'source'/name)==wanted,name
    for name,wanted in value['tools'].items(): assert pin(TOOLS.parent/'attention-owned-source'/name)==wanted,name
    assert (SOURCE/'source'/source_tool.TARGET).read_bytes()==source_tool.change((ROOT/source_tool.TARGET).read_bytes())[0]
    assert pin(PROFILE/'closed.json')['sha256']=='378f7c1979052e4dcf06af4d2c9edad04cf899cc3d40de5f7a491f2c5f0c13b2'
    proof=read(PROFILE/'closed.json')
    assert proof['passed'] and proof['files']['bundle/spec.json']==pin(PROFILE/'bundle/spec.json')
    old=read(PROFILE/'bundle/spec.json')
    runtime={n.removeprefix('runtime-control/'):v for n,v in old['files'].items() if n.startswith('runtime-control/')}
    for name,wanted in runtime.items(): assert pin(PROFILE/'bundle/runtime-control'/name)==wanted,name
    return value,runtime


def prepare():
    assert not BASE.exists()
    source,runtime=references()
    assert (TOOLS/'review.py').exists() and (TOOLS/'README.md').exists()
    for p in TOOLS.glob('*.py'): ast.parse(p.read_text(encoding='utf8'))
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,data):
        path=bundle/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    for name in source['source']: put('source/'+name,(SOURCE/'source'/name).read_bytes())
    for name in ['Program.cs','Bridge.csproj','global.json']:
        put('bridge-source/'+name,(QUALIFIED/'bundle/bridge-source'/name).read_bytes())
    put('source/tests/Lokad.Onnx.Backend.Tests/OwnedPackedRuntimeIdentityTests.cs',
        (TOOLS.parent/'packed-final-row-build/RuntimeIdentityTests.cs.txt').read_bytes())
    put('remote.py',(TOOLS/'vm.py').read_bytes())
    put('common.py',(TOOLS.parent/'managed-phase-amd/remote.py').read_bytes())
    put('prospective-plan.md',(SOURCE/'prospective-plan.md').read_bytes())
    put('protocol.md',(TOOLS/'README.md').read_bytes())
    spec=dict(boot=1789634288.0,prior=REMOTE_RUNTIME,
        external={REMOTE_RUNTIME+'/'+n:v for n,v in runtime.items()},
        before_product={n:runtime[n] for n in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']},
        qualified_root=pin(QUALIFIED/'closed.json'),source_prepared=pin(SOURCE/'prepared.json'),release_admitted=False,
        feed='/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed',
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=180),
        capture_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=300),
        minimum_free=1024**3,output_limit=512*1024**2,
        expected_classes=dict(normal=dict(MatMulDestinationTests=16,MatMulEmptyTests=4,MatMulVectorTests=3,
            OwnedPackedRuntimeIdentityTests=1,OwnedPackedWeightTests=40,OwnedAttentionPreparationTests=29),
            scalar=dict(MatMulDestinationTests=16,MatMulEmptyTests=4,MatMulVectorTests=3,
            OwnedPackedRuntimeIdentityTests=1,OwnedPackedUnavailableTests=1,OwnedAttentionUnavailableTests=1)),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    helpers=['weight-ownership-probe/run.py','managed-phase-amd/run.py','managed-phase-amd/remote.py',
        'ort-diagnosis-amd/run.py','feed-forward-cost-diagnostic/isolated_baseline.py',
        'packed-final-row-build/RuntimeIdentityTests.cs.txt']
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        source_prepared=pin(SOURCE/'prepared.json'),tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},
        helpers={n:pin(TOOLS.parent/n) for n in helpers}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'))))


def prepared():
    references();value=read(BASE/'prepared.json')
    assert value['source_prepared']==pin(SOURCE/'prepared.json')
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json')
    for name,wanted in value['tools'].items(): assert pin(TOOLS/name)==wanted,name
    for name,wanted in value['helpers'].items(): assert pin(TOOLS.parent/name)==wanted,name
    for name,wanted in read(BASE/'bundle/spec.json')['files'].items(): assert pin(BASE/'bundle'/name)==wanted,name


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare': prepare()
    else:
        prepared()
        if action=='stage': transport.stage()
        else:
            kind=sys.argv[2];assert kind in ['build','capture']
            if action=='launch':
                if kind=='capture': assert read(BASE/'build-review-transferred.json')['passed']
                transport.launch(kind)
            else: {'observe':prior.observe,'collect':prior.collect}[action](kind)
