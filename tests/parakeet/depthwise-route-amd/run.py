"""Freeze the exact-product observer build and one complete public count capture."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
from expected import expected

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
loader=importlib.util.spec_from_file_location('depthwise_transport',TOOLS.parent/'weight-ownership-probe/run.py')
prior=importlib.util.module_from_spec(loader);loader.loader.exec_module(prior)
pin,read,write,ssh=prior.pin,prior.read,prior.write,prior.ssh
BASE=ROOT/'artifacts/parakeet-depthwise-route-amd-20260925';REMOTE='/dev/shm/lokad-parakeet-depthwise-route-20260925'
PRELUDE=prior.PRELUDE.replace(prior.REMOTE,REMOTE)
prior.BASE,prior.REMOTE,prior.PRELUDE=BASE,REMOTE,PRELUDE
transport=prior.transport;transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE
SOURCE=ROOT/'artifacts/parakeet-depthwise-route-source-20260925'
PROFILE=ROOT/'artifacts/parakeet-packed-final-row-profile-amd-20260925'
RUNTIME=PROFILE/'build-collected/runtime-control'
REMOTE_RUNTIME='/dev/shm/lokad-parakeet-packed-final-row-profile-20260925/runtime-control'
APP=ROOT/'artifacts/parakeet-packed-final-row-release-app-amd-20260925'
REMOTE_APP='/dev/shm/lokad-parakeet-packed-final-row-release-app-20260925'
DIAGNOSIS=ROOT/'artifacts/parakeet-stem-diagnosis-20260925'


def references():
    source=read(SOURCE/'prepared.json');assert source['passed'] and source['diagnostic_only'] and not source['root_product_changed']
    for name,wanted in source['source'].items():assert pin(SOURCE/'source'/name)==wanted,name
    for name,wanted in source['tools'].items():assert pin(TOOLS.parent/'depthwise-route-source'/name)==wanted,name
    assert source['plan']==pin(SOURCE/'prospective-plan.md') and source['patch']==pin(SOURCE/'observer.patch')
    assert source['diagnosis']==pin(DIAGNOSIS/'closed.json') and read(DIAGNOSIS/'closed.json')['analysis']==pin(DIAGNOSIS/'analysis.json')
    expected(read(DIAGNOSIS/'analysis.json'))
    runtime={p.name:pin(p) for p in RUNTIME.iterdir() if p.is_file()}
    built=read(PROFILE/'build-collected/built.json')
    assert runtime=={n.removeprefix('runtime-control/'):v for n,v in built['runtime_files'].items() if n.startswith('runtime-control/')}
    assert runtime['Lokad.Onnx.dll']==built['core'] and runtime['SampledAudio.dll']==built['consumer']
    return source,runtime


def prepare():
    assert not BASE.exists();source,runtime=references()
    for name in ['audit.py','vm.py','README.md','expected.py','test_expected.py']:assert (TOOLS/name).is_file()
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,data):
        path=bundle/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    for name in source['source']:put('source/'+name,(SOURCE/'source'/name).read_bytes())
    put('bridge-source/Program.cs',(TOOLS.parent/'owned-packed-weight-build/Bridge.cs.txt').read_bytes())
    put('bridge-source/Bridge.csproj',(TOOLS.parent/'selected-profile-build-amd/Bridge.csproj').read_bytes())
    put('bridge-source/global.json',(ROOT/'global.json').read_bytes())
    put('remote.py',(TOOLS/'vm.py').read_bytes());put('common.py',(TOOLS.parent/'managed-phase-amd/remote.py').read_bytes())
    put('campaign_processes.py',(APP/'collected/runtime/campaign_processes.py').read_bytes())
    put('protocol.py',(APP/'collected/runtime/protocol.py').read_bytes())
    put('diagnosis.json',(DIAGNOSIS/'analysis.json').read_bytes())
    put('protocol.md',(TOOLS/'README.md').read_bytes())
    for name in ['prospective-plan.md','observer.patch']:put(name,(SOURCE/name).read_bytes())
    put('reference-public.json',(PROFILE/'bundle/evidence/candidate-public.json').read_bytes())
    oldspec=read(PROFILE/'bundle/spec.json')
    external={name:value for name,value in oldspec['external'].items() if not name.startswith('/dev/shm/lokad-parakeet-packed-final-row-profile-20260925/')}
    external.update({REMOTE_RUNTIME+'/'+name:value for name,value in runtime.items()})
    spec=dict(boot=1789634288.0,external=external,runtime=runtime,prior=REMOTE_RUNTIME,app=REMOTE_APP,
        source=pin(SOURCE/'prepared.json'),diagnosis=pin(DIAGNOSIS/'closed.json'),diagnostic_only=True,release_admitted=False,
        before_product={n:runtime[n] for n in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']},consumer=runtime['SampledAudio.dll'],
        feed='/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed',
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=180),
        capture_limits=dict(available_before=11*1024**3,tmpfs_before=2*1024**3,rss=12*1024**3,seconds=900),
        minimum_free=1024**3,output_limit=512*1024**2,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    helpers=['weight-ownership-probe/run.py','managed-phase-amd/run.py','managed-phase-amd/remote.py','ort-diagnosis-amd/run.py',
        'feed-forward-cost-diagnostic/isolated_baseline.py','owned-packed-weight-build/Bridge.cs.txt','selected-profile-build-amd/Bridge.csproj']
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},
        helpers={(TOOLS.parent/n).relative_to(ROOT).as_posix():pin(TOOLS.parent/n) for n in helpers}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'))))


def prepared():
    references();value=read(BASE/'prepared.json');spec=read(BASE/'bundle/spec.json')
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
