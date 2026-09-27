"""Freeze the single candidate and raw contracts using the existing VM transport."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
from cases import cases, FALLBACK

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-pointwise-tail-contracts-amd-20260927'
REMOTE='/dev/shm/lokad-pointwise-tail-contracts-20260927'
SOURCE=ROOT/'artifacts/parakeet-pointwise-tail-source-20260927'
QUALIFIED=ROOT/'artifacts/parakeet-decoder-lstm-layout-root-amd-20260927'
PROFILE=ROOT/'artifacts/parakeet-decoder-lstm-layout-profile-amd-20260927'


def load(name,path):
    loader=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(loader);loader.loader.exec_module(value);return value


prior=load('tail_contract_transport',TOOLS.parent/'weight-ownership-probe/run.py')
pin,read,write,ssh=prior.pin,prior.read,prior.write,prior.ssh
PRELUDE=prior.PRELUDE.replace(prior.REMOTE,REMOTE)
prior.BASE,prior.REMOTE,prior.PRELUDE=BASE,REMOTE,PRELUDE
transport=prior.transport;transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE


def references():
    value=read(SOURCE/'prepared.json')
    assert value['passed'] and not value['release_admitted'] and value['full_panel_source_unchanged']
    assert value['baseline']==pin(QUALIFIED/'closed.json')
    assert value['baseline']['sha256']=='efb99eea455647c64bbe0811c1b7d46387add59049ac81e48a926b250e7c42da'
    for name,wanted in value['source'].items():assert pin(SOURCE/'source'/name)==wanted,name
    for name,wanted in value['source_before'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in value['tools'].items():assert pin(TOOLS.parent/'pointwise-tail-source'/name)==wanted,name
    proof=read(PROFILE/'closed.json')
    assert proof['passed'] and proof['files']['bundle/spec.json']==pin(PROFILE/'bundle/spec.json')
    old=read(PROFILE/'bundle/spec.json')
    runtime={n.removeprefix('runtime-control/'):v for n,v in old['files'].items() if n.startswith('runtime-control/')}
    for name,wanted in runtime.items():assert pin(PROFILE/'bundle/runtime-control'/name)==wanted,name
    assert runtime['Lokad.Onnx.dll']['sha256']=='47984318b082710c3a4f57a85b1500d49d7c1236c04b1234477d19e48d11207c'
    return value,runtime


def prepare():
    assert not BASE.exists() and (TOOLS/'audit.py').exists()
    source,runtime=references()
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,data):
        p=bundle/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    for name in source['source']:put('source/'+name,(SOURCE/'source'/name).read_bytes())
    for name in ['Program.cs','Bridge.csproj','global.json']:put('bridge-source/'+name,(QUALIFIED/'bundle/bridge-source'/name).read_bytes())
    put('contract-source/Program.cs',(TOOLS/'Contracts.cs.txt').read_bytes())
    put('contract-source/global.json',(ROOT/'global.json').read_bytes())
    refs=''.join(f'<Reference Include="{n.removesuffix(".dll")}"><HintPath>$(FrozenProductDirectory)/{n}</HintPath></Reference>' for n in runtime if n.endswith('.dll') and n!='SampledAudio.dll')
    project='<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><Nullable>enable</Nullable><ImplicitUsings>enable</ImplicitUsings><AllowUnsafeBlocks>true</AllowUnsafeBlocks><UseAppHost>false</UseAppHost></PropertyGroup><ItemGroup>'+refs+'</ItemGroup></Project>'
    put('contract-source/TailContracts.csproj',project.encode())
    for target,path in [('remote.py',TOOLS/'vm.py'),('common.py',TOOLS.parent/'managed-phase-amd/remote.py'),
        ('protocol.md',TOOLS/'README.md'),('prospective-plan.md',SOURCE/'prospective-plan.md'),('candidate.patch',SOURCE/'candidate.patch')]:put(target,path.read_bytes())
    original='/dev/shm/lokad-lstmlayout-profile-20260927/runtime-control'
    spec=dict(boot=1789634288.0,prior=original,runtime=runtime,external={original+'/'+n:v for n,v in runtime.items()},
        source=pin(SOURCE/'prepared.json'),baseline=source['baseline'],diagnosis=source['diagnosis'],
        changed_methods=source['changed_methods'],added_methods=source['added_methods'],
        raw_cases=cases(),fallback_cases=FALLBACK,modes=['normal','avx512-disabled','scalar-baseline','scalar-candidate'],
        disasm='*PackedColumn* *mm_unsafe_vectorized_intrinsics_2x4packed_bump*',
        feed='/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed',
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=180),
        capture_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=600),
        minimum_free=1024**3,output_limit=512*1024**2,release_admitted=False,no_model_execution=True,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    helpers=['weight-ownership-probe/run.py','managed-phase-amd/run.py','managed-phase-amd/remote.py',
        'ort-diagnosis-amd/run.py','feed-forward-cost-diagnostic/isolated_baseline.py']
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},helpers={n:pin(TOOLS.parent/n) for n in helpers}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),raw_cases=len(spec['raw_cases']))))


def prepared():
    references();value=read(BASE/'prepared.json');spec=read(BASE/'bundle/spec.json')
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json')
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in value['helpers'].items():assert pin(TOOLS.parent/name)==wanted,name
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
