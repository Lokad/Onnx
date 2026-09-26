"""Freeze one prospective complete-call screen; reuse closed product binaries."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
from scope import census, references, instrument

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
loader=importlib.util.spec_from_file_location('sigmoid_screen_transport',TOOLS.parent/'weight-ownership-probe/run.py')
prior=importlib.util.module_from_spec(loader);loader.loader.exec_module(prior)
sys.path.insert(0,str(TOOLS))
pin,read,write,ssh=prior.pin,prior.read,prior.write,prior.ssh
BASE=ROOT/'artifacts/parakeet-sigmoid-execution-diagnostic-amd-20260927'
REMOTE='/dev/shm/lokad-parakeet-sigmoid-execution-diagnostic-20260927'
PRELUDE=prior.PRELUDE.replace(prior.REMOTE,REMOTE)
prior.BASE,prior.REMOTE,prior.PRELUDE=BASE,REMOTE,PRELUDE
transport=prior.transport;transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE
RUNTIMES={'current':ROOT/'artifacts/parakeet-pad-current-root-amd-20260926/collected/runtime',
          'candidate':ROOT/'artifacts/parakeet-vector-sigmoid-build-amd-20260925/build-collected/runtime'}
REMOTE_RUNTIMES={'current':'/dev/shm/lokad-parakeet-pad-current-root-20260926/runtime',
                 'candidate':'/dev/shm/lokad-parakeet-vector-sigmoid-build-20260925/runtime'}
DIAGNOSTIC_FLAGS={'DOTNET_JitDisasm':'Lokad.Onnx.CPUExecutionProvider:Sigmoid Lokad.Onnx.MathOps:ExpVector',
                 'DOTNET_JitDisasmWithCodeBytes':'1'}


def prepare():
    assert not BASE.exists();products,evidence=references();cases=census()
    for name in ['audit.py','vm.py','README.md','scope.py','CounterProbe.cs','score.py','codegen.py','test_scope.py']:assert (TOOLS/name).is_file()
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,data):
        path=bundle/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    original=(TOOLS.parent/'vector-sigmoid-screen/Screen.cs').read_text(encoding='utf8')
    put('source/Screen.cs',instrument(original).encode())
    put('source/CounterProbe.cs',(TOOLS/'CounterProbe.cs').read_bytes())
    put('evidence/OriginalScreen.cs.txt',original.encode())
    put('source/global.json',(ROOT/'global.json').read_bytes())
    refs=''.join(f'<Reference Include="{n}"><HintPath>$(FrozenProductDirectory)/{n}.dll</HintPath></Reference>' for n in ['Lokad.Onnx','Google.Protobuf'])
    project='<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable><UseAppHost>false</UseAppHost></PropertyGroup><ItemGroup>'+refs+'</ItemGroup></Project>'
    put('source/Screen.csproj',project.encode())
    put('common.py',(TOOLS.parent/'managed-phase-amd/remote.py').read_bytes())
    put('remote.py',(TOOLS/'vm.py').read_bytes())
    put('campaign_processes.py',(TOOLS.parent/'prepared-recurrence-timing-amd/campaign_processes.py').read_bytes())
    put('protocol.md',(TOOLS/'README.md').read_bytes());write(bundle/'census.json',cases)
    external={REMOTE_RUNTIMES[role]+'/Lokad.Onnx.dll':products[role] for role in products}
    protobuf=pin(RUNTIMES['current']/'Google.Protobuf.dll')
    assert pin(RUNTIMES['candidate']/'Google.Protobuf.dll')==protobuf
    external[REMOTE_RUNTIMES['current']+'/Google.Protobuf.dll']=protobuf
    spec=dict(boot=1789634288.0,products=products,evidence=evidence,external=external,runtimes=REMOTE_RUNTIMES,
        diagnostic_only=True,release_admitted=False,failed_graph_cases=[],diagnostic_flags=DIAGNOSTIC_FLAGS,
        retained_failed_screen='aaca2b2fd3b54f66dec841b948c0673279b833f90733770348fcbe41dfc4201f',census=pin(bundle/'census.json'),
        feed='/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed',
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=180),
        capture_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=900),
        minimum_free=1024**3,output_limit=512*1024**2,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    helpers=['weight-ownership-probe/run.py','managed-phase-amd/run.py','managed-phase-amd/remote.py','ort-diagnosis-amd/run.py',
             'feed-forward-cost-diagnostic/isolated_baseline.py','prepared-recurrence-timing-amd/campaign_processes.py',
             'vector-sigmoid-screen/Screen.cs','vector-sigmoid-screen/census.py','vector-sigmoid-screen/score.py']
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},
        helpers={(TOOLS.parent/n).relative_to(ROOT).as_posix():pin(TOOLS.parent/n) for n in helpers}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),cases=len(cases['cases']))))


def prepared():
    products,evidence=references();value=read(BASE/'prepared.json');spec=read(BASE/'bundle/spec.json')
    assert spec['products']==products and spec['evidence']==evidence
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json')
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in value['helpers'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in spec['files'].items():assert pin(BASE/'bundle'/name)==wanted,name
    assert read(BASE/'bundle/census.json')==census()


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
