"""Freeze one representation probe against unchanged products; reuse transport."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
loader = importlib.util.spec_from_file_location('ownership_transport', TOOLS.parent/'weight-ownership-probe/run.py')
prior = importlib.util.module_from_spec(loader); loader.loader.exec_module(prior)
pin, read, write, ssh = prior.pin, prior.read, prior.write, prior.ssh
BASE = ROOT/'artifacts/parakeet-packed-logical-weight-amd-20260925'
REMOTE = '/dev/shm/lokad-parakeet-packed-logical-weight-20260925'
PRELUDE = prior.PRELUDE.replace(prior.REMOTE, REMOTE)
prior.BASE, prior.REMOTE, prior.PRELUDE = BASE, REMOTE, PRELUDE
transport = prior.transport
transport.BASE, transport.REMOTE, transport.PRELUDE = BASE, REMOTE, PRELUDE
OWNERSHIP = ROOT/'artifacts/parakeet-weight-ownership-amd-v2-20260925'
RUNTIME = '/dev/shm/lokad-parakeet-weight-ownership-v2-20260925/runtime'


def references():
    isolated, costs, cost_closure = prior.references()
    closed = read(OWNERSHIP/'closed.json')
    assert pin(OWNERSHIP/'closed.json')['sha256'] == 'abe97a41fbb60648a6ee55968cd5e36c861ac908603336bd2f465379cbb1f3cd'
    assert closed['passed'] and closed['prediction_passed']
    for name, wanted in closed['files'].items(): assert pin(OWNERSHIP/name) == wanted, name
    return isolated, costs, cost_closure


def cases(mode):
    rows = []
    if mode == 'native':
        for n,k in [(7,33),(8,64),(1024,4096),(4096,1024)]:
            rows.append(dict(id=f'layout-{n}-{k}',kind='layout',n=n,k=k))
        rows.append(dict(id='bounds',kind='bounds'))
        for n,k in [(1024,4096),(4096,1024)]:
            for m in [167,225]:rows.append(dict(id=f'packed-{n}-{k}-{m}',kind='packed',n=n,k=k,m=m))
            rows.append(dict(id=f'ownership-{n}-{k}',kind='ownership',n=n,k=k))
    for n,k in [(7,33),(8,64)]:
        for m in [1,2,3]:
            for option in ['scalar','simd', 'intrinsics' if mode=='native' else 'auto']:
                rows.append(dict(id=f'fallback-{n}-{k}-{m}-{option}',kind='fallback',n=n,k=k,m=m,mode=option))
    if mode != 'native':rows.append(dict(id='explicit-intrinsics-unavailable',kind='unavailable'))
    assert len(rows) == (29 if mode=='native' else 19)
    return rows


def prepare():
    assert not BASE.exists(); isolated, costs, cost_closure = references()
    assert (TOOLS/'review.py').exists()
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(encoding='utf8'),str(p))
    BASE.mkdir(); bundle=BASE/'bundle'; (bundle/'source').mkdir(parents=True)
    for name in ['Program','PackedLogicalWeight']:
        (bundle/'source'/(name+'.cs')).write_bytes((TOOLS/(name+'.cs.txt')).read_bytes())
    (bundle/'source/global.json').write_bytes((ROOT/'global.json').read_bytes())
    names=['Lokad.Onnx','Lokad.Onnx.Data','Google.Protobuf','FastBertTokenizer','Lokad.Tokenizers','SixLabors.ImageSharp']
    refs=''.join(f'<Reference Include="{n}"><HintPath>$(FrozenProductDirectory)/{n}.dll</HintPath></Reference>' for n in names)
    project='<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><Nullable>enable</Nullable><ImplicitUsings>enable</ImplicitUsings><AllowUnsafeBlocks>true</AllowUnsafeBlocks><LangVersion>11.0</LangVersion><UseAppHost>false</UseAppHost></PropertyGroup><ItemGroup>'+refs+'</ItemGroup></Project>'
    (bundle/'source/PackedLogicalWeightProbe.csproj').write_text(project)
    (bundle/'remote.py').write_bytes((TOOLS/'vm.py').read_bytes())
    (bundle/'common.py').write_bytes((TOOLS.parent/'managed-phase-amd/remote.py').read_bytes())
    (bundle/'plan.md').write_bytes((ROOT/'.agent/m75-parakeet-packed-logical-weight-20260925.md').read_bytes())
    runtime={n+'.dll':pin(OWNERSHIP/'capture-collected/runtime'/(n+'.dll')) for n in names}
    assert all(runtime[n]==v for n,v in isolated['product'].items())
    spec=dict(boot=1789634288.0,original_runtime=RUNTIME,runtime_files=runtime,product=isolated['product'],
        isolated_evidence=isolated['evidence'],failed_release_controls=isolated['failed_release_controls'],
        ownership_closure=pin(OWNERSHIP/'closed.json'),cost_report=pin(prior.COSTS),cost_closure=pin(cost_closure),
        release_admitted=False,diagnostic_only=True,no_model_execution=True,cases={m:cases(m) for m in ['native','hardware-disabled']},
        external={RUNTIME+'/'+n:v for n,v in runtime.items()},
        feed='/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed',
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=2*1024**3,seconds=180),
        capture_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=2*1024**3,seconds=300),
        minimum_free=1024**3,output_limit=64*1024**2,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in bundle.rglob('*'):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    helpers=['weight-ownership-probe/run.py','managed-phase-amd/run.py','managed-phase-amd/remote.py',
             'ort-diagnosis-amd/run.py','feed-forward-cost-diagnostic/isolated_baseline.py']
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},
        helpers={str((TOOLS.parent/n).relative_to(ROOT)).replace('\\','/'):pin(TOOLS.parent/n) for n in helpers}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),cases={m:len(v) for m,v in spec['cases'].items()})))


def prepared():
    isolated, costs, cost_closure=references(); value=read(BASE/'prepared.json');spec=read(BASE/'bundle/spec.json')
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json')
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in value['helpers'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in spec['files'].items():assert pin(BASE/'bundle'/name)==wanted,name
    assert spec['isolated_evidence']==isolated['evidence'] and spec['product']==isolated['product']
    assert spec['ownership_closure']==pin(OWNERSHIP/'closed.json') and spec['cost_report']==pin(prior.COSTS)


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
