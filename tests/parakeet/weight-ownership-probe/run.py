"""Freeze the ownership probe, preserving the isolated product and prior verdicts."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
sys.path.insert(0,str(TOOLS.parent/'feed-forward-cost-diagnostic'))
from isolated_baseline import qualify
loader=importlib.util.spec_from_file_location('ownership_transport',TOOLS.parent/'managed-phase-amd/run.py')
transport=importlib.util.module_from_spec(loader);loader.loader.exec_module(transport)
pin,read,write,ssh,SSH=transport.pin,transport.read,transport.write,transport.ssh,transport.SSH
BASE=ROOT/'artifacts/parakeet-weight-ownership-amd-20260925'
REMOTE='/dev/shm/lokad-parakeet-weight-ownership-20260925'
PRELUDE=transport.PRELUDE.replace(transport.REMOTE,REMOTE)
transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE
MODELS=ROOT/'artifacts/parakeet-slice-dense-conversion-models-amd-20260925'
REMOTE_MODELS='/dev/shm/lokad-parakeet-slice-dense-conversion-models-20260925/runtimes/candidate'
APP=ROOT/'artifacts/parakeet-observed-dense-where-app-amd-20260924'
REMOTE_APP='/dev/shm/lokad-parakeet-observed-dense-where-app-20260924'
MANIFEST='manifests/candidate-parakeet.json'
COSTS=ROOT/'tests/parakeet/feed-forward-cost-results/costs-20260925.json'
NPY=ROOT/'artifacts/parakeet-slice-materialization-profile-amd-20260924/bundle/consumer-source/NpySupport.cs'


def references():
    isolated=qualify();costs=read(COSTS)
    assert pin(COSTS)['sha256']=='233a927b373399a6c110c59a300c84799873d12e3d71eec6c099a09858d88d06'
    assert costs['passed'] and costs['diagnostic_only'] and not costs['release_admitted']
    closure=ROOT/'artifacts/parakeet-feed-forward-cost-joint-20260925/closed.json'
    assert costs['closure']==pin(closure) and read(closure)['usable_for_candidate_selection']
    for name,wanted in costs['managed_sources'].items():assert pin(ROOT/name)==wanted
    return isolated,costs,closure


def prepare():
    assert not BASE.exists();isolated,costs,closure=references()
    assert (TOOLS/'review.py').exists(),'Finish both reviews before freezing'
    manifest=read(APP/'collected'/MANIFEST)
    case=max(manifest['cases'],key=lambda x:x['expected']['encoded_frames'])
    assert case['name']=='1188-133604-0002' and case['samples']==287360
    route_path=ROOT/'artifacts/parakeet-projection-route-resume-amd-20260924/matched-projection-calls.json'
    mapped={r['name'] for r in read(route_path) if '/feed_forward' in r['name'] and r['route']!='unmapped'}
    assert len(mapped)==9
    weights=[dict(name=w['name'],dims=w['dims'],mapped=w['consumers'][0]['node'] in mapped) for w in costs['feed_forward_weights']]
    assert len(weights)==96 and sum(w['mapped'] for w in weights)==9
    BASE.mkdir();bundle=BASE/'bundle';(bundle/'source').mkdir(parents=True)
    (bundle/'source/Program.cs').write_bytes((TOOLS/'Program.cs.txt').read_bytes())
    (bundle/'source/NpySupport.cs').write_bytes(NPY.read_bytes())
    (bundle/'source/global.json').write_bytes((ROOT/'global.json').read_bytes())
    names=['Lokad.Onnx','Lokad.Onnx.Data','Google.Protobuf','FastBertTokenizer','Lokad.Tokenizers','SixLabors.ImageSharp']
    refs=''.join(f'<Reference Include="{n}"><HintPath>$(FrozenProductDirectory)/{n}.dll</HintPath></Reference>' for n in names)
    project='<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><Nullable>enable</Nullable><ImplicitUsings>enable</ImplicitUsings><LangVersion>11.0</LangVersion><UseAppHost>false</UseAppHost></PropertyGroup><ItemGroup>'+refs+'</ItemGroup></Project>'
    (bundle/'source/WeightOwnershipProbe.csproj').write_text(project)
    (bundle/'remote.py').write_bytes((TOOLS/'vm.py').read_bytes())
    (bundle/'common.py').write_bytes((TOOLS.parent/'managed-phase-amd/remote.py').read_bytes())
    (bundle/'plan.md').write_bytes((ROOT/'.agent/m74-parakeet-weight-ownership-20260925.md').read_bytes())
    runtime={n+'.dll':pin(MODELS/'collected/runtimes/candidate'/(n+'.dll')) for n in names}
    assert all(runtime[n]==v for n,v in isolated['product'].items())
    external={REMOTE_MODELS+'/'+n:v for n,v in runtime.items()}
    external.update({r['path']:{k:r[k] for k in ['bytes','sha256']} for r in manifest['models'].values()})
    for row in [manifest['reference'],case['pcm']]:
        external[REMOTE_APP+'/assets/'+row['path']]={k:row[k] for k in ['bytes','sha256']}
    external[REMOTE_APP+'/'+MANIFEST]=pin(APP/'collected'/MANIFEST)
    specification=dict(boot=1789634288.0,external=external,product=isolated['product'],runtime_files=runtime,
        original_runtime=REMOTE_MODELS,isolated_evidence=isolated['evidence'],failed_release_controls=isolated['failed_release_controls'],
        release_admitted=False,diagnostic_only=True,cost_report=pin(COSTS),cost_closure=pin(closure),
        sources=costs['managed_sources'],case=case,weights=weights,
        pcm_path=REMOTE_APP+'/assets/'+case['pcm']['path'],model_directory=str(Path(manifest['models']['nemo128.onnx']['path']).parent).replace('\\','/'),
        feed='/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed',
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=180),
        capture_limits=dict(available_before=11*1024**3,tmpfs_before=2*1024**3,rss=12*1024**3,seconds=900),
        minimum_free=1024**3,output_limit=128*1024**2,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',specification)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in bundle.rglob('*'):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(encoding='utf8'),str(p))
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},
        helpers={p.relative_to(ROOT).as_posix():pin(p) for p in [TOOLS.parent/'managed-phase-amd/remote.py',TOOLS.parent/'managed-phase-amd/run.py',TOOLS.parent/'ort-diagnosis-amd/run.py']},
        npy_reader=pin(NPY),routes=pin(route_path)))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),weights=len(weights),mapped=sum(w['mapped'] for w in weights))))


def prepared():
    isolated,costs,closure=references();value=read(BASE/'prepared.json');spec=read(BASE/'bundle/spec.json')
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json')
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in value['helpers'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in spec['files'].items():assert pin(BASE/'bundle'/name)==wanted,name
    assert spec['isolated_evidence']==isolated['evidence'] and spec['product']==isolated['product']
    assert spec['cost_report']==pin(COSTS) and spec['cost_closure']==pin(closure)


def observe(kind):
    result=ssh(PRELUDE+f'''
from remote import read,live
p=base/({kind!r}+'-state.json');state=read(p) if p.exists() else None
ids=[{read(BASE/(kind+'-deployment.json'))!r}]
if state:ids += [dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
print(json.dumps(dict(live=[i for i in ids if live(i)],complete=state and state['complete'],code=state and state['code'],
 latest=None if not state or not state['runs'] else {{k:state['runs'][-1].get(k) for k in ['name','samples','complete','code']}},error=state and state.get('error'))))
''')
    with (BASE/(kind+'-observations.jsonl')).open('a') as stream:stream.write(json.dumps(result)+'\n')
    print(json.dumps(result))


def collect(kind):
    target=BASE/(kind+'-collected');assert not target.exists()
    script=PRELUDE+f'''
from remote import read,live,pin,verify
kind={kind!r};state=read(base/(kind+'-state.json'))
ids=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
assert state['complete'] and not any(live(i) for i in ids)
spec=verify();paths={{base/n for n in spec['files']}}
for folder in ['logs','runtime']+(['probe'] if kind=='capture' else []):
 paths.update(p for p in (base/folder).rglob('*') if p.is_file())
paths.update(p for p in base.iterdir() if p.is_file() and p.name!='transfer.tar.gz')
files={{p.relative_to(base).as_posix():pin(p) for p in sorted(paths)}}
with (base/(kind+'-collection.json')).open('x') as f:json.dump(dict(files=files,state=pin(base/(kind+'-state.json')),terminal=True,code=state['code'],identities=ids),f)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name in [*files,kind+'-collection.json']:tar.add(base/name,arcname=name,recursive=False)
'''
    archive=BASE/(kind+'-results.tar.gz')
    with archive.open('xb') as out,(BASE/(kind+'-collection.stderr')).open('x') as err:
        result=subprocess.run(SSH+['python3','-B','-'],input=script.encode(),stdout=out,stderr=err,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode==0,'Preserve incomplete collection; do not repeat workload'
    target.mkdir()
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert len({m.name for m in members})==len(members)
        tar.extractall(target,filter='data')
    receipt=read(target/(kind+'-collection.json'))
    for name,wanted in receipt['files'].items():assert pin(target/name)==wanted,name
    write(BASE/(kind+'-transfer.json'),dict(passed=True,archive=pin(archive),collection=pin(target/(kind+'-collection.json'))))
    print(json.dumps(dict(code=receipt['code'],files=len(receipt['files']))))


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
            else:dict(observe=observe,collect=collect)[action](kind)
