"""Freeze a consumer-only accounting check after complete model correctness."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
loader=importlib.util.spec_from_file_location('ownership_transport',TOOLS.parent/'weight-ownership-probe/run.py')
prior=importlib.util.module_from_spec(loader);loader.loader.exec_module(prior)
pin,read,write,ssh,SSH=prior.pin,prior.read,prior.write,prior.ssh,prior.SSH
BASE=ROOT/'artifacts/parakeet-owned-packed-weight-counters-amd-20260925'
REMOTE='/dev/shm/lokad-parakeet-owned-packed-weight-counters-20260925'
PRELUDE=prior.PRELUDE.replace(prior.REMOTE,REMOTE)
prior.BASE,prior.REMOTE,prior.PRELUDE=BASE,REMOTE,PRELUDE
transport=prior.transport;transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE
MODELS=ROOT/'artifacts/parakeet-owned-packed-weight-models-amd-20260925'
REMOTE_MODELS='/dev/shm/lokad-parakeet-owned-packed-weight-models-20260925'
CENSUS=ROOT/'artifacts/parakeet-owned-packed-weight-scope-census-amd-20260925'
JOBS=[role+'-'+mode for mode in ['512','256'] for role in ['selected','candidate']]


def references():
    for folder,digest in [(MODELS,'bb1a758c6936fa251d921992c67dd7c65ff075fe20829af72f4455b529f1b2ca'),
                          (CENSUS,'577a07a60d901cf57c4ba46d70b2c6fc4c02148ab67a1c28fd78d9b4fbcc3d08')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed'] and proof['analysis']==pin(folder/'analysis.json')
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    models=read(MODELS/'analysis.json');census=read(CENSUS/'analysis.json')
    assert models['passed'] and models['no_performance_measurement'] and len(models['results'])==8
    assert census['product']==models['identities']['candidate']
    for name,result in models['results'].items():
        assert result['passed']
        if '-native-' in name:
            assert result['native']['arrays']==784 and result['native']['values']==3090494
            if name.startswith('candidate-'):assert all(r['bit_identical'] for r in result['native']['exact_selected_comparisons'])
        else:assert result['public_requests']==20
    return models,census


def prepare():
    assert not BASE.exists();models,census=references();BASE.mkdir();bundle=BASE/'bundle';(bundle/'source').mkdir(parents=True)
    def put(name,content):
        p=bundle/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(content if isinstance(content,bytes) else content.encode())
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    put('source/Program.cs',(TOOLS/'Program.cs.txt').read_bytes())
    put('source/NpySupport.cs',(CENSUS/'bundle/source/NpySupport.cs').read_bytes())
    put('source/global.json',(ROOT/'global.json').read_bytes())
    names=['Lokad.Onnx','Lokad.Onnx.Data','Google.Protobuf','FastBertTokenizer','Lokad.Tokenizers','SixLabors.ImageSharp']
    refs=''.join(f'<Reference Include="{n}"><HintPath>$(FrozenProductDirectory)/{n}.dll</HintPath></Reference>' for n in names)
    put('source/OwnedWeightCounters.csproj','<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><Nullable>enable</Nullable><ImplicitUsings>enable</ImplicitUsings><LangVersion>11.0</LangVersion><UseAppHost>false</UseAppHost></PropertyGroup><ItemGroup>'+refs+'</ItemGroup></Project>')
    for name,source in [('common.py',TOOLS.parent/'managed-phase-amd/remote.py'),('remote.py',TOOLS/'vm.py'),
                        ('README.md',TOOLS/'README.md'),('plan.md',ROOT/'.agent/m76-parakeet-owned-packed-weights-20260925.md'),
                        ('evidence/models-closed.json',MODELS/'closed.json'),('evidence/census-closed.json',CENSUS/'closed.json')]:put(name,source.read_bytes())
    manifest=read(MODELS/'collected/manifests/candidate-parakeet.json');old=read(CENSUS/'bundle/spec.json')
    runtime={role:{n+'.dll':pin(MODELS/'collected/runtimes'/role/(n+'.dll')) for n in names} for role in ['selected','candidate']}
    external={REMOTE_MODELS+'/runtimes/'+role+'/'+n:v for role,files in runtime.items() for n,v in files.items()}
    for info in manifest['models'].values():external[info['path']]={k:info[k] for k in ['bytes','sha256']}
    for case in manifest['cases']:external[REMOTE_MODELS+'/assets/'+case['pcm']['path']]={k:case['pcm'][k] for k in ['bytes','sha256']}
    external[REMOTE_MODELS+'/collection.json']=pin(MODELS/'collected/collection.json')
    assert len(manifest['cases'])==20
    odd=[c['expected']['encoded_frames'] for c in manifest['cases'] if c['expected']['encoded_frames']%2 and c['expected']['encoded_frames']%3]
    assert odd==[167,89,157,83,61,151,169]
    specification=dict(boot=old['boot'],external=external,products=models['identities'],runtime_files=runtime,
        original_runtime=REMOTE_MODELS+'/runtimes',model_collection=REMOTE_MODELS+'/collection.json',jobs=JOBS,
        model_closure=pin(MODELS/'closed.json'),census_closure=pin(CENSUS/'closed.json'),
        failed_release_controls=census['failed_release_controls'],release_admitted=False,diagnostic_only=True,
        weights=old['weights'],cases=manifest['cases'],assets=REMOTE_MODELS+'/assets',model_directory=old['model_directory'],
        feed=old['feed'],build_limits=old['build_limits'],capture_limits=old['capture_limits'],minimum_free=old['minimum_free'],output_limit=128*1024**2,
        prediction=dict(weights=87,bytes_per_weight=16777216,packs_per_corpus=1740,reconstructions_per_corpus=609,
            scratch_reduction=29192355840,copy_increase=10217324544),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',specification)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in bundle.rglob('*'):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    helpers=['weight-ownership-probe/run.py','managed-phase-amd/run.py','managed-phase-amd/remote.py','ort-diagnosis-amd/run.py']
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},helpers={n:pin(TOOLS.parent/n) for n in helpers}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),jobs=JOBS)))


def prepared():
    references();value=read(BASE/'prepared.json')
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json')
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in value['helpers'].items():assert pin(TOOLS.parent/name)==wanted,name
    for name,wanted in read(BASE/'bundle/spec.json')['files'].items():assert pin(BASE/'bundle'/name)==wanted,name


def collect(kind):
    # Retained collector, with dereference=True for the two hardlinked runtime trees.
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
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',dereference=True) as tar:
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
            else:{'observe':prior.observe,'collect':collect}[action](kind)
