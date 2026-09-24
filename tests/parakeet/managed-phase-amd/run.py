"""Prepare an isolated Data observer; keep the qualified Core and application flow."""
import ast
import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-managed-phase-amd-20260924'
REMOTE = '/dev/shm/lokad-parakeet-managed-phase-20260924'
PRIOR = ROOT/'artifacts/parakeet-selected-profile-build-amd-20260924'
APP = ROOT/'artifacts/parakeet-prepared-recurrence-app-amd-20260924'
REMOTE_PRIOR = '/dev/shm/lokad-parakeet-selected-profile-build-20260924/runtime'
REMOTE_APP = '/dev/shm/lokad-parakeet-prepared-recurrence-app-20260924'
spec = importlib.util.spec_from_file_location('native_transport', TOOLS.parent/'ort-diagnosis-amd/run.py')
transport = importlib.util.module_from_spec(spec); spec.loader.exec_module(transport)
pin, read, write, ssh, SSH = transport.pin, transport.read, transport.write, transport.ssh, transport.SSH
PRELUDE = transport.PRELUDE+f"\nbase=Path({REMOTE!r})\nsys.path.insert(0,str(base))\n"


def prepare():
    assert not BASE.exists()
    selected = read(ROOT/'artifacts/parakeet-wide-entry-first-use-source-v2-20260923/prepared.json')
    for name, wanted in selected['source'].items(): assert pin(ROOT/name) == wanted, name
    assert read(PRIOR/'closed.json')['passed'] and read(APP/'closed.json')['passed']
    BASE.mkdir(); bundle=BASE/'bundle'; bundle.mkdir()
    def put(name, content):
        path=bundle/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(content if isinstance(content,bytes) else content.encode())
    original={}
    for path in sorted((ROOT/'src/Lokad.Onnx.Data').glob('*.cs')):
        original[path.name]=pin(path);content=path.read_text(encoding='utf8')
        if path.name=='ParakeetTranscriber.cs':
            needle='    static IReadOnlyDictionary<string, ITensor> Execute(GraphExecution context, Dictionary<string, ITensor> feeds)\n    {\n'
            assert content.count(needle)==1
            changed=content.replace(needle,needle+'        using var observation = ParakeetPhaseProbe.Enter(context);\n')
            assert changed.replace('        using var observation = ParakeetPhaseProbe.Enter(context);\n','')==content
            content=changed
        put('data-source/'+path.name,content)
    put('data-source/PhaseProbe.cs',(TOOLS/'PhaseProbe.cs.txt').read_bytes())
    refs=['Lokad.Onnx','Google.Protobuf','FastBertTokenizer','Lokad.Tokenizers','SixLabors.ImageSharp']
    references=''.join(f'<Reference Include="{name}"><HintPath>$(FrozenProductDirectory)/{name}.dll</HintPath></Reference>' for name in refs)
    put('data-source/ObserverData.csproj','<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net10.0</TargetFramework><Nullable>enable</Nullable><LangVersion>11.0</LangVersion><AssemblyName>Lokad.Onnx.Data</AssemblyName></PropertyGroup><ItemGroup>'+references+'</ItemGroup></Project>')
    for name in ['Program.cs','Diagnostic.cs','NpySupport.cs','SampledAudio.csproj']:
        content=(PRIOR/'bundle/source'/name).read_text(encoding='utf8')
        if name=='Program.cs':
            edits=[('=="065b7a7f28561a37174f9d38ac13ef77bc59c010702e2b64200e9542376eb4c5"',
                    '==Environment.GetEnvironmentVariable("PARAKEET_PHASE_DATA_SHA")'),
                   ('long setupStart=Stopwatch.GetTimestamp();','PhaseConsumer.Initialize();\nlong setupStart=Stopwatch.GetTimestamp();'),
                   ('    JsonElement normalized=actual switch','    PhaseConsumer.Save(output,records.Count,c.Name,pass);\n    JsonElement normalized=actual switch')]
            for before,after in edits:
                assert content.count(before)==1;content=content.replace(before,after)
            restored=content
            for before,after in reversed(edits):restored=restored.replace(after,before)
            assert restored==(PRIOR/'bundle/source'/name).read_text(encoding='utf8')
        if name=='SampledAudio.csproj':content=content.replace('<Compile Include="Program.cs"/>','<Compile Include="Program.cs"/><Compile Include="PhaseConsumer.cs"/>')
        put('consumer-source/'+name,content)
    put('consumer-source/PhaseConsumer.cs',(TOOLS/'PhaseConsumer.cs.txt').read_bytes())
    bridge=(TOOLS.parent/'selected-profile-build-amd/Bridge.cs.txt').read_text(encoding='utf8')
    assert bridge.count('new[] { "SampledAudio.dll" }')==1
    put('bridge-source/Program.cs',bridge.replace('new[] { "SampledAudio.dll" }','new[] { "SampledAudio.dll", "Lokad.Onnx.Data.dll" }'))
    put('bridge-source/Bridge.csproj',(TOOLS.parent/'selected-profile-build-amd/Bridge.csproj').read_bytes())
    for folder in ['data-source','consumer-source','bridge-source']:put(folder+'/global.json',(ROOT/'global.json').read_bytes())
    for name in ['remote.py','README.md']:
        if name.endswith('.py'):ast.parse((TOOLS/name).read_text())
        put(name,(TOOLS/name).read_bytes())
    external={str(Path(REMOTE_PRIOR)/p.name).replace('\\','/'):pin(p) for p in (PRIOR/'collected/runtime').iterdir() if p.is_file()}
    manifest=read(APP/'collected/manifests/current-parakeet.json')
    for value in manifest['models'].values():external[value['path']]={k:value[k] for k in ['bytes','sha256']}
    for value in [manifest['reference'],*[c['pcm'] for c in manifest['cases']]]:
        external[REMOTE_APP+'/assets/'+value['path']]={k:value[k] for k in ['bytes','sha256']}
    for name in ['manifests/current-parakeet.json','runtime/protocol.py','runtime/campaign_processes.py']:
        external[REMOTE_APP+'/'+name]=pin(APP/'collected'/name)
    specification=dict(boot=1789634288.0,prior=REMOTE_PRIOR,app=REMOTE_APP,external=external,
        data_source=original,source_receipt=pin(ROOT/'artifacts/parakeet-wide-entry-first-use-source-v2-20260923/prepared.json'),
        original_consumer=pin(PRIOR/'collected/runtime/SampledAudio.dll'),
        core=pin(PRIOR/'collected/runtime/Lokad.Onnx.dll'),data=pin(PRIOR/'collected/runtime/Lokad.Onnx.Data.dll'),
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=180),
        capture_limits=dict(available_before=11*1024**3,tmpfs_before=2*1024**3,rss=12*1024**3,seconds=900),
        minimum_free=1024**3,output_limit=512*1024**2,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',specification)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in bundle.rglob('*'):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),data_source_files=len(original))))


def stage():
    receipt=read(BASE/'prepared.json');assert not (BASE/'staged.json').exists()
    result=ssh(PRELUDE+'''
assert not base.exists()
assert psutil.boot_time()==1789634288.0
assert psutil.virtual_memory().available>=2*1024**3 and psutil.disk_usage('/dev/shm').free>=1024**3
base.mkdir()
print(json.dumps(dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free)))
''')
    write(BASE/'stage-started.json',result)
    subprocess.run(['scp',*SSH[1:-1],str(BASE/'payload.tar.gz'),SSH[-1]+':'+REMOTE+'/transfer.tar.gz'],check=True,timeout=90,creationflags=subprocess.CREATE_NO_WINDOW)
    result=ssh(PRELUDE+f'''
with (base/'transfer.tar.gz').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=={receipt['archive']['sha256']!r}
with tarfile.open(base/'transfer.tar.gz') as tar:tar.extractall(base,filter='data')
from remote import verify,idle,pin
verify();idle()
assert pin(base/'spec.json')=={receipt['spec']!r}
print(json.dumps(dict(passed=True,spec=pin(base/'spec.json'))))
''')
    write(BASE/'staged.json',result);print(json.dumps(result))


def launch(kind):
    assert kind in ['build','capture'] and not (BASE/(kind+'-deployment.json')).exists()
    result=ssh(PRELUDE+f'''
from remote import verify,idle,pin
verify();idle();kind={kind!r}
assert not (base/(kind+'-state.json')).exists()
env=dict(os.environ,PYTHONPATH={transport.SITE!r},PYTHONDONTWRITEBYTECODE='1')
env.pop('PYTHONOPTIMIZE',None)
with (base/(kind+'-supervisor.stdout')).open('x') as out,(base/(kind+'-supervisor.stderr')).open('x') as err:
 p=subprocess.Popen([sys.executable,'-B',str(base/'remote.py'),kind],cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 value=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time())
print(json.dumps(value))
''')
    write(BASE/(kind+'-deployment.json'),result);print(json.dumps(result))


def observe(kind):
    result=ssh(PRELUDE+f'''
from remote import read,live
path=base/({kind!r}+'-state.json');state=read(path) if path.exists() else None
ids=[{read(BASE/(kind+'-deployment.json'))!r}]
if state:ids += [dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
print(json.dumps(dict(state=state,live=[i for i in ids if live(i)],stderr=(base/({kind!r}+'-supervisor.stderr')).read_text()[-5000:])))
''')
    with (BASE/(kind+'-observations.jsonl')).open('a') as stream:stream.write(json.dumps(result)+'\n')
    state=result['state'];print(json.dumps(dict(live=result['live'],complete=state and state['complete'],code=state and state['code'],latest=state and state['runs'][-1],error=state and state.get('error'),stderr=result['stderr'])))


def collect(kind):
    target=BASE/(kind+'-collected');assert not target.exists()
    script=PRELUDE+f'''
from remote import read,live,pin,verify
kind={kind!r};state=read(base/(kind+'-state.json'))
assert state['complete'] and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
verify()
paths=[p for p in base.rglob('*') if p.is_file() and ('logs' in p.parts or p.parent==base or (kind=='build' and p.parent.name in ['inventory','runtime-control','runtime-observed']) or (kind=='capture' and p.parent.name in ['control','phase','wall'])) and p.name!='transfer.tar.gz']
files={{p.relative_to(base).as_posix():pin(p) for p in paths}}
(base/(kind+'-collection.json')).write_text(json.dumps(dict(files=files,state=pin(base/(kind+'-state.json')),terminal=True,code=state['code'])))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name in [*files,kind+'-collection.json']:tar.add(base/name,arcname=name,recursive=False)
'''
    archive=BASE/(kind+'-results.tar.gz')
    with archive.open('xb') as out,(BASE/(kind+'-collection.stderr')).open('x') as err:
        result=subprocess.run(SSH+['python3','-B','-'],input=script.encode(),stdout=out,stderr=err,timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode==0
    target.mkdir()
    with tarfile.open(archive) as tar:tar.extractall(target,filter='data')
    value=read(target/(kind+'-collection.json'))
    for name,wanted in value['files'].items():assert pin(target/name)==wanted,name
    write(BASE/(kind+'-transfer.json'),dict(passed=True,archive=pin(archive),collection=pin(target/(kind+'-collection.json'))))
    print(json.dumps(dict(code=value['code'],files=len(value['files']),archive=pin(archive))))


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='stage':stage()
    else:dict(launch=launch,observe=observe,collect=collect)[action](sys.argv[2])
