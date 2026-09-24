"""Freeze an observation-only Data build after M66 release qualification."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
PHASE_TOOLS = TOOLS.parent/'managed-phase-amd'
loader = importlib.util.spec_from_file_location('phase_transport', PHASE_TOOLS/'run.py')
transport = importlib.util.module_from_spec(loader); loader.loader.exec_module(transport)
pin, read, write, ssh = transport.pin, transport.read, transport.write, transport.ssh
BASE = ROOT/'artifacts/parakeet-masking-padding-layout-amd-20260924'
REMOTE = '/dev/shm/lokad-parakeet-masking-padding-layout-20260924'
PRIOR = transport.PRIOR
APP = ROOT/'artifacts/parakeet-validated-composition-app-amd-20260924'
REMOTE_APP = '/dev/shm/lokad-parakeet-validated-composition-app-20260924'
SOURCE = ROOT/'artifacts/parakeet-validated-composition-source-v2-20260924'
PRODUCT = ROOT/'artifacts/parakeet-validated-composition-build-amd-v2-20260924/build-collected/source/runtime-observed'
RELEASE = ROOT/'artifacts/parakeet-validated-composition-root-amd-20260924'
PHASE = ROOT/'artifacts/parakeet-managed-phase-amd-20260924'
MANIFEST = 'manifests/candidate-parakeet.json'
CORE_SHA = '37c243756bbe5e5d563e79627a0a4d20ff027598801849da9606549c7ac60286'
DATA_SHA = 'cc37b19eb41cf728061c35bcdb7e06a6ab4d370bfd4c47555eec86c86b2ef6d6'
PRELUDE = transport.PRELUDE.replace(transport.REMOTE, REMOTE)
transport.BASE, transport.REMOTE, transport.PRELUDE = BASE, REMOTE, PRELUDE


def release_closed():
    closure = read(RELEASE/'closed.json')
    assert closure['passed'] and closure['analysis'] == pin(RELEASE/'analysis.json')
    for name, wanted in closure['files'].items(): assert pin(RELEASE/name) == wanted, name
    analysis = read(RELEASE/'analysis.json')
    assert analysis['passed'] and analysis['root_source_verified']
    assert analysis['measured'] == {name: pin(PRODUCT/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
    return pin(RELEASE/'closed.json')


def consumer_edits():
    return [
        ('=="672e5f303b011e27bb23097a49252c38ddee334938c75895e3c2341df0f3be35"', '=="'+CORE_SHA+'"'),
        ('=="065b7a7f28561a37174f9d38ac13ef77bc59c010702e2b64200e9542376eb4c5"',
         '==Environment.GetEnvironmentVariable("PARAKEET_MASKING_DATA_SHA")'),
        ('long setupStart=Stopwatch.GetTimestamp();', 'MaskingConsumer.Initialize();\nlong setupStart=Stopwatch.GetTimestamp();'),
        ('    JsonElement normalized=actual switch',
         '    MaskingConsumer.Save(output,records.Count,c.Name,pass);\n    JsonElement normalized=actual switch')]


def source_files():
    """Build the exact source inputs in memory; inspect performs no remote work."""
    receipt = read(SOURCE/'prepared.json')
    assert receipt['passed'] and pin(SOURCE/'prepared.json')['sha256'] == 'af68e6c2c28f5794f3eece39bd9e78fc6809e2ca8925a5616ff2956c405061c4'
    for name, wanted in receipt['source'].items(): assert pin(SOURCE/'source'/name) == wanted, name
    assert pin(PRODUCT/'Lokad.Onnx.dll')['sha256'] == CORE_SHA
    assert pin(PRODUCT/'Lokad.Onnx.Data.dll')['sha256'] == DATA_SHA
    files = {}
    def put(name, content):
        assert name not in files
        files[name] = content if isinstance(content, bytes) else content.encode()
    for path in sorted((SOURCE/'source/src/Lokad.Onnx.Data').glob('*.cs')):
        content = path.read_text(encoding='utf8')
        if path.name == 'ParakeetTranscriber.cs':
            needle = '    static IReadOnlyDictionary<string, ITensor> Execute(GraphExecution context, Dictionary<string, ITensor> feeds)\n    {\n'
            assert content.count(needle) == 1
            addition = '        using var observation = ParakeetMaskingProbe.Enter(context);\n'
            changed = content.replace(needle, needle+addition)
            assert changed.replace(addition,'') == content
            content = changed
        put('data-source/'+path.name, content)
    put('data-source/MaskingProbe.cs', (TOOLS/'MaskingProbe.cs.txt').read_bytes())
    references = ''.join(f'<Reference Include="{name}"><HintPath>$(FrozenProductDirectory)/{name}.dll</HintPath></Reference>'
        for name in ['Lokad.Onnx','Google.Protobuf','FastBertTokenizer','Lokad.Tokenizers','SixLabors.ImageSharp'])
    put('data-source/ObserverData.csproj','<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net10.0</TargetFramework><Nullable>enable</Nullable><LangVersion>11.0</LangVersion><AssemblyName>Lokad.Onnx.Data</AssemblyName></PropertyGroup><ItemGroup>'+references+'</ItemGroup></Project>')
    for name in ['Program.cs','Diagnostic.cs','NpySupport.cs','SampledAudio.csproj']:
        original = (PRIOR/'bundle/source'/name).read_text(encoding='utf8'); content = original
        if name == 'Program.cs':
            for before, after in consumer_edits():
                assert content.count(before) == 1; content = content.replace(before, after)
            restored = content
            for before, after in reversed(consumer_edits()): restored = restored.replace(after, before)
            assert restored == original
        if name == 'SampledAudio.csproj':
            needle = '<Compile Include="Program.cs"/>'; assert content.count(needle) == 1
            content = content.replace(needle, needle+'<Compile Include="MaskingConsumer.cs"/>')
        put('consumer-source/'+name, content)
    put('consumer-source/MaskingConsumer.cs', (TOOLS/'MaskingConsumer.cs.txt').read_bytes())
    bridge = (TOOLS.parent/'selected-profile-build-amd/Bridge.cs.txt').read_text(encoding='utf8')
    assert bridge.count('new[] { "SampledAudio.dll" }') == 1
    put('bridge-source/Program.cs', bridge.replace('new[] { "SampledAudio.dll" }','new[] { "SampledAudio.dll", "Lokad.Onnx.Data.dll" }'))
    put('bridge-source/Bridge.csproj', (TOOLS.parent/'selected-profile-build-amd/Bridge.csproj').read_bytes())
    for folder in ['data-source','consumer-source','bridge-source']:
        put(folder+'/global.json', (SOURCE/'source/global.json').read_bytes())
    for original in sorted((PRIOR/'collected/runtime').iterdir()):
        if original.is_file():
            path = PRODUCT/original.name if original.name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll'] else original
            put('reference-runtime/'+original.name, path.read_bytes())
    put('evidence/graphs.json', (PHASE/'capture-collected/phase/graphs.json').read_bytes())
    put('common.py', (PHASE_TOOLS/'remote.py').read_bytes())
    for name in ['remote.py','README.md']:
        if name.endswith('.py'): ast.parse((TOOLS/name).read_text())
        put(name, (TOOLS/name).read_bytes())
    return files


def prepare():
    assert not BASE.exists()
    release = release_closed()
    assert read(PRIOR/'closed.json')['passed'] and read(APP/'closed.json')['passed'] and read(PHASE/'closed.json')['passed']
    files = source_files(); BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir()
    for name, content in files.items():
        path = bundle/name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream: stream.write(content)
    external = {}; manifest = read(APP/'collected'/MANIFEST)
    for value in manifest['models'].values(): external[value['path']] = {k:value[k] for k in ['bytes','sha256']}
    for value in [manifest['reference'], *[c['pcm'] for c in manifest['cases']]]:
        external[REMOTE_APP+'/assets/'+value['path']] = {k:value[k] for k in ['bytes','sha256']}
    for name in [MANIFEST,'runtime/protocol.py','runtime/campaign_processes.py']:
        external[REMOTE_APP+'/'+name] = pin(APP/'collected'/name)
    spec = dict(boot=1789634288.0, prior=REMOTE+'/reference-runtime', app=REMOTE_APP, manifest=MANIFEST, external=external,
        release_closure=release, source_receipt=pin(SOURCE/'prepared.json'), application_closure=pin(APP/'closed.json'),
        original_consumer=pin(PRIOR/'collected/runtime/SampledAudio.dll'),
        core=pin(PRODUCT/'Lokad.Onnx.dll'), data=pin(PRODUCT/'Lokad.Onnx.Data.dll'),
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=180),
        capture_limits=dict(available_before=11*1024**3,tmpfs_before=2*1024**3,rss=12*1024**3,seconds=900),
        minimum_free=1024**3,output_limit=512*1024**2,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},transport=pin(PHASE_TOOLS/'run.py')))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),files=len(files))))


def prepared():
    receipt = read(BASE/'prepared.json')
    assert receipt['archive'] == pin(BASE/'payload.tar.gz') and receipt['spec'] == pin(BASE/'bundle/spec.json')
    assert release_closed() == read(BASE/'bundle/spec.json')['release_closure']
    assert receipt['transport'] == pin(PHASE_TOOLS/'run.py')
    for name, wanted in receipt['tools'].items(): assert pin(TOOLS/name) == wanted, name
    for name, wanted in read(BASE/'bundle/spec.json')['files'].items(): assert pin(BASE/'bundle'/name) == wanted, name


def observe(kind):
    result = ssh(PRELUDE+f'''
from remote import read,live
path=base/({kind!r}+'-state.json');state=read(path) if path.exists() else None
ids=[{read(BASE/(kind+'-deployment.json'))!r}]
if state:ids += [dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
print(json.dumps(dict(live=[i for i in ids if live(i)],complete=state and state['complete'],code=state and state['code'],
 latest=None if not state or not state['runs'] else {{k:state['runs'][-1].get(k) for k in ['name','samples','complete','code']}},
 error=state and state.get('error'),stderr=(base/({kind!r}+'-supervisor.stderr')).read_text()[-3000:])))
''')
    with (BASE/(kind+'-observations.jsonl')).open('a') as stream: stream.write(json.dumps(result)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'inspect':
        files = source_files()
        print(json.dumps(dict(source_verified=True,files=len(files),bytes=sum(map(len,files.values())),
            sources={k:hashlib.sha256(v).hexdigest() for k,v in files.items() if k.endswith('.cs')},
            release_closed=(RELEASE/'closed.json').exists(),no_build_or_inference=True)))
    elif action == 'prepare': prepare()
    elif action == 'observe': observe(sys.argv[2])
    elif action == 'collect': transport.collect(sys.argv[2])
    else:
        prepared()
        if action == 'stage': transport.stage()
        elif action == 'launch':
            if sys.argv[2] == 'capture':
                assert read(BASE/'build-review.json')['passed'] and read(BASE/'build-review-transferred.json')['passed']
            transport.launch(sys.argv[2])
        else: raise ValueError(action)
