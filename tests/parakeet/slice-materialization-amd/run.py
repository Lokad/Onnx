"""Prepare the layout observer and reuse the bounded, frozen phase transport."""
import ast
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
BASE = ROOT/'artifacts/parakeet-slice-layout-amd-20260924'
REMOTE = '/dev/shm/lokad-parakeet-slice-layout-20260924'
PRIOR, APP = transport.PRIOR, transport.APP
REMOTE_PRIOR, REMOTE_APP = transport.REMOTE_PRIOR, transport.REMOTE_APP
pin, read, write, ssh = transport.pin, transport.read, transport.write, transport.ssh
PRELUDE = transport.PRELUDE.replace(transport.REMOTE, REMOTE)
transport.BASE, transport.REMOTE, transport.PRELUDE = BASE, REMOTE, PRELUDE


def prepare():
    assert not BASE.exists()
    selected_path = ROOT/'artifacts/parakeet-wide-entry-first-use-source-v2-20260923/prepared.json'
    selected = read(selected_path)
    for name, wanted in selected['source'].items(): assert pin(ROOT/name) == wanted, name
    phase = ROOT/'artifacts/parakeet-managed-phase-amd-20260924'
    assert read(phase/'closed.json')['passed']
    assert read(PRIOR/'closed.json')['passed'] and read(APP/'closed.json')['passed']
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir()
    def put(name, content):
        path = bundle/name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream: stream.write(content if isinstance(content, bytes) else content.encode())
    originals = {}
    for name, identity in selected['source'].items():
        if not (name.startswith('src/Lokad.Onnx/') or name in ['global.json','LICENSE.txt','icon.png','README.md','CHANGELOG.md']): continue
        content = (ROOT/name).read_bytes(); originals[name] = identity
        if name == 'src/Lokad.Onnx/TensorSlice.cs':
            text = content.decode()
            needle = 'public override Tensor<T> Reshape(ReadOnlySpan<int> dimensions) => Clone().Reshape(dimensions);'
            assert text.count(needle) == 1
            changed = 'public override Tensor<T> Reshape(ReadOnlySpan<int> dimensions) { SliceLayoutProbe.Observe(this, dimensions); return Clone().Reshape(dimensions); }'
            content = text.replace(needle, changed)
        put('source/'+name, content)
    put('source/src/Lokad.Onnx/SliceLayoutProbe.cs', (TOOLS/'LayoutProbe.cs.txt').read_bytes())
    for name in ['Program.cs','Diagnostic.cs','NpySupport.cs','SampledAudio.csproj']:
        original = (PRIOR/'bundle/source'/name).read_text(encoding='utf8'); content = original
        if name == 'Program.cs':
            edits = [('=="672e5f303b011e27bb23097a49252c38ddee334938c75895e3c2341df0f3be35"',
                      '==Environment.GetEnvironmentVariable("PARAKEET_LAYOUT_CORE_SHA")'),
                     ('long setupStart=Stopwatch.GetTimestamp();','LayoutConsumer.Initialize();\nlong setupStart=Stopwatch.GetTimestamp();'),
                     ('    JsonElement normalized=actual switch','    LayoutConsumer.Save(output,records.Count,c.Name,pass);\n    JsonElement normalized=actual switch')]
            for before, after in edits:
                assert content.count(before) == 1; content = content.replace(before, after)
            restored = content
            for before, after in reversed(edits): restored = restored.replace(after,before)
            assert restored == original
        if name == 'SampledAudio.csproj':
            content = content.replace('<Compile Include="Program.cs"/>','<Compile Include="Program.cs"/><Compile Include="LayoutConsumer.cs"/>')
        put('consumer-source/'+name, content)
    put('consumer-source/LayoutConsumer.cs', (TOOLS/'LayoutConsumer.cs.txt').read_bytes())
    bridge = (TOOLS.parent/'selected-profile-build-amd/Bridge.cs.txt').read_text(encoding='utf8')
    assert bridge.count('new[] { "SampledAudio.dll" }') == 1
    put('bridge-source/Program.cs', bridge.replace('new[] { "SampledAudio.dll" }','new[] { "SampledAudio.dll", "Lokad.Onnx.dll" }'))
    put('bridge-source/Bridge.csproj', (TOOLS.parent/'selected-profile-build-amd/Bridge.csproj').read_bytes())
    for folder in ['consumer-source','bridge-source']: put(folder+'/global.json', (ROOT/'global.json').read_bytes())
    put('common.py', (PHASE_TOOLS/'remote.py').read_bytes())
    for name in ['remote.py','README.md']:
        if name.endswith('.py'): ast.parse((TOOLS/name).read_text())
        put(name, (TOOLS/name).read_bytes())
    external = {REMOTE_PRIOR+'/'+p.name: pin(p) for p in (PRIOR/'collected/runtime').iterdir() if p.is_file()}
    manifest = read(APP/'collected/manifests/current-parakeet.json')
    for value in manifest['models'].values(): external[value['path']] = {k:value[k] for k in ['bytes','sha256']}
    for value in [manifest['reference'], *[c['pcm'] for c in manifest['cases']]]:
        external[REMOTE_APP+'/assets/'+value['path']] = {k:value[k] for k in ['bytes','sha256']}
    for name in ['manifests/current-parakeet.json','runtime/protocol.py','runtime/campaign_processes.py']:
        external[REMOTE_APP+'/'+name] = pin(APP/'collected'/name)
    spec = dict(boot=1789634288.0, prior=REMOTE_PRIOR, app=REMOTE_APP, external=external,
        original_source=originals, source_receipt=pin(selected_path), previous_closure=pin(phase/'closed.json'),
        original_consumer=pin(PRIOR/'collected/runtime/SampledAudio.dll'),
        core=pin(PRIOR/'collected/runtime/Lokad.Onnx.dll'), data=pin(PRIOR/'collected/runtime/Lokad.Onnx.Data.dll'),
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=180),
        capture_limits=dict(available_before=11*1024**3,tmpfs_before=2*1024**3,rss=12*1024**3,seconds=900),
        minimum_free=1024**3,output_limit=512*1024**2,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in bundle.rglob('*'):
            if path.is_file(): archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},transport=pin(PHASE_TOOLS/'run.py')))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),source_files=len(originals))))


def observe(kind):
    assert not (BASE/'closed.json').exists()
    result = ssh(PRELUDE+f'''
from remote import read,live
path=base/({kind!r}+'-state.json');state=read(path) if path.exists() else None
ids=[{read(BASE/(kind+'-deployment.json'))!r}]
if state:ids += [dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
print(json.dumps(dict(live=[i for i in ids if live(i)],complete=state and state['complete'],code=state and state['code'],
 latest=None if not state or not state['runs'] else {{k:state['runs'][-1].get(k) for k in ['name','samples','complete','code']}},
 error=state and state.get('error'),stderr=(base/({kind!r}+'-supervisor.stderr')).read_text()[-5000:])))
''')
    with (BASE/(kind+'-observations.jsonl')).open('a') as stream: stream.write(json.dumps(result)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'prepare': prepare()
    elif action == 'stage': transport.stage()
    elif action == 'observe': observe(sys.argv[2])
    else: dict(launch=transport.launch,collect=transport.collect)[action](sys.argv[2])
