"""Freeze and transport the single slice-copy candidate and tensor contracts."""
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
LAYOUT_TOOLS = TOOLS.parent/'slice-materialization-amd'
loader = importlib.util.spec_from_file_location('layout_transport',LAYOUT_TOOLS/'run.py')
layout = importlib.util.module_from_spec(loader); loader.loader.exec_module(layout)
transport = layout.transport
BASE = ROOT/'artifacts/parakeet-slice-materialization-build-amd-20260924'
REMOTE = '/dev/shm/lokad-parakeet-slice-materialization-build-20260924'
SOURCE = ROOT/'artifacts/parakeet-slice-materialization-source-20260924'
PRIOR = layout.PRIOR
pin, read, write, ssh = layout.pin, layout.read, layout.write, layout.ssh
PRELUDE = layout.PRELUDE.replace(layout.REMOTE,REMOTE)
transport.BASE, transport.REMOTE, transport.PRELUDE = BASE, REMOTE, PRELUDE


def prepare():
    source = read(SOURCE/'prepared.json'); assert source['passed'] and not BASE.exists()
    for name,wanted in source['source'].items(): assert pin(SOURCE/'source'/name) == wanted, name
    for name,wanted in source['before'].items(): assert pin(ROOT/name) == wanted, name
    assert source['layout_closure'] == pin(layout.BASE/'closed.json') and read(layout.BASE/'closed.json')['passed']
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir()
    def put(name,data):
        path=bundle/name; path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream: stream.write(data if isinstance(data,bytes) else data.encode())
    for name in source['source']: put('source/'+name,(SOURCE/'source'/name).read_bytes())
    put('source/tests/Lokad.Onnx.Tensors.Tests/SliceCandidateIdentityTests.cs',(TOOLS/'IdentityTests.cs.txt').read_bytes())
    bridge = (TOOLS.parent/'selected-profile-build-amd/Bridge.cs.txt').read_text(encoding='utf8')
    assert bridge.count('new[] { "SampledAudio.dll" }') == 1
    put('bridge-source/Program.cs',bridge.replace('new[] { "SampledAudio.dll" }','new[] { "Lokad.Onnx.dll" }'))
    put('bridge-source/Bridge.csproj',(TOOLS.parent/'selected-profile-build-amd/Bridge.csproj').read_bytes())
    put('bridge-source/global.json',(ROOT/'global.json').read_bytes())
    put('common.py',(TOOLS.parent/'managed-phase-amd/remote.py').read_bytes())
    for name in ['remote.py','README.md']: put(name,(TOOLS/name).read_bytes())
    external = {layout.REMOTE_PRIOR+'/'+p.name:pin(p) for p in (PRIOR/'collected/runtime').iterdir() if p.is_file()}
    specification = dict(boot=1789634288.0,prior=layout.REMOTE_PRIOR,external=external,
        source_prepared=pin(SOURCE/'prepared.json'),layout_closure=source['layout_closure'],
        core=pin(PRIOR/'collected/runtime/Lokad.Onnx.dll'),
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=180),
        capture_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=300),
        minimum_free=1024**3,output_limit=512*1024**2,
        expected_tests=369,expected_skipped=0,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',specification)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in bundle.rglob('*'):
            if path.is_file(): archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},transport=pin(TOOLS.parent/'managed-phase-amd/run.py')))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),source_files=len(source['source']))))


def observe(kind):
    assert not (BASE/'closed.json').exists()
    result=ssh(PRELUDE+f'''
from remote import read,live
path=base/({kind!r}+'-state.json');state=read(path) if path.exists() else None
ids=[{read(BASE/(kind+'-deployment.json'))!r}]
if state:ids += [dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
print(json.dumps(dict(live=[i for i in ids if live(i)],complete=state and state['complete'],code=state and state['code'],
 latest=None if not state or not state['runs'] else {{k:state['runs'][-1].get(k) for k in ['name','samples','complete','code']}},
 error=state and state.get('error'),stderr=(base/({kind!r}+'-supervisor.stderr')).read_text()[-5000:])))
''')
    with (BASE/(kind+'-observations.jsonl')).open('a') as stream:stream.write(json.dumps(result)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='stage':transport.stage()
    elif action=='observe':observe(sys.argv[2])
    else:dict(launch=transport.launch,collect=transport.collect)[action](sys.argv[2])
