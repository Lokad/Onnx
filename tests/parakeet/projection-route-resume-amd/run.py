"""Observe only the unstarted phase using the already qualified binaries."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
ORIGINAL_TOOLS = TOOLS.parent/'projection-route-amd'
sys.path.insert(0, str(ORIGINAL_TOOLS))
loader = importlib.util.spec_from_file_location('original_projection', ORIGINAL_TOOLS/'run.py')
original = importlib.util.module_from_spec(loader); loader.loader.exec_module(original)
pin, read, write, ssh = original.pin, original.read, original.write, original.ssh
ORIGINAL = original.BASE
BASE = ROOT/'artifacts/parakeet-projection-route-resume-amd-20260924'
REMOTE = '/dev/shm/lokad-parakeet-projection-route-resume-20260924'
APP, MANIFEST = original.APP, original.MANIFEST
PRELUDE = original.PRELUDE.replace(original.REMOTE, REMOTE)
transport = original.transport
transport.BASE, transport.REMOTE, transport.PRELUDE = BASE, REMOTE, PRELUDE


def initial():
    original.prepared()
    folder = ORIGINAL/'capture-collected'
    receipt = read(folder/'capture-collection.json'); transfer = read(ORIGINAL/'capture-transfer.json')
    state = read(folder/'capture-state.json')
    assert transfer['passed'] and transfer['archive'] == pin(ORIGINAL/'capture-results.tar.gz')
    assert transfer['collection'] == pin(folder/'capture-collection.json')
    assert receipt['terminal'] and receipt['code'] == 1 and receipt['state'] == pin(folder/'capture-state.json')
    assert state['complete'] and state['code'] == 1 and state['supervisor'] == read(ORIGINAL/'capture-deployment.json')
    for name, wanted in receipt['files'].items(): assert pin(folder/name) == wanted, name
    assert [r['name'] for r in state['runs']] == ['control','phase']
    control, missing = state['runs']; limits = read(ORIGINAL/'bundle/spec.json')['capture_limits']
    assert control['complete'] and control['code'] == 0 and control['samples'] > 0
    assert not missing['complete'] and missing['code'] is None and missing['samples'] == 0 and not missing['members']
    assert not any(k in missing for k in ['owner','ready','seconds'])
    assert missing['preflight']['available'] < limits['available_before']
    assert missing['preflight']['tmpfs'] >= limits['tmpfs_before']
    assert 'assert preflight[' in state['error'] and state['error'].endswith('AssertionError\n')
    assert not (folder/'phase').exists()
    assert not any(p.name.startswith('phase.') for p in (folder/'logs').iterdir())
    return dict(collection=pin(folder/'capture-collection.json'), transfer=pin(ORIGINAL/'capture-transfer.json'),
        archive=pin(ORIGINAL/'capture-results.tar.gz'), state=pin(folder/'capture-state.json'),
        spec=pin(ORIGINAL/'bundle/spec.json'), phase_never_started=True, original_code=1)


def remote_source():
    source = (ORIGINAL_TOOLS/'remote.py').read_text(encoding='utf8')
    edits = [
        ("    review = read(BASE/'build-review.json')", "    product = Path(spec['projection_product'])\n    review = read(product/'build-review.json')"),
        ("pin(BASE/'built.json')", "pin(product/'built.json')"),
        ("read(BASE/'built.json')", "read(product/'built.json')"),
        ("pin(BASE/name)", "pin(product/name)"),
        ("runtime = BASE/'runtime-observed'", "runtime = product/'runtime-observed'"),
        ("for mode in ['control','phase']:", "for mode in ['phase']:")]
    for before, after in edits:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    return source


def prepare():
    assert not BASE.exists(); first = initial()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir()
    contents = {'remote.py': remote_source(), 'common.py': (ORIGINAL/'bundle/common.py').read_bytes(),
        'README.md': (TOOLS/'README.md').read_bytes()}
    for name, content in contents.items():
        with (bundle/name).open('xb') as stream: stream.write(content if isinstance(content,bytes) else content.encode())
    spec = read(ORIGINAL/'bundle/spec.json')
    spec.update(initial=first, projection_product=original.REMOTE)
    for name in ['capture-collection.json','capture-state.json','built.json','build-review.json']:
        spec['external'][original.REMOTE+'/'+name] = pin(ORIGINAL/'capture-collected'/name)
    built = read(ORIGINAL/'build-collected/built.json')
    for name, wanted in built['runtime_files'].items():
        assert pin(ORIGINAL/'build-collected'/name) == wanted
        spec['external'][original.REMOTE+'/'+name] = wanted
    spec['files'] = {p.name: pin(p) for p in bundle.iterdir()}
    write(bundle/'spec.json', spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in bundle.iterdir(): archive.add(path,arcname=path.name,recursive=False)
    for path in TOOLS.glob('*.py'): ast.parse(path.read_text(encoding='utf8'),str(path))
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),initial=first,
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()}, original_prepared=pin(ORIGINAL/'prepared.json')))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),phase_only=True,no_rebuild=True)))


def prepared():
    value = read(BASE/'prepared.json'); assert value['initial'] == initial()
    assert value['original_prepared'] == pin(ORIGINAL/'prepared.json')
    assert value['archive'] == pin(BASE/'payload.tar.gz') and value['spec'] == pin(BASE/'bundle/spec.json')
    for name, wanted in value['tools'].items(): assert pin(TOOLS/name) == wanted, name
    spec = read(BASE/'bundle/spec.json'); before = read(ORIGINAL/'bundle/spec.json')
    assert {k:v for k,v in spec.items() if k not in ['files','external','initial','projection_product']} == {
        k:v for k,v in before.items() if k not in ['files','external']}
    assert all(spec['external'][k] == v for k,v in before['external'].items())
    for name, wanted in spec['files'].items(): assert pin(BASE/'bundle'/name) == wanted, name
    assert (BASE/'bundle/remote.py').read_text(encoding='utf8') == remote_source()


def observe():
    result = ssh(PRELUDE+f'''
from remote import read,live
state=read(base/'capture-state.json') if (base/'capture-state.json').exists() else None
ids=[{read(BASE/'capture-deployment.json')!r}]
if state:ids += [dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
print(json.dumps(dict(live=[i for i in ids if live(i)],complete=state and state['complete'],code=state and state['code'],
 latest=None if not state or not state['runs'] else {{k:state['runs'][-1].get(k) for k in ['name','samples','complete','code']}},
 error=state and state.get('error'),stderr=(base/'capture-supervisor.stderr').read_text()[-3000:])))
''')
    with (BASE/'capture-observations.jsonl').open('a') as stream: stream.write(json.dumps(result)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'prepare': prepare()
    else:
        prepared()
        if action == 'stage': transport.stage()
        elif action == 'observe': observe()
        else: dict(launch=transport.launch,collect=transport.collect)[action]('capture')
