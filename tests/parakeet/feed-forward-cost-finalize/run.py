"""Finish observer nullability and reuse separate Core and Data inspectors."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent / 'feed-forward-cost-diagnostic'))
loader = importlib.util.spec_from_file_location('frozen_cost_transport', TOOLS.parent / 'feed-forward-cost-diagnostic/run.py')
original = importlib.util.module_from_spec(loader)
loader.loader.exec_module(original)

BASE = ROOT / 'artifacts/parakeet-feed-forward-cost-observer-finalize-20260925'
KIND = 'observer-finalize'
FIRST = ROOT / 'artifacts/parakeet-feed-forward-cost-observer-recovery-20260925'
OBSERVER = ROOT / 'artifacts/parakeet-managed-phase-amd-20260924'
pin, read, write, ssh = original.pin, original.read, original.write, original.ssh
PRELUDE = original.PRELUDE


def prepare():
    original.prepared()
    assert not BASE.exists()
    before = FIRST / 'collected'
    receipt = read(before / 'observer-recovery-collection.json')
    assert receipt['terminal'] and receipt['code'] == 1
    for name, wanted in receipt['files'].items(): assert pin(before / name) == wanted, name
    state = read(before / 'observer-recovery-state.json')
    assert state['complete'] and state['code'] == 1
    assert [r['name'] for r in state['runs']] == ['observer-data-restore', 'observer-data-build', 'observer-inventory']
    assert [r['code'] for r in state['runs']] == [0, 0, -6]
    assert 'Unexpected Data method change.' in (before / 'logs/observer-inventory.stderr').read_text()
    assert 'warning CS8604' in (before / 'logs/observer-data-build.stdout').read_text()
    origin = before / 'observer-recovery-source/PhaseProbe.cs'
    old = origin.read_bytes()
    needle = b'Require(wall.All(n => graphStart'
    replacement = b'Require(wall!.All(n => graphStart'
    assert old.count(needle) == 1
    corrected = old.replace(needle, replacement)
    assert corrected.replace(replacement, needle) == old
    core_path = 'core-source/src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll'
    inspector_source = OBSERVER / 'bundle/bridge-source/Program.cs'
    core_inspector_source = ROOT / 'tests/parakeet/first-use-kernels-build-amd/Bridge.cs.txt'
    prefix = 'var observations = new List<object>();'
    shared = inspector_source.read_text(encoding='utf8').split(prefix, 1)[0]
    assert shared == core_inspector_source.read_text(encoding='utf8').split(prefix, 1)[0]
    old_proof = read(OBSERVER / 'build-review.json')
    assert old_proof['passed'] and old_proof['inventory'] == pin(OBSERVER / 'build-collected/inventory/instructions.json')
    assert old_proof['spec'] == pin(OBSERVER / 'bundle/spec.json')
    assert read(OBSERVER / 'bundle/spec.json')['files']['bridge-source/Program.cs'] == pin(inspector_source)
    assert 'new[] { "SampledAudio.dll", "Lokad.Onnx.Data.dll" }' in inspector_source.read_text(encoding='utf8')
    remote = ssh(PRELUDE + f'''
import base64
from remote import verify,read,live,pin,idle
verify();idle();state=read(base/'observer-recovery-state.json')
assert pin(base/'observer-recovery-state.json')=={pin(before / 'observer-recovery-state.json')!r}
assert state['complete'] and state['code']==1 and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
old=Path('/dev/shm/lokad-parakeet-managed-phase-20260924')
assert pin(old/'bridge-source/Program.cs')=={pin(inspector_source)!r}
files={{}}
for name in ['Bridge.dll','Bridge.deps.json','Bridge.runtimeconfig.json']:
 p=old/'bridge-source/bin/Release/net10.0'/name
 files[name]=dict(identity=pin(p),content=base64.b64encode(p.read_bytes()).decode())
print(json.dumps(dict(core=pin(base/{core_path!r}),inspector=files)))
''')
    assert remote['core'] == read(FIRST / 'bundle/observer-recovery-spec.json')['core']
    assert remote['inspector']['Bridge.dll']['identity']['sha256'] == 'd31414683e281ae937f9d9dfac6da52764dfcdf49603e073a36766b146d24dc8'
    BASE.mkdir(); bundle = BASE / 'bundle'; bundle.mkdir()
    def put(name, data):
        path = bundle / name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream: stream.write(data)
    for path in (before / 'observer-recovery-source').iterdir():
        assert path.is_file()
        put('observer-finalize-source/' + path.name, corrected if path.name == 'PhaseProbe.cs' else path.read_bytes())
    put('observer-finalize-source.cs', corrected)
    put('observer-finalize.py', (TOOLS / 'remote.py').read_bytes())
    for name, row in remote['inspector'].items():
        put('observer-finalize-inspector/' + name, base64.b64decode(row['content']))
        assert pin(bundle / 'observer-finalize-inspector' / name) == row['identity']
    put('observer-finalize-inspector/Program.cs', inspector_source.read_bytes())
    put('observer-finalize-inspector/qualified-build.json', (OBSERVER / 'build-review.json').read_bytes())
    write(bundle / 'observer-finalize-source-review.json', dict(passed=True,
        before=pin(origin), after=pin(bundle / 'observer-finalize-source.cs'),
        exact_inverse=True, existing_non_null_guard_preserved=True, no_product_source_changed=True,
        previous_failed_collection=pin(before / 'observer-recovery-collection.json'),
        identical_inspector_helpers_sha256=hashlib.sha256(shared.encode()).hexdigest(), reviewer=pin(Path(__file__))))
    spec = dict(original_spec=pin(original.BASE / 'bundle/spec.json'),
        original_state=pin(original.BASE / 'build-collected/build-state.json'),
        recovery_state=pin(before / 'observer-recovery-state.json'),
        recovery_spec=pin(FIRST / 'bundle/observer-recovery-spec.json'),
        core_path=core_path, core=remote['core'], core_rebuilt=False, consumer_rebuilt=False,
        previous_runtimes={p.relative_to(before).as_posix():pin(p) for role in ['runtime-control','runtime-observed'] for p in (before / role).iterdir()},
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle / 'observer-finalize-spec.json', spec)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as tar:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): tar.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    for path in TOOLS.glob('*.py'): ast.parse(path.read_text(encoding='utf8'), str(path))
    write(BASE / 'prepared.json', dict(archive=pin(BASE / 'payload.tar.gz'), spec=pin(bundle / 'observer-finalize-spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()}))
    print(json.dumps(dict(prepared=True, core_reused=remote['core'], archive=pin(BASE / 'payload.tar.gz'))))


def prepared():
    original.prepared()
    value = read(BASE / 'prepared.json')
    assert value['archive'] == pin(BASE / 'payload.tar.gz') and value['spec'] == pin(BASE / 'bundle/observer-finalize-spec.json')
    for name, wanted in value['tools'].items(): assert pin(TOOLS / name) == wanted, name
    for name, wanted in read(BASE / 'bundle/observer-finalize-spec.json')['files'].items(): assert pin(BASE / 'bundle' / name) == wanted, name
    return value


def stage():
    value = prepared(); assert not (BASE / 'staged.json').exists()
    subprocess.run(['scp', *original.SSH[1:-1], str(BASE / 'payload.tar.gz'),
        original.SSH[-1] + ':' + original.REMOTE + '/observer-finalize-transfer.tar.gz'],
        check=True, timeout=90, creationflags=subprocess.CREATE_NO_WINDOW)
    result = ssh(PRELUDE + f'''
from remote import verify,idle,pin
verify();idle();archive=base/'observer-finalize-transfer.tar.gz'
assert pin(archive)=={value['archive']!r}
with tarfile.open(archive) as tar:
 members=tar.getmembers()
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and not (base/m.name).exists() for m in members)
 assert len({{m.name for m in members}})==len(members)
 tar.extractall(base,filter='data')
assert pin(base/'observer-finalize-spec.json')=={value['spec']!r}
import importlib.util
loader=importlib.util.spec_from_file_location('observer_recovery',base/'observer-finalize.py')
module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module);module.verify()
print(json.dumps(dict(passed=True,spec=pin(base/'observer-finalize-spec.json'))))
''')
    write(BASE / 'staged.json', result); print(json.dumps(result))


def launch():
    prepared(); assert read(BASE / 'staged.json')['passed'] and not (BASE / 'deployment.json').exists()
    result = ssh(PRELUDE + '''
from remote import idle,verify
verify();idle();assert not (base/'observer-finalize-state.json').exists()
with (base/'observer-finalize-supervisor.stdout').open('x') as out,(base/'observer-finalize-supervisor.stderr').open('x') as err:
 env=dict(os.environ,PYTHONPATH='/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
 p=subprocess.Popen([sys.executable,'-B',str(base/'observer-finalize.py')],cwd=base,env=env,
   stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
print(json.dumps(dict(pid=p.pid,birth=psutil.Process(p.pid).create_time())))
''')
    write(BASE / 'deployment.json', result); print(json.dumps(result))


def observe():
    prepared()
    result = ssh(PRELUDE + f'''
from remote import read,live
state=read(base/'observer-finalize-state.json')
ids=[{read(BASE / 'deployment.json')!r}]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
print(json.dumps(dict(live=[i for i in ids if live(i)],complete=state['complete'],code=state['code'],
 latest=None if not state['runs'] else {{k:state['runs'][-1].get(k) for k in ['name','samples','complete','code']}},error=state.get('error'))))
''')
    with (BASE / 'observations.jsonl').open('a') as stream: stream.write(json.dumps(result) + '\n')
    print(json.dumps(result))


def collect():
    prepared(); target = BASE / 'collected'; assert not target.exists()
    script = PRELUDE + '''
from remote import read,live,pin,verify
spec=verify();repair=read(base/'observer-finalize-spec.json');state=read(base/'observer-finalize-state.json')
ids=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
assert state['complete'] and not any(live(i) for i in ids)
paths={base/name for name in spec['files']}|{base/name for name in repair['files']}
for folder in ['logs','inventory','runtime-original','runtime-control','runtime-observed']:
 paths.update(p for p in (base/folder).rglob('*') if p.is_file())
paths.update(p for p in base.iterdir() if p.is_file())
files={p.relative_to(base).as_posix():pin(p) for p in sorted(paths)}
receipt=base/'observer-finalize-collection.json'
with receipt.open('x') as stream:json.dump(dict(files=files,state=pin(base/'observer-finalize-state.json'),terminal=True,code=state['code'],identities=ids),stream)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name in [*files,receipt.name]:tar.add(base/name,arcname=name,recursive=False)
'''
    archive = BASE / 'results.tar.gz'
    with archive.open('xb') as out, (BASE / 'collection.stderr').open('x') as err:
        result = subprocess.run(original.SSH + ['python3 -B -'], input=script.encode(), stdout=out, stderr=err,
            timeout=300, creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0, 'Preserve partial collection'
    target.mkdir()
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert len({m.name for m in members}) == len(members)
        tar.extractall(target, filter='data')
    receipt = read(target / 'observer-finalize-collection.json')
    for name, wanted in receipt['files'].items(): assert pin(target / name) == wanted, name
    write(BASE / 'transfer.json', dict(passed=True, archive=pin(archive), collection=pin(target / 'observer-finalize-collection.json')))
    print(json.dumps(dict(code=receipt['code'], files=len(receipt['files']))))


if __name__ == '__main__':
    assert sys.argv[1] in ['prepare', 'stage', 'launch', 'observe', 'collect']
    globals()[sys.argv[1]]()
