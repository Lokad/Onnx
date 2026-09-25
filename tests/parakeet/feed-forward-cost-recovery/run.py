"""Preserve the failed build; stage one observer correction and reuse its Core."""
import ast
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

BASE = ROOT / 'artifacts/parakeet-feed-forward-cost-observer-recovery-20260925'
KIND = 'observer-recovery'
pin, read, write, ssh = original.pin, original.read, original.write, original.ssh
PRELUDE = original.PRELUDE


def prepare():
    original.prepared()
    assert not BASE.exists()
    before = original.BASE / 'build-collected'
    state = read(before / 'build-state.json')
    assert state['complete'] and state['code'] == 1
    assert [r['name'] for r in state['runs']] == ['sdk-version', 'core-restore', 'core-build', 'data-restore', 'data-build']
    assert [r['code'] for r in state['runs']] == [0, 0, 0, 0, 1]
    assert 'error CS1673' in (before / 'logs/data-build.stdout').read_text()
    origin = before / 'data-source/PhaseProbe.cs'
    old = origin.read_bytes()
    needle = b'                        Require(wall.All(n => start <= n.StartTicks'
    replacement = b'                        long graphStart = start;\r\n                        Require(wall.All(n => graphStart <= n.StartTicks'
    assert old.count(needle) == 1
    corrected = old.replace(needle, replacement)
    assert corrected.replace(replacement, needle) == old
    core_path = 'core-source/src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll'
    remote = ssh(PRELUDE + f'''
from remote import verify,read,live,pin,idle
verify();idle();state=read(base/'build-state.json')
assert pin(base/'build-state.json')=={pin(before / 'build-state.json')!r}
assert state['complete'] and state['code']==1 and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
print(json.dumps(dict(core=pin(base/{core_path!r}))))
''')
    BASE.mkdir(); bundle = BASE / 'bundle'; bundle.mkdir()
    def put(name, data):
        path = bundle / name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream: stream.write(data)
    for path in (before / 'data-source').iterdir():
        assert path.is_file()
        put('observer-recovery-source/' + path.name, corrected if path.name == 'PhaseProbe.cs' else path.read_bytes())
    put('observer-recovery-source.cs', corrected)
    put('observer-recovery.py', (TOOLS / 'remote.py').read_bytes())
    write(bundle / 'observer-recovery-source-review.json', dict(passed=True,
        before=pin(origin), after=pin(bundle / 'observer-recovery-source.cs'),
        exact_inverse=True, field_read_copied_to_local=True, no_product_source_changed=True,
        original_failed_collection=pin(before / 'build-collection.json'), reviewer=pin(Path(__file__))))
    spec = dict(original_spec=pin(original.BASE / 'bundle/spec.json'), original_state=pin(before / 'build-state.json'),
        core_path=core_path, core=remote['core'], core_rebuilt=False, consumer_rebuilt=False,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle / 'observer-recovery-spec.json', spec)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as tar:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): tar.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    for path in TOOLS.glob('*.py'): ast.parse(path.read_text(encoding='utf8'), str(path))
    write(BASE / 'prepared.json', dict(archive=pin(BASE / 'payload.tar.gz'), spec=pin(bundle / 'observer-recovery-spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()}))
    print(json.dumps(dict(prepared=True, core_reused=remote['core'], archive=pin(BASE / 'payload.tar.gz'))))


def prepared():
    original.prepared()
    value = read(BASE / 'prepared.json')
    assert value['archive'] == pin(BASE / 'payload.tar.gz') and value['spec'] == pin(BASE / 'bundle/observer-recovery-spec.json')
    for name, wanted in value['tools'].items(): assert pin(TOOLS / name) == wanted, name
    for name, wanted in read(BASE / 'bundle/observer-recovery-spec.json')['files'].items(): assert pin(BASE / 'bundle' / name) == wanted, name
    return value


def stage():
    value = prepared(); assert not (BASE / 'staged.json').exists()
    subprocess.run(['scp', *original.SSH[1:-1], str(BASE / 'payload.tar.gz'),
        original.SSH[-1] + ':' + original.REMOTE + '/observer-recovery-transfer.tar.gz'],
        check=True, timeout=90, creationflags=subprocess.CREATE_NO_WINDOW)
    result = ssh(PRELUDE + f'''
from remote import verify,idle,pin
verify();idle();archive=base/'observer-recovery-transfer.tar.gz'
assert pin(archive)=={value['archive']!r}
with tarfile.open(archive) as tar:
 members=tar.getmembers()
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and not (base/m.name).exists() for m in members)
 assert len({{m.name for m in members}})==len(members)
 tar.extractall(base,filter='data')
assert pin(base/'observer-recovery-spec.json')=={value['spec']!r}
import importlib.util
loader=importlib.util.spec_from_file_location('observer_recovery',base/'observer-recovery.py')
module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module);module.verify()
print(json.dumps(dict(passed=True,spec=pin(base/'observer-recovery-spec.json'))))
''')
    write(BASE / 'staged.json', result); print(json.dumps(result))


def launch():
    prepared(); assert read(BASE / 'staged.json')['passed'] and not (BASE / 'deployment.json').exists()
    result = ssh(PRELUDE + '''
from remote import idle,verify
verify();idle();assert not (base/'observer-recovery-state.json').exists()
with (base/'observer-recovery-supervisor.stdout').open('x') as out,(base/'observer-recovery-supervisor.stderr').open('x') as err:
 env=dict(os.environ,PYTHONPATH='/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
 p=subprocess.Popen([sys.executable,'-B',str(base/'observer-recovery.py')],cwd=base,env=env,
   stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
print(json.dumps(dict(pid=p.pid,birth=psutil.Process(p.pid).create_time())))
''')
    write(BASE / 'deployment.json', result); print(json.dumps(result))


def observe():
    prepared()
    result = ssh(PRELUDE + f'''
from remote import read,live
state=read(base/'observer-recovery-state.json')
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
spec=verify();repair=read(base/'observer-recovery-spec.json');state=read(base/'observer-recovery-state.json')
ids=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
assert state['complete'] and not any(live(i) for i in ids)
paths={base/name for name in spec['files']}|{base/name for name in repair['files']}
for folder in ['logs','inventory','runtime-original','runtime-control','runtime-observed']:
 paths.update(p for p in (base/folder).rglob('*') if p.is_file())
paths.update(p for p in base.iterdir() if p.is_file())
files={p.relative_to(base).as_posix():pin(p) for p in sorted(paths)}
receipt=base/'observer-recovery-collection.json'
with receipt.open('x') as stream:json.dump(dict(files=files,state=pin(base/'observer-recovery-state.json'),terminal=True,code=state['code'],identities=ids),stream)
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
    receipt = read(target / 'observer-recovery-collection.json')
    for name, wanted in receipt['files'].items(): assert pin(target / name) == wanted, name
    write(BASE / 'transfer.json', dict(passed=True, archive=pin(archive), collection=pin(target / 'observer-recovery-collection.json')))
    print(json.dumps(dict(code=receipt['code'], files=len(receipt['files']))))


if __name__ == '__main__':
    assert sys.argv[1] in ['prepare', 'stage', 'launch', 'observe', 'collect']
    globals()[sys.argv[1]]()
