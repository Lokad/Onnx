"""One offline extraction from a closed trace, with no application/build process."""
import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-sigmoid-address-review-20260928'
REMOTE = '/dev/shm/lokad-parakeet-sigmoid-address-review-20260928'
CAPTURE = ROOT/'artifacts/parakeet-sigmoid-residual-diagnostic-amd-20260928'
REMOTE_CAPTURE = '/dev/shm/lokad-parakeet-sigmoid-residual-diagnostic-20260928'
EVENTS = ROOT/'artifacts/e5-direct-tier-diagnostic-v2-amd-20260925/collected/export-runtime'
REMOTE_EVENTS = '/dev/shm/lokad-e5-direct-tier-diagnostic-v2-20260925/export-runtime'
loader = importlib.util.spec_from_file_location('frozen_residual', TOOLS.parent/'sigmoid-residual-diagnostic-amd/run.py')
prior = importlib.util.module_from_spec(loader); loader.loader.exec_module(prior)
pin, read, write, transport = prior.pin, prior.read, prior.write, prior.transport
PRELUDE = prior.PRELUDE+f'\nbase=Path({REMOTE!r})\n'


def checked():
    value = read(BASE/'prepared.json')
    assert value['passed'] and value['spec'] == pin(BASE/'spec.json')
    for name, wanted in value['inputs'].items():
        assert pin(ROOT/name) == wanted, name
    return value


def stage():
    assert not (BASE/'prepared.json').exists()
    assert pin(CAPTURE/'closed.json')['sha256'] == '31b91fba1f513c219a7f1c4b9c44b637ff90389d676522856b64be83c5280a54'
    closure = read(CAPTURE/'closed.json'); assert closure['passed']
    for name, wanted in closure['files'].items():
        assert pin(CAPTURE/name) == wanted, name
    external = {REMOTE_CAPTURE+'/'+name:pin(CAPTURE/'collected'/name)
                for name in ['remote.py', 'spec.json', 'sampled/capture.nettrace']}
    external.update({REMOTE_EVENTS+'/'+p.name:pin(p) for p in EVENTS.iterdir() if p.is_file()})
    external['/dev/shm/lokad-transpose-axis-profile-20260928/remote.py'] = pin(prior.PROFILE/'bundle/remote.py')
    files = {name:pin(TOOLS/name) for name in ['Read-Addresses.ps1', 'remote.py']}
    spec = dict(boot=1789634288.0, trace=REMOTE_CAPTURE+'/sampled/capture.nettrace',
        monitor=REMOTE_CAPTURE+'/remote.py', libraries=REMOTE_EVENTS, external=external, files=files,
        limits=dict(available_before=11*1024**3, tmpfs_before=2*1024**3, rss=12*1024**3,
                    minimum_free=1024**3, output=512*1024**2, seconds=900),
        inference_calls=0, builds=0, local_collection_excludes=['output/trace.etlx'])
    write(BASE/'spec.json', spec)
    inputs = {p.relative_to(ROOT).as_posix():pin(p) for p in TOOLS.iterdir() if p.is_file()}
    inputs.update({p.relative_to(ROOT).as_posix():pin(p) for p in [CAPTURE/'closed.json', BASE/'upstream.json']})
    write(BASE/'prepared.json', dict(passed=True, spec=pin(BASE/'spec.json'), inputs=inputs))
    payload = {name:base64.b64encode((TOOLS/name).read_bytes()).decode() for name in files}
    payload['spec.json'] = base64.b64encode((BASE/'spec.json').read_bytes()).decode()
    result = transport.ssh(PRELUDE+f'''
import base64,importlib.util
assert not base.exists()
p=Path({REMOTE_CAPTURE!r})/'remote.py'
s=importlib.util.spec_from_file_location('prior_monitor',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.idle();m.preflight()
base.mkdir()
for name,data in {payload!r}.items():(base/name).write_bytes(base64.b64decode(data))
sys.path.insert(0,str(base));from remote import verify,pin
verify();print(json.dumps(dict(passed=True,spec=pin(base/'spec.json'))))
''')
    assert result['spec'] == pin(BASE/'spec.json')
    write(BASE/'staged.json', result); print(json.dumps(result))


def launch():
    checked(); assert read(BASE/'staged.json')['passed'] and not (BASE/'deployment.json').exists()
    result = transport.ssh(PRELUDE+'''
sys.path.insert(0,str(base));from remote import verify,monitor
verify();monitor.idle();assert not (base/'state.json').exists()
with (base/'supervisor.stdout').open('x') as out,(base/'supervisor.stderr').open('x') as err:
 p=subprocess.Popen([sys.executable,'-B',str(base/'remote.py')],cwd=base,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
value=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time())
(base/'deployment.json').write_text(json.dumps(value));print(json.dumps(value))
''')
    write(BASE/'deployment.json', result); print(json.dumps(result))


def observe():
    print(json.dumps(transport.ssh(PRELUDE+'''
sys.path.insert(0,str(base));from remote import read,live
s=read(base/'state.json') if (base/'state.json').exists() else None
owners=[read(base/'deployment.json')]+([] if s is None else [i for r in s['runs'] for i in r['processes'].values()])
print(json.dumps(dict(live=[i for i in owners if live(i)],state=s,
 stderr=(base/'supervisor.stderr').read_text()[-5000:],log=(base/'logs/addresses.log').read_text()[-3000:] if (base/'logs/addresses.log').exists() else '')))
''')))


def collect():
    checked(); assert not (BASE/'results.tar.gz').exists()
    script = PRELUDE+'''
sys.path.insert(0,str(base));from remote import read,live,pin,verify
s=read(base/'state.json');owners=[s['supervisor']]+[i for r in s['runs'] for i in r['processes'].values()]
assert s['complete'] and not any(live(i) for i in owners);verify()
files={p.relative_to(base).as_posix():pin(p) for p in base.rglob('*') if p.is_file()}
selected={n:v for n,v in files.items() if n!='output/trace.etlx'}
assert sum(v['bytes'] for v in selected.values())<32*1024**2
(base/'collection.json').write_text(json.dumps(dict(terminal=True,code=s['code'],identities=owners,files=selected,retained_remote={n:v for n,v in files.items() if n not in selected})))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name in [*selected,'collection.json']:tar.add(base/name,arcname=name,recursive=False)
'''
    with (BASE/'results.tar.gz').open('xb') as out, (BASE/'collection.stderr').open('x') as err:
        result = subprocess.run(transport.SSH+['python3','-B','-'], input=script.encode(), stdout=out, stderr=err,
            timeout=300, creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0, 'Preserve failure; do not repeat the offline reader unchanged'
    target = BASE/'collected'; target.mkdir()
    with tarfile.open(BASE/'results.tar.gz') as archive:
        rows = archive.getmembers()
        assert all(r.isfile() and not Path(r.name).is_absolute() and '..' not in Path(r.name).parts for r in rows)
        assert len(rows) == len({r.name for r in rows}); archive.extractall(target, filter='data')
    receipt = read(target/'collection.json')
    for name, wanted in receipt['files'].items():
        assert pin(target/name) == wanted, name
    write(BASE/'transfer.json', dict(passed=True, archive=pin(BASE/'results.tar.gz'), collection=pin(target/'collection.json')))
    print(json.dumps(dict(terminal=True, code=receipt['code'], files=len(receipt['files']))))


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['stage', 'launch', 'observe', 'collect']
    globals()[sys.argv[1]]()
