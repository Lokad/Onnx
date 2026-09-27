"""Bind one diagnostic to the rejected screen and reuse its bounded worker."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
from source import changed

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
PARENT = TOOLS.parent/'decoder-packed-row-screen'
loader = importlib.util.spec_from_file_location('original_screen', PARENT/'run.py')
original = importlib.util.module_from_spec(loader); loader.loader.exec_module(original)
pin, read, write, ssh = original.pin, original.read, original.write, original.ssh
PREVIOUS = original.BASE
BASE = ROOT/'artifacts/parakeet-decoder-unmapped-calls-amd-20260927'
REMOTE = '/dev/shm/lokad-unmapped-calls-20260927'
PRELUDE = original.PRELUDE.replace(original.REMOTE, REMOTE)
transport, observer = original.transport, original.prior
transport.BASE, transport.REMOTE, transport.PRELUDE = BASE, REMOTE, PRELUDE
observer.BASE, observer.REMOTE, observer.PRELUDE = BASE, REMOTE, PRELUDE
# The reused build auditor imports score only for its unused capture function.
sys.path.append(str(PARENT))


def references():
    original.prepared()
    assert pin(PREVIOUS/'closed.json')['sha256'] == '3e6a7562db1946954f2cddba10b2ee31958854e66c50fbb55848ed04e3e75148'
    closed = read(PREVIOUS/'closed.json'); assert closed['passed'] and not closed['admitted']
    for name, wanted in closed['files'].items(): assert pin(PREVIOUS/name) == wanted, name
    return read(PREVIOUS/'bundle/spec.json')


def prepare():
    assert not BASE.exists(); previous = references()
    for name in ['audit.py', 'observations.py', 'test_observations.py', 'README.md']: assert (TOOLS/name).is_file()
    for path in TOOLS.glob('*.py'): ast.parse(path.read_text())
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir()
    for name in previous['files']:
        path = bundle/name; path.parent.mkdir(parents=True, exist_ok=True)
        data = (PREVIOUS/'bundle'/name).read_bytes()
        if name == 'source/Screen.cs': data = changed(data.decode()).encode()
        if name == 'protocol.md': data = (TOOLS/'README.md').read_bytes()
        path.write_bytes(data)
    assert read(bundle/'census.json') == original.census()
    spec = dict(previous)
    spec.update(diagnostic_only=True, release_admitted=False,
        prior_screen=pin(PREVIOUS/'closed.json'), consumer_change='Individual clocks only for unmapped calls; preallocated arrays.',
        evidence=dict(previous['evidence'], rejected_screen=pin(PREVIOUS/'closed.json')),
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json', spec)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): tar.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    helpers = {**read(PREVIOUS/'prepared.json')['helpers'],
        **{(PARENT/name).relative_to(ROOT).as_posix(): wanted for name, wanted in read(PREVIOUS/'prepared.json')['tools'].items()}}
    write(BASE/'prepared.json', dict(archive=pin(BASE/'payload.tar.gz'), spec=pin(bundle/'spec.json'),
        tools={p.name: pin(p) for p in TOOLS.iterdir() if p.is_file()}, helpers=helpers))
    print(json.dumps(dict(prepared=True, archive=pin(BASE/'payload.tar.gz'), spec=pin(bundle/'spec.json'))))


def prepared():
    previous = references(); value = read(BASE/'prepared.json'); spec = read(BASE/'bundle/spec.json')
    assert value['archive'] == pin(BASE/'payload.tar.gz') and value['spec'] == pin(BASE/'bundle/spec.json')
    for name, wanted in value['tools'].items(): assert pin(TOOLS/name) == wanted, name
    for name, wanted in value['helpers'].items(): assert pin(ROOT/name) == wanted, name
    for name, wanted in spec['files'].items(): assert pin(BASE/'bundle'/name) == wanted, name
    assert spec['products'] == previous['products'] and spec['external'] == previous['external']
    assert (BASE/'bundle/source/Screen.cs').read_text() == changed((PREVIOUS/'bundle/source/Screen.cs').read_text())
    assert pin(BASE/'bundle/census.json') == pin(PREVIOUS/'bundle/census.json')


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'prepare': prepare()
    else:
        prepared()
        if action == 'stage': transport.stage()
        else:
            kind = sys.argv[2]; assert kind in ['build', 'capture']
            if action == 'launch':
                if kind == 'capture': assert read(BASE/'build-review-transferred.json')['passed']
                transport.launch(kind)
            else: {'observe': observer.observe, 'collect': observer.collect}[action](kind)
