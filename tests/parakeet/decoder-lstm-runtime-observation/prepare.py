"""Freeze one diagnostic of the failed screen; preserve every original verdict."""
import ast
import difflib
import json
from pathlib import Path
import tarfile
from protocol import TOOLS, PARENT, pin, read, save
from source import instrument, SCREEN

ROOT = TOOLS.parents[2]
BASE = ROOT/'artifacts/parakeet-decoder-lstm-runtime-observation-amd-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-decoder-packed-row-root-amd-20260927'
OBSERVATION = ROOT/'artifacts/parakeet-decoder-projection-observation-v3-amd-20260927'
REMOTE_SCREEN = '/dev/shm/lokad-lstmlayout-timing-20260927'
REMOTE_ROOT = '/dev/shm/lokad-parakeet-decoder-packed-row-root-20260927'
REMOTE_OBSERVATION = '/dev/shm/lokad-decmap-20260927'


def previous_closed():
    for folder, digest in [(SCREEN, 'ac1b5e0e5400d03d84250d6148253208a6d25d3bc9e565a49718721861772fe2'),
        (QUALIFIED, 'd0a78cdd3106d6a72a41303f879bbb2f9ea3bd6a298d778333015f38ccdac246')]:
        assert pin(folder/'closed.json')['sha256'] == digest and read(folder/'closed.json')['passed']
        for name, wanted in read(folder/'closed.json')['files'].items(): assert pin(folder/name) == wanted, name
    assert not read(SCREEN/'closed.json')['admitted']
    assert sum(not r['passed'] for r in read(SCREEN/'analysis.json')['performance']['controls']) == 100
    assert read(OBSERVATION/'closed.json')['passed']
    for name, wanted in read(QUALIFIED/'bundle/evidence/root-applied.json')['source_files'].items():
        assert pin(ROOT/name) == wanted, name


def prepare():
    assert not BASE.exists(); previous_closed()
    timing, project = instrument()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = {}; links = {}
    def put(name, data):
        target = bundle/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    def copy(source, name):
        originals[source.relative_to(ROOT).as_posix()] = pin(source); put(name, source.read_bytes())
    def link(source, remote, target, wanted):
        assert pin(source) == wanted, source
        originals[source.relative_to(ROOT).as_posix()] = wanted
        links[target] = dict(source=remote, identity=wanted)
    copy(ROOT/'global.json', 'source/global.json')
    put('source/Timing.cs', timing.encode()); put('source/Timing.csproj', project.encode())
    for filename in ['Timing.cs', 'Timing.csproj']:
        copy(SCREEN/'bundle/source'/filename, 'original/'+filename)
    put('diagnostic.patch', ''.join(difflib.unified_diff((SCREEN/'bundle/source/Timing.cs').read_text().splitlines(True),
        timing.splitlines(True), fromfile='original/Timing.cs', tofile='source/Timing.cs')).encode())
    copy(TOOLS.parent/'decoder-projection-observation/Driver.cs', 'original/marker-driver.cs')
    for name in ['protocol.py', 'checks.py', 'remote.py', 'remote_prepare.py']:
        copy(TOOLS/name, 'tools/'+name)
    for name in ['protocol', 'remote']: copy(PARENT/(name+'.py'), 'tools/'+name+'_base.py')
    copy(TOOLS/'README.md', 'prospective-observation.md')
    put('prospective-plan.md', (ROOT/'PLAN.md').read_bytes())
    terminals = []
    for label, folder, remote in [('screen', SCREEN, REMOTE_SCREEN), ('root', QUALIFIED, REMOTE_ROOT),
        ('observation', OBSERVATION, REMOTE_OBSERVATION)]:
        copy(folder/'closed.json', 'evidence/'+label+'/closed.json')
        copy(folder/'collected/collection.json', 'evidence/'+label+'/collection.json')
        terminals.append(dict(remote=remote+'/collection.json', local='evidence/'+label+'/collection.json'))
    copy(SCREEN/'analysis.json', 'evidence/screen/analysis.json')
    prior = read(SCREEN/'payload.json'); product = prior['identities']['selectedfallback']
    for name, wanted in product.items():
        path = 'products/selectedfallback/'+name
        link(SCREEN/'collected'/path, REMOTE_SCREEN+'/'+path, 'runtime/'+name, wanted)
    for name, wanted in prior['files'].items():
        if name.startswith('fixtures/'):
            link(SCREEN/'collected'/name, REMOTE_SCREEN+'/'+name, name, wanted)
    assert sum(n.startswith('fixtures/') for n in links) == 444
    observed = read(OBSERVATION/'payload.json')
    for name, wanted in observed['files'].items():
        if name.startswith(('tracer/', 'export-runtime/')):
            remote = '/dev/shm/lokad-parakeet-dispatch-events-20260923' if name.startswith('tracer/') else '/dev/shm/lokad-parakeet-decoder-projection-observation-20260927'
            assert read(OBSERVATION/'collected/collection.json')['files'][name] == wanted
            link(OBSERVATION/'collected'/name, remote+'/'+name, name, wanted)
    assert 'tracer/dotnet-trace.dll' in links and 'export-runtime/DispatchEventsExport.dll' in links
    save(bundle/'spec.json', dict(identities=dict(selectedfallback=product)))
    save(bundle/'stage.json', dict(passed=True, product=product, links=links, terminals=terminals,
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for path in [*TOOLS.iterdir(), PARENT/'run.py', TOOLS.parent/'decoder-projection-observation/run.py']:
        if path.is_file():
            if path.suffix == '.py': ast.parse(path.read_text(), str(path))
            originals[path.relative_to(ROOT).as_posix()] = pin(path)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(passed=True, archive=pin(BASE/'payload.tar.gz'), links=len(links), diagnostic_calls=3800)))


if __name__ == '__main__': prepare()
