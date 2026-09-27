"""Freeze one unchanged-candidate diagnostic using the retained trace tools."""
import ast
import difflib
import json
from pathlib import Path
import tarfile
from protocol import TOOLS, PARENT, pin, read, save
from source import instrument, SCREEN
from adapters import PRIOR, adapted

ROOT = TOOLS.parents[2]
BASE = ROOT/'artifacts/parakeet-pointwise-tail-runtime-observation-amd-20260927'
OBSERVATION = ROOT/'artifacts/parakeet-decoder-lstm-runtime-observation-amd-20260927'
# The transport uses this collection's terminal owners for its initial preflight.
QUALIFIED = OBSERVATION
REMOTE_SCREEN = '/dev/shm/lokad-pointwise-tail-timing-20260927'
REMOTE_OBSERVATION = '/dev/shm/lokad-lstm-runtime-20260927'


def previous_closed():
    for folder, digest in [(SCREEN, 'b9ff0d6050068ad8ad9ed9fa7a50350bb7e99079e10316f077183bea8ef0765c'),
        (OBSERVATION, 'a00f9195ab9be512b1230357d68e3e1142593014a79680198cc5d517e95b6c85')]:
        assert pin(folder/'closed.json')['sha256'] == digest
        for name, wanted in read(folder/'closed.json')['files'].items(): assert pin(folder/name) == wanted, name
    assert read(SCREEN/'closed.json')['completed'] and not read(SCREEN/'closed.json')['component_admitted']
    assert read(OBSERVATION/'closed.json')['passed']


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
    copy(SCREEN/'bundle/contract-source/Program.cs', 'original/Program.cs')
    copy(SCREEN/'bundle/contract-source/TailContracts.csproj', 'original/TailContracts.csproj')
    put('diagnostic.patch', ''.join(difflib.unified_diff((SCREEN/'bundle/contract-source/Program.cs').read_text().splitlines(True),
        timing.splitlines(True), fromfile='original/Program.cs', tofile='source/Timing.cs')).encode())
    copy(TOOLS.parent/'decoder-projection-observation/Driver.cs', 'original/marker-driver.cs')
    for name in ['protocol.py', 'checks.py', 'remote_prepare.py']: copy(TOOLS/name, 'tools/'+name)
    put('tools/remote.py', adapted('remote.py').encode())
    for name in ['protocol', 'remote']: copy(PARENT/(name+'.py'), 'tools/'+name+'_base.py')
    copy(TOOLS/'README.md', 'prospective-observation.md')
    put('prospective-plan.md', (ROOT/'PLAN.md').read_bytes())
    terminals = []
    for label, folder, remote, receipt in [
        ('screen', SCREEN, REMOTE_SCREEN, 'capture-collected/capture-collection.json'),
        ('observation', OBSERVATION, REMOTE_OBSERVATION, 'collected/collection.json')]:
        copy(folder/'closed.json', 'evidence/'+label+'/closed.json')
        copy(folder/receipt, 'evidence/'+label+'/collection.json')
        terminals.append(dict(remote=remote+'/'+Path(receipt).name, local='evidence/'+label+'/collection.json'))
    copy(SCREEN/'analysis.json', 'evidence/screen/analysis.json')
    prior = read(SCREEN/'bundle/spec.json'); built = read(SCREEN/'capture-collected/built.json')
    for name, wanted in built['runtime'].items():
        if name.startswith('candidate/') and name.endswith('.dll') and not name.endswith(('TailContracts.dll', 'SampledAudio.dll')):
            link(SCREEN/'capture-collected/runtime'/name, REMOTE_SCREEN+'/runtime/'+name,
                'runtime/'+Path(name).name, wanted)
    observed = read(OBSERVATION/'payload.json')
    for name, wanted in observed['files'].items():
        if name.startswith(('tracer/', 'export-runtime/')):
            link(OBSERVATION/'collected'/name, REMOTE_OBSERVATION+'/'+name, name, wanted)
    assert 'tracer/dotnet-trace.dll' in links and 'export-runtime/DispatchEventsExport.dll' in links
    output = read(SCREEN/'capture-collected/probe/candidate-1/result.json')
    hashes = [r['output_sha256'] for r in output['records'][:40]]
    assert len(hashes) == 40
    external = dict(prior['external'])
    # Retain exact SDK/runtime/offline-feed identities; omit unused audio/model inputs.
    external.update({n:w for n,w in observed['external'].items()
        if n.startswith('/home/vermorel/.dotnet/') or n.startswith(prior['feed']+'/')})
    save(bundle/'spec.json', prior)
    save(bundle/'stage.json', dict(passed=True, products=prior['products'], links=links, terminals=terminals,
        output_hashes=hashes, previous_owner=read(SCREEN/'capture-collected/capture-collection.json')['identities'][0],
        feed=prior['feed'], interpreter=observed['interpreter'], external=external,
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for path in [*TOOLS.iterdir(), *PRIOR.glob('*.py'), PARENT/'run.py', TOOLS.parent/'decoder-projection-observation/run.py']:
        if path.is_file():
            if path.suffix == '.py': ast.parse(path.read_text(), str(path))
            originals[path.relative_to(ROOT).as_posix()] = pin(path)
    for name in ['events.py', 'audit.py', 'remote.py']: ast.parse(adapted(name), name)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=originals, archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(passed=True, archive=pin(BASE/'payload.tar.gz'), links=len(links), diagnostic_calls=400)))


if __name__ == '__main__': prepare()
