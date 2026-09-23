"""Freeze ordinary selected/M47 binaries with additive request instrumentation."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-pad-runtime-diagnostic-amd-20260923'
SCREEN = ROOT / 'artifacts/parakeet-pad-dispatch-screen-amd-20260923'
EVENTS = ROOT / 'artifacts/graph-startup-diagnostic-amd-20260923'
REMOTE_SCREEN = '/dev/shm/lokad-parakeet-pad-dispatch-screen-20260923'
REMOTE_EVENTS = '/dev/shm/lokad-graph-startup-diagnostic-20260923'


def previous_closed():
    for folder, digest in [
        (SCREEN, '8e757188ca7a29c9c78a9fdc1803eb64d16dab18c476e0e433419eafd5245d73'),
        (EVENTS, 'c2f4961396ec577711ef3ac48e629e901414a21a327594921b59fb015bda9a1f')]:
        assert pin(folder / 'closed.json')['sha256'] == digest
        proof = read(folder / 'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items():
            assert pin(folder / name) == wanted, name
    assert not read(SCREEN / 'closed.json')['admitted']
    source = ROOT / 'artifacts/parakeet-pad-dispatch-source-20260923/prepared.json'
    assert pin(source)['sha256'] == '727fa49ee92dfe2cfe8ff34396d27fee82b961f826eb2499dbf1a8d818e77b0d'
    selected = read(source)['before']; assert len(selected) == 420
    for name, wanted in selected.items(): assert pin(ROOT / name) == wanted, name
    assert pin(TOOLS / 'census.py') == pin(ROOT / 'tests/parakeet/pad-dispatch-screen/census.py')


def prepare():
    assert not BASE.exists(); previous_closed()
    BASE.mkdir(); bundle = BASE / 'bundle'; bundle.mkdir(); originals = {}

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)

    for name in ['Screen.cs', 'Producer.csproj']:
        copy(TOOLS / name, bundle / 'source/consumer' / name)
    copy(ROOT / 'global.json', bundle / 'source/global.json')
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py']:
        copy(TOOLS / name, bundle / 'tools' / name)
    copy(ROOT / 'tests/parakeet/dispatch-events-amd/remote.py', bundle / 'tools/remote_base.py')
    copy(TOOLS / 'README.md', bundle / 'README.md')
    shutil.copy2(ROOT / '.agent/m48-parakeet-pad-runtime-20260923.md', bundle / 'prospective-plan.md')
    for label, folder in [('screen', SCREEN), ('events', EVENTS)]:
        for name in ['closed.json', 'collected/collection.json']:
            copy(folder / name, bundle / 'evidence' / (label + '-' + Path(name).name))
    copy(SCREEN / 'bundle/census.json', bundle / 'census.json')
    copy(SCREEN / 'collected/current-screen0-512/result.json', bundle / 'evidence/original-current.json')
    copy(SCREEN / 'collected/candidate-screen1-512/result.json', bundle / 'evidence/original-candidate.json')
    payload = read(SCREEN / 'payload.json'); old = read(EVENTS / 'payload.json')
    links = {name: dict(source=REMOTE_SCREEN + '/' + name, identity=wanted)
             for name, wanted in payload['files'].items() if name.startswith('runtimes/')}
    for name, wanted in old['files'].items():
        if name.startswith('tracer/'):
            links[name] = dict(source=REMOTE_EVENTS + '/' + name, identity=wanted)
    built = read(EVENTS / 'collected/built.json')
    for name, wanted in built['files'].items():
        if name.startswith('export-runtime/'):
            links[name] = dict(source=REMOTE_EVENTS + '/' + name, identity=wanted)
    assert links['export-runtime/DispatchEventsExport.dll']['identity'] == built['exporter']
    save(bundle / 'stage.json', dict(passed=True, diagnostic_only=True, links=links,
        products=payload['products'], external=payload['external'], feed=payload['feed'],
        interpreter=payload['interpreter'], exporter=built['exporter'],
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'), str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    # Preserve the independent oracle, case setup and every behavior check.
    original = (ROOT / 'tests/parakeet/pad-dispatch-screen/Screen.cs').read_text()
    actual = (TOOLS / 'Screen.cs').read_text()
    for start, end in [('    static float[] Oracle(', '    static void Main('),
                       ('            long setup =', '            for (int iteration'),
                       ('                if (iteration == 0)', '        Require(index == 12')]:
        assert original[original.index(start):original.index(end)] in actual
    assert actual.count('var returned = CPUExecutionProvider.Pad(source, padTensor, fillTensor, mode, null, null, null);') == 1
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): tar.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE / 'prepared.json', dict(passed=True, files=originals,
        stage=pin(bundle / 'stage.json'), archive=pin(BASE / 'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE / 'payload.tar.gz'), links=len(links))))


if __name__ == '__main__': prepare()
