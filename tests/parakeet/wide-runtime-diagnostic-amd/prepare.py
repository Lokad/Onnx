"""Freeze M52 diagnostics using the unchanged compiled matrix trace consumer."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-wide-runtime-diagnostic-amd-20260923'
SCREEN = ROOT / 'artifacts/parakeet-wide-projection-isolation-screen-amd-20260923'
EVENTS = ROOT / 'artifacts/graph-startup-diagnostic-amd-20260923'
CONSUMER = ROOT / 'artifacts/parakeet-dispatch-events-amd-20260923'
REMOTE_SCREEN = '/dev/shm/lokad-parakeet-wide-projection-isolation-screen-20260923'
REMOTE_EVENTS = '/dev/shm/lokad-graph-startup-diagnostic-20260923'
REMOTE_CONSUMER = '/dev/shm/lokad-parakeet-dispatch-events-20260923'


def previous_closed():
    for folder, digest in [
        (SCREEN, 'd0d447e6a943be6100d3ec83a1bb7f4a147706a5b3879e2e2ff602b9e8fe70dc'),
        (EVENTS, 'c2f4961396ec577711ef3ac48e629e901414a21a327594921b59fb015bda9a1f'),
        (CONSUMER, 'c6e1e4d42d3f377f77382a4091ad1150c1a62335a72c7d63a38c728d7a753e19')]:
        assert pin(folder / 'closed.json')['sha256'] == digest
        proof = read(folder / 'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder / name) == wanted, name
    assert not read(SCREEN / 'closed.json')['admitted']
    source = ROOT / 'artifacts/parakeet-wide-projection-isolation-source-20260923/prepared.json'
    assert pin(source)['sha256'] == '2714b31148e581466fad1802f1d13560d860950a39481eee40eb59100b8550ee'
    selected = read(source)['before']; assert len(selected) == 420
    for name, wanted in selected.items(): assert pin(ROOT / name) == wanted, name
    assert pin(CONSUMER / 'bundle/source/consumer/Driver.cs') == pin(ROOT / 'tests/parakeet/dispatch-events-amd/Driver.cs')
    assert read(CONSUMER / 'collected/built.json')['consumer']['sha256'] == '38fe4a8701d9f2cbf4b4efe47ddd76459b51f0868b98f3bb77bb8ecdc9948234'
    for name in ['checks.py','test_checks.py']:
        assert pin(TOOLS/name)==pin(ROOT/'tests/parakeet/isolated-runtime-diagnostic-amd'/name)


def prepare():
    assert not BASE.exists(); previous_closed()
    BASE.mkdir(); bundle = BASE / 'bundle'; bundle.mkdir(); originals = {}

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)

    copy(ROOT / 'global.json', bundle / 'source/global.json')
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py']: copy(TOOLS / name, bundle / 'tools' / name)
    copy(ROOT / 'tests/parakeet/dispatch-events-amd/remote.py', bundle / 'tools/remote_base.py')
    copy(TOOLS / 'README.md', bundle / 'README.md')
    shutil.copy2(ROOT / '.agent/m53-wide-projection-runtime-20260923.md', bundle / 'prospective-plan.md')
    for label, folder in [('screen', SCREEN), ('events', EVENTS), ('consumer', CONSUMER)]:
        for name in ['closed.json', 'collected/collection.json']:
            copy(folder / name, bundle / 'evidence' / (label + '-' + Path(name).name))
    copy(SCREEN / 'bundle/fixtures/result.json', bundle / 'evidence/capture.json')
    copy(SCREEN / 'collected/current-screen0-512/result.json', bundle / 'evidence/original-current.json')
    copy(SCREEN / 'collected/candidate-screen1-512/result.json', bundle / 'evidence/original-candidate.json')
    copy(CONSUMER / 'bundle/source/consumer/Driver.cs', bundle / 'evidence/trace-consumer-source.cs')
    payload = read(SCREEN / 'payload.json'); old = read(EVENTS / 'payload.json')
    links = {name: dict(source=REMOTE_SCREEN + '/' + name, identity=wanted)
             for name, wanted in payload['files'].items() if name.startswith(('runtimes/', 'fixtures/'))}
    for name, wanted in old['files'].items():
        if name.startswith('tracer/'): links[name] = dict(source=REMOTE_EVENTS + '/' + name, identity=wanted)
    built = read(EVENTS / 'collected/built.json')
    for name, wanted in built['files'].items():
        if name.startswith('export-runtime/'): links[name] = dict(source=REMOTE_EVENTS + '/' + name, identity=wanted)
    trace = read(CONSUMER / 'collected/built.json')
    for role in ['current', 'candidate']:
        for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
            name = 'runtimes/' + role + '/ParakeetDispatchEvents.' + suffix
            assert name not in links
            links[name] = dict(source=REMOTE_CONSUMER + '/' + name, identity=trace['files'][name])
    assert links['export-runtime/DispatchEventsExport.dll']['identity'] == built['exporter']
    assert sum(n.startswith('fixtures/') and n.endswith('.bin') for n in links) == 45
    save(bundle / 'stage.json', dict(passed=True, diagnostic_only=True, links=links,
        products=payload['products'], external=payload['external'], feed=payload['feed'],
        interpreter=payload['interpreter'], exporter=built['exporter'], capture_consumer=trace['consumer'],
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'), str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): tar.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE / 'prepared.json', dict(passed=True, files=originals,
        stage=pin(bundle / 'stage.json'), archive=pin(BASE / 'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE / 'payload.tar.gz'), links=len(links), linked_bytes=sum(v['identity']['bytes'] for v in links.values()))))


if __name__ == '__main__': prepare()
