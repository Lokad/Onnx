"""Freeze a prospective M46 build only after the exact M43 parent is selected."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-last-axis-pad-build-amd-20260923'
SOURCE = ROOT / 'artifacts/parakeet-last-axis-pad-source-20260923'
QUALIFIED = ROOT / 'artifacts/parakeet-first-use-kernels-root-amd-20260923'


def previous_closed():
    source = read(SOURCE / 'prepared.json')
    assert source['passed'] and not source['root_product_changed'] and not source['built']
    assert len(source['source']) == 423 and len(source['before']) == 421
    assert source['changed'] == ['src/Lokad.Onnx/CPUExecutionProvider.Shape.cs',
        'src/Lokad.Onnx/Zzz.LastAxisPad.cs', 'tests/Lokad.Onnx.Backend.Tests/LastAxisPadTests.cs']
    for name, wanted in source['source'].items():
        assert pin(SOURCE / 'source' / name) == wanted, name
    for name, wanted in source['before'].items():
        assert pin(ROOT / name) == wanted, name
    for name, key in [('candidate.patch', 'patch'), ('prospective-plan.md', 'plan')]:
        assert pin(SOURCE / name) == source[key]
    for name, wanted in source['tools'].items():
        assert pin(ROOT / 'tests/parakeet/last-axis-pad-source' / name) == wanted, name
    assert pin(QUALIFIED / 'closed.json') == source['qualified_parent']
    proof = read(QUALIFIED / 'closed.json')
    assert proof['passed']
    for name, wanted in proof['files'].items():
        assert pin(QUALIFIED / name) == wanted, name
    analysis = read(QUALIFIED / 'analysis.json')
    assert analysis['passed'] and analysis['root_source_verified']


def prepare():
    assert not BASE.exists()
    previous_closed()
    BASE.mkdir()
    bundle = BASE / 'bundle'
    bundle.mkdir()
    originals = {}

    def copy(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)

    source = read(SOURCE / 'prepared.json')
    for name in source['source']:
        copy(SOURCE / 'source' / name, bundle / 'source' / name)
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'checks.py']:
        copy(TOOLS / name, bundle / 'tools' / name)
    for name in ['Bridge.dll', 'Bridge.deps.json', 'Bridge.runtimeconfig.json']:
        copy(QUALIFIED / 'collected/bridge' / name, bundle / 'bridge' / name)
    for name in ['closed.json', 'analysis.json', 'payload.json']:
        copy(QUALIFIED / name, bundle / 'evidence' / name)
    copy(QUALIFIED / 'collected/collection.json', bundle / 'evidence/collection.json')
    for name in ['prepared.json', 'candidate.patch', 'prospective-plan.md']:
        copy(SOURCE / name, bundle / 'evidence' / ('source-' + name))
    copy(TOOLS / 'README.md', bundle / 'prospective-build.md')
    current = QUALIFIED / 'collected/runtime'
    stage = dict(passed=True, source_prepared=pin(SOURCE / 'prepared.json'),
        measured={n: pin(current / n) for n in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']},
        measured_files={p.name: pin(p) for p in current.iterdir() if p.is_file()},
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    assert stage['measured'] == read(QUALIFIED / 'analysis.json')['built']
    assert sum(n.startswith('source/') for n in stage['files']) == 423
    save(bundle / 'stage.json', stage)
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix == '.py':
                ast.parse(p.read_text(encoding='utf8'), str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():
                archive.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE / 'prepared.json', dict(passed=True, files=originals,
        stage=pin(bundle / 'stage.json'), archive=pin(BASE / 'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE / 'payload.tar.gz'), stage=pin(bundle / 'stage.json'), source_files=423)))


if __name__ == '__main__':
    prepare()
