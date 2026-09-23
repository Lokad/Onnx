"""Freeze the M49 padding build against selected root 94a550de, without M43."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-pad-first-use-build-amd-20260923'
SOURCE = ROOT / 'artifacts/parakeet-pad-first-use-source-20260923'
QUALIFIED = ROOT / 'artifacts/pyannote-winograd-product-root-amd-20260923'
INSPECTOR = ROOT / 'artifacts/parakeet-first-use-kernels-build-amd-20260923'


def previous_closed():
    source = read(SOURCE / 'prepared.json')
    assert source['passed'] and not source['root_product_changed'] and not source['built']
    diagnostic = ROOT / 'artifacts/parakeet-pad-runtime-diagnostic-amd-20260923/closed.json'
    assert source['diagnostic'] == pin(diagnostic)
    assert source['diagnostic']['sha256'] == '2e718d3095eaea8ec79fb263d5ca504564a5e5a83e986ef5535c53505b3864de'
    assert len(source['source']) == 422 and len(source['before']) == 420
    assert source['changed'] == ['src/Lokad.Onnx/CPUExecutionProvider.Shape.cs',
        'src/Lokad.Onnx/Zzz.LastAxisPadDispatch.cs', 'tests/Lokad.Onnx.Backend.Tests/LastAxisPadTests.cs']
    for name, wanted in source['source'].items():
        assert pin(SOURCE / 'source' / name) == wanted, name
    for name, wanted in source['before'].items():
        assert pin(ROOT / name) == wanted, name
    for name, key in [('candidate.patch', 'patch'), ('prospective-plan.md', 'plan')]:
        assert pin(SOURCE / name) == source[key]
    for name, wanted in source['tools'].items():
        assert pin(ROOT / 'tests/parakeet/pad-first-use-source' / name) == wanted, name
    assert pin(QUALIFIED / 'closed.json') == source['qualified_parent']
    assert source['qualified_parent']['sha256'] == '62141a2a722548697c106e42b2c0d9425b4f0c6ce166611a5bc3ca26a4fccdd0'
    proof = read(QUALIFIED / 'closed.json')
    assert proof['passed']
    for name, wanted in proof['files'].items():
        assert pin(QUALIFIED / name) == wanted, name
    analysis = read(QUALIFIED / 'analysis.json')
    assert analysis['passed'] and analysis['root_source_verified']
    # Reuse the already built inspector with exact method-flag inventory. Its
    # qualification is independent of the rejected product's speed verdict.
    assert pin(INSPECTOR / 'closed.json')['sha256'] == '2fb4e3e587ab463a965d7cd4290ffe3f37674182bb47b7ee529d040825c3f243'
    inspector = read(INSPECTOR / 'closed.json')
    assert inspector['passed']
    for name in ['Bridge.dll', 'Bridge.deps.json', 'Bridge.runtimeconfig.json']:
        relative = 'collected/bridge/' + name
        assert pin(INSPECTOR / relative) == inspector['files'][relative]


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
        copy(INSPECTOR / 'collected/bridge' / name, bundle / 'bridge' / name)
    copy(INSPECTOR / 'closed.json', bundle / 'evidence/inspector-closed.json')
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
    assert sum(n.startswith('source/') for n in stage['files']) == 422
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
    print(json.dumps(dict(archive=pin(BASE / 'payload.tar.gz'), stage=pin(bundle / 'stage.json'), source_files=422)))


if __name__ == '__main__':
    prepare()
