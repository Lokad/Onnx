"""Freeze the public Pad workload against the qualified padding build."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from census import census
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-last-axis-pad-screen-amd-20260923'
SOURCE = ROOT / 'artifacts/parakeet-last-axis-pad-source-20260923'
BUILD = ROOT / 'artifacts/parakeet-last-axis-pad-build-amd-20260923'
CURRENT = ROOT / 'artifacts/parakeet-winograd-baseline-amd-20260923'
REVIEW = ROOT / 'tests/parakeet/last-axis-pad-results/composition-20260923.json'
FRAMES = ROOT / 'artifacts/parakeet-first-use-kernels-source-20260923/census.json'
GRAPH = ROOT / 'tests/parakeet/current-profile-results/memory-source-observations-20260923.json'


def previous_closed():
    for folder, digest in [
        (BUILD, '62043d100a96d418b02e25d09a65d8f89655ea21147c27d15a5efd861cce7e1c'),
        (CURRENT, '2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3')]:
        assert pin(folder / 'closed.json')['sha256'] == digest
        proof = read(folder / 'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items():
            assert pin(folder / name) == wanted, name
    assert pin(SOURCE / 'prepared.json')['sha256'] == 'aaa54105b582d11fd62b6a3d02218a4e58e7e8e42750dbe3e42cea980a6ad4b0'
    for name, wanted in read(SOURCE / 'prepared.json')['before'].items():
        assert pin(ROOT / name) == wanted, name
    review = read(REVIEW)
    assert review['passed'] and review['build'] == pin(BUILD / 'closed.json')
    assert review['inventory'] == pin(BUILD / 'collected/inventory/instructions.json')
    assert pin(REVIEW.parent / 'review.py') == review['generator']
    assert review['original_fill_materialization_fallback_exact'] and review['branch_targets_preserved']
    assert review['candidate'] == read(BUILD / 'analysis.json')['built']
    assert [read(BUILD / 'analysis.json')['suites'][name]['passed'] for name in ['pad-tests', 'pad-tests-256']] == [6, 6]
    assert {51, 106, 167, 225}.issubset({r['frames'] for r in read(FRAMES)['rows']})
    assert pin(GRAPH)['sha256'] == '68ff8129363e5cc0a4b2a0c4610c7586ff284d82a33590063685bb8effaf70ec'


def prepare():
    assert not BASE.exists(); previous_closed()
    BASE.mkdir(); bundle = BASE / 'bundle'; bundle.mkdir(); originals = {}

    def copy(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)

    for name in ['Screen.cs', 'Prototype.csproj']:
        copy(TOOLS / name, bundle / 'source/consumer' / name)
    copy(ROOT / 'global.json', bundle / 'source/global.json')
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py']:
        copy(TOOLS / name, bundle / 'tools' / name)
    copy(TOOLS / 'README.md', bundle / 'README.md')
    shutil.copy2(ROOT / '.agent/m46-parakeet-last-axis-pad-20260923.md', bundle / 'prospective-plan.md')
    products = {}
    for role, folder in [('current', CURRENT / 'collected/runtimes/current'), ('candidate', BUILD / 'collected/runtime')]:
        for p in folder.iterdir():
            if p.is_file():
                copy(p, bundle / 'runtimes' / role / p.name)
        products[role] = {name: pin(bundle / 'runtimes' / role / name) for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']}
    assert products['current']['Lokad.Onnx.dll']['sha256'] == '521bae1702849ca23dda586515e7cbabaac2d1eabdff04dc90a7ba76059e93fb'
    assert products['current']['Lokad.Onnx.Data.dll']['sha256'] == 'f3b9aa81ee9766797e95216714dec559c5d8b020f8df510cf5f7ee0dda82693a'
    assert products['candidate'] == read(BUILD / 'analysis.json')['built']
    for name in ['closed.json', 'payload.json']:
        copy(BUILD / name, bundle / 'evidence' / ('build-' + name))
    copy(BUILD / 'collected/collection.json', bundle / 'evidence/build-collection.json')
    copy(REVIEW, bundle / 'evidence/composition.json')
    copy(FRAMES, bundle / 'evidence/frames.json')
    copy(GRAPH, bundle / 'evidence/graph-pads.json')
    save(bundle / 'census.json', census())
    save(bundle / 'stage.json', dict(passed=True, products=products, root_product_changed=False,
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'), str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): archive.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE / 'prepared.json', dict(passed=True, files=originals,
        stage=pin(bundle / 'stage.json'), archive=pin(BASE / 'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE / 'payload.tar.gz'), stage=pin(bundle / 'stage.json'), products=products)))


if __name__ == '__main__':
    prepare()
