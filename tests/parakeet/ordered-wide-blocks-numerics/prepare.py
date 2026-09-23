"""Freeze normally built M54/M55 binaries and the unchanged independent arithmetic oracle."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save
from fixtures import verify_prefixes
from consumer_scope import verify_consumer

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-ordered-wide-blocks-numerics-amd-20260923'
BUILD = ROOT / 'artifacts/parakeet-ordered-wide-blocks-build-amd-20260923'
CURRENT = ROOT / 'artifacts/parakeet-wide-entry-first-use-build-amd-20260923'
SOURCE = ROOT / 'artifacts/parakeet-ordered-wide-blocks-source-20260923'
RELEASE = ROOT / 'artifacts/parakeet-wide-entry-first-use-root-amd-20260923'
CAPTURE = ROOT / 'artifacts/parakeet-wide-matmul-v3-20260921'
FIXTURES = ROOT / 'artifacts/parakeet-short-wide-pack-numerics-amd-20260923'


def previous_closed():
    verify_consumer()
    assert pin(SOURCE / 'prepared.json')['sha256'] == '2e546cad0194b575a139fa013a86411e8b5b9aba6c190ae9f639544a2bc7cf62'
    source = read(SOURCE / 'prepared.json')
    assert source['passed'] and len(source['source']) == 423 and len(source['before']) == 422
    for name, wanted in source['source'].items(): assert pin(SOURCE / 'source' / name) == wanted, name
    for name, wanted in source['before'].items(): assert pin(ROOT / name) == wanted, name
    assert pin(CURRENT / 'closed.json')['sha256'] == 'da923692f2c97cbff2774006f4a3dc911aaca39e0af5b35a89639443a75ead58'
    assert pin(FIXTURES / 'closed.json')['sha256'] == '335cb9bc6727f3845f12aeaf47e2c0e207f8b6992bfdd2b2c9bb35595b008335'
    for folder in [BUILD, CURRENT, RELEASE, FIXTURES]:
        proof = read(folder / 'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder / name) == wanted, name
    build = read(BUILD / 'analysis.json'); parent = read(CURRENT / 'analysis.json')
    assert build['source_prepared'] == pin(SOURCE / 'prepared.json') and build['measured'] == parent['built']
    scope = build['inventory']
    assert scope['passed'] and scope['candidate_core_methods'] == 3192 and scope['unchanged_core_methods'] == 3188
    assert scope['all_existing_flags_exact'] and scope['original_edges_locals_exceptions_preserved']
    release = read(RELEASE / 'analysis.json')
    assert release['root_source_verified'] and release['measured'] == parent['built']
    assert read(RELEASE / 'bundle/evidence/root-applied.json')['source_files'] == source['before']
    assert pin(CAPTURE / 'capture-closed.json')['sha256'] == '41c4bf08f7050935e2e7cec4d2a9f1aa8e57eed4e521862bfebb3b0b2a8c67df'
    proof = read(CAPTURE / 'capture-closed.json'); assert proof['passed'] and proof['independent_onnx_weights']
    for name, wanted in proof['files'].items(): assert pin(ROOT / name) == wanted, name


def prepare():
    assert not BASE.exists(); previous_closed(); BASE.mkdir(); bundle = BASE / 'bundle'; bundle.mkdir(); originals = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in ['Driver.cs', 'Prototype.csproj']: copy(TOOLS / name, bundle / 'source/consumer' / name)
    copy(ROOT / 'global.json', bundle / 'source/global.json')
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py']: copy(TOOLS / name, bundle / 'tools' / name)
    copy(TOOLS / 'README.md', bundle / 'README.md')
    copy(ROOT / '.agent/m55-ordered-wide-blocks-20260923.md', bundle / 'prospective-plan.md')
    products = {}
    for role, folder in [('current', CURRENT / 'collected/runtime'), ('candidate', BUILD / 'collected/runtime')]:
        for p in folder.iterdir():
            if p.is_file(): copy(p, bundle / 'runtimes' / role / p.name)
        products[role] = {name: pin(bundle / 'runtimes' / role / name) for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']}
    assert products['current'] == read(CURRENT / 'analysis.json')['built']
    assert products['candidate'] == read(BUILD / 'analysis.json')['built']
    for label, folder in [('build', BUILD), ('current-build', CURRENT), ('release', RELEASE)]:
        for name in ['closed.json', 'analysis.json', 'payload.json']:
            copy(folder / name, bundle / 'evidence' / (label + '-' + name))
        copy(folder / 'collected/collection.json', bundle / 'evidence' / (label + '-collection.json'))
    copy(SOURCE / 'prepared.json', bundle / 'evidence/source-prepared.json')
    copy(CAPTURE / 'capture-closed.json', bundle / 'evidence/capture-closed.json')
    capture = read(FIXTURES / 'bundle/fixtures/result.json'); assert capture['passed'] and len(capture['entries']) == 21
    copy(FIXTURES / 'bundle/evidence/original-capture.json', bundle / 'evidence/original-capture.json')
    original = read(FIXTURES / 'payload.json')
    for p in sorted((FIXTURES / 'bundle/fixtures').iterdir()):
        if p.is_file():
            assert pin(p) == original['files']['fixtures/' + p.name]
            copy(p, bundle / 'fixtures' / p.name)
    assert verify_prefixes(read(bundle / 'evidence/original-capture.json'), capture, bundle / 'fixtures')
    arrays = list((bundle / 'fixtures').glob('*.bin')); assert len(arrays) == 45
    retained = ROOT / 'artifacts/parakeet-isolated-short-kernels-screen-amd-20260923'
    retained_payload = read(retained / 'payload.json')
    copy(retained / 'collected/collection.json', bundle / 'evidence/fixture-collection.json')
    fixture_links = {}
    for p in (bundle / 'fixtures').iterdir():
        if p.is_file():
            name = p.relative_to(bundle).as_posix(); assert pin(p) == retained_payload['files'][name]
            fixture_links[name] = dict(source='/dev/shm/lokad-parakeet-isolated-short-kernels-screen-20260923/' + name, identity=pin(p))
    save(bundle / 'stage.json', dict(passed=True, products=products, root_product_changed=False, fixture_links=fixture_links,
         files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in [*TOOLS.iterdir(), *(TOOLS.parent / 'wide-entry-first-use-numerics' / name for name in ['Driver.cs', 'Prototype.csproj', 'fixtures.py', 'protocol.py'])]:
        if p.is_file():
            if p.suffix == '.py': ast.parse(p.read_text(), str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file() and not p.relative_to(bundle).as_posix().startswith('fixtures/'):
                archive.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE / 'prepared.json', dict(passed=True, files=originals, stage=pin(bundle / 'stage.json'), archive=pin(BASE / 'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE / 'payload.tar.gz'), stage=pin(bundle / 'stage.json'), products=products, fixture_arrays=len(arrays))))


if __name__ == '__main__':
    prepare()
