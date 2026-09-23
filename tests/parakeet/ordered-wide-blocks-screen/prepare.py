"""Freeze the fixed complete-call comparison using the already compiled M54 consumer."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save
from fixtures import verify_prefixes
from consumer_scope import verify_scope

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-ordered-wide-blocks-screen-amd-20260923'
BUILD = ROOT / 'artifacts/parakeet-ordered-wide-blocks-build-amd-20260923'
NUMERICS = ROOT / 'artifacts/parakeet-ordered-wide-blocks-numerics-amd-20260923'
SOURCE = ROOT / 'artifacts/parakeet-ordered-wide-blocks-source-20260923'
CURRENT = ROOT / 'artifacts/parakeet-wide-entry-first-use-build-amd-20260923'
RELEASE = ROOT / 'artifacts/parakeet-wide-entry-first-use-root-amd-v2-20260923'
OLD = ROOT / 'artifacts/parakeet-wide-entry-first-use-screen-amd-20260923'
REVIEW = ROOT / 'tests/parakeet/ordered-wide-blocks-results/codegen-review-20260923.json'
CONSUMER = dict(bytes=20480, sha256='e1a75f61a0cb69064dd5e9a87a91feea1df0c42f173e054bf6fada5a16c1d78a')


def previous_closed():
    verify_scope()
    assert pin(SOURCE / 'prepared.json')['sha256'] == '2e546cad0194b575a139fa013a86411e8b5b9aba6c190ae9f639544a2bc7cf62'
    source = read(SOURCE / 'prepared.json'); assert source['passed']
    for name, wanted in source['before'].items(): assert pin(ROOT / name) == wanted, name
    for name, wanted in source['source'].items(): assert pin(SOURCE / 'source' / name) == wanted, name
    assert pin(CURRENT / 'closed.json')['sha256'] == 'da923692f2c97cbff2774006f4a3dc911aaca39e0af5b35a89639443a75ead58'
    assert pin(OLD / 'closed.json')['sha256'] == '5ca99bc1ac4b647c083104fc7f72a3cd19871c5c72499ff085c9475ca4b8cb8a'
    for folder in [BUILD, NUMERICS, CURRENT, RELEASE, OLD]:
        proof = read(folder / 'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder / name) == wanted, name
    build = read(BUILD / 'analysis.json'); current = read(CURRENT / 'analysis.json')['built']
    assert build['source_prepared'] == pin(SOURCE / 'prepared.json') and build['measured'] == current
    numeric = read(NUMERICS / 'analysis.json')
    assert numeric['numerically_admitted'] and numeric['products'] == dict(current=current, candidate=build['built'])
    release = read(RELEASE / 'analysis.json')
    assert release['root_source_verified'] and release['measured'] == current
    assert read(RELEASE / 'bundle/evidence/root-applied.json')['source_files'] == source['before']
    review = read(REVIEW)
    assert review['passed'] and review['mechanism_admitted'] and review['no_performance_measurement']
    assert review['closure'] == pin(NUMERICS / 'closed.json')
    assert review['numerical_values_per_worker'] == 8416305 and review['numerical_groups_per_worker'] == 76
    for name, wanted in review['files'].items(): assert pin(ROOT / name) == wanted, name
    assert read(OLD / 'collected/built.json')['consumer'] == CONSUMER
    assert read(OLD / 'analysis.json')['consumer'] == CONSUMER
    for name in ['Screen.cs', 'Prototype.csproj']:
        assert pin(TOOLS / name) == pin(OLD / 'bundle/source/consumer' / name), name


def prepare():
    assert not BASE.exists(); previous_closed(); BASE.mkdir(); bundle = BASE / 'bundle'; bundle.mkdir(); originals = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    copy(ROOT / 'global.json', bundle / 'source/global.json')
    for name in ['Screen.cs', 'Prototype.csproj']: copy(TOOLS / name, bundle / 'source/consumer' / name)
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py']: copy(TOOLS / name, bundle / 'tools' / name)
    copy(TOOLS / 'README.md', bundle / 'README.md')
    # Freeze the prospective snapshot in stage.json while the living plan advances.
    shutil.copy2(ROOT / '.agent/m55-ordered-wide-blocks-20260923.md', bundle / 'prospective-plan.md')
    products = {}; consumer_files = {}
    for role, folder in [('current', CURRENT / 'collected/runtime'), ('candidate', BUILD / 'collected/runtime')]:
        for p in folder.iterdir():
            if p.is_file(): copy(p, bundle / 'runtimes' / role / p.name)
        products[role] = {name: pin(bundle / 'runtimes' / role / name) for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']}
        for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
            name = 'ParakeetShortWideScreen.' + suffix
            target = bundle / 'runtimes' / role / name; assert not target.exists()
            copy(OLD / 'collected/runtimes/current' / name, target)
            consumer_files[target.relative_to(bundle).as_posix()] = pin(target)
        assert pin(bundle / 'runtimes' / role / 'ParakeetShortWideScreen.dll') == CONSUMER
    assert products == read(NUMERICS / 'analysis.json')['products']
    save(bundle / 'built.json', dict(passed=True, reused_consumer=True, consumer=CONSUMER,
         original_build=pin(OLD / 'collected/built.json'), files=consumer_files))
    for label, folder in [('build', BUILD), ('numerics', NUMERICS), ('current', CURRENT), ('release', RELEASE), ('consumer', OLD)]:
        for name in ['closed.json', 'analysis.json', 'payload.json']:
            copy(folder / name, bundle / 'evidence' / (label + '-' + name))
        copy(folder / 'collected/collection.json', bundle / 'evidence' / (label + '-collection.json'))
    copy(OLD / 'collected/built.json', bundle / 'evidence/consumer-built.json')
    copy(REVIEW, bundle / 'evidence/codegen-review.json')
    copy(SOURCE / 'prepared.json', bundle / 'evidence/source-prepared.json')
    capture = read(NUMERICS / 'bundle/fixtures/result.json'); assert capture['passed'] and len(capture['entries']) == 21
    copy(NUMERICS / 'bundle/evidence/original-capture.json', bundle / 'evidence/original-capture.json')
    numeric_payload = read(NUMERICS / 'payload.json')
    for p in sorted((NUMERICS / 'bundle/fixtures').iterdir()):
        if p.is_file():
            assert pin(p) == numeric_payload['files']['fixtures/' + p.name]
            copy(p, bundle / 'fixtures' / p.name)
    assert verify_prefixes(read(bundle / 'evidence/original-capture.json'), capture, bundle / 'fixtures')
    arrays = list((bundle / 'fixtures').glob('*.bin')); assert len(arrays) == 45
    links = {p.relative_to(bundle).as_posix(): dict(source='/dev/shm/lokad-parakeet-ordered-wide-blocks-numerics-20260923/' + p.relative_to(bundle).as_posix(), identity=pin(p))
             for p in (bundle / 'fixtures').iterdir() if p.is_file()}
    save(bundle / 'stage.json', dict(passed=True, products=products, consumer=CONSUMER, root_product_changed=False, fixture_links=links,
         files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in [*TOOLS.iterdir(), *(TOOLS.parent / 'wide-entry-first-use-screen' / name for name in ['Screen.cs', 'Prototype.csproj', 'fixtures.py', 'remote.py', 'protocol.py', 'run.py', 'audit.py'])]:
        if p.is_file():
            if p.suffix == '.py': ast.parse(p.read_text(), str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file() and not p.relative_to(bundle).as_posix().startswith('fixtures/'):
                archive.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE / 'prepared.json', dict(passed=True, files=originals, stage=pin(bundle / 'stage.json'), archive=pin(BASE / 'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE / 'payload.tar.gz'), stage=pin(BASE / 'stage.json'), products=products,
                          consumer=CONSUMER, reused_consumer=True, fixture_arrays=len(arrays))))


if __name__ == '__main__':
    prepare()
