"""Pin qualified selected/candidate binaries and the 123-case prospective census."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save
from fixtures import cases

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-scalar-where-numerics-amd-20260924'
BUILD = ROOT / 'artifacts/parakeet-scalar-where-build-amd-v2-20260924'
CURRENT = ROOT / 'artifacts/parakeet-wide-entry-first-use-build-amd-20260923'
SOURCE = ROOT / 'artifacts/parakeet-scalar-where-source-v2-20260924'
RELEASE = ROOT / 'artifacts/parakeet-wide-entry-first-use-root-amd-v2-20260923'
CAPTURE = ROOT / 'artifacts/parakeet-scalar-where-layout-amd-20260923'


def previous_closed():
    for folder, digest in [(BUILD, '5a9cc8c861ef9da4d9464d9eb05a01de33576b77c0f7085ed74dbd45537e05ed'),
        (CURRENT, 'da923692f2c97cbff2774006f4a3dc911aaca39e0af5b35a89639443a75ead58'),
        (RELEASE, '16d570819ab69915fe34fa6c5a4efb79d45ae792dbd5b1448c65645fa0d55f73'),
        (CAPTURE, 'bc0942f51e27e70f33ae9fdc994c345fe3c620159478ced1a40d5433777f66d8')]:
        assert pin(folder / 'closed.json')['sha256'] == digest
        proof = read(folder / 'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder / name) == wanted, name
    assert pin(SOURCE / 'prepared.json')['sha256'] == '86022fda5f1c9d076133f9c12062fc5ecfe57f4fdba84fd36e2c8d2ee7ff51e1'
    source = read(SOURCE / 'prepared.json'); assert source['passed']
    for name, wanted in source['source'].items(): assert pin(SOURCE / 'source' / name) == wanted, name
    for name, wanted in source['before'].items(): assert pin(ROOT / name) == wanted, name
    build = read(BUILD / 'analysis.json'); assert build['source_prepared'] == pin(SOURCE / 'prepared.json')
    assert build['measured'] == read(CURRENT / 'analysis.json')['built'] == read(RELEASE / 'analysis.json')['measured']
    scope = build['inventory']; assert scope['candidate_core_methods'] == 3190 and scope['unchanged_core_methods'] == 3188
    assert scope['original_body_and_edges_exact'] and scope['all_existing_flags_exact'] and scope['public_surface_equal']


def prepare():
    assert not BASE.exists(); previous_closed(); BASE.mkdir(); bundle = BASE / 'bundle'; bundle.mkdir(); originals = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in ['Driver.cs', 'Prototype.csproj']: copy(TOOLS / name, bundle / 'source/consumer' / name)
    copy(ROOT / 'global.json', bundle / 'source/global.json')
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py']: copy(TOOLS / name, bundle / 'tools' / name)
    copy(TOOLS / 'README.md', bundle / 'README.md')
    shutil.copy2(ROOT / '.agent/m56-parakeet-scalar-where-20260923.md', bundle / 'prospective-plan.md')
    products = {}
    for role, folder in [('current', CURRENT / 'collected/runtime'), ('candidate', BUILD / 'collected/runtime')]:
        for p in folder.iterdir():
            if p.is_file(): copy(p, bundle / 'runtimes' / role / p.name)
        products[role] = {name: pin(bundle / 'runtimes' / role / name) for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']}
    for label, folder in [('build', BUILD), ('current-build', CURRENT), ('release', RELEASE), ('capture', CAPTURE)]:
        for name in ['closed.json', 'analysis.json', 'payload.json']:
            copy(folder / name, bundle / 'evidence' / (label + '-' + name))
        copy(folder / 'collected/collection.json', bundle / 'evidence' / (label + '-collection.json'))
    copy(SOURCE / 'prepared.json', bundle / 'evidence/source-prepared.json')
    capture = read(CAPTURE / 'collected/capture/result.json'); census = cases(capture)
    save(bundle / 'cases.json', census)
    fixture_links = {}
    for fixture in capture['fixtures']:
        for item in [*fixture['inputs'], fixture['output']]:
            name = item['file']; source = CAPTURE / 'collected/capture' / name
            assert pin(source) == {k: item[k] for k in ['bytes', 'sha256']}
            copy(source, bundle / 'fixtures' / name)
            fixture_links['fixtures/' + name] = dict(source='/dev/shm/lokad-parakeet-scalar-where-layout-20260923/capture/' + name, identity=pin(source))
    copy(CAPTURE / 'collected/collection.json', bundle / 'evidence/fixture-collection.json')
    copy(CAPTURE / 'collected/capture/result.json', bundle / 'evidence/capture-result.json')
    save(bundle / 'stage.json', dict(passed=True, products=products, root_product_changed=False, fixture_links=fixture_links,
        cases=len(census), codegen_cases=32, files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix == '.py': ast.parse(p.read_text(), str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file() and not p.relative_to(bundle).as_posix().startswith('fixtures/'):
                archive.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    save(BASE / 'prepared.json', dict(passed=True, files=originals, stage=pin(bundle / 'stage.json'), archive=pin(BASE / 'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE / 'payload.tar.gz'), stage=pin(bundle / 'stage.json'), cases=len(census), arrays=len(fixture_links))))


if __name__ == '__main__': prepare()
