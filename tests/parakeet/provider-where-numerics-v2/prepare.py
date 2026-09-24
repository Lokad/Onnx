"""Pin qualified selected/candidate binaries and the 203-case prospective census."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save
from fixtures import cases

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-provider-where-numerics-amd-v2-20260924'
BUILD = ROOT / 'artifacts/parakeet-provider-where-build-amd-20260924'
CURRENT = ROOT / 'artifacts/parakeet-wide-entry-first-use-build-amd-20260923'
SOURCE = ROOT / 'artifacts/parakeet-provider-where-source-20260924'
RELEASE = ROOT / 'artifacts/parakeet-wide-entry-first-use-root-amd-v2-20260923'
CAPTURE = ROOT / 'artifacts/parakeet-scalar-where-layout-amd-20260923'
REFERENCE = ROOT / 'artifacts/parakeet-scalar-where-numerics-amd-v3-20260924'


def previous_closed():
    failed = ROOT / 'artifacts/parakeet-provider-where-numerics-amd-20260924'
    assert pin(failed / 'closed.json')['sha256'] == '9bc68920404a05fd330454a9dab65c092cb28f9e82875ea409900f1e0939b550'
    proof = read(failed / 'closed.json'); assert not proof['passed']
    for name, wanted in proof['files'].items(): assert pin(failed / name) == wanted, name
    for folder, digest in [(BUILD, 'ae46692fdec50bb0184c79b38dfb603a897546b4d68735977c6ac7fb96245a89'),
        (CURRENT, 'da923692f2c97cbff2774006f4a3dc911aaca39e0af5b35a89639443a75ead58'),
        (RELEASE, '16d570819ab69915fe34fa6c5a4efb79d45ae792dbd5b1448c65645fa0d55f73'),
        (CAPTURE, 'bc0942f51e27e70f33ae9fdc994c345fe3c620159478ced1a40d5433777f66d8'),
        (REFERENCE, '6493136351aa15a5b96b62e98a557d2fcc4f09191712bccf056e6a827e8705dc')]:
        assert pin(folder / 'closed.json')['sha256'] == digest
        proof = read(folder / 'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder / name) == wanted, name
    assert pin(SOURCE / 'prepared.json')['sha256'] == '64460d0440c095691f80f0bfde61487a2d3654ac0263d5a34303a4edb3b292e8'
    source = read(SOURCE / 'prepared.json'); assert source['passed']
    for name, wanted in source['source'].items(): assert pin(SOURCE / 'source' / name) == wanted, name
    for name, wanted in source['before'].items(): assert pin(ROOT / name) == wanted, name
    build = read(BUILD / 'analysis.json'); assert build['source_prepared'] == pin(SOURCE / 'prepared.json')
    assert build['measured'] == read(CURRENT / 'analysis.json')['built'] == read(RELEASE / 'analysis.json')['measured']
    scope = build['inventory']; assert scope['candidate_core_methods'] == 3190 and scope['unchanged_core_methods'] == 3188
    assert scope['generic_tensor_where_exact'] and scope['existing_validation_and_nonfloat_edges_exact'] and scope['qualified_uniform_helper_exact'] and scope['all_existing_flags_exact'] and scope['public_surface_equal']


def consumer_scope():
    qualified = (ROOT / 'tests/parakeet/scalar-where-numerics-v3/Driver.cs').read_text()
    actual = (TOOLS / 'Driver.cs').read_text()
    # The complete layout construction, value generator and coordinate oracle stay exact.
    for start, end in [('    static void Require(', '    static void Run<T>('),
                       ('        string name = Text(spec,', '        var before = ')]:
        assert actual.split(start,1)[1].split(end,1)[0] == qualified.split(start,1)[1].split(end,1)[0]


def prepare():
    consumer_scope()
    assert not BASE.exists(); previous_closed(); BASE.mkdir(); bundle = BASE / 'bundle'; bundle.mkdir(); originals = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in ['Driver.cs', 'Contracts.cs', 'Prototype.csproj']: copy(TOOLS / name, bundle / 'source/consumer' / name)
    copy(ROOT / 'global.json', bundle / 'source/global.json')
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py']: copy(TOOLS / name, bundle / 'tools' / name)
    copy(TOOLS / 'README.md', bundle / 'README.md')
    shutil.copy2(ROOT / '.agent/m57-parakeet-provider-where-20260924.md', bundle / 'prospective-plan.md')
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
    failed = ROOT / 'artifacts/parakeet-provider-where-numerics-amd-20260924'
    copy(failed / 'closed.json', bundle / 'evidence/failed-numerical-closed.json')
    copy(failed / 'collected/collection.json', bundle / 'evidence/failed-numerical-collection.json')
    copy(REFERENCE / 'closed.json', bundle / 'evidence/numerical-reference-closed.json')
    copy(REFERENCE / 'collected/current-numerics-256/result.json', bundle / 'evidence/numerical-reference.json')
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
        cases=len(census), codegen_cases=188, files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()}))
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
