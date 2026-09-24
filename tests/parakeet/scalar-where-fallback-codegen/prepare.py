"""Pin qualified selected/candidate binaries and the untimed122-case diagnostic census."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save
from fixtures import screen_cases

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-scalar-where-fallback-codegen-amd-20260924'
BUILD = ROOT / 'artifacts/parakeet-scalar-where-build-amd-v3-20260924'
CURRENT = ROOT / 'artifacts/parakeet-wide-entry-first-use-build-amd-20260923'
SOURCE = ROOT / 'artifacts/parakeet-scalar-where-source-v3-20260924'
RELEASE = ROOT / 'artifacts/parakeet-wide-entry-first-use-root-amd-v2-20260923'
CAPTURE = ROOT / 'artifacts/parakeet-scalar-where-layout-amd-20260923'
NUMERICAL = ROOT / 'artifacts/parakeet-scalar-where-numerics-amd-v3-20260924'
CODEGEN = ROOT / 'tests/parakeet/scalar-where-results/codegen-review-v3-20260924.json'
SCREEN = ROOT / 'artifacts/parakeet-scalar-where-screen-amd-20260924'


def previous_closed():
    for folder, digest in [(BUILD, '8f5958cb468b3a35582150de113cdaec61bb4389afea98ca3a7c40fa67071909'),
        (CURRENT, 'da923692f2c97cbff2774006f4a3dc911aaca39e0af5b35a89639443a75ead58'),
        (RELEASE, '16d570819ab69915fe34fa6c5a4efb79d45ae792dbd5b1448c65645fa0d55f73'),
        (CAPTURE, 'bc0942f51e27e70f33ae9fdc994c345fe3c620159478ced1a40d5433777f66d8'),
        (NUMERICAL, '6493136351aa15a5b96b62e98a557d2fcc4f09191712bccf056e6a827e8705dc'),
        (SCREEN, 'fd8477677a34b9064fa5294d1834b91a59a2284b6bbfbb0ff470c612a2c4949b')]:
        assert pin(folder / 'closed.json')['sha256'] == digest
        proof = read(folder / 'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder / name) == wanted, name
    assert read(SCREEN / 'closed.json')['performance_admitted'] is False
    assert pin(SOURCE / 'prepared.json')['sha256'] == '6971271ecaa50346aa43b5040dab55cf6c9a812282aedbe34b5f8d117a700c89'
    source = read(SOURCE / 'prepared.json'); assert source['passed']
    for name, wanted in source['source'].items(): assert pin(SOURCE / 'source' / name) == wanted, name
    for name, wanted in source['before'].items(): assert pin(ROOT / name) == wanted, name
    build = read(BUILD / 'analysis.json'); assert build['source_prepared'] == pin(SOURCE / 'prepared.json')
    assert build['measured'] == read(CURRENT / 'analysis.json')['built'] == read(RELEASE / 'analysis.json')['measured']
    scope = build['inventory']; assert scope['candidate_core_methods'] == 3190 and scope['unchanged_core_methods'] == 3188
    assert scope['original_body_and_edges_exact'] and scope['all_existing_flags_exact'] and scope['public_surface_equal']
    assert pin(CODEGEN)['sha256'] == 'b9181070de6d9421f69ac3251556e0e80a5aaeffd8aa5c44e2a963beaf5cb3a8'
    codegen = read(CODEGEN); assert codegen['passed'] and codegen['ready_for_fixed_screen']
    assert codegen['closure'] == pin(NUMERICAL / 'closed.json')
    qualified = (ROOT / 'tests/parakeet/scalar-where-numerics-v3/Driver.cs').read_text()
    setup = qualified.replace('static class Program', 'static partial class Program').replace('static int Main(string[] args)', 'static int NumericalMain(string[] args)')
    assert (TOOLS / 'QualifiedSetup.cs').read_text() == setup
    construction = qualified.split('    static void Run<T>(JsonElement spec) where T : unmanaged\n    {\n', 1)[1].split('        var before = ', 1)[0]
    actual = (TOOLS / 'PrepareCase.cs').read_text().split('    static Prepared<T> Prepare<T>(JsonElement spec) where T : unmanaged\n    {\n', 1)[1].split('        var expected = ', 1)[0]
    assert actual == construction


def prepare():
    assert not BASE.exists(); previous_closed(); BASE.mkdir(); bundle = BASE / 'bundle'; bundle.mkdir(); originals = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in ['QualifiedSetup.cs', 'PrepareCase.cs', 'Diagnostic.cs', 'Prototype.csproj']: copy(TOOLS / name, bundle / 'source/consumer' / name)
    copy(ROOT / 'global.json', bundle / 'source/global.json')
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py']: copy(TOOLS / name, bundle / 'tools' / name)
    copy(TOOLS / 'README.md', bundle / 'README.md')
    shutil.copy2(ROOT / '.agent/m56-parakeet-scalar-where-20260923.md', bundle / 'prospective-plan.md')
    products = {}
    for role, folder in [('current', CURRENT / 'collected/runtime'), ('candidate', BUILD / 'collected/runtime')]:
        for p in folder.iterdir():
            if p.is_file(): copy(p, bundle / 'runtimes' / role / p.name)
        products[role] = {name: pin(bundle / 'runtimes' / role / name) for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']}
    for label, folder in [('build', BUILD), ('current-build', CURRENT), ('release', RELEASE), ('capture', CAPTURE), ('numerical', NUMERICAL), ('screen', SCREEN)]:
        for name in ['closed.json', 'analysis.json', 'payload.json']:
            copy(folder / name, bundle / 'evidence' / (label + '-' + name))
        copy(folder / 'collected/collection.json', bundle / 'evidence' / (label + '-collection.json'))
    copy(SOURCE / 'prepared.json', bundle / 'evidence/source-prepared.json')
    copy(CODEGEN, bundle / 'evidence/codegen-review.json')
    copy(NUMERICAL / 'collected/current-numerics-256/result.json', bundle / 'evidence/qualified-reference.json')
    capture = read(CAPTURE / 'collected/capture/result.json'); census = screen_cases(capture, read(bundle / 'evidence/qualified-reference.json'))
    assert census == read(SCREEN / 'bundle/cases.json')
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
        cases=len(census), diagnostic_processes=2, files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()}))
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
