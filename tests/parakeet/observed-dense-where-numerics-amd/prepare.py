"""Bind the unchanged 235-case numerical census to the observed-mask composition."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save
from fixtures import cases

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-observed-dense-where-numerics-amd-20260924'
BUILD = ROOT/'artifacts/parakeet-observed-dense-where-inventory-amd-20260924'
SOURCE = ROOT/'artifacts/parakeet-observed-dense-where-source-20260924'
CAPTURE = ROOT/'artifacts/parakeet-scalar-where-layout-amd-20260923'
REFERENCE = ROOT/'artifacts/parakeet-scalar-where-numerics-amd-v3-20260924'
RETAINED = ROOT/'artifacts/parakeet-dense-scalar-where-numerics-amd-20260924'
ORIGINAL = TOOLS.parent/'dense-scalar-where-numerics-amd'


def previous_closed():
    for folder,digest in [(BUILD,'7d5296b266feeacbf9c52bf57a689de2cdf0a839045130f15912912f5da1b7bd'),
        (CAPTURE,'bc0942f51e27e70f33ae9fdc994c345fe3c620159478ced1a40d5433777f66d8'),
        (REFERENCE,'6493136351aa15a5b96b62e98a557d2fcc4f09191712bccf056e6a827e8705dc'),
        (RETAINED,'edd4cf6e9bc9eccb1154df66f9e26419f0c28ebe789962d1d2618d85c1802832')]:
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); assert proof['passed']
        for name,wanted in proof['files'].items(): assert pin(folder/name) == wanted,name
    assert pin(SOURCE/'prepared.json')['sha256'] == '41f2477c7060c3f543471bebe991c77ea3b853d724708fec098a2cbb505cb0a2'
    source = read(SOURCE/'prepared.json'); assert source['passed']
    for name,wanted in source['source'].items(): assert pin(SOURCE/'source'/name) == wanted,name
    for name,wanted in source['before'].items(): assert pin(ROOT/name) == wanted,name
    build = read(BUILD/'analysis.json'); assert build['source_prepared'] == pin(SOURCE/'prepared.json')
    assert build['measured'] == source['measured']
    scope = build['inventory']
    assert scope['candidate_core_methods'] == 3253 and scope['unchanged_core_methods'] == 3250
    assert scope['generic_tensor_where_exact'] and scope['retained_provider_and_helper_bodies_exact']
    assert scope['all_existing_flags_exact'] and scope['public_surface_equal'] and not scope['new_composition_numerically_qualified']
    for name in ['Driver.cs','Contracts.cs','Prototype.csproj','fixtures.py','remote.py','audit.py']:
        assert pin(TOOLS/name) == pin(ORIGINAL/name),name
    expected = (ORIGINAL/'protocol.py').read_text().replace('preflight_available=12*GIB, build_preflight_available=12*GIB','preflight_available=11*GIB, build_preflight_available=11*GIB')
    assert (TOOLS/'protocol.py').read_text() == expected


def prepare():
    assert not BASE.exists(); previous_closed(); BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = {}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in ['Driver.cs','Contracts.cs','Prototype.csproj']: copy(TOOLS/name,bundle/'source/consumer'/name)
    copy(ROOT/'global.json',bundle/'source/global.json')
    for name in ['protocol.py','remote.py','remote_prepare.py']: copy(TOOLS/name,bundle/'tools'/name)
    copy(TOOLS/'README.md',bundle/'README.md')
    shutil.copy2(ROOT/'.agent/m70-parakeet-observed-dense-where-20260924.md',bundle/'prospective-plan.md')
    products = {}
    for role,folder in [('current',BUILD/'collected/measured'),('candidate',BUILD/'collected/runtime')]:
        for p in folder.iterdir():
            if p.is_file(): copy(p,bundle/'runtimes'/role/p.name)
        products[role] = {name:pin(bundle/'runtimes'/role/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
    build = read(BUILD/'analysis.json'); assert products == dict(current=build['measured'],candidate=build['built'])
    for label,folder in [('build',BUILD),('capture',CAPTURE)]:
        for name in ['closed.json','analysis.json','payload.json']: copy(folder/name,bundle/'evidence'/(label+'-'+name))
        copy(folder/'collected/collection.json',bundle/'evidence'/(label+'-collection.json'))
    copy(SOURCE/'prepared.json',bundle/'evidence/source-prepared.json')
    copy(REFERENCE/'closed.json',bundle/'evidence/numerical-reference-closed.json')
    copy(REFERENCE/'collected/current-numerics-256/result.json',bundle/'evidence/numerical-reference.json')
    copy(RETAINED/'closed.json',bundle/'evidence/retained-numerics-closed.json')
    capture = read(CAPTURE/'collected/capture/result.json'); census = cases(capture)
    assert census == read(RETAINED/'bundle/cases.json') and len(census) == 235
    save(bundle/'cases.json',census)
    links = {}
    for fixture in capture['fixtures']:
        for item in [*fixture['inputs'],fixture['output']]:
            name = item['file']; path = CAPTURE/'collected/capture'/name
            assert pin(path) == {k:item[k] for k in ['bytes','sha256']}
            copy(path,bundle/'fixtures'/name)
            links['fixtures/'+name] = dict(source='/dev/shm/lokad-parakeet-scalar-where-layout-20260923/capture/'+name,identity=pin(path))
    copy(CAPTURE/'collected/collection.json',bundle/'evidence/fixture-collection.json')
    copy(CAPTURE/'collected/capture/result.json',bundle/'evidence/capture-result.json')
    save(bundle/'stage.json',dict(passed=True,products=products,root_product_changed=False,fixture_links=links,
        cases=len(census),codegen_cases=220,files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix == '.py': ast.parse(p.read_text(encoding='utf8'),str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    for name in ['Driver.cs','Contracts.cs','Prototype.csproj','fixtures.py','remote.py','audit.py','protocol.py']:
        originals[(ORIGINAL/name).relative_to(ROOT).as_posix()] = pin(ORIGINAL/name)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file() and not p.relative_to(bundle).as_posix().startswith('fixtures/'):
                archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),cases=len(census),arrays=len(links),products=products)))


if __name__ == '__main__': prepare()
