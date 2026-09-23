"""Freeze actual DLLs, preserved capture provenance and the common numerical consumer."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin,read,save
from fixtures import verify_prefixes

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-wide-entry-first-use-screen-amd-20260923'
NUMERICS=ROOT/'artifacts/parakeet-wide-entry-first-use-numerics-amd-20260923'
BUILD=ROOT/'artifacts/parakeet-wide-entry-first-use-build-amd-20260923'
CURRENT=ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
SOURCE=ROOT/'artifacts/parakeet-wide-entry-first-use-source-v2-20260923'
CAPTURE=ROOT/'artifacts/parakeet-wide-matmul-v3-20260921'
QUALIFIED=ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923'


def previous_closed():
    for folder,digest in [(BUILD,'da923692f2c97cbff2774006f4a3dc911aaca39e0af5b35a89639443a75ead58'),
                          (QUALIFIED,'62141a2a722548697c106e42b2c0d9425b4f0c6ce166611a5bc3ca26a4fccdd0'),
                          (CURRENT,'2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert pin(CAPTURE/'capture-closed.json')['sha256']=='41c4bf08f7050935e2e7cec4d2a9f1aa8e57eed4e521862bfebb3b0b2a8c67df'
    proof=read(CAPTURE/'capture-closed.json');assert proof['passed'] and proof['independent_onnx_weights']
    for name,wanted in proof['files'].items():assert pin(ROOT/name)==wanted,name
    assert pin(SOURCE/'prepared.json')['sha256']=='72c22bee93652ed8d6a759c2a984d965fa341697e869cf7a2461a908727483f6'
    source=read(SOURCE/'prepared.json')
    for name,wanted in source['before'].items():assert pin(ROOT/name)==wanted,name
    assert read(BUILD/'analysis.json')['inventory']['unchanged_core_bodies']==3181
    assert read(BUILD/'analysis.json')['inventory']['shared_flags_exact']
    for name in ['Screen.cs','Prototype.csproj','fixtures.py','score.py','test_score.py','protocol.py']:
        assert pin(TOOLS/name)==pin(ROOT/'tests/parakeet/short-dispatch-screen'/name),name
    assert pin(NUMERICS/'closed.json')['sha256']=='f5613c62f9bda671777ec0e0efca867a887400029339cc022cf7e8ef14ff0e7e'
    numeric=read(NUMERICS/'closed.json');assert numeric['passed'] and numeric['numerically_admitted']
    for name,wanted in numeric['files'].items():assert pin(NUMERICS/name)==wanted,name
    review=ROOT/'tests/parakeet/wide-entry-first-use-results/codegen-review-20260923.json'
    assert pin(review)['sha256']=='8b769467aec1819cd4bbbf0fca60c3069c3d3e161e71ccb71429ee46292370a2'
    assert read(review)['passed'] and read(review)['closure']==pin(NUMERICS/'closed.json')
    assert read(review)['mechanism_admitted']
    for name,wanted in read(review)['files'].items():assert pin(ROOT/name)==wanted,name


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['Screen.cs','Prototype.csproj']:copy(TOOLS/name,bundle/'source/consumer'/name)
    copy(ROOT/'global.json',bundle/'source/global.json')
    for name in ['protocol.py','remote.py','remote_prepare.py']:copy(TOOLS/name,bundle/'tools'/name)
    copy(TOOLS/'README.md',bundle/'README.md')
    shutil.copy2(ROOT/'.agent/m54-wide-projection-first-use-entry-20260923.md',bundle/'prospective-plan.md')
    products={}
    for role,folder in [('current',CURRENT/'collected/runtimes/current'),('candidate',BUILD/'collected/runtime')]:
        for p in folder.iterdir():
            if p.is_file():copy(p,bundle/'runtimes'/role/p.name)
        products[role]={name:pin(bundle/'runtimes'/role/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
    qualified=read(QUALIFIED/'analysis.json')
    assert qualified['inventory']==dict(passed=True,core_methods=3179,data_methods=697,public_surface_equal=True)
    assert products['current']==qualified['measured']
    assert read(BUILD/'analysis.json')['measured']==read(ROOT/'artifacts/parakeet-wide-projection-isolation-build-amd-20260923/analysis.json')['built']
    assert products['candidate']==read(BUILD/'analysis.json')['built']
    for name in ['closed.json','payload.json']:copy(BUILD/name,bundle/'evidence'/('build-'+name))
    for name in ['closed.json','analysis.json']:copy(QUALIFIED/name,bundle/'evidence'/('qualified-'+name))
    copy(BUILD/'collected/collection.json',bundle/'evidence/build-collection.json')
    for name in ['closed.json','payload.json']:copy(NUMERICS/name,bundle/'evidence'/('numerics-'+name))
    copy(NUMERICS/'collected/collection.json',bundle/'evidence/numerics-collection.json')
    copy(ROOT/'tests/parakeet/wide-entry-first-use-results/codegen-review-20260923.json',bundle/'evidence/codegen-review.json')
    copy(CAPTURE/'capture-closed.json',bundle/'evidence/capture-closed.json')
    capture=read(NUMERICS/'bundle/fixtures/result.json');assert capture['passed'] and len(capture['entries'])==21
    copy(NUMERICS/'bundle/evidence/original-capture.json',bundle/'evidence/original-capture.json')
    numeric_payload=read(NUMERICS/'payload.json')
    for p in sorted((NUMERICS/'bundle/fixtures').iterdir()):
        if p.is_file():
            assert pin(p)==numeric_payload['files']['fixtures/'+p.name]
            copy(p,bundle/'fixtures'/p.name)
    assert verify_prefixes(read(bundle/'evidence/original-capture.json'),capture,bundle/'fixtures')
    arrays=[p for p in (bundle/'fixtures').iterdir() if p.suffix=='.bin'];assert len(arrays)==45
    links={p.relative_to(bundle).as_posix():dict(source='/dev/shm/lokad-parakeet-wide-entry-first-use-numerics-20260923/'+p.relative_to(bundle).as_posix(),identity=pin(p))
        for p in (bundle/'fixtures').iterdir() if p.is_file()}
    save(bundle/'stage.json',dict(passed=True,products=products,root_product_changed=False,
        fixture_links=links,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file() and not p.relative_to(bundle).as_posix().startswith('fixtures/'):
                archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),products=products,fixture_arrays=len(arrays))))


if __name__=='__main__':prepare()
