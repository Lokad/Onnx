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
BASE=ROOT/'artifacts/parakeet-first-use-kernels-screen-amd-20260923'
NUMERICS=ROOT/'artifacts/parakeet-first-use-kernels-numerics-amd-20260923'
BUILD=ROOT/'artifacts/parakeet-first-use-kernels-build-amd-20260923'
CURRENT=ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
SOURCE=ROOT/'artifacts/parakeet-first-use-kernels-source-20260923'
CAPTURE=ROOT/'artifacts/parakeet-wide-matmul-v3-20260921'


def previous_closed():
    for folder,digest in [(BUILD,'2fb4e3e587ab463a965d7cd4290ffe3f37674182bb47b7ee529d040825c3f243'),
                          (CURRENT,'2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert pin(CAPTURE/'capture-closed.json')['sha256']=='41c4bf08f7050935e2e7cec4d2a9f1aa8e57eed4e521862bfebb3b0b2a8c67df'
    proof=read(CAPTURE/'capture-closed.json');assert proof['passed'] and proof['independent_onnx_weights']
    for name,wanted in proof['files'].items():assert pin(ROOT/name)==wanted,name
    assert pin(SOURCE/'prepared.json')['sha256']=='829e26d7acd55ccf969f4292949abc19385a014f562517344a3265d42a0f51c0'
    source=read(SOURCE/'prepared.json')
    for name,wanted in source['before'].items():assert pin(ROOT/name)==wanted,name
    assert read(BUILD/'analysis.json')['inventory']['unchanged_core_methods']==3178
    assert read(BUILD/'analysis.json')['previous_inventory']['only_four_float_flags']
    for name in ['Screen.cs','Prototype.csproj','fixtures.py','score.py','test_score.py','protocol.py']:
        assert pin(TOOLS/name)==pin(ROOT/'tests/parakeet/short-dispatch-screen'/name),name
    assert pin(NUMERICS/'closed.json')['sha256']=='8d59a4d948ed0dc60be07e3427acd6b832199e2f3e598e18bde4c0abe90497ed'
    numeric=read(NUMERICS/'closed.json');assert numeric['passed'] and numeric['numerically_admitted']
    for name,wanted in numeric['files'].items():assert pin(NUMERICS/name)==wanted,name
    review=ROOT/'tests/parakeet/first-use-kernels-results/codegen-review-20260923.json'
    assert pin(review)['sha256']=='21d2097cbcb7b2c58a5b2223a6edebf36c32928c76956a3923f8822144b11f0d'
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
    shutil.copy2(ROOT/'.agent/m43-parakeet-first-use-kernels-20260923.md',bundle/'prospective-plan.md')
    products={}
    for role,folder in [('current',CURRENT/'collected/runtimes/current'),('candidate',BUILD/'collected/runtime')]:
        for p in folder.iterdir():
            if p.is_file():copy(p,bundle/'runtimes'/role/p.name)
        products[role]={name:pin(bundle/'runtimes'/role/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
    assert products['current']==read(BUILD/'analysis.json')['measured']
    assert products['candidate']==read(BUILD/'analysis.json')['built']
    for name in ['closed.json','payload.json']:copy(BUILD/name,bundle/'evidence'/('build-'+name))
    copy(BUILD/'collected/collection.json',bundle/'evidence/build-collection.json')
    for name in ['closed.json','payload.json']:copy(NUMERICS/name,bundle/'evidence'/('numerics-'+name))
    copy(NUMERICS/'collected/collection.json',bundle/'evidence/numerics-collection.json')
    copy(ROOT/'tests/parakeet/first-use-kernels-results/codegen-review-20260923.json',bundle/'evidence/codegen-review.json')
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
    save(bundle/'stage.json',dict(passed=True,products=products,root_product_changed=False,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),products=products,fixture_arrays=len(arrays))))


if __name__=='__main__':prepare()
