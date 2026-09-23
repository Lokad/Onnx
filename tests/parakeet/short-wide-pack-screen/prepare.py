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
BASE=ROOT/'artifacts/parakeet-short-wide-pack-screen-amd-20260923'
NUMERICS=ROOT/'artifacts/parakeet-short-wide-pack-numerics-amd-20260923'
BUILD=ROOT/'artifacts/parakeet-short-wide-pack-build-amd-20260923'
CURRENT=ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
SOURCE=ROOT/'artifacts/parakeet-short-wide-pack-source-20260923'
CAPTURE=ROOT/'artifacts/parakeet-wide-matmul-v3-20260921'


def previous_closed():
    for folder,digest in [(BUILD,'23746895d6c89a887f55cc7e15a2a19907d9753e3b348a9959e515aa6c060578'),
                          (CURRENT,'2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert pin(CAPTURE/'capture-closed.json')['sha256']=='41c4bf08f7050935e2e7cec4d2a9f1aa8e57eed4e521862bfebb3b0b2a8c67df'
    proof=read(CAPTURE/'capture-closed.json');assert proof['passed'] and proof['independent_onnx_weights']
    for name,wanted in proof['files'].items():assert pin(ROOT/name)==wanted,name
    assert pin(SOURCE/'prepared.json')['sha256']=='682e600da428c03b80411a5ae72150ac6dca9bddf64b1886e14805b2b6ff41cb'
    source=read(SOURCE/'prepared.json')
    for name,wanted in source['before'].items():assert pin(ROOT/name)==wanted,name
    assert read(BUILD/'analysis.json')['inventory']['unchanged_core_methods']==3178
    assert pin(NUMERICS/'closed.json')['sha256']=='335cb9bc6727f3845f12aeaf47e2c0e207f8b6992bfdd2b2c9bb35595b008335'
    numeric=read(NUMERICS/'closed.json');assert numeric['passed'] and numeric['numerically_admitted']
    for name,wanted in numeric['files'].items():assert pin(NUMERICS/name)==wanted,name
    review=ROOT/'tests/parakeet/short-wide-pack-results/codegen-review-20260923.json'
    assert pin(review)['sha256']=='56410c54d0d555892f1bf57b382dfdfc2ee8a83c2b3d4730b2da4875471c3e6f'
    assert read(review)['passed'] and read(review)['closure']==pin(NUMERICS/'closed.json')


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['Screen.cs','Prototype.csproj']:copy(TOOLS/name,bundle/'source/consumer'/name)
    copy(ROOT/'global.json',bundle/'source/global.json')
    for name in ['protocol.py','remote.py','remote_prepare.py']:copy(TOOLS/name,bundle/'tools'/name)
    copy(TOOLS/'README.md',bundle/'README.md')
    shutil.copy2(ROOT/'.agent/m40-parakeet-short-wide-pack-20260923.md',bundle/'prospective-plan.md')
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
    copy(ROOT/'tests/parakeet/short-wide-pack-results/codegen-review-20260923.json',bundle/'evidence/codegen-review.json')
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
