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
BASE=ROOT/'artifacts/parakeet-isolated-short-kernels-numerics-amd-20260923'
BUILD=ROOT/'artifacts/parakeet-isolated-short-kernels-build-amd-v2-20260923'
CURRENT=ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
SOURCE=ROOT/'artifacts/parakeet-isolated-short-kernels-source-20260923'
CAPTURE=ROOT/'artifacts/parakeet-wide-matmul-v3-20260921'
FIXTURES=ROOT/'artifacts/parakeet-short-wide-pack-numerics-amd-20260923'
QUALIFIED=ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923'


def previous_closed():
    for folder,digest in [(BUILD,'0adb72e2eae376df7d4f3dd6eb7f4a97b15dd57c5d4c3fa962f8a4cd3a9c45c6'),
                          (QUALIFIED,'62141a2a722548697c106e42b2c0d9425b4f0c6ce166611a5bc3ca26a4fccdd0'),
                          (CURRENT,'2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert pin(CAPTURE/'capture-closed.json')['sha256']=='41c4bf08f7050935e2e7cec4d2a9f1aa8e57eed4e521862bfebb3b0b2a8c67df'
    proof=read(CAPTURE/'capture-closed.json');assert proof['passed'] and proof['independent_onnx_weights']
    for name,wanted in proof['files'].items():assert pin(ROOT/name)==wanted,name
    assert pin(SOURCE/'prepared.json')['sha256']=='f6635fac6f21ec4832ea105fc7d4e64888db8716b1f4c345776652cc30989104'
    source=read(SOURCE/'prepared.json')
    for name,wanted in source['before'].items():assert pin(ROOT/name)==wanted,name
    assert read(BUILD/'analysis.json')['inventory']['unchanged_core_methods']==3175
    assert read(BUILD/'analysis.json')['inventory']['original_general_body_exact']
    assert read(BUILD/'analysis.json')['inventory']['existing_implementation_flags_equal']
    from consumer_scope import verify_consumer
    assert verify_consumer()
    for name in ['Prototype.csproj','fixtures.py']:
        assert pin(TOOLS/name)==pin(ROOT/'tests/parakeet/short-dispatch-numerics-v2'/name),name
    assert pin(FIXTURES/'closed.json')['sha256']=='335cb9bc6727f3845f12aeaf47e2c0e207f8b6992bfdd2b2c9bb35595b008335'
    for name,wanted in read(FIXTURES/'closed.json')['files'].items():assert pin(FIXTURES/name)==wanted,name


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['Driver.cs','Prototype.csproj']:copy(TOOLS/name,bundle/'source/consumer'/name)
    copy(ROOT/'global.json',bundle/'source/global.json')
    for name in ['protocol.py','remote.py','remote_prepare.py']:copy(TOOLS/name,bundle/'tools'/name)
    copy(TOOLS/'README.md',bundle/'README.md')
    shutil.copy2(ROOT/'.agent/m50-isolated-short-projection-kernels-20260923.md',bundle/'prospective-plan.md')
    products={}
    for role,folder in [('current',CURRENT/'collected/runtimes/current'),('candidate',BUILD/'collected/runtime')]:
        for p in folder.iterdir():
            if p.is_file():copy(p,bundle/'runtimes'/role/p.name)
        products[role]={name:pin(bundle/'runtimes'/role/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
    qualified=read(QUALIFIED/'analysis.json')
    assert qualified['inventory']==dict(passed=True,core_methods=3179,data_methods=697,public_surface_equal=True)
    assert products['current']==qualified['measured']
    assert read(BUILD/'analysis.json')['measured']==qualified['built']
    assert products['candidate']==read(BUILD/'analysis.json')['built']
    for name in ['closed.json','payload.json']:copy(BUILD/name,bundle/'evidence'/('build-'+name))
    for name in ['closed.json','analysis.json']:copy(QUALIFIED/name,bundle/'evidence'/('qualified-'+name))
    copy(BUILD/'collected/collection.json',bundle/'evidence/build-collection.json')
    copy(CAPTURE/'capture-closed.json',bundle/'evidence/capture-closed.json')
    capture=read(FIXTURES/'bundle/fixtures/result.json');assert capture['passed'] and len(capture['entries'])==21
    copy(FIXTURES/'bundle/evidence/original-capture.json',bundle/'evidence/original-capture.json')
    original=read(FIXTURES/'payload.json')
    for p in sorted((FIXTURES/'bundle/fixtures').iterdir()):
        if p.is_file():
            assert pin(p)==original['files']['fixtures/'+p.name]
            copy(p,bundle/'fixtures'/p.name)
    assert verify_prefixes(read(bundle/'evidence/original-capture.json'),capture,bundle/'fixtures')
    arrays=list((bundle/'fixtures').glob('*.bin'));assert len(arrays)==45
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
