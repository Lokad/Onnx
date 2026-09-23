"""Freeze actual DLLs, preserved capture provenance and the common numerical consumer."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin,read,save

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-wide-per-call-numerics-amd-20260923'
BUILD=ROOT/'artifacts/parakeet-wide-per-call-build-amd-20260923'
CURRENT=ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
SOURCE=ROOT/'artifacts/parakeet-wide-per-call-source-20260923'
CAPTURE=ROOT/'artifacts/parakeet-wide-matmul-v3-20260921'


def previous_closed():
    for folder,digest in [(BUILD,'3359287fcbde0cd4742744270437ce2b76ea53ea9107e878d72d41d869cc290d'),
                          (CURRENT,'2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert pin(CAPTURE/'capture-closed.json')['sha256']=='41c4bf08f7050935e2e7cec4d2a9f1aa8e57eed4e521862bfebb3b0b2a8c67df'
    proof=read(CAPTURE/'capture-closed.json');assert proof['passed'] and proof['independent_onnx_weights']
    for name,wanted in proof['files'].items():assert pin(ROOT/name)==wanted,name
    assert pin(SOURCE/'prepared.json')['sha256']=='60032db2e8892c170002516cd20f4fe972b4e0d78bf213f45ed45fed230a3baf'
    source=read(SOURCE/'prepared.json')
    for name,wanted in source['before'].items():assert pin(ROOT/name)==wanted,name
    assert read(BUILD/'analysis.json')['inventory']['unchanged_core_methods']==3178


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['Driver.cs','Prototype.csproj']:copy(TOOLS/name,bundle/'source/consumer'/name)
    copy(ROOT/'global.json',bundle/'source/global.json')
    for name in ['protocol.py','remote.py','remote_prepare.py']:copy(TOOLS/name,bundle/'tools'/name)
    copy(TOOLS/'README.md',bundle/'README.md')
    shutil.copy2(ROOT/'.agent/m39-parakeet-wide-per-call-20260923.md',bundle/'prospective-plan.md')
    products={}
    for role,folder in [('current',CURRENT/'collected/runtimes/current'),('candidate',BUILD/'collected/runtime')]:
        for p in folder.iterdir():
            if p.is_file():copy(p,bundle/'runtimes'/role/p.name)
        products[role]={name:pin(bundle/'runtimes'/role/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
    assert products['current']==read(BUILD/'analysis.json')['measured']
    assert products['candidate']==read(BUILD/'analysis.json')['built']
    for name in ['closed.json','payload.json']:copy(BUILD/name,bundle/'evidence'/('build-'+name))
    copy(BUILD/'collected/collection.json',bundle/'evidence/build-collection.json')
    copy(CAPTURE/'capture-closed.json',bundle/'evidence/capture-closed.json')
    capture=read(CAPTURE/'capture/result.json');assert capture['passed'] and len(capture['entries'])==12
    copy(CAPTURE/'capture/result.json',bundle/'fixtures/result.json')
    arrays=dict(capture['weights'])
    for i,entry in enumerate(capture['entries']):
        for role in ['a','y']:arrays[str(i)+role]=entry[role]
    assert len(arrays)==27
    for entry in arrays.values():
        assert entry['dtype']=='<f4' and not entry['reverse']
        p=CAPTURE/'capture'/entry['file'];assert pin(p)=={k:entry[k] for k in ['bytes','sha256']}
        copy(p,bundle/'fixtures'/entry['file'])
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
