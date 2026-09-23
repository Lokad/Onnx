"""Freeze current/candidate Parakeet conformance only after component admission."""
import ast
from pathlib import Path
import json
import shutil
import tarfile
from protocol import pin,read,save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-wide-entry-first-use-models-amd-20260923'
SCREEN=ROOT/'artifacts/parakeet-wide-entry-first-use-screen-amd-20260923'
BUILD=ROOT/'artifacts/parakeet-wide-entry-first-use-build-amd-20260923'
CURRENT=ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
PREVIOUS=ROOT/'artifacts/pyannote-winograd-product-parakeet-amd-20260923'
SOURCE=ROOT/'artifacts/parakeet-wide-entry-first-use-source-v2-20260923'


def previous_closed():
    for folder,digest in [(SCREEN,'5ca99bc1ac4b647c083104fc7f72a3cd19871c5c72499ff085c9475ca4b8cb8a'),(BUILD,'da923692f2c97cbff2774006f4a3dc911aaca39e0af5b35a89639443a75ead58'),
        (CURRENT,'2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3'),
        (PREVIOUS,'38b346aec0df29b4e99163282253012a5f066b2d80615cd7517a730b199ac14f')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert read(SCREEN/'analysis.json')['admitted']
    source=read(SOURCE/'prepared.json')
    assert pin(SOURCE/'prepared.json')['sha256']=='72c22bee93652ed8d6a759c2a984d965fa341697e869cf7a2461a908727483f6'
    for name,wanted in source['before'].items():assert pin(ROOT/name)==wanted,name


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','native_audit.py','public_audit.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    for target,source in [('native_audit.py',ROOT/'tests/parakeet/transcribe/audit.py'),('public_audit.py',ROOT/'tests/audio/comparison/audit.py')]:
        assert (TOOLS/target).read_bytes()==source.read_bytes();originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for folder,label in [(SCREEN,'screen'),(BUILD,'build'),(CURRENT,'current'),(PREVIOUS,'previous')]:
        for name in ['closed.json','payload.json']:copy(folder/name,bundle/'evidence'/(label+'-'+name))
        copy(folder/'collected/collection.json',bundle/'evidence'/(label+'-collection.json'))
    copy(CURRENT/'collected/manifests/current-parakeet.json',bundle/'evidence/original-manifest.json')
    identities={}
    for role,folder in [('selected',CURRENT/'collected/runtimes/current'),('candidate',BUILD/'collected/runtime')]:
        identities[role]={name:pin(folder/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
        for name in identities[role]:copy(folder/name,bundle/'products'/role/name)
    consumers={name:pin(CURRENT/'collected/runtimes/current'/(name+'.dll')) for name in ['TranscribeReplay','AudioBenchmark']}
    assert consumers['TranscribeReplay']['sha256']=='335ca09d0e45e344068c484c92af9d0db43a6ae0accd1895d7ae7bb88b0afcf9'
    assert consumers['AudioBenchmark']['sha256']=='7eca033a1b986a4cb90621392639d230c95097cb703dd25274fd72d66c5ba4f1'
    shutil.copy2(ROOT/'.agent/m54-wide-projection-first-use-entry-20260923.md',bundle/'prospective-plan.md')
    save(bundle/'stage.json',dict(passed=True,identities=identities,consumers=consumers,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),identities=identities)))


if __name__=='__main__':prepare()
