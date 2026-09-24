"""Freeze full Parakeet conformance after M63 scope, contracts and residency pass."""
import ast
from pathlib import Path
import json
import shutil
import tarfile
from protocol import pin,read,save

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-inclusive-packing-models-amd-20260924'
RESIDENCY=ROOT/'artifacts/parakeet-inclusive-packing-residency-amd-v2-20260924'
CONTRACTS=ROOT/'artifacts/parakeet-inclusive-packing-contracts-amd-v2-20260924'
BUILD=ROOT/'artifacts/parakeet-inclusive-packing-build-amd-20260924'
CURRENT=ROOT/'artifacts/parakeet-wide-entry-first-use-models-amd-20260923'
PREVIOUS=ROOT/'artifacts/pyannote-winograd-product-parakeet-amd-20260923'
SOURCE=ROOT/'artifacts/parakeet-inclusive-packing-source-20260924'


def previous_closed():
    for folder,digest in [(RESIDENCY,'82342a58fddab63974e8bff009916a91376beb1d91f9db264480c2763c8e7b0a'),
        (CONTRACTS,'e85e5f5f7d435141b41b19957cc9bc119850c0593a120956002c798e66d01b76'),
        (BUILD,'50f2a3a8a2ebbe20d41315bc0be4242e24fb7f3eb74e78533091a750791a4078'),
        (CURRENT,'f30100534cbb79db790aac30365d533e6d3dcce79e779abc5feb8f7cf3fc1e22'),
        (PREVIOUS,'38b346aec0df29b4e99163282253012a5f066b2d80615cd7517a730b199ac14f')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert read(RESIDENCY/'analysis.json')['comparison']['both_modes_exact']
    assert read(CONTRACTS/'analysis.json')['candidate_passes']==124
    source=read(SOURCE/'prepared.json')
    assert pin(SOURCE/'prepared.json')['sha256']=='a535266a48631edc8578eeab679df644884d08f1f1cb08e84dbb27415b1cbd7c'
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
    for folder,label in [(RESIDENCY,'residency'),(CONTRACTS,'contracts'),(BUILD,'build'),(CURRENT,'current'),(PREVIOUS,'previous')]:
        for name in ['closed.json','payload.json']:copy(folder/name,bundle/'evidence'/(label+'-'+name))
        copy(folder/'collected/collection.json',bundle/'evidence'/(label+'-collection.json'))
    copy(CURRENT/'collected/manifests/candidate-parakeet.json',bundle/'evidence/original-manifest.json')
    identities={}
    for role,folder in [('selected',CURRENT/'collected/runtimes/candidate'),('candidate',BUILD/'collected/runtime')]:
        identities[role]={name:pin(folder/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
        for name in identities[role]:copy(folder/name,bundle/'products'/role/name)
    consumers={name:pin(CURRENT/'collected/runtimes/candidate'/(name+'.dll')) for name in ['TranscribeReplay','AudioBenchmark']}
    assert consumers['TranscribeReplay']['sha256']=='335ca09d0e45e344068c484c92af9d0db43a6ae0accd1895d7ae7bb88b0afcf9'
    assert consumers['AudioBenchmark']['sha256']=='7eca033a1b986a4cb90621392639d230c95097cb703dd25274fd72d66c5ba4f1'
    assert identities['selected']==read(CURRENT/'analysis.json')['identities']['candidate']
    assert identities['candidate']==read(BUILD/'analysis.json')['built']
    copy(TOOLS/'README.md',bundle/'prospective-models.md')
    shutil.copy2(ROOT/'.agent/m63-parakeet-inclusive-packing-20260924.md',bundle/'prospective-plan.md')
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
