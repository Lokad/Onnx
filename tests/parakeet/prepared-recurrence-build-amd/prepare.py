"""Freeze the bounded recurrence build after actual complete-call capture."""
import ast
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-prepared-recurrence-build-amd-20260924'
SOURCE=ROOT/'artifacts/parakeet-prepared-recurrence-source-20260924'
QUALIFIED=ROOT/'artifacts/parakeet-wide-entry-first-use-build-amd-20260923'
RELEASE=ROOT/'artifacts/parakeet-wide-entry-first-use-root-amd-v2-20260923'
CAPTURE=ROOT/'artifacts/parakeet-decoder-lstm-capture-amd-20260924'
PREVIOUS=ROOT/'artifacts/parakeet-inclusive-packing-build-amd-20260924'


def previous_closed():
    for folder,digest in [(QUALIFIED,'da923692f2c97cbff2774006f4a3dc911aaca39e0af5b35a89639443a75ead58'),
        (RELEASE,'16d570819ab69915fe34fa6c5a4efb79d45ae792dbd5b1448c65645fa0d55f73'),
        (CAPTURE,'28c7afe448ed16e3bb19d29232c3f90eb2d72c096196ae27792f5261afa5b64f'),
        (PREVIOUS,'50f2a3a8a2ebbe20d41315bc0be4242e24fb7f3eb74e78533091a750791a4078')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert pin(SOURCE/'prepared.json')['sha256']=='b52a89c1043165de1c376b37fc5307cd003a7c8b76f0f52508c4cdefcc669eab'
    source=read(SOURCE/'prepared.json');assert source['passed'] and len(source['source'])==424 and len(source['before'])==422
    for name,wanted in source['source'].items():assert pin(SOURCE/'source'/name)==wanted,name
    for name,wanted in source['before'].items():assert pin(ROOT/name)==wanted,name
    for name,key in [('candidate.patch','patch'),('prospective-plan.md','plan')]:assert pin(SOURCE/name)==source[key]
    for name,entry in source['added_sources'].items():assert pin(ROOT/entry['path'])=={k:entry[k] for k in ['bytes','sha256']}
    assert pin(ROOT/'tests/parakeet/prepared-recurrence-source/prepare.py')==source['generator']


def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    source=read(SOURCE/'prepared.json')
    for name in source['source']:copy(SOURCE/'source'/name,bundle/'source'/name)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','il_body.py']:copy(TOOLS/name,bundle/'tools'/name)
    for name in ['Bridge.dll','Bridge.deps.json','Bridge.runtimeconfig.json']:copy(QUALIFIED/'collected/bridge'/name,bundle/'bridge'/name)
    for name in ['closed.json','analysis.json','payload.json']:
        copy(QUALIFIED/name,bundle/'evidence'/name);copy(RELEASE/name,bundle/'evidence/release'/name)
    copy(QUALIFIED/'collected/collection.json',bundle/'evidence/collection.json')
    copy(RELEASE/'collected/collection.json',bundle/'evidence/release/collection.json')
    copy(CAPTURE/'collected/collection.json',bundle/'evidence/capture-collection.json')
    copy(CAPTURE/'closed.json',bundle/'evidence/capture-closed.json')
    for name in ['prepared.json','candidate.patch','prospective-plan.md']:copy(SOURCE/name,bundle/'evidence'/('source-'+name))
    copy(TOOLS/'README.md',bundle/'prospective-build.md')
    path=PREVIOUS/'collected/inventory/instructions.json';rows=read(path)['observations']
    prior=dict(passed=True,inventory=pin(path),source_prepared=pin(SOURCE/'prepared.json'),
        methods={r['assembly']:r['normalized_methods'] for r in rows},flags={r['assembly']:r['method_flags_before'] for r in rows})
    measured={name:pin(QUALIFIED/'collected/runtime'/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
    assert all(r['before_sha256']==measured[r['assembly']]['sha256'] for r in rows)
    save(bundle/'evidence/prior-composition.json',prior);originals[path.relative_to(ROOT).as_posix()]=pin(path)
    stage=dict(passed=True,source_prepared=pin(SOURCE/'prepared.json'),parent_release=pin(RELEASE/'closed.json'),measured=measured,
        measured_files={p.name:pin(p) for p in (QUALIFIED/'collected/runtime').iterdir() if p.is_file()},
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz'),source_files=424)))


if __name__=='__main__':prepare()
