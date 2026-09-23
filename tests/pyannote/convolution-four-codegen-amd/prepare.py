"""Reuse the qualified diagnostic driver and graph callers with the exact two products."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-convolution-four-codegen-amd-20260923'
NUM=ROOT/'artifacts/pyannote-convolution-four-numerics-amd-20260923'
CURRENT=ROOT/'artifacts/parakeet-current-baseline-amd-20260922'
EXT=ROOT/'artifacts/pyannote-convolution-four-extremes-amd-20260923'
OLD=ROOT/'artifacts/pyannote-spatial-weight-codegen-20260922'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('codegen_monitor',MONITOR)
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)


def previous_closed():
    for folder,digest in [(EXT,'402b7b2a5a68e0a888d250e9232a750d9ec6de3fe311a29b9ebec648bea9611e'),(NUM,'643657102dab0406b2f55ec92df446c01b6586030663708f1d5325735a0729a7'),
                          (CURRENT,'6c65419f54f93ac43cc9ca26886dcf4bb9b6535e85f40c4db291fb9b9e1ea4bb'),
                          (OLD,'7c32ceccb3349efcde4c614ba00f10a1ea3d7c2ece57633d43609d6048cbd4bf')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name


def prepare():
    assert not BASE.exists();previous_closed()
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for suffix in ['dll','deps.json','runtimeconfig.json']:
        copy(OLD/'payload/runtime/production'/('SpatialWeightCodegen.'+suffix),bundle/'driver'/('SpatialWeightCodegen.'+suffix))
    copy(NUM/'collected/layers-512/result.json',bundle/'reference.json')
    copy(NUM/'bundle/fixtures/result.json',bundle/'fixtures/result.json')
    for folder,label in [(EXT,'extremes'),(NUM,'numerical'),(CURRENT,'current'),(OLD,'driver')]:
        copy(folder/'closed.json',bundle/'evidence'/(label+'-closed.json'))
    for p in TOOLS.glob('*.py'):
        ast.parse(p.read_text(),str(p))
        if p.name in ['protocol.py','remote.py','remote_prepare.py','checks.py']:copy(p,bundle/'tools'/p.name)
    copy(ROOT/'.agent/m29-pyannote-four-dispatch-20260923.md',bundle/'prospective-plan.md')
    originals.pop('.agent/m29-pyannote-four-dispatch-20260923.md')
    stage=dict(passed=True,current_core=pin(CURRENT/'collected/runtimes/current/Lokad.Onnx.dll'),
        candidate_core=read(NUM/'analysis.json')['core'],probe=read(NUM/'analysis.json')['consumers']['layers'],
        driver=pin(bundle/'driver/SpatialWeightCodegen.dll'),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for p in [*TOOLS.iterdir(),MONITOR]:
        if p.is_file():originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))


if __name__=='__main__':prepare()
