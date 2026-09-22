"""Reuse the qualified diagnostic driver and graph callers with the exact two products."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save
from score import iteration_manifest

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-kernel-loop-screen-amd-20260922'
NUM=ROOT/'artifacts/pyannote-kernel-loop-numerics-amd-v2-20260922'
CURRENT=ROOT/'artifacts/parakeet-current-baseline-amd-20260922'
OLD=ROOT/'artifacts/pyannote-spatial-weight-screen-20260922'
CODEGEN=ROOT/'artifacts/pyannote-kernel-loop-codegen-amd-20260922'
REVIEW=ROOT/'artifacts/pyannote-kernel-loop-codegen-review-20260922/review.json'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
spec=importlib.util.spec_from_file_location('codegen_monitor',MONITOR)
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)


def previous_closed():
    assert pin(CODEGEN/'closed.json')['sha256']=='bf25713d67d4a4bf72f688e0f5b92cc46bc0618afad2065dc5b7939f2d650322'
    proof=read(CODEGEN/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(CODEGEN/name)==wanted,name
    assert pin(REVIEW)['sha256']=='2c2850a29a9aec59681415276ef79b062ab23badba8532668ceaa0aed9dd01bc'
    assert read(REVIEW)['passed']
    for folder,digest in [(NUM,'551fb5db9dd20c9e5e9c3737a4f311779226bdbc92c53f4cff835f6277799dda'),
                          (CURRENT,'6c65419f54f93ac43cc9ca26886dcf4bb9b6535e85f40c4db291fb9b9e1ea4bb'),
                          (OLD,'51a8ae8307eae17ba4f0fed16b5905846ebe8102042f8a581fd3d6f828e282e3')]:
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
        copy(OLD/'payload/runtime/production'/('SpatialWeightScreen.'+suffix),bundle/'driver'/('SpatialWeightScreen.'+suffix))
    copy(NUM/'collected/layers-512/result.json',bundle/'reference.json')
    copy(NUM/'bundle/fixtures/result.json',bundle/'fixtures/result.json')
    copy(OLD/'payload/iterations.json',bundle/'iterations.json')
    assert read(bundle/'iterations.json')==iteration_manifest(read(bundle/'fixtures/result.json')['calls'])
    copy(CODEGEN/'closed.json',bundle/'evidence/codegen-closed.json')
    copy(REVIEW,bundle/'evidence/codegen-review.json')
    for folder,label in [(NUM,'numerical'),(CURRENT,'current'),(OLD,'driver')]:
        copy(folder/'closed.json',bundle/'evidence'/(label+'-closed.json'))
    for p in TOOLS.glob('*.py'):
        ast.parse(p.read_text(),str(p))
        if p.name in ['protocol.py','remote.py','remote_prepare.py','checks.py','score.py']:copy(p,bundle/'tools'/p.name)
    copy(ROOT/'.agent/m23-pyannote-kernel-loops-20260922.md',bundle/'prospective-plan.md')
    originals.pop('.agent/m23-pyannote-kernel-loops-20260922.md')
    stage=dict(passed=True,current_core=pin(CURRENT/'collected/runtimes/current/Lokad.Onnx.dll'),
        candidate_core=read(NUM/'analysis.json')['core'],probe=read(NUM/'analysis.json')['consumers']['layers'],
        driver=pin(bundle/'driver/SpatialWeightScreen.dll'),
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
