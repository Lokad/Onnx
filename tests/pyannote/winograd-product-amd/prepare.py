"""Freeze the admitted isolated source and the unchanged full-suite/package consumers."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin,read,save

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-winograd-product-amd-20260923'
SOURCE=ROOT/'artifacts/pyannote-winograd-product-source-v2-20260923'
SCREEN=ROOT/'artifacts/pyannote-winograd-range-screen-amd-20260923'
BUILD=ROOT/'artifacts/pyannote-winograd-product-build-amd-v2-20260923'
SELECTED=ROOT/'artifacts/pyannote-lstm-input-root-amd-20260922'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
module=importlib.util.spec_from_file_location('product_monitor',MONITOR)
monitor=importlib.util.module_from_spec(module);module.loader.exec_module(monitor)


def previous_closed():
    for folder,digest in [
        (SCREEN,'302de6c9b75bf2a60f2b09b0e79f0c5d8ff846a149a1f1657f5fa094820c919a'),
        (BUILD,'50e50f01912ae2ae3accf38e9fa4657ea4a405be3dadb0c2f1a1fa8ea0a09f87'),
        (SELECTED,'5cc03093982964beb44776b7d64cf561e5a6b49c1947b1d6b9a80b790cf41a21')]:
        assert pin(folder/'closed.json')['sha256']==digest
        value=read(folder/'closed.json');assert value['passed']
        for name,wanted in value['files'].items():assert pin(folder/name)==wanted,name
    assert read(SCREEN/'analysis.json')['performance']['admitted']
    assert pin(SOURCE/'prepared.json')['sha256']=='3c7e4d2432558ef6044f6a484b64994c990762e192b4e77e93b74584e7dab8cd'
    source=read(SOURCE/'prepared.json');assert source['passed'] and not source['root_product_changed']
    for name,wanted in source['files'].items():assert pin(SOURCE/name)==wanted,name
    for name,wanted in source['before'].items():assert pin(ROOT/name)==wanted,name


def prepare():
    assert not BASE.exists();previous_closed()
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name,wanted in read(SOURCE/'prepared.json')['files'].items():
        if name.startswith('source/'):
            assert pin(SOURCE/name)==wanted
            copy(SOURCE/name,bundle/name)
    assert len(originals)==420
    previous=read(SELECTED/'bundle/stage.json')
    for name,wanted in previous['files'].items():
        if name.startswith(('consumer/','bridge/')):
            assert pin(SELECTED/'bundle'/name)==wanted
            if name=='consumer/Program.cs':copy(TOOLS/'PackageProgram.cs',bundle/name)
            else:copy(SELECTED/'bundle'/name,bundle/name)
    copy(SELECTED/'bundle/evidence/tensor-source.tar',bundle/'evidence/tensor-source.tar')
    # These selected full suites already include M22's 36 input-row cases.
    for name in ['backend','tensors']:
        copy(SELECTED/'collected'/(name+'-tests')/(name+'.trx'),bundle/'evidence'/('selected-'+name+'.trx'))
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    copy(SCREEN/'closed.json',bundle/'evidence/screen-closed.json')
    copy(SCREEN/'analysis.json',bundle/'evidence/screen-analysis.json')
    copy(SOURCE/'prepared.json',bundle/'evidence/source-prepared.json')
    copy(ROOT/'.agent/m34-pyannote-winograd-product-20260923.md',bundle/'prospective-plan.md')
    originals.pop('.agent/m34-pyannote-winograd-product-20260923.md')
    copy(BUILD/'payload.json',bundle/'evidence/build-payload.json')
    stage=dict(passed=True,measured_core=pin(BUILD/'collected/runtime/Lokad.Onnx.dll'),
        measured_data=pin(BUILD/'collected/runtime/Lokad.Onnx.Data.dll'),
        build_collection=pin(BUILD/'collected/collection.json'),
        source_commit=read(SOURCE/'prepared.json')['source_commit'],
        source_scope='420 isolated source files: three changed, four added, admitted Winograd arithmetic and optional preparation/dispatch; root unchanged.',
        screen=pin(SCREEN/'closed.json'),source=pin(SOURCE/'prepared.json'),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    files=dict(originals)
    for p in [*TOOLS.iterdir(),MONITOR,SCREEN/'closed.json',BUILD/'closed.json',SELECTED/'closed.json',SOURCE/'prepared.json']:
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=files,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),source_files=420)))


if __name__=='__main__':prepare()
