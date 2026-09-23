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
BASE=ROOT/'artifacts/pyannote-convolution-pointer-product-amd-20260923'
SOURCE=ROOT/'artifacts/pyannote-convolution-pointer-unroll-20260923'
SCREEN=ROOT/'artifacts/pyannote-convolution-pointer-screen-amd-20260923'
BUILD=ROOT/'artifacts/pyannote-convolution-pointer-build-amd-20260923'
SELECTED=ROOT/'artifacts/pyannote-lstm-input-root-amd-20260922'
MONITOR=ROOT/'tests/parakeet/packing-budgets/common.py'
module=importlib.util.spec_from_file_location('product_monitor',MONITOR)
monitor=importlib.util.module_from_spec(module);module.loader.exec_module(monitor)


def previous_closed():
    for folder,digest in [
        (SCREEN,'ead36d62874432715bdaeaa0a15cc03ec5eecf69d365c552af81eb109febb7fd'),
        (BUILD,'edff508abee43ccd09c098f0b72325482c797bf3d2a772365706c959f3ceccc9'),
        (SELECTED,'5cc03093982964beb44776b7d64cf561e5a6b49c1947b1d6b9a80b790cf41a21')]:
        assert pin(folder/'closed.json')['sha256']==digest
        value=read(folder/'closed.json');assert value['passed']
        for name,wanted in value['files'].items():assert pin(folder/name)==wanted,name
    assert read(SCREEN/'analysis.json')['admitted']
    assert pin(SOURCE/'prepared.json')['sha256']=='863f9c85c35c6e8a3610eedcaa1b0014c6a6a6a069fc0ad814b589f1733d10b9'
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
    assert len(originals)==416
    previous=read(SELECTED/'bundle/stage.json')
    for name,wanted in previous['files'].items():
        if name.startswith(('consumer/','bridge/')):
            assert pin(SELECTED/'bundle'/name)==wanted
            copy(SELECTED/'bundle'/name,bundle/name)
    copy(SELECTED/'bundle/evidence/tensor-source.tar',bundle/'evidence/tensor-source.tar')
    # These selected full suites already include M22's 36 input-row cases.
    for name in ['backend','tensors']:
        copy(SELECTED/'collected'/(name+'-tests')/(name+'.trx'),bundle/'evidence'/('selected-'+name+'.trx'))
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    copy(SCREEN/'closed.json',bundle/'evidence/screen-closed.json')
    copy(SCREEN/'analysis.json',bundle/'evidence/screen-analysis.json')
    copy(SOURCE/'prepared.json',bundle/'evidence/source-prepared.json')
    copy(ROOT/'.agent/m28-pyannote-pointer-unroll-20260923.md',bundle/'prospective-plan.md')
    originals.pop('.agent/m28-pyannote-pointer-unroll-20260923.md')
    stage=dict(passed=True,measured_core=read(SCREEN/'payload.json')['job_details']['candidate']['core'],
        source_commit=read(SOURCE/'prepared.json')['source_commit'],
        source_scope='All 416 qualified source files with the isolated one-file M28 change; root unchanged.',
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
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),source_files=416)))


if __name__=='__main__':prepare()
