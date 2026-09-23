"""Freeze an isolated numerical consumer; preserve every selected source byte."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin,read,save
from generate import generate

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-winograd-numerics-amd-20260923'
PROOF=ROOT/'artifacts/pyannote-lstm-input-root-amd-20260922'
SCREEN=ROOT/'artifacts/pyannote-convolution-four-screen-amd-20260923'
FIXTURES=ROOT/'artifacts/pyannote-blocked-spatial-fixtures-20260922/output'
PLAN=ROOT/'.agent/m30-pyannote-winograd-20260923.md'
DIRECT=['Zzz.ConvBlockedSpatial.cs','Zzz.ConvBlockedSpatial.Input.cs','Zzz.ConvBlockedSpatial.Kernels.cs','Zzz.ConvBlockedSpatial.Output.cs']


def previous_closed():
    for folder,digest in [(PROOF,'5cc03093982964beb44776b7d64cf561e5a6b49c1947b1d6b9a80b790cf41a21'),
                          (SCREEN,'0101c5d045bd6aa93526852e263106b7eeae4eb80b272c1efdd82c59c6830a43')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    expected={n.removeprefix('source/'):v for n,v in read(PROOF/'bundle/stage.json')['files'].items() if n.startswith('source/')}
    assert len(expected)==416
    for name,wanted in expected.items():assert pin(ROOT/name)==wanted,name
    assert not read(SCREEN/'analysis.json')['admitted']


def prepare():
    assert not BASE.exists();previous_closed()
    assert (TOOLS/'Winograd.Kernels.cs').read_text()==generate()
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in DIRECT:copy(ROOT/'src/Lokad.Onnx'/name,bundle/'source/consumer'/name)
    for name in ['Winograd.cs','Winograd.Kernels.cs','Driver.cs','Prototype.csproj']:copy(TOOLS/name,bundle/'source/consumer'/name)
    copy(ROOT/'global.json',bundle/'source/global.json')
    for name in ['protocol.py','remote.py','remote_prepare.py']:copy(TOOLS/name,bundle/'tools'/name)
    copy(PLAN,bundle/'prospective-plan.md');copy(TOOLS/'README.md',bundle/'README.md')
    copy(SCREEN/'closed.json',bundle/'evidence/screen-closed.json')
    copy(SCREEN/'collected/collection.json',bundle/'evidence/screen-collection.json')
    copy(FIXTURES/'result.json',bundle/'evidence/fixtures.json')
    copy(FIXTURES/'capture-spec.json',bundle/'evidence/capture-spec.json')
    fixture_pins={name:pin(FIXTURES/name) for name in ['result.json','capture-spec.json']}
    for entry in read(FIXTURES/'result.json')['tensors'].values():
        name=entry['path'];wanted={k:entry[k] for k in ['bytes','sha256']}
        assert pin(FIXTURES/name)==wanted,name;fixture_pins[name]=wanted
    save(bundle/'stage.json',dict(passed=True,fixtures=fixture_pins,root_product_changed=False,
        direct={name:pin(ROOT/'src/Lokad.Onnx'/name) for name in DIRECT},
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),fixtures=len(fixture_pins),root_product_changed=False)))


if __name__=='__main__':prepare()
