"""Prepare an isolated caller/new-kernel change, retaining qualified fallback bytes."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from transform import SOURCE,KERNEL,ADDED,GENERATOR,transform

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/pyannote-convolution-four-dispatch-20260923'
PROOF=ROOT/'artifacts/pyannote-lstm-input-root-amd-20260922'
APPLICATION=ROOT/'artifacts/pyannote-convolution-pointer-app-amd-20260923'
PLAN=ROOT/'.agent/m29-pyannote-four-dispatch-20260923.md'


def read(path):return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def main():
    assert not BASE.exists()
    for folder,digest in [(PROOF,'5cc03093982964beb44776b7d64cf561e5a6b49c1947b1d6b9a80b790cf41a21'),
                          (APPLICATION,'ae998ef9106ce143e26e83a856f8575af4bff1240049ecdd08edbfa57501d54b')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    assert not read(APPLICATION/'analysis.json')['performance']['admitted']
    BASE.mkdir();source=BASE/'source';source.mkdir();before={}
    for name,wanted in read(PROOF/'bundle/stage.json')['files'].items():
        if not name.startswith('source/'):continue
        relative=name.removeprefix('source/');assert pin(ROOT/relative)==wanted,relative
        target=source/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target);before[relative]=wanted
    assert len(before)==416 and ADDED not in before
    caller,added,difference=transform((source/SOURCE).read_text(encoding='utf8'),(source/KERNEL).read_text(encoding='utf8'))
    (source/SOURCE).write_text(caller,encoding='utf8',newline='\n')
    (source/ADDED).write_text(added,encoding='utf8',newline='\n')
    (BASE/'candidate.patch').write_text(difference,encoding='utf8',newline='\n')
    shutil.copy2(PLAN,BASE/'prospective-plan.md')
    changed=[name for name,wanted in before.items() if pin(source/name)!=wanted]
    assert changed==[SOURCE] and pin(source/KERNEL)==before[KERNEL]
    for name,wanted in before.items():assert pin(ROOT/name)==wanted,name
    assert not (ROOT/ADDED).exists()
    value=dict(passed=True,root_product_changed=False,built=False,numerically_qualified=False,
        source_commit=subprocess.check_output(['git','rev-parse','fe4eb657'],cwd=ROOT,text=True).strip(),
        product_closure=pin(PROOF/'closed.json'),preceding_application=pin(APPLICATION/'closed.json'),
        changed=changed,added=[ADDED],before=before,
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},
        tools={p.relative_to(ROOT).as_posix():pin(p) for p in [*Path(__file__).parent.glob('*.py'),GENERATOR]},
        allowed_compiled_change='Lokad.Onnx.ConvBlockedSpatial::Execute',
        allowed_compiled_addition='Lokad.Onnx.ConvBlockedSpatial::Kernel512Four',
        expected_unchanged_core_methods=3162,expected_unchanged_data_methods=697)
    (BASE/'prepared.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),patch=pin(BASE/'candidate.patch'),changed=changed,added=[ADDED],root_product_changed=False)))


if __name__=='__main__':main()
