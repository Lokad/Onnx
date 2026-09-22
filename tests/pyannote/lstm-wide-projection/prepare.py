"""Prepare isolated source and tests without compiling or changing the root product."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from transform import PANELS,ROWS,WIDE,transform

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-lstm-wide-projection-20260922'
PROOF=ROOT/'artifacts/pyannote-lstm-input-root-amd-20260922'
PROFILE=ROOT/'artifacts/pyannote-current-profile-amd-v2-20260922'


def read(p):return json.loads(p.read_text(encoding='utf8'))


def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def main():
    assert not BASE.exists()
    for folder,digest,relative in [(PROOF,'5cc03093982964beb44776b7d64cf561e5a6b49c1947b1d6b9a80b790cf41a21',False),
            (PROFILE,'5c96b85e3618053574e815f94bfa7a8a6b6e912e2817c08e293281cd56a40ad0',True)]:
        assert pin(folder/'closed.json')['sha256']==digest
        value=read(folder/'closed.json');assert value['passed']
        for name,wanted in value['files'].items():assert pin((ROOT if relative else folder)/name)==wanted,name
    BASE.mkdir();source=BASE/'source';source.mkdir();before={}
    for name,wanted in read(PROOF/'bundle/stage.json')['files'].items():
        if not name.startswith('source/'):continue
        relative=name.removeprefix('source/');assert pin(ROOT/relative)==wanted,relative
        target=source/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target);before[relative]=wanted
    assert len(before)==416
    changed,patch=transform((source/PANELS).read_text(),(source/ROWS).read_text())
    for name,text in changed.items():(source/name).write_text(text,encoding='utf8',newline='\n')
    test='tests/Lokad.Onnx.Backend.Tests/LstmWideProjectionTests.cs';shutil.copy2(TOOLS/'LstmWideProjectionTests.cs',source/test)
    patch+=''.join(difflib.unified_diff([],(source/test).read_text().splitlines(True),fromfile='/dev/null',tofile=test))
    (BASE/'candidate.patch').write_text(patch,encoding='utf8',newline='\n')
    assert [name for name,wanted in before.items() if pin(source/name)!=wanted]==[PANELS]
    for name,wanted in before.items():assert pin(ROOT/name)==wanted,name
    shutil.copy2(ROOT/'.agent/m25-pyannote-wide-lstm-20260922.md',BASE/'prospective-plan.md')
    ort='onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc';revision='2e2543fbe9fae542f921d47a72d21d5a4ef0b710'
    (BASE/'ort-lstm.cc').write_bytes(subprocess.check_output(['git','-C',str(ROOT/'external/onnxruntime'),'show',revision+':'+ort]))
    value=dict(passed=True,root_product_changed=False,built=False,numerically_qualified=False,source_commit='fe4eb657',
        changed=[PANELS],added=[WIDE,test],before=before,profile=pin(PROFILE/'closed.json'),product=pin(PROOF/'closed.json'),
        ort=dict(revision=revision,path=ort,file=pin(BASE/'ort-lstm.cc')),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},
        tools={p.relative_to(ROOT).as_posix():pin(p) for p in TOOLS.iterdir() if p.is_file()},
        compiled_scope='Only panel Create/Input/InputBlock/Recurrent change; add LstmProjectOrdered512 and LstmProjectOrderedRows512; preserve every other existing method/public declaration')
    (BASE/'prepared.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),patch=pin(BASE/'candidate.patch'),changed=value['changed'],added=value['added'],root_product_changed=False,built=False)))


if __name__=='__main__':main()
