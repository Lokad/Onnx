"""Prepare the distinct M26 source experiment without building or editing root files."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from transform import RECURRENT,HELPER,transform

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-lstm-default-gates-20260923'
PROOF=ROOT/'artifacts/pyannote-lstm-input-root-amd-20260922'
SCREEN=ROOT/'artifacts/pyannote-lstm-wide-screen-amd-20260923'
CODEGEN=ROOT/'artifacts/pyannote-lstm-wide-codegen-amd-20260923'

def read(p):return json.loads(p.read_text(encoding='utf8'))
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())

def main():
    assert not BASE.exists()
    for folder,name,digest in [(PROOF,'closed.json','5cc03093982964beb44776b7d64cf561e5a6b49c1947b1d6b9a80b790cf41a21'),
        (SCREEN,'closed.json','0cde23e5719eac520b57e6123e5caa76cbed3a586948446b13b8262479a1c3db'),
        (CODEGEN,'reconciled-closed.json','55dfae31762b55fd2c124404506a270662571eaf3aef138bf534e160c0b16f0b')]:
        assert pin(folder/name)['sha256']==digest;proof=read(folder/name);assert proof['passed']
        for key,wanted in proof['files'].items():assert pin(folder/key)==wanted,key
    assert not read(SCREEN/'analysis.json')['decision']['admitted']
    BASE.mkdir();source=BASE/'source';source.mkdir();before={}
    for name,wanted in read(PROOF/'bundle/stage.json')['files'].items():
        if not name.startswith('source/'):continue
        relative=name.removeprefix('source/');assert pin(ROOT/relative)==wanted,relative
        target=source/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target);before[relative]=wanted
    assert len(before)==416
    changed,patch=transform((source/RECURRENT).read_text())
    for name,text in changed.items():(source/name).write_text(text,encoding='utf8',newline='\n')
    test='tests/Lokad.Onnx.Backend.Tests/LstmDefaultGateTests.cs';shutil.copy2(TOOLS/'LstmDefaultGateTests.cs',source/test)
    patch+=''.join(difflib.unified_diff([],(source/test).read_text().splitlines(True),fromfile='/dev/null',tofile=test))
    (BASE/'candidate.patch').write_text(patch,encoding='utf8',newline='\n')
    assert [name for name,wanted in before.items() if pin(source/name)!=wanted]==[RECURRENT]
    for name,wanted in before.items():assert pin(ROOT/name)==wanted,name
    shutil.copy2(ROOT/'.agent/m26-pyannote-default-gates-20260923.md',BASE/'prospective-plan.md')
    revision='2e2543fbe9fae542f921d47a72d21d5a4ef0b710';ort=[]
    for filename in ['uni_directional_lstm.cc','rnn_helpers.cc']:
        path='onnxruntime/core/providers/cpu/rnn/'+filename;target=BASE/('ort-'+filename)
        target.write_bytes(subprocess.check_output(['git','-C',str(ROOT/'external/onnxruntime'),'show',revision+':'+path]))
        ort.append(dict(revision=revision,path=path,file=pin(target)))
    value=dict(passed=True,root_product_changed=False,built=False,numerically_qualified=False,source_commit='fe4eb657',
        changed=[RECURRENT],added=[HELPER,test],before=before,product=pin(PROOF/'closed.json'),rejected_screen=pin(SCREEN/'closed.json'),
        codegen=pin(CODEGEN/'reconciled-closed.json'),ort=ort,
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},
        tools={p.relative_to(ROOT).as_posix():pin(p) for p in TOOLS.iterdir() if p.is_file()},
        compiled_scope='Only Lstm changes; add private LstmUpdateDefaultGates; all3162otherCore/697Data methods and public declarations unchanged')
    (BASE/'prepared.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),patch=pin(BASE/'candidate.patch'),changed=value['changed'],added=value['added'],root_product_changed=False,built=False)))

if __name__=='__main__':main()
