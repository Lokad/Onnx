"""Prepare a reviewable patch without changing root product files."""
import ast
import difflib
import subprocess
from run import ROOT, TOOLS, MODEL, PACKAGE, PATCHBASE, NAMES, pin, read, save, terminal, rel


def main():
    assert not (PATCHBASE/'prepared.json').exists()
    if PATCHBASE.exists(): assert {p.name for p in PATCHBASE.iterdir()} == {'candidate.patch'}
    for folder, expected in [(MODEL,'e7a9a30d88ef2a425c0c51b007e2b7d89428d54e50ef7dc4e446e191f95719e3'),
        (PACKAGE,'fc4f4811032ff38ea837b8d53ea68325a63cd9959b7e5d61b9fbb89ba7fefc66')]:
        assert pin(folder/'closed.json')['sha256'] == expected
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
        for identity in proof['identities']:terminal(identity)
    assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)==''
    subprocess.run(['git','diff','--exit-code','a0cc7741','--','src','tests/Lokad.Onnx.Backend.Tests','tests/Lokad.Onnx.Tensors.Tests'],cwd=ROOT,check=True,stdout=subprocess.PIPE)
    # All pre-existing product/test sources outside the twelve reviewed files
    # must still equal the isolated qualified tree. New candidate files are
    # exactly the additions named in that patch.
    tracked=subprocess.check_output(['git','ls-files','src','tests/Lokad.Onnx.Backend.Tests','tests/Lokad.Onnx.Tensors.Tests'],cwd=ROOT,text=True).splitlines()
    changed=[]
    for name in tracked:
        current,candidate=ROOT/name,MODEL/'source'/name
        assert candidate.exists(),name
        if current.read_bytes()!=candidate.read_bytes():
            if current.suffix in ['.cs','.csproj','.props','.targets','.md','.txt','.proto','.py']:
                if current.read_text(encoding='utf-8-sig')==candidate.read_text(encoding='utf-8-sig'):continue
            changed.append(name)
    assert set(changed)<=set(NAMES),changed
    patch=[];before={}
    for name in NAMES:
        current,candidate=ROOT/name,MODEL/'source'/name
        before[name]=pin(current) if current.exists() else None
        left=current.read_text(encoding='utf-8-sig') if current.exists() else ''
        right=candidate.read_text(encoding='utf-8-sig');assert left!=right,name
        patch.extend(difflib.unified_diff(left.splitlines(True),right.splitlines(True),fromfile='a/'+name if current.exists() else '/dev/null',tofile='b/'+name))
    PATCHBASE.mkdir(exist_ok=True);path=PATCHBASE/'candidate.patch'
    if path.exists(): assert path.read_text() == ''.join(patch)
    else: path.write_text(''.join(patch),encoding='utf8')
    subprocess.run(['git','apply','--check','--ignore-space-change',str(path)],cwd=ROOT,check=True)
    assert subprocess.check_output(['git','apply','--numstat',str(path)],cwd=ROOT,text=True).count('\n')==12
    inputs=[path,MODEL/'closed.json',PACKAGE/'closed.json',*[MODEL/'source'/n for n in NAMES],
        PACKAGE/'consumer/Program.cs',PACKAGE/'consumer/PackageProbe.csproj',ROOT/'tests/pyannote/combined-avx512/Inventory.cs.txt',
        ROOT/'artifacts/pyannote-blocked-spatial-composition-20260922/bridge/Bridge.csproj',*TOOLS.iterdir()]
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    save(PATCHBASE/'prepared.json',dict(passed=True,files={rel(p):pin(p) for p in inputs if p.is_file()},before=before,names=NAMES,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        product_files_changed=False,admission_required=True,
        preparation_note='Pre-freeze default git-apply check rejected three CRLF context files. The patch was retained unchanged; ignore-space-change handles context newlines, with exact source pins and post-apply full-text equality mandatory.'))
    print(dict(prepared=pin(PATCHBASE/'prepared.json'),patch=pin(path),files=len(NAMES),product_files_changed=False))


if __name__=='__main__':main()
