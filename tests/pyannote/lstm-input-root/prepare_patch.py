"""Prepare the exact qualified five-file patch without changing root product files."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/pyannote-lstm-input-root-patch-20260922'
PRODUCT = ROOT/'artifacts/pyannote-lstm-input-product-amd-v2-20260922'
NAMES = [
    'src/Lokad.Onnx/CPUExecutionProvider.LstmPanels.cs',
    'src/Lokad.Onnx/CPUExecutionProvider.Recurrent.cs',
    'src/Lokad.Onnx/Zzz.LstmInputBlocks.cs',
    'tests/Lokad.Onnx.Backend.Tests/LstmInputBlockTests.cs',
    'tests/Lokad.Onnx.Backend.Tests/LstmOutputLaneTests.cs']
TEXT = {'.cs','.csproj','.props','.targets','.json','.slnx','.md','.txt','.config'}


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def canonical(path):
    data = path.read_bytes()
    return data.decode('utf-8-sig').replace('\r\n','\n').encode('utf8') if path.suffix in TEXT else data


def main():
    assert not BASE.exists()
    assert pin(PRODUCT/'closed.json')['sha256'] == '7d442059d85e5df31f9a616b4e5e5d5203a612affb41b59a0c1128e6e046529f'
    proof = read(PRODUCT/'closed.json'); assert proof['passed']
    for name,wanted in proof['files'].items(): assert pin(PRODUCT/name) == wanted,name
    qualified = read(PRODUCT/'analysis.json')
    assert qualified['passed'] and qualified['measured']['Lokad.Onnx.dll']['sha256'] == '208371f620fcfce5ff709db54525fa158b4c9442685430afbeb730619e1ded81'
    stage = read(PRODUCT/'bundle/stage.json'); source = PRODUCT/'bundle/source'
    before = {}; changed = []
    for name,wanted in stage['files'].items():
        if not name.startswith('source/'): continue
        relative = name.removeprefix('source/'); incoming = source/relative; current = ROOT/relative
        assert pin(incoming) == wanted,name
        before[relative] = pin(current) if current.exists() else None
        if not current.exists() or canonical(current) != canonical(incoming): changed.append(relative)
    assert sorted(changed) == sorted(NAMES),changed
    patch = []
    for name in NAMES:
        current = ROOT/name; incoming = source/name
        prior = subprocess.run(['git','show','9533cd67:'+name],cwd=ROOT,capture_output=True)
        if current.exists():
            assert prior.returncode == 0 and canonical(current) == prior.stdout.replace(b'\r\n',b'\n')
        else: assert prior.returncode != 0
        left = canonical(current).decode('utf8').splitlines(True) if current.exists() else []
        right = canonical(incoming).decode('utf8').splitlines(True)
        patch.append('diff --git a/'+name+' b/'+name+'\n')
        patch.extend(difflib.unified_diff(left,right,fromfile='a/'+name if current.exists() else '/dev/null',tofile='b/'+name))
    BASE.mkdir(); path = BASE/'candidate.patch'; path.write_text(''.join(patch),encoding='utf8',newline='\n')
    subprocess.run(['git','apply','--check','--ignore-space-change',str(path)],cwd=ROOT,check=True)
    rows = subprocess.check_output(['git','apply','--numstat',str(path)],cwd=ROOT,text=True).splitlines()
    assert sorted(row.split('\t',2)[2] for row in rows) == sorted(NAMES)
    value = dict(passed=True,conditional_only=True,root_product_changed=False,names=NAMES,before=before,
        after={name:pin(source/name) for name in NAMES},patch=pin(path),product_closure=pin(PRODUCT/'closed.json'),
        source=source.relative_to(ROOT).as_posix(),tool=pin(Path(__file__)),
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        condition='Apply only after full M22 application closure passes every native/meeting/ownership/repeatability/speed gate; then verify a normal root build/package on AMD.')
    (BASE/'prepared.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),patch=value['patch'],files=NAMES,root_product_changed=False)))


if __name__ == '__main__': main()
