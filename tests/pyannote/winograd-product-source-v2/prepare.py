"""Prepare the isolated one-file prototype without building or changing root source."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import difflib

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/pyannote-winograd-product-source-v2-20260923'
PROOF = ROOT/'artifacts/pyannote-lstm-input-root-amd-20260922'
PLAN = ROOT/'.agent/m34-pyannote-winograd-product-20260923.md'


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def main():
    assert not BASE.exists()
    assert pin(PROOF/'closed.json')['sha256'] == '5cc03093982964beb44776b7d64cf561e5a6b49c1947b1d6b9a80b790cf41a21'
    proof = read(PROOF/'closed.json'); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(PROOF/name) == wanted, name
    BASE.mkdir(); source = BASE/'source'; source.mkdir(); before = {}
    stage = read(PROOF/'bundle/stage.json')
    for name, wanted in stage['files'].items():
        if not name.startswith('source/'): continue
        relative = name.removeprefix('source/')
        assert pin(ROOT/relative) == wanted, relative
        target = source/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, target); before[relative] = wanted
    assert len(before) == 416
    tools = Path(__file__).resolve().parent
    changes = ['src/Lokad.Onnx/GraphConvPacking.cs', 'src/Lokad.Onnx/TensorOps.ConvBlocked.cs',
        'tests/Lokad.Onnx.Backend.Tests/ConvBlockedSpatialTests.cs']
    added = ['src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Winograd.cs',
        'src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Winograd.Kernels.cs',
        'tests/Lokad.Onnx.Backend.Tests/ConvWinogradTests.cs', 'src/Lokad.Onnx/Zzz.ConvWinogradDispatch.cs']
    differences = []
    for name in changes:
        old = (source/name).read_text(encoding='utf8'); new = (tools/Path(name).name).read_text(encoding='utf8')
        differences.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=name,tofile=name))
        (source/name).write_text(new,encoding='utf8',newline='\n')
    prototype = ROOT/'tests/pyannote/winograd-range-prototype'
    for original,name in [(prototype/'Winograd.cs',added[0]),(prototype/'Winograd.Kernels.cs',added[1]),(tools/'ConvWinogradTests.cs',added[2]),(tools/'ConvWinogradDispatch.cs',added[3])]:
        assert not (source/name).exists();shutil.copy2(original,source/name)
        assert pin(original)==pin(source/name)
        differences.extend(difflib.unified_diff([], (source/name).read_text().splitlines(True),fromfile='/dev/null',tofile=name))
    (BASE/'candidate.patch').write_text(''.join(differences),encoding='utf8',newline='\n')
    shutil.copy2(PLAN, BASE/'prospective-plan.md')
    changed = [name for name,wanted in before.items() if pin(source/name)!=wanted]
    assert set(changed)==set(changes)
    screen=ROOT/'artifacts/pyannote-winograd-range-screen-amd-20260923'
    assert pin(screen/'closed.json')['sha256']=='302de6c9b75bf2a60f2b09b0e79f0c5d8ff846a149a1f1657f5fa094820c919a'
    result=read(screen/'closed.json');assert result['passed'] and result['admitted']
    for name,wanted in result['files'].items():assert pin(screen/name)==wanted,name
    for name, wanted in before.items(): assert pin(ROOT/name) == wanted, name
    value = dict(passed=True, root_product_changed=False, built=False, numerically_qualified=False,
        source_commit=subprocess.check_output(['git','rev-parse','fe4eb657'], cwd=ROOT, text=True).strip(),
        product_closure=pin(PROOF/'closed.json'), screen=pin(screen/'closed.json'), changed=changed, added=added, before=before,
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},
        tools={p.relative_to(ROOT).as_posix():pin(p) for p in tools.iterdir() if p.is_file()},
        prototype={p.relative_to(ROOT).as_posix():pin(p) for p in [prototype/'Winograd.cs',prototype/'Winograd.Kernels.cs']},
        allowed_compiled_types=['Lokad.Onnx.GraphConvPacking','Lokad.Onnx.PackedConvWeight','Lokad.Onnx.Tensor`1','Lokad.Onnx.ConvBlockedSpatial','Lokad.Onnx.ConvWinogradDispatch'],
        expected_unchanged_data_methods=697)
    (BASE/'prepared.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'), patch=pin(BASE/'candidate.patch'), changed=changed,
        root_product_changed=False, built=False, numerically_qualified=False)))


if __name__ == '__main__': main()
