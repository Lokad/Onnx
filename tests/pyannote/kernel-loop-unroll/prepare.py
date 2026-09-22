"""Prepare the isolated one-file prototype without building or changing root source."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from transform import SOURCE, transform

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/pyannote-kernel-loop-unroll-20260922'
PROOF = ROOT/'artifacts/pyannote-lstm-input-root-amd-20260922'
PLAN = ROOT/'.agent/m23-pyannote-kernel-loops-20260922.md'


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
    original = (source/SOURCE).read_text(encoding='utf8')
    candidate, difference = transform(original)
    (source/SOURCE).write_text(candidate, encoding='utf8', newline='\n')
    (BASE/'candidate.patch').write_text(difference, encoding='utf8', newline='\n')
    shutil.copy2(PLAN, BASE/'prospective-plan.md')
    changed = [name for name, wanted in before.items() if pin(source/name) != wanted]
    assert changed == [SOURCE]
    for name, wanted in before.items(): assert pin(ROOT/name) == wanted, name
    value = dict(passed=True, root_product_changed=False, built=False, numerically_qualified=False,
        source_commit=subprocess.check_output(['git','rev-parse','fe4eb657'], cwd=ROOT, text=True).strip(),
        product_closure=pin(PROOF/'closed.json'), changed=changed, before=before,
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},
        tools={p.relative_to(ROOT).as_posix():pin(p) for p in Path(__file__).parent.glob('*.py')},
        allowed_compiled_change='Lokad.Onnx.ConvBlockedSpatial::Kernel512',
        expected_unchanged_core_methods=3162, expected_unchanged_data_methods=697)
    (BASE/'prepared.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'), patch=pin(BASE/'candidate.patch'), changed=changed,
        root_product_changed=False, built=False, numerically_qualified=False)))


if __name__ == '__main__': main()
