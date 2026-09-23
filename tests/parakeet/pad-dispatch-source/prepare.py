"""Prepare M47 against qualified selected source 94a550de after M43 rejection."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-pad-dispatch-source-20260923'
PARENT = ROOT / 'artifacts/pyannote-winograd-product-root-integration-20260923'
QUALIFIED = ROOT / 'artifacts/pyannote-winograd-product-root-amd-20260923'
REJECTED = ROOT / 'artifacts/parakeet-last-axis-pad-screen-amd-20260923'
FILE = 'src/Lokad.Onnx/CPUExecutionProvider.Shape.cs'
HELPER = 'src/Lokad.Onnx/Zzz.LastAxisPadDispatch.cs'
TESTS = 'tests/Lokad.Onnx.Backend.Tests/LastAxisPadTests.cs'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,
                    sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def transform(source):
    needle = 'return Success(op, PadCore('
    replacement = 'return Success(op, PadDispatch('
    assert source.count(needle) == 4 and 'PadDispatch' not in source
    result = source.replace(needle, replacement)
    assert result.replace(replacement, needle) == source
    marker = '    static DenseTensor<T> PadCore<T>('
    assert result[result.index(marker):] == source[source.index(marker):]
    return result


def main():
    assert not BASE.exists()
    assert pin(REJECTED / 'closed.json')['sha256'] == '3473f6ef880173d6581255e790d8776f50d17560818f9a9abae273e3ae0f0abe'
    previous = read(REJECTED / 'closed.json')
    assert previous['passed'] and not previous['admitted']
    for name, wanted in previous['files'].items():
        assert pin(REJECTED / name) == wanted, name
    assert pin(PARENT / 'applied.json')['sha256'] == 'f367e28d3180fc0bf7d17600c353db534c7b8188b5c3e5457942bed7c83000eb'
    parent = read(PARENT / 'applied.json')
    assert parent['passed'] and len(parent['source_files']) == 420
    assert pin(QUALIFIED / 'closed.json')['sha256'] == '62141a2a722548697c106e42b2c0d9425b4f0c6ce166611a5bc3ca26a4fccdd0'
    proof = read(QUALIFIED / 'closed.json')
    assert proof['passed']
    for name, wanted in proof['files'].items():
        assert pin(QUALIFIED / name) == wanted, name
    analysis = read(QUALIFIED / 'analysis.json')
    assert analysis['passed'] and analysis['root_source_verified']
    assert analysis['root_integration'] == pin(PARENT / 'applied.json')
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'src'], cwd=ROOT, check=True)
    for name, wanted in parent['source_files'].items():
        assert pin(ROOT / name) == wanted, name
    assert not (ROOT / HELPER).exists()
    assert not (ROOT / TESTS).exists()
    before = (ROOT / FILE).read_text(encoding='utf8')
    after = transform(before)
    BASE.mkdir()
    source = BASE / 'source'
    for name in parent['source_files']:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    (source / FILE).write_text(after, encoding='utf8', newline='\n')
    shutil.copy2(TOOLS / Path(HELPER).name, source / HELPER)
    shutil.copy2(TOOLS / Path(TESTS).name, source / TESTS)
    patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        fromfile=FILE, tofile=FILE))
    for name in [HELPER, TESTS]:
        added = (source / name).read_text(encoding='utf8')
        patch += ''.join(difflib.unified_diff([], added.splitlines(True),
                                            fromfile='/dev/null', tofile=name))
    (BASE / 'candidate.patch').write_text(patch, encoding='utf8')
    shutil.copy2(ROOT / '.agent/m47-parakeet-pad-dispatch-20260923.md', BASE / 'prospective-plan.md')
    changed = [name for name, wanted in parent['source_files'].items() if pin(source / name) != wanted]
    assert changed == [FILE]
    for name, wanted in parent['source_files'].items():
        assert pin(ROOT / name) == wanted, name
    result = dict(passed=True, built=False, numerically_qualified=False,
                  root_product_changed=False, parent=pin(PARENT / 'applied.json'),
                  qualified_parent=pin(QUALIFIED / 'closed.json'),
                  rejected_predecessor=pin(REJECTED / 'closed.json'),
                  before=parent['source_files'], changed=[FILE, HELPER, TESTS],
                  source={p.relative_to(source).as_posix(): pin(p)
                          for p in source.rglob('*') if p.is_file()},
                  patch=pin(BASE / 'candidate.patch'), plan=pin(BASE / 'prospective-plan.md'),
                  tools={p.name: pin(p) for p in TOOLS.iterdir() if p.is_file()})
    assert len(result['source']) == 422
    (BASE / 'prepared.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE / 'prepared.json'), changed=result['changed'],
                         root_product_changed=False)))


if __name__ == '__main__':
    main()
