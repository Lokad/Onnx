"""Extend the isolated projection guard; retain every other source byte."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/parakeet-wide-projection-isolation-source-20260923'
PARENT = ROOT / 'artifacts/parakeet-isolated-short-kernels-source-20260923'
DIAG = ROOT / 'artifacts/parakeet-isolated-runtime-diagnostic-amd-20260923'
FILE = 'src/Lokad.Onnx/Zzz.IsolatedShortMatMul.cs'


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as f: return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(f, 'sha256').hexdigest())


def transform(text):
    guard = 'if (m >= 48 && m < 64 && n >= 1024 && k >= 1024'
    comment = '// Keep short, wide projections separate from the general dispatch body.'
    assert text.count(guard) == text.count(comment) == 1
    return text.replace(guard, 'if (m >= 48 && n >= 1024 && k >= 1024').replace(comment,
        '// Isolate wide projections at every packed row count; shared kernels stay unchanged.')


def main():
    assert not BASE.exists()
    assert pin(PARENT / 'prepared.json')['sha256'] == 'f6635fac6f21ec4832ea105fc7d4e64888db8716b1f4c345776652cc30989104'
    prior = read(PARENT / 'prepared.json')
    assert prior['passed'] and not prior['root_product_changed'] and len(prior['source']) == 421
    for n, v in prior['source'].items(): assert pin(PARENT / 'source' / n) == v, n
    assert len(prior['before']) == 420
    for n, v in prior['before'].items(): assert pin(ROOT / n) == v, n
    assert pin(DIAG / 'closed.json')['sha256'] == 'ecfea393d0475617f32de3b87419cd6354f6d3b61edd4883adbdba877caa0f2c'
    diag = read(DIAG / 'closed.json'); assert diag['passed'] and diag['diagnostic_only']
    for n, v in diag['files'].items(): assert pin(DIAG / n) == v, n
    before = (PARENT / 'source' / FILE).read_text(encoding='utf8'); after = transform(before)
    BASE.mkdir(); source = BASE / 'source'; source.mkdir()
    for n in prior['source']:
        target = source / n; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(PARENT / 'source' / n, target)
    (source / FILE).write_text(after, encoding='utf8', newline='\n')
    assert [n for n, v in prior['source'].items() if pin(source / n) != v] == [FILE]
    (BASE / 'candidate.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=FILE, tofile=FILE)), encoding='utf8')
    shutil.copy2(ROOT / '.agent/m52-wide-projection-isolation-20260923.md', BASE / 'prospective-plan.md')
    shutil.copy2(PARENT / 'census.json', BASE / 'census.json')
    result = dict(passed=True, built=False, root_product_changed=False, before=prior['before'],
        source={n: pin(source / n) for n in prior['source']}, delta_from_parent=[FILE],
        parent=pin(PARENT / 'prepared.json'), diagnostic=pin(DIAG / 'closed.json'), generator=pin(Path(__file__)),
        patch=pin(BASE / 'candidate.patch'), census=pin(BASE / 'census.json'), plan=pin(BASE / 'prospective-plan.md'),
        scope='Relative to M50, remove m<64 from isolated dispatch guard; explanatory comment only otherwise. All shared bodies, flags, arithmetic and method names exact.')
    (BASE / 'prepared.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE / 'prepared.json'), source_files=len(result['source']), delta=result['delta_from_parent'])))


if __name__ == '__main__': main()
