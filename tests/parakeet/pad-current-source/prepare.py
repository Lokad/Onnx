"""Apply the existing M47 dispatcher to the exact qualified current root snapshot."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-pad-current-source-20260926'
QUALIFIED = ROOT / 'artifacts/parakeet-owned-batch-isolation-root-policy-amd-20260925'
APPLIED = ROOT / 'artifacts/parakeet-owned-batch-isolation-root-policy-20260925/integration/applied.json'
DIAGNOSTIC = ROOT / 'artifacts/parakeet-pad-warmup-diagnostic-amd-20260926'
OLD = ROOT / 'tests/parakeet/pad-dispatch-source'
TARGET = 'src/Lokad.Onnx/CPUExecutionProvider.Shape.cs'
HELPER = 'src/Lokad.Onnx/Zzz.LastAxisPadDispatch.cs'
TEST = 'tests/Lokad.Onnx.Backend.Tests/LastAxisPadTests.cs'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def transform(before):
    needle = 'return Success(op, PadCore('
    assert before.count(needle) == 4 and 'PadDispatch' not in before
    after = before.replace(needle, 'return Success(op, PadDispatch(')
    assert after.replace('return Success(op, PadDispatch(', needle) == before
    marker = '    static DenseTensor<T> PadCore<T>('
    assert after[after.index(marker):] == before[before.index(marker):]
    return after


def main():
    assert not BASE.exists()
    assert pin(QUALIFIED / 'closed.json')['sha256'] == 'c77ef606c76508144c5656bdbe48aeac8948ce28396c8bee645587d31609c475'
    proof = read(QUALIFIED / 'closed.json'); assert proof['passed']
    analysis = read(QUALIFIED / 'analysis.json')
    assert pin(QUALIFIED / 'analysis.json') == proof['analysis']
    assert analysis['passed'] and analysis['root_source_verified'] and analysis['root_integration'] == pin(APPLIED)
    source = read(APPLIED)['source_files']; assert len(source) == 435
    for name, wanted in source.items(): assert pin(ROOT / name) == wanted, name
    assert pin(DIAGNOSTIC / 'closed.json')['sha256'] == '4402da8effb9467d71fd6472678ead08b714ba147d9e3fe3f8788bb715ec9b4f'
    diagnostic = read(DIAGNOSTIC / 'closed.json')
    assert diagnostic['passed'] and not diagnostic['admitted'] and diagnostic['original_screens_remain_rejected']
    observed = read(DIAGNOSTIC / 'analysis.json')
    assert pin(DIAGNOSTIC / 'analysis.json') == diagnostic['files']['analysis.json']
    assert observed['state_condition_met'] and all(not v['later_loads'] for v in observed['compilation_state'].values())
    old = ROOT / 'artifacts/parakeet-pad-dispatch-source-20260923/prepared.json'
    assert pin(old)['sha256'] == '727fa49ee92dfe2cfe8ff34396d27fee82b961f826eb2499dbf1a8d818e77b0d'
    previous = read(old)
    for name in (HELPER, TEST):
        assert name not in source and not (ROOT / name).exists()
        assert pin(OLD / Path(name).name) == previous['source'][name]
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'src'], cwd=ROOT, check=True)
    before = (ROOT / TARGET).read_bytes()
    after = transform(before.decode('utf8')).encode('utf8')
    BASE.mkdir()
    for name in source:
        target = BASE / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    (BASE / 'source' / TARGET).write_bytes(after)
    for name in (HELPER, TEST): shutil.copy2(OLD / Path(name).name, BASE / 'source' / name)
    patch = ''.join(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True),
                                       fromfile=TARGET, tofile=TARGET))
    for name in (HELPER, TEST):
        patch += ''.join(difflib.unified_diff([], (BASE / 'source' / name).read_text().splitlines(True),
                                            fromfile='/dev/null', tofile=name))
    (BASE / 'candidate.patch').write_text(patch, encoding='utf8')
    shutil.copy2(ROOT / 'PLAN.md', BASE / 'prospective-plan.md')
    current = {name: pin(BASE / 'source' / name) for name in (*source, HELPER, TEST)}
    assert [name for name in source if current[name] != source[name]] == [TARGET]
    value = dict(passed=True, built=False, root_product_changed=False, release_admitted=False,
        qualified_parent=pin(QUALIFIED / 'closed.json'), root_integration=pin(APPLIED),
        parent_product=analysis['built'], diagnosis=pin(DIAGNOSTIC / 'closed.json'),
        original_proposal=pin(old), before=source, source=current,
        changed=[TARGET, HELPER, TEST], patch=pin(BASE / 'candidate.patch'), plan=pin(BASE / 'prospective-plan.md'),
        tools={p.name: pin(p) for p in TOOLS.iterdir() if p.is_file()})
    (BASE / 'prepared.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE / 'prepared.json'), source_files=len(current), root_product_changed=False)))


if __name__ == '__main__': main()
