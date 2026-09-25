"""Prepare one direct-depthwise candidate over the exact measured M78 snapshot."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]; TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-direct-depthwise-source-v2-20260925'
PRIOR = ROOT/'artifacts/parakeet-packed-final-row-source-20260925'
TARGET = 'src/Lokad.Onnx/TensorOps.ConvPool.cs'
HELPER = 'src/Lokad.Onnx/Zzz.DirectDepthwise.cs'
TEST = 'tests/Lokad.Onnx.Backend.Tests/DirectDepthwiseTests.cs'
INSERTION = '''        if (TryConvDirectDepthwise(xd, wd, bd, output, N, group, C, H, W, M,
            kH, kW, dH, dW, sH, sW, pad, outH, outW, options)) return output;
'''
ANCHOR = '        var spatial = PlanConvSpatialScratch(C, kH, kW, M, outH, outW);'


def pin(path): return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def read(path): return json.loads(path.read_text(encoding='utf8'))
def write(path, value):
    with path.open('x', encoding='utf8') as f: json.dump(value, f, indent=2); f.write('\n')


def changed(original):
    assert original.count(ANCHOR) == 1 and 'TryConvDirectDepthwise' not in original
    after = original.replace(ANCHOR, INSERTION+ANCHOR)
    assert after.replace(INSERTION, '') == original
    assert after.index('if (options.UseSegmentedConvolution') < after.index(INSERTION) < after.index(ANCHOR)
    return after


def main():
    assert not BASE.exists()
    assert pin(PRIOR/'prepared.json')['sha256'] == 'e3ca64b50ea4a5dd276d90a19779d602f8ed78b6275b9e5448fe3b021bec5e1b'
    previous = read(PRIOR/'prepared.json'); assert len(previous['source']) == 433
    values = {}
    for name, wanted in previous['source'].items():
        p = PRIOR/'source'/name; assert pin(p) == wanted; values[name] = p.read_bytes()
    diagnosis = ROOT/'artifacts/parakeet-depthwise-route-amd-20260925/closed.json'
    native = ROOT/'artifacts/parakeet-ort-depthwise-kernels-20260925/corrected-closed.json'
    assert read(diagnosis)['passed'] and read(native)['passed']
    failed = ROOT/'artifacts/parakeet-direct-depthwise-build-amd-20260925'
    receipt = read(failed/'build-collected/build-collection.json')
    assert receipt['terminal'] and receipt['code'] == 1
    for name,wanted in receipt['files'].items(): assert pin(failed/'build-collected'/name) == wanted
    original = ROOT/'artifacts/parakeet-direct-depthwise-source-20260925'
    first = read(original/'prepared.json')
    assert pin(TOOLS/'DirectDepthwise.cs.txt') == first['source'][HELPER]
    fixed_test = (TOOLS/'DirectDepthwiseTests.cs.txt').read_text(encoding='utf8')
    reversed_test = fixed_test.replace('if (!OperatingSystem.IsLinux()) throw new PlatformNotSupportedException("AMD Linux qualification only.");', 'Assert.True(OperatingSystem.IsLinux());').replace('MemoryMarshal.TryGetArray<float>(output.Buffer, out var memory)', 'MemoryMarshal.TryGetArray(output.Buffer, out var memory)')
    assert reversed_test == (original/'source'/TEST).read_text(encoding='utf8')
    before = values[TARGET].decode(); after = changed(before); values[TARGET] = after.encode()
    assert HELPER not in values and TEST not in values
    values[HELPER] = (TOOLS/'DirectDepthwise.cs.txt').read_bytes()
    values[TEST] = (TOOLS/'DirectDepthwiseTests.cs.txt').read_bytes()
    BASE.mkdir()
    for name, content in values.items():
        p = BASE/'source'/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(content)
    (BASE/'candidate.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=TARGET, tofile=TARGET)), encoding='utf8')
    (BASE/'prospective-plan.md').write_bytes((ROOT/'PLAN.md').read_bytes())
    source = {n:pin(BASE/'source'/n) for n in values}
    assert [n for n,v in previous['source'].items() if source[n] != v] == [TARGET]
    write(BASE/'prepared.json', dict(passed=True, root_product_changed=False, release_admitted=False,
        failed_build_collection=pin(failed/'build-collected/build-collection.json'), original_candidate=pin(original/'prepared.json'), test_only_correction=True,
        baseline=pin(PRIOR/'prepared.json'), diagnosis=pin(diagnosis), native=pin(native), source=source,
        changed_product_files=[TARGET], added_product_files=[HELPER], added_tests=[TEST],
        changed_methods=['Tensor.Conv2DFloatCore'], added_methods=['TryConvDirectDepthwise', 'RunDirectDepthwiseLine',
            'RunDirectDepthwiseSpatial', 'DirectDepthwisePoint'],
        plan=pin(BASE/'prospective-plan.md'), patch=pin(BASE/'candidate.patch'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()}))
    print(json.dumps(dict(source=pin(BASE/'prepared.json'), files=len(source))))


if __name__ == '__main__': main()
