"""Copy exact short-wide arithmetic while preserving all shared original methods."""
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-isolated-short-kernels-source-20260923'
OLD = ROOT / 'artifacts/parakeet-first-use-kernels-source-20260923'
QUALIFIED = ROOT / 'artifacts/pyannote-winograd-product-root-amd-20260923'
FILE = 'src/Lokad.Onnx/TensorOps.MatMul.cs'
ADDED = 'src/Lokad.Onnx/Zzz.IsolatedShortMatMul.cs'
PAIRS = [('PackPanelsB', 'ShortWidePackPanelsB'),
         ('mm_unsafe_vectorized_intrinsics_2x4packed_bump', 'ShortWideMultiply2Rows'),
         ('mm_unsafe_vectorized_intrinsics_3x4packed', 'ShortWideMultiply3Rows'),
         ('mm_unsafe_vectorized_intrinsics', 'ShortWideMultiplyRemainder')]


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def method(text, header):
    start = text.index(header); opened = text.index('{', start); depth = 1; end = opened + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}'); end += 1
    return text[start:end]


def transform(source, math, prior_dispatch, prior_helpers):
    declaration = '    static unsafe void RunFloatMatMulKernel('
    original = method(source, declaration)
    assert source.count('RunFloatMatMulKernel(') == 5 and source.count(declaration) == 1
    changed = source.replace('RunFloatMatMulKernel(', 'RunIsolatedShortWideKernel(')
    changed = changed.replace('    static unsafe void RunIsolatedShortWideKernel(', declaration, 1)
    assert changed.replace('RunIsolatedShortWideKernel(', 'RunFloatMatMulKernel(') == source
    assert method(changed, declaration) == original
    wrapper = method(prior_dispatch, declaration)
    wrapper = wrapper.replace(declaration, '    static unsafe void RunIsolatedShortWideKernel(', 1)
    wrapper = wrapper.replace('RunShortWidePackedRows(', 'RunIsolatedShortWidePackedRows(')
    wrapper = wrapper.replace('RunGeneralFloatMatMulKernel(', 'RunFloatMatMulKernel(')
    short = method(prior_helpers, '    static unsafe void RunShortWidePackedRows(')
    short = short.replace('RunShortWidePackedRows(', 'RunIsolatedShortWidePackedRows(', 1)
    for old, new in PAIRS:
        assert short.count(old + '(') == 1
        short = short.replace(old + '(', new + '(')
    prefix = math[:math.index('namespace Lokad.Onnx;')]
    extra = (prefix + 'using System.Buffers;\nusing static Lokad.Onnx.MathOps;\n\nnamespace Lokad.Onnx;\n\n'
             'public abstract partial class Tensor<T> where T : unmanaged\n{\n'
             '    [MethodImpl(MethodImplOptions.AggressiveInlining)]\n' + wrapper + '\n\n'
             '    [MethodImpl(MethodImplOptions.NoInlining)]\n' + short + '\n}\n\n'
             'public partial class MathOps\n{\n'
             '    // These copies serve only the guarded short-wide projection route.\n'
             '    // Shared kernels retain their original bodies and compilation flags.\n')
    for old, new in PAIRS:
        header = '    public unsafe static void ' + old + '('
        body = method(math, header); assert 'float*' in body[:body.index('{')]
        clone = body.replace(header, '    internal unsafe static void ' + new + '(', 1)
        assert clone.replace('    internal unsafe static void ' + new + '(', header, 1) == body
        extra += '    [MethodImpl(MethodImplOptions.AggressiveOptimization)]\n' + clone + '\n\n'
    return changed, extra + '}\n'


def main():
    assert not BASE.exists()
    review_path = ROOT / 'tests/parakeet/short-projection-isolation/source-observations-20260923.json'
    review = read(review_path); assert review['passed'] and review['static_only']
    assert review['generator'] == pin(review_path.parent / 'inspect_source.py')
    assert pin(OLD / 'prepared.json') == review['m43_source']
    prior = read(OLD / 'prepared.json'); assert len(prior['before']) == 420 and len(prior['source']) == 421
    for name, wanted in prior['before'].items(): assert pin(ROOT / name) == wanted, name
    for name, wanted in prior['source'].items(): assert pin(OLD / 'source' / name) == wanted, name
    assert pin(QUALIFIED / 'closed.json')['sha256'] == '62141a2a722548697c106e42b2c0d9425b4f0c6ce166611a5bc3ca26a4fccdd0'
    qualified = read(QUALIFIED / 'closed.json'); assert qualified['passed']
    for name, wanted in qualified['files'].items(): assert pin(QUALIFIED / name) == wanted, name
    for folder, identity in [('parakeet-first-use-kernels-app-amd-20260923', review['m43_application']),
                             ('warmed-release-amd-v2-20260923', review['rejected_release'])]:
        assert pin(ROOT / 'artifacts' / folder / 'closed.json') == identity
    assert not (ROOT / ADDED).exists()
    before = (ROOT / FILE).read_text(encoding='utf8')
    after, extra = transform(before, (ROOT / 'src/Lokad.Onnx/MathOps.cs').read_text(),
        (OLD / 'source' / FILE).read_text(), (OLD / 'source/src/Lokad.Onnx/Zzz.ShortWideMatMul.cs').read_text())
    BASE.mkdir(); source = BASE / 'source'; source.mkdir()
    for name in prior['before']:
        target = source / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, target)
    (source / FILE).write_text(after, encoding='utf8', newline='\n')
    (source / ADDED).write_text(extra, encoding='utf8', newline='\n')
    assert [n for n, v in prior['before'].items() if pin(source / n) != v] == [FILE]
    patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=FILE, tofile=FILE))
    patch += ''.join(difflib.unified_diff([], extra.splitlines(True), fromfile='/dev/null', tofile=ADDED))
    (BASE / 'candidate.patch').write_text(patch, encoding='utf8')
    shutil.copy2(ROOT / '.agent/m50-isolated-short-projection-kernels-20260923.md', BASE / 'prospective-plan.md')
    shutil.copy2(OLD / 'census.json', BASE / 'census.json'); assert pin(BASE / 'census.json') == prior['census']
    result = dict(passed=True, built=False, numerically_qualified=False, root_product_changed=False,
        before=prior['before'], changed=[FILE], added=[ADDED], pairs=PAIRS,
        source={p.relative_to(source).as_posix(): pin(p) for p in source.rglob('*') if p.is_file()},
        qualified_parent=pin(QUALIFIED / 'closed.json'), prior_source=pin(OLD / 'prepared.json'), review=pin(review_path),
        patch=pin(BASE / 'candidate.patch'), census=pin(BASE / 'census.json'), plan=pin(BASE / 'prospective-plan.md'),
        generator=pin(Path(__file__)), scope='Four call operands only; original general dispatcher and MathOps file intact; six new methods with flags 256,8,512,512,512,512.')
    assert len(result['source']) == 421
    for name, wanted in prior['before'].items(): assert pin(ROOT / name) == wanted, name
    (BASE / 'prepared.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE / 'prepared.json'), changed=result['changed'], added=result['added'])))


if __name__ == '__main__': main()
