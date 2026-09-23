"""Isolate short-wide packing while retaining the original general dispatcher."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-short-dispatch-source-v3-20260923'
ADDED = 'src/Lokad.Onnx/Zzz.ShortWideMatMul.cs'
PROOF = ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923'
SCREEN = ROOT/'artifacts/parakeet-short-wide-pack-screen-amd-20260923'
FILE = 'src/Lokad.Onnx/TensorOps.MatMul.cs'


def read(p): return json.loads(p.read_text(encoding='utf8'))


def pin(p):
    with p.open('rb') as f:
        return dict(bytes=p.stat().st_size, sha256=hashlib.file_digest(f, 'sha256').hexdigest())


WRAPPER = '''    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static unsafe void RunFloatMatMulKernel(int m, int n, int k, float* x, float* y, float* output, TensorExecutionOptions options)
    {
        // Keep short, wide projections separate from the general dispatch body.
        if (m >= 48 && m < 64 && n >= 1024 && k >= 1024
            && (long)n * k <= 67108864
            && options.UseSimd && options.UseIntrinsics && Fma.IsSupported)
        {
            RunShortWidePackedRows(m, n, k, x, y, output, options);
            return;
        }
        RunGeneralFloatMatMulKernel(m, n, k, x, y, output, options);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    static unsafe void RunShortWidePackedRows(int m, int n, int k, float* x, float* y, float* output, TensorExecutionOptions options)
    {
        bool threeRows = m % 3 == 0;
        int rows = threeRows ? m : m - m % 2;
        float[] packed = RentScratch<float>(n * k, options);
        try
        {
            fixed (float* pp = packed)
            {
                PackPanelsB(n, k, y, pp);
                if (!AblationSwitches.EnablePackedAvx512Dynamic
                    || !TryPackedAvx512Rows(rows, n, k, x, pp, output))
                {
                    if (threeRows)
                        mm_unsafe_vectorized_intrinsics_3x4packed(rows, n, k, x, pp, output);
                    else
                        mm_unsafe_vectorized_intrinsics_2x4packed_bump(rows, n, k, x, pp, output);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(packed);
        }
        // The last odd row still reads the original operand, after scratch returns.
        if (rows != m)
            mm_unsafe_vectorized_intrinsics(1, n, k, x + rows * n, y, output + rows * k);
    }

'''


def transform(source):
    start = source.index('    static unsafe void RunFloatMatMulKernel(')
    end = source.index('\n    public static ', start)
    before = source[start:end]
    general = before.replace('RunFloatMatMulKernel(', 'RunGeneralFloatMatMulKernel(', 1)
    assert general.replace('RunGeneralFloatMatMulKernel(', 'RunFloatMatMulKernel(', 1) == before
    wrapper, helper = WRAPPER.split('    [MethodImpl(MethodImplOptions.NoInlining)]', 1)
    extra = 'namespace Lokad.Onnx;\n\nusing System.Buffers;\nusing System.Runtime.CompilerServices;\nusing System.Runtime.Intrinsics.X86;\nusing static Lokad.Onnx.MathOps;\n\npublic abstract partial class Tensor<T> where T : unmanaged\n{\n' + '    [MethodImpl(MethodImplOptions.NoInlining)]' + helper + '    [MethodImpl(MethodImplOptions.NoInlining)]\n' + general + '\n}\n'
    return source[:start] + wrapper.rstrip() + '\n' + source[end:], extra


def main():
    assert not BASE.exists()
    for folder, digest in [(PROOF, '62141a2a722548697c106e42b2c0d9425b4f0c6ce166611a5bc3ca26a4fccdd0'),
                           (SCREEN, 'ca7c7cb28e20f2e6a9163c5d33f9c5d6bab2cf0c97f094f269d0249ba05ad1ef')]:
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
    screen = read(SCREEN/'analysis.json')
    assert not screen['admitted'] and all(c['passed'] for c in screen['controls'])
    expected = {name.removeprefix('source/'): wanted
                for name, wanted in read(PROOF/'bundle/stage.json')['files'].items()
                if name.startswith('source/')}
    assert len(expected) == 420
    for name, wanted in expected.items(): assert pin(ROOT/name) == wanted, name
    before = (ROOT/FILE).read_text(encoding='utf8'); after, extra = transform(before)
    BASE.mkdir(); source = BASE/'source'; source.mkdir()
    for name in expected:
        p = source/name; p.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, p)
    (source/FILE).write_text(after, encoding='utf8', newline='\n')
    (source/ADDED).write_text(extra, encoding='utf8', newline='\n')
    assert not (ROOT/ADDED).exists()
    changed = [name for name, wanted in expected.items() if pin(source/name) != wanted]
    assert changed == [FILE]
    for name, wanted in expected.items(): assert pin(ROOT/name) == wanted, name
    (BASE/'candidate.patch').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile=FILE, tofile=FILE)) + ''.join(difflib.unified_diff([], extra.splitlines(True), fromfile='/dev/null', tofile=ADDED)), encoding='utf8')
    shutil.copy2(ROOT/'.agent/m41-parakeet-short-dispatch-20260923.md', BASE/'prospective-plan.md')
    census = ROOT/'artifacts/parakeet-short-wide-pack-source-20260923/census.json'
    assert pin(census) == read(census.parent/'prepared.json')['census']
    shutil.copy2(census, BASE/'census.json')
    value = dict(passed=True, built=False, numerically_qualified=False, root_product_changed=False,
        before=expected, changed=changed, added=[ADDED],
        permitted_compiled_change='RunFloatMatMulKernel wrapper; added original RunGeneralFloatMatMulKernel and RunShortWidePackedRows',
        source={p.relative_to(source).as_posix(): pin(p) for p in source.rglob('*') if p.is_file()},
        patch=pin(BASE/'candidate.patch'), plan=pin(BASE/'prospective-plan.md'), generator=pin(Path(__file__)),
        root_closure=pin(PROOF/'closed.json'), screen_closure=pin(SCREEN/'closed.json'), census=pin(BASE/'census.json'),
        scope='Isolate48..63wide rows; retain exact original general body, arithmetic, budgets and all other routes.')
    (BASE/'prepared.json').write_text(json.dumps(value, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'), patch=value['patch'], changed=changed, added=[ADDED], root_product_changed=False)))


if __name__ == '__main__': main()
