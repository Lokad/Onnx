"""Extend only the selected AVX512 output blocking; preserve ordered arithmetic."""
import difflib
import hashlib

SOURCE = 'src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Kernels.cs'
ADDED = 'src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Four.cs'


def once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def transform(original):
    original = original.replace('\r\n', '\n')
    assert hashlib.sha256(original.encode()).hexdigest() == 'b7f2730aca76bdeea61bf6ad168c9b9713e4dce942045725dfa34485bd4187cf'
    start = original.index('    static void Kernel512(')
    assert original.endswith('    }\n}\n')
    method = original[start:-2]
    four = once(method, 'static void Kernel512(', 'static void Kernel512Four(')
    four = once(four, 'oc += 2 * lanes', 'oc += 4 * lanes')
    for spatial in range(6):
        old = f'                Vector512<float> a{spatial}1 = Vector512<float>.Zero;'
        four = once(four, old, old + ''.join(f'\n                Vector512<float> a{spatial}{block} = Vector512<float>.Zero;' for block in [2, 3]))
    old = '                float* w1 = w0 + lanes * c * 9;'
    four = once(four, old, old + '\n                float* w2 = w1 + lanes * c * 9; float* w3 = w2 + lanes * c * 9;')
    old = 'var wv0 = *(Vector512<float>*)w0; var wv1 = *(Vector512<float>*)w1;'
    assert four.count(old) == 2
    four = four.replace(old, old + '\n                    var wv2 = *(Vector512<float>*)w2; var wv3 = *(Vector512<float>*)w3;')
    for spatial in range(6):
        old = f'                    a{spatial}1 = Avx512F.FusedMultiplyAdd(i{spatial}, wv1, a{spatial}1);'
        four = once(four, old, old + ''.join(f'\n                    a{spatial}{block} = Avx512F.FusedMultiplyAdd(i{spatial}, wv{block}, a{spatial}{block});' for block in [2, 3]))
    assert four.count('w0 += lanes; w1 += lanes;') == 2
    four = four.replace('w0 += lanes; w1 += lanes;', 'w0 += lanes; w1 += lanes; w2 += lanes; w3 += lanes;')
    for spatial in range(6):
        old = f'                if (oc + lanes < m) *(Vector512<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col + {spatial}) * lanes) = a{spatial}1;'
        four = once(four, old, '\n'.join(f'                *(Vector512<float>*)(output + ((oc / lanes + {block}) * spatial + y * ow + col + {spatial}) * lanes) = a{spatial}{block};' for block in [1, 2, 3]))
    old = '                var a0 = Vector512<float>.Zero; var a1 = Vector512<float>.Zero;'
    four = once(four, old, old + '\n                var a2 = Vector512<float>.Zero; var a3 = Vector512<float>.Zero;')
    old = '                float* w0 = weights + oc * c * 9; float* w1 = w0 + lanes * c * 9;'
    four = once(four, old, old + '\n                float* w2 = w1 + lanes * c * 9; float* w3 = w2 + lanes * c * 9;')
    old = '                        a0 = Avx512F.FusedMultiplyAdd(input, wv0, a0); a1 = Avx512F.FusedMultiplyAdd(input, wv1, a1);'
    four = once(four, old, old + '\n                        a2 = Avx512F.FusedMultiplyAdd(input, wv2, a2); a3 = Avx512F.FusedMultiplyAdd(input, wv3, a3);')
    old = '                        a0 = a0 + input * wv0; a1 = a1 + input * wv1;'
    four = once(four, old, old + '\n                        a2 = a2 + input * wv2; a3 = a3 + input * wv3;')
    old = '                if (oc + lanes < m) *(Vector512<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col) * lanes) = a1;'
    four = once(four, old, '\n'.join(f'                *(Vector512<float>*)(output + ((oc / lanes + {block}) * spatial + y * ow + col) * lanes) = a{block};' for block in [1, 2, 3]))
    candidate = once(original, '        const int lanes = 16;', '''        if (m % 64 == 0)
        {
            Kernel512Four(x, weights, output, c, m, h, w, stride, oh, ow);
            return;
        }
        const int lanes = 16;''')
    added = '''using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
namespace Lokad.Onnx;

internal static unsafe partial class ConvBlockedSpatial
{
''' + four + '}\n'
    diff = ''.join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True), fromfile='a/'+SOURCE, tofile='b/'+SOURCE))
    diff += ''.join(difflib.unified_diff([], added.splitlines(True), fromfile='/dev/null', tofile='b/'+ADDED))
    return candidate, added, diff
