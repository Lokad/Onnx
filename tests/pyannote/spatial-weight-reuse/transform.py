"""Extend only the selected AVX512 spatial tile, preserving every reduction."""
import difflib
import hashlib

SOURCE = 'src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Kernels.cs'


def once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def accumulators(count):
    return ''.join(f'                Vector512<float> a{p}{b} = Vector512<float>.Zero;\n'
                   for p in range(count) for b in range(2))


def arithmetic(count):
    return ''.join(f'''                    var i{p} = Vector512.Create(input[{p} * stride * lanes]);
                    a{p}0 = Avx512F.FusedMultiplyAdd(i{p}, wv0, a{p}0);
                    a{p}1 = Avx512F.FusedMultiplyAdd(i{p}, wv1, a{p}1);
''' for p in range(count))


def stores(count):
    return ''.join(f'''                *(Vector512<float>*)(output + (oc / lanes * spatial + y * ow + col + {p}) * lanes) = a{p}0;
                if (oc + lanes < m) *(Vector512<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col + {p}) * lanes) = a{p}1;
''' for p in range(count))


def transform(original):
    original = original.replace('\r\n', '\n')
    assert hashlib.sha256(original.encode()).hexdigest() == 'b7f2730aca76bdeea61bf6ad168c9b9713e4dce942045725dfa34485bd4187cf'
    start = original.index('    static void Kernel512(')
    prefix, kernel = original[:start], original[start:]
    tail = kernel[kernel.index('            for (; col < ow; col++)'):]
    kernel = once(kernel, 'for (; col + 6 <= ow && y * ow + col + 6 <= fusedEnd; col += 6)',
                  'for (; col + 12 <= ow && y * ow + col + 12 <= fusedEnd; col += 12)')
    kernel = once(kernel, accumulators(6), accumulators(12))
    kernel = once(kernel, arithmetic(6), arithmetic(12))
    kernel = once(kernel, stores(6), stores(12))
    assert kernel[kernel.index('            for (; col < ow; col++)'):] == tail
    candidate = prefix+kernel
    diff = ''.join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True),
                                      fromfile='a/'+SOURCE, tofile='b/'+SOURCE))
    return candidate, diff
