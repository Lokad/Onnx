"""Expand only the nine fixed full-tile AVX512 positions, preserving reduction order."""
import difflib
import hashlib

SOURCE = 'src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Kernels.cs'
SELECTED = 'b7f2730aca76bdeea61bf6ad168c9b9713e4dce942045725dfa34485bd4187cf'


def transform(text):
    original = text.replace('\r\n', '\n')
    assert hashlib.sha256(original.encode()).hexdigest() == SELECTED
    start = original.index('    static void Kernel512(')
    store = original.index('                *(Vector512<float>*)(output', start)
    prefix, kernel, suffix = original[:start], original[start:store], original[store:]
    loops = ('                for (int ic = 0; ic < c; ic++)\n'
             '                for (int ky = 0; ky < 3; ky++)\n'
             '                for (int kx = 0; kx < 3; kx++)\n')
    assert kernel.count(loops) == 1
    before, body = kernel.split(loops)
    assert body.startswith('                {\n') and body.endswith('                }\n')
    body = body[len('                {\n'):-len('                }\n')]
    assert body.count(' + ky)') == body.count(' + kx)') == 1
    expanded = ['                for (int ic = 0; ic < c; ic++)\n                {\n']
    for ky in range(3):
        for kx in range(3):
            step = body.replace(' + ky)', f' + {ky})').replace(' + kx)', f' + {kx})')
            expanded.append(f'                    // Kernel row {ky}, column {kx}: preserve the selected FMA chain.\n                    {{\n')
            expanded.extend('    '+line for line in step.splitlines(True))
            expanded.append('                    }\n')
    expanded.append('                }\n')
    candidate = prefix+before+''.join(expanded)+suffix
    assert candidate[:start] == prefix
    assert candidate[candidate.index('                *(Vector512<float>*)(output', start):] == suffix
    assert candidate.count('Avx512F.FusedMultiplyAdd') == original.count('Avx512F.FusedMultiplyAdd')+8*12
    difference = ''.join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True), fromfile='a/'+SOURCE, tofile='b/'+SOURCE))
    return candidate, difference
