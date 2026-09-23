"""Combine row pointers with nine ordered steps; preserve stores, tails and AVX2."""
import difflib
import hashlib

SOURCE='src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Kernels.cs'
SELECTED='b7f2730aca76bdeea61bf6ad168c9b9713e4dce942045725dfa34485bd4187cf'

def transform(text):
    original=text.replace('\r\n','\n')
    assert hashlib.sha256(original.encode()).hexdigest()==SELECTED
    start=original.index('    static void Kernel512(')
    store=original.index('                *(Vector512<float>*)(output',start)
    prefix,kernel,suffix=original[:start],original[start:store],original[store:]
    declaration='        int ph = h + 2, pw = w + 2, spatial = oh * ow, fusedEnd = spatial / 8 * 8;'
    assert kernel.count(declaration)==1
    kernel=kernel.replace(declaration,declaration+'\n        int rowStride = pw * lanes, blockStride = ph * rowStride, step = stride * lanes;')
    loops=('                for (int ic = 0; ic < c; ic++)\n'
           '                for (int ky = 0; ky < 3; ky++)\n'
           '                for (int kx = 0; kx < 3; kx++)\n')
    assert kernel.count(loops)==1
    before,body=kernel.split(loops)
    assert body.startswith('                {\n') and body.endswith('                }\n')
    body=body[len('                {\n'):-len('                }\n')]
    address='                    float* input = x + ((ic / lanes * ph + y * stride + ky) * pw + col * stride + kx) * lanes + ic % lanes;\n'
    assert body.count(address)==1;body=body.replace(address,'')
    for position in range(6):
        old=f'input[{position} * stride * lanes]';assert body.count(old)==1
        body=body.replace(old,f'input[{position} * step]')
    expanded=['                float* tileInput = x + (y * stride * pw + col * stride) * lanes;\n',
        '                for (int ic = 0; ic < c; ic++)\n                {\n',
        '                    float* row0 = tileInput + (ic >> 4) * blockStride + (ic & 15);\n',
        '                    float* row1 = row0 + rowStride;\n',
        '                    float* row2 = row1 + rowStride;\n']
    for ky in range(3):
        for kx in range(3):
            expanded.append(f'                    // Ordered kernel row {ky}, column {kx}.\n                    {{\n')
            expanded.append(f'                        float* input = row{ky} + {kx} * lanes;\n')
            expanded.extend('    '+line for line in body.splitlines(True))
            expanded.append('                    }\n')
    expanded.append('                }\n')
    candidate=prefix+before+''.join(expanded)+suffix
    assert candidate[:start]==prefix
    assert candidate[candidate.index('                *(Vector512<float>*)(output',start):]==suffix
    assert candidate.count('Avx512F.FusedMultiplyAdd')==original.count('Avx512F.FusedMultiplyAdd')+8*12
    assert candidate.count('for (int ic = 0; ic < c; ic++)')==original.count('for (int ic = 0; ic < c; ic++)')
    difference=''.join(difflib.unified_diff(original.splitlines(True),candidate.splitlines(True),fromfile='a/'+SOURCE,tofile='b/'+SOURCE))
    return candidate,difference
