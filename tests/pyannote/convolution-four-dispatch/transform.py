"""Keep original kernels exact; select a separate four-block pointer kernel."""
import difflib
import hashlib
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
SOURCE='src/Lokad.Onnx/Zzz.ConvBlockedSpatial.cs'
KERNEL='src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Kernels.cs'
ADDED='src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Four.cs'
GENERATOR=ROOT/'tests/pyannote/filter-block-reuse/transform.py'
GENERATOR_HASH='80ae11784ecafeb095aeab3bf108e57bf089b2d0a742c90680266da329bc7b6c'


def transform(caller, original_kernel):
    caller=caller.replace('\r\n','\n')
    assert hashlib.sha256(caller.encode()).hexdigest()=='5128ad60bf98dfd85989803d783d4b3e375623e81954abf353d4c2f5840e98ca'
    assert hashlib.sha256(GENERATOR.read_bytes()).hexdigest()==GENERATOR_HASH
    spec=importlib.util.spec_from_file_location('qualified_four_generator',GENERATOR)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    unused_kernel,added,unused_patch=module.transform(original_kernel)
    original_added=added
    start=added.index('    static void Kernel512Four(')
    store=added.index('                *(Vector512<float>*)(output',start)
    prefix,body,suffix=added[:start],added[start:store],added[store:]
    declaration='        int ph = h + 2, pw = w + 2, spatial = oh * ow, fusedEnd = spatial / 8 * 8;'
    body=module.once(body,declaration,declaration+'\n        int rowStride = pw * lanes, blockStride = ph * rowStride, step = stride * lanes;')
    loops=('                for (int ic = 0; ic < c; ic++)\n'
           '                for (int ky = 0; ky < 3; ky++)\n'
           '                for (int kx = 0; kx < 3; kx++)\n')
    assert body.count(loops)==1
    before,step=body.split(loops)
    assert step.startswith('                {\n') and step.endswith('                }\n')
    step=step[len('                {\n'):-len('                }\n')]
    address='                    float* input = x + ((ic / lanes * ph + y * stride + ky) * pw + col * stride + kx) * lanes + ic % lanes;\n'
    step=module.once(step,address,'')
    for pos in range(6):step=module.once(step,f'input[{pos} * stride * lanes]',f'input[{pos} * step]')
    expansion=['                float* tileInput = x + (y * stride * pw + col * stride) * lanes;\n',
        '                for (int ic = 0; ic < c; ic++)\n                {\n',
        '                    float* row0 = tileInput + (ic >> 4) * blockStride + (ic & 15);\n',
        '                    float* row1 = row0 + rowStride;\n',
        '                    float* row2 = row1 + rowStride;\n']
    for ky in range(3):
        for kx in range(3):
            expansion.append(f'                    // Ordered kernel row {ky}, column {kx}.\n                    {{\n')
            expansion.append(f'                        float* input = row{ky} + {kx} * lanes;\n')
            expansion.extend('    '+line for line in step.splitlines(True))
            expansion.append('                    }\n')
    expansion.append('                }\n')
    added=prefix+before+''.join(expansion)+suffix
    assert added[added.index('                *(Vector512<float>*)(output',start):]==suffix
    assert added.count('Avx512F.FusedMultiplyAdd')==original_added.count('Avx512F.FusedMultiplyAdd')+8*24
    old='            if (lanes == 16) Kernel512(x, weights, output, c, m, h, w, stride, oh, ow);'
    new=('            if (lanes == 16 && m % 64 == 0) Kernel512Four(x, weights, output, c, m, h, w, stride, oh, ow);\n'
         '            else if (lanes == 16) Kernel512(x, weights, output, c, m, h, w, stride, oh, ow);')
    candidate=module.once(caller,old,new)
    diff=''.join(difflib.unified_diff(caller.splitlines(True),candidate.splitlines(True),fromfile='a/'+SOURCE,tofile='b/'+SOURCE))
    diff+=''.join(difflib.unified_diff([],added.splitlines(True),fromfile='/dev/null',tofile='b/'+ADDED))
    return candidate,added,diff
