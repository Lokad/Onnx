"""Hoist integer addressing only; preserve the selected two-block arithmetic."""
import difflib
import hashlib

SOURCE = 'src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Kernels.cs'


def once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def transform(original):
    original = original.replace('\r\n', '\n')
    assert hashlib.sha256(original.encode()).hexdigest() == 'b7f2730aca76bdeea61bf6ad168c9b9713e4dce942045725dfa34485bd4187cf'
    start = original.index('    static void Kernel512(')
    prefix, kernel = original[:start], original[start:]
    old = '        int ph = h + 2, pw = w + 2, spatial = oh * ow, fusedEnd = spatial / 8 * 8;'
    kernel = once(kernel, old, old+'\n        int rowStride = pw * lanes, blockStride = ph * rowStride, step = stride * lanes;')
    loops = '''                for (int ic = 0; ic < c; ic++)
                for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++)
                {
'''
    assert kernel.count(loops) == 2
    for tail in [False, True]:
        begin = kernel.index(loops)
        end = kernel.index('                }\n                *(Vector512<float>*)', begin)+len('                }\n')
        body = kernel[begin+len(loops):end-len('                }\n')]
        if not tail:
            body = once(body, '                    float* input = x + ((ic / lanes * ph + y * stride + ky) * pw + col * stride + kx) * lanes + ic % lanes;\n', '')
            for position in range(6):
                body = once(body, f'input[{position} * stride * lanes]', f'input[{position} * step]')
        else:
            body = once(body, 'float value = x[((ic / lanes * ph + y * stride + ky) * pw + col * stride + kx) * lanes + ic % lanes];', 'float value = *source;')
        cursor = 'source' if tail else 'input'
        replacement = '''                float* tileInput = x + (y * stride * pw + col * stride) * lanes;
                for (int ic = 0; ic < c; ic++)
                {
                    float* rowInput = tileInput + (ic >> 4) * blockStride + (ic & 15);
                    for (int ky = 0; ky < 3; ky++, rowInput += rowStride)
                    {
'''+f'''                        float* {cursor} = rowInput;
                        for (int kx = 0; kx < 3; kx++, {cursor} += lanes)
                        {{
'''+''.join('        '+line+'\n' for line in body.splitlines())+'''                        }
                    }
                }
'''
        kernel = kernel[:begin]+replacement+kernel[end:]
    assert 'ic / lanes' not in kernel and 'ic % lanes' not in kernel
    candidate = prefix+kernel
    diff = ''.join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True), fromfile='a/'+SOURCE, tofile='b/'+SOURCE))
    return candidate, diff
