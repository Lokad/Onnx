"""Emit the same twelve-accumulator spatial loop for 256/512-bit instructions."""
from pathlib import Path


def generate(path):
    assert not path.exists()
    lines=['using System.Runtime.Intrinsics;','using System.Runtime.Intrinsics.X86;',
           'internal static unsafe partial class BlockedSpatial','{']
    for bits,lanes,fma in [(256,8,'Fma.MultiplyAdd'),(512,16,'Avx512F.FusedMultiplyAdd')]:
        v=f'Vector{bits}'
        lines += [f'    static void Kernel{bits}(float* x, float* weights, float* output, int c, int m, int h, int w, int stride, int oh, int ow)',
            '    {',f'        const int lanes = {lanes};',
            '        int ph = h + 2, pw = w + 2, spatial = oh * ow, fusedEnd = spatial / 8 * 8;',
            '        for (int oc = 0; oc < m; oc += 2 * lanes)',
            '        for (int y = 0; y < oh; y++)',
            '        {', '            int col = 0;',
            '            for (; col + 6 <= ow && y * ow + col + 6 <= fusedEnd; col += 6)',
            '            {']
        for p in range(6):
            for b in range(2):lines += [f'                {v}<float> a{p}{b} = {v}<float>.Zero;']
        lines += ['                float* w0 = weights + oc * c * 9;',
                  '                float* w1 = w0 + lanes * c * 9;',
                  '                for (int ic = 0; ic < c; ic++)',
                  '                for (int ky = 0; ky < 3; ky++)',
                  '                for (int kx = 0; kx < 3; kx++)',
                  '                {',
                  '                    float* input = x + ((ic / lanes * ph + y * stride + ky) * pw + col * stride + kx) * lanes + ic % lanes;',
                  f'                    var wv0 = *({v}<float>*)w0; var wv1 = *({v}<float>*)w1;']
        for p in range(6):
            lines += [f'                    var i{p} = {v}.Create(input[{p} * stride * lanes]);']
            for b in range(2):lines += [f'                    a{p}{b} = {fma}(i{p}, wv{b}, a{p}{b});']
        lines += ['                    w0 += lanes; w1 += lanes;', '                }']
        for p in range(6):
            lines += [f'                *({v}<float>*)(output + (oc / lanes * spatial + y * ow + col + {p}) * lanes) = a{p}0;',
                      f'                if (oc + lanes < m) *({v}<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col + {p}) * lanes) = a{p}1;']
        lines += ['            }', '            for (; col < ow; col++)', '            {',
                  f'                var a0 = {v}<float>.Zero; var a1 = {v}<float>.Zero;',
                  '                float* w0 = weights + oc * c * 9; float* w1 = w0 + lanes * c * 9;',
                  '                bool fused = y * ow + col < fusedEnd;',
                  '                for (int ic = 0; ic < c; ic++)',
                  '                for (int ky = 0; ky < 3; ky++)',
                  '                for (int kx = 0; kx < 3; kx++)',
                  '                {',
                  '                    float value = x[((ic / lanes * ph + y * stride + ky) * pw + col * stride + kx) * lanes + ic % lanes];',
                  f'                    var input = {v}.Create(value);',
                  f'                    var wv0 = *({v}<float>*)w0; var wv1 = *({v}<float>*)w1;',
                  '                    if (fused)', '                    {',
                  f'                        a0 = {fma}(input, wv0, a0); a1 = {fma}(input, wv1, a1);',
                  '                    }', '                    else', '                    {',
                  '                        a0 = a0 + input * wv0; a1 = a1 + input * wv1;',
                  '                    }', '                    w0 += lanes; w1 += lanes;', '                }',
                  f'                *({v}<float>*)(output + (oc / lanes * spatial + y * ow + col) * lanes) = a0;',
                  f'                if (oc + lanes < m) *({v}<float>*)(output + ((oc / lanes + 1) * spatial + y * ow + col) * lanes) = a1;',
                  '            }', '        }', '    }']
    lines += ['}']
    path.write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    import sys
    assert len(sys.argv)==2
    generate(Path(sys.argv[1]))
