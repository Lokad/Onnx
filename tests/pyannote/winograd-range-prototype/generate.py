"""Generate two explicit SIMD widths of the same eight-tile arithmetic."""
from pathlib import Path


def generate():
    lines=['using System.Runtime.Intrinsics;', 'using System.Runtime.Intrinsics.X86;',
           'namespace Lokad.Onnx;', 'internal static unsafe partial class ConvBlockedSpatial', '{']
    for width,lanes,isa,fma in [(256,8,'Avx','Fma.MultiplyAdd'),(512,16,'Avx512F','Avx512F.FusedMultiplyAdd')]:
        v=f'Vector{width}<float>'
        lines += [f'    static void MultiplyWinograd{width}(float* v, float* u, float* p, int c, int m)', '    {',
            f'        const int lanes = {lanes};', '        for (int k = 0; k < 16; k++)',
            '        for (int oc = 0; oc < m; oc += lanes)', '        {']
        lines += [f'            var a{t} = {v}.Zero;' for t in range(8)]
        lines += ['            float* weights = u + k * c * m + oc;',
            '            float* inputs = v + k * c * WinogradBatch;',
            '            for (int ic = 0; ic < c; ic++)', '            {',
            f'                var weight = *({v}*)weights;']
        lines += [f'                a{t} = {fma}(Vector{width}.Create(inputs[{t}]), weight, a{t});' for t in range(8)]
        lines += ['                weights += m; inputs += WinogradBatch;', '            }',
            '            float* output = p + (k * m + oc) * WinogradBatch;']
        lines += [f'            *({v}*)(output + {t} * lanes) = a{t};' for t in range(8)]
        lines += ['        }', '    }',
            f'    static void OutputWinograd{width}(float* p, float* output, int m, int h, int w, int tileWidth, int first, int count)',
            '    {',f'        const int lanes = {lanes};', '        int spatial = h * w;',
            '        for (int oc = 0; oc < m; oc += lanes)',
            '        for (int tile = 0; tile < count; tile++)', '        {',
            '            float* src = p + oc * WinogradBatch + tile * lanes;',
            '            int step = m * WinogradBatch;',
            '            int y = (first + tile) / tileWidth * 2, x = (first + tile) % tileWidth * 2;']
        for col in range(4):
            get=lambda r:f'*({v}*)(src + {r*4+col} * step)'
            lines += [f'            var a{col} = {isa}.Add({isa}.Add({get(0)}, {get(1)}), {get(2)});',
                      f'            var b{col} = {isa}.Subtract({isa}.Subtract({get(1)}, {get(2)}), {get(3)});']
        lines += [f'            var r00 = {isa}.Add({isa}.Add(a0, a1), a2);',
                  f'            var r01 = {isa}.Subtract({isa}.Subtract(a1, a2), a3);',
                  f'            var r10 = {isa}.Add({isa}.Add(b0, b1), b2);',
                  f'            var r11 = {isa}.Subtract({isa}.Subtract(b1, b2), b3);',
                  '            float* dst = output + oc * spatial + (y * w + x) * lanes;',
                  f'            *({v}*)dst = r00;',
                  f'            if (x + 1 < w) *({v}*)(dst + lanes) = r01;',
                  '            if (y + 1 < h)', '            {',
                  f'                *({v}*)(dst + w * lanes) = r10;',
                  f'                if (x + 1 < w) *({v}*)(dst + (w + 1) * lanes) = r11;',
                  '            }', '        }', '    }']
    lines += ['}', '']
    return '\n'.join(lines)


if __name__=='__main__':
    path=Path(__file__).with_name('Winograd.Kernels.cs')
    assert not path.exists()
    path.write_text(generate(),encoding='utf8',newline='\n')
