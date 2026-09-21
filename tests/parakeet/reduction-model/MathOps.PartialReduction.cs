namespace Lokad.Onnx;

using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

public partial class MathOps
{
    // Isolated candidate: preserve C += A*B, with 256-term partial sums.
    // Non-admitted geometry remains entirely on the original kernels.
    static unsafe bool TryPackedPartialSums(int m, int reduction, int columns, float* a, float* packed, float* c)
    {
        if (!Fma.IsSupported || m < 1 || reduction < 1024 || columns < 32 || columns % 32 != 0)
            return false;
        for (int column = 0; column < columns; column += 32)
        for (int begin = 0; begin < reduction; begin += 256)
        {
            int end = Math.Min(reduction, begin + 256);
            float* panel = packed + column * reduction + begin * 32;
            for (int row = 0; row < m; row += 2)
            {
                bool second = row + 1 < m;
                float* a0 = a + row * reduction + begin;
                float* a1 = a0 + reduction;
                float* bp = panel;
                var cp0 = (Vector256<float>*)(c + row * columns + column);
                var cp1 = (Vector256<float>*)(c + (row + 1) * columns + column);
                var c00 = Vector256<float>.Zero; var c01 = Vector256<float>.Zero;
                var c02 = Vector256<float>.Zero; var c03 = Vector256<float>.Zero;
                var c10 = Vector256<float>.Zero; var c11 = Vector256<float>.Zero;
                var c12 = Vector256<float>.Zero; var c13 = Vector256<float>.Zero;
                for (int j = begin; j < end; j++)
                {
                    var b = (Vector256<float>*)bp;
                    var av0 = Vector256.Create(*a0++);
                    var av1 = second ? Vector256.Create(*a1++) : Vector256<float>.Zero;
                    var bv0 = b[0]; var bv1 = b[1]; var bv2 = b[2]; var bv3 = b[3];
                    c00 = Fma.MultiplyAdd(bv0, av0, c00); c10 = Fma.MultiplyAdd(bv0, av1, c10);
                    c01 = Fma.MultiplyAdd(bv1, av0, c01); c11 = Fma.MultiplyAdd(bv1, av1, c11);
                    c02 = Fma.MultiplyAdd(bv2, av0, c02); c12 = Fma.MultiplyAdd(bv2, av1, c12);
                    c03 = Fma.MultiplyAdd(bv3, av0, c03); c13 = Fma.MultiplyAdd(bv3, av1, c13);
                    bp += 32;
                }
                cp0[0] = Avx.Add(cp0[0], c00); cp0[1] = Avx.Add(cp0[1], c01);
                cp0[2] = Avx.Add(cp0[2], c02); cp0[3] = Avx.Add(cp0[3], c03);
                if (second)
                {
                    cp1[0] = Avx.Add(cp1[0], c10); cp1[1] = Avx.Add(cp1[1], c11);
                    cp1[2] = Avx.Add(cp1[2], c12); cp1[3] = Avx.Add(cp1[3], c13);
                }
            }
        }
        return true;
    }
}
