using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// Diagnostic only: overwrite C, even rows, whole 32-column panels, finite data.
static unsafe class Partial
{
    public static void Run(int m, int k, int n, float* a, float* packed, float* c, int block)
    {
        if (!Fma.IsSupported || m < 2 || m % 2 != 0 || k < 1 || n < 32 || n % 32 != 0 || block < 1)
            throw new ArgumentException("Unsupported diagnostic geometry");
        for (int column = 0; column < n; column += 32)
        for (int begin = 0; begin < k; begin += block)
        {
            int end = Math.Min(k, begin + block);
            float* panel = packed + column * k + begin * 32;
            for (int row = 0; row < m; row += 2)
            {
                float* a0 = a + row * k + begin, a1 = a0 + k, bp = panel;
                var cp0 = (Vector256<float>*)(c + row * n + column);
                var cp1 = (Vector256<float>*)(c + (row + 1) * n + column);
                var c00 = Vector256<float>.Zero; var c01 = Vector256<float>.Zero;
                var c02 = Vector256<float>.Zero; var c03 = Vector256<float>.Zero;
                var c10 = Vector256<float>.Zero; var c11 = Vector256<float>.Zero;
                var c12 = Vector256<float>.Zero; var c13 = Vector256<float>.Zero;
                for (int j = begin; j < end; j++)
                {
                    var b = (Vector256<float>*)bp;
                    var av0 = Vector256.Create(*a0++); var av1 = Vector256.Create(*a1++);
                    var bv0 = b[0]; var bv1 = b[1]; var bv2 = b[2]; var bv3 = b[3];
                    c00 = Fma.MultiplyAdd(bv0, av0, c00); c10 = Fma.MultiplyAdd(bv0, av1, c10);
                    c01 = Fma.MultiplyAdd(bv1, av0, c01); c11 = Fma.MultiplyAdd(bv1, av1, c11);
                    c02 = Fma.MultiplyAdd(bv2, av0, c02); c12 = Fma.MultiplyAdd(bv2, av1, c12);
                    c03 = Fma.MultiplyAdd(bv3, av0, c03); c13 = Fma.MultiplyAdd(bv3, av1, c13);
                    bp += 32;
                }
                if (begin == 0)
                {
                    cp0[0] = c00; cp0[1] = c01; cp0[2] = c02; cp0[3] = c03;
                    cp1[0] = c10; cp1[1] = c11; cp1[2] = c12; cp1[3] = c13;
                }
                else
                {
                    cp0[0] = Avx.Add(cp0[0], c00); cp0[1] = Avx.Add(cp0[1], c01);
                    cp0[2] = Avx.Add(cp0[2], c02); cp0[3] = Avx.Add(cp0[3], c03);
                    cp1[0] = Avx.Add(cp1[0], c10); cp1[1] = Avx.Add(cp1[1], c11);
                    cp1[2] = Avx.Add(cp1[2], c12); cp1[3] = Avx.Add(cp1[3], c13);
                }
            }
        }
    }
}
