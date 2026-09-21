using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

static unsafe class Blocked
{
    public static bool TryRun(int m, int k, int n, float* a, float* packed, float* c, int block)
    {
        if (!Fma.IsSupported || m < 2 || (m % 2 != 0 && m % 3 != 0) || k < 1 || n < 32 || n % 32 != 0 || block < 1) return false;
        if (m % 3 == 0) Rows3(m,k,n,a,packed,c,block);
        else Rows2(m,k,n,a,packed,c,block);
        return true;
    }
    static void Rows2(int m, int k, int n, float* a, float* packed, float* c, int block)
    {
        for (int column = 0; column < n; column += 32)
        for (int begin = 0; begin < k; begin += block)
        {
            int end = Math.Min(k, begin + block);
            float* panel = packed + column*k + begin*32;
            for (int row = 0; row < m; row += 2)
            {
                float* a0 = a + (row+0)*k + begin;
                var cp0 = (Vector256<float>*)(c + (row+0)*n + column);
                var c00 = cp0[0];
                var c01 = cp0[1];
                var c02 = cp0[2];
                var c03 = cp0[3];
                float* a1 = a + (row+1)*k + begin;
                var cp1 = (Vector256<float>*)(c + (row+1)*n + column);
                var c10 = cp1[0];
                var c11 = cp1[1];
                var c12 = cp1[2];
                var c13 = cp1[3];
                float* bp = panel;
                for (int j = begin; j < end; j++)
                {
                    var b = (Vector256<float>*)bp;
                    var av0 = Vector256.Create(*a0++);
                    var av1 = Vector256.Create(*a1++);
                    var bv0 = b[0];
                    c00 = Fma.MultiplyAdd(bv0, av0, c00);
                    c10 = Fma.MultiplyAdd(bv0, av1, c10);
                    var bv1 = b[1];
                    c01 = Fma.MultiplyAdd(bv1, av0, c01);
                    c11 = Fma.MultiplyAdd(bv1, av1, c11);
                    var bv2 = b[2];
                    c02 = Fma.MultiplyAdd(bv2, av0, c02);
                    c12 = Fma.MultiplyAdd(bv2, av1, c12);
                    var bv3 = b[3];
                    c03 = Fma.MultiplyAdd(bv3, av0, c03);
                    c13 = Fma.MultiplyAdd(bv3, av1, c13);
                    bp += 32;
                }
                cp0[0] = c00;
                cp0[1] = c01;
                cp0[2] = c02;
                cp0[3] = c03;
                cp1[0] = c10;
                cp1[1] = c11;
                cp1[2] = c12;
                cp1[3] = c13;
            }
        }
    }
    static void Rows3(int m, int k, int n, float* a, float* packed, float* c, int block)
    {
        for (int column = 0; column < n; column += 32)
        for (int begin = 0; begin < k; begin += block)
        {
            int end = Math.Min(k, begin + block);
            float* panel = packed + column*k + begin*32;
            for (int row = 0; row < m; row += 3)
            {
                float* a0 = a + (row+0)*k + begin;
                var cp0 = (Vector256<float>*)(c + (row+0)*n + column);
                var c00 = cp0[0];
                var c01 = cp0[1];
                var c02 = cp0[2];
                var c03 = cp0[3];
                float* a1 = a + (row+1)*k + begin;
                var cp1 = (Vector256<float>*)(c + (row+1)*n + column);
                var c10 = cp1[0];
                var c11 = cp1[1];
                var c12 = cp1[2];
                var c13 = cp1[3];
                float* a2 = a + (row+2)*k + begin;
                var cp2 = (Vector256<float>*)(c + (row+2)*n + column);
                var c20 = cp2[0];
                var c21 = cp2[1];
                var c22 = cp2[2];
                var c23 = cp2[3];
                float* bp = panel;
                for (int j = begin; j < end; j++)
                {
                    var b = (Vector256<float>*)bp;
                    var av0 = Vector256.Create(*a0++);
                    var av1 = Vector256.Create(*a1++);
                    var av2 = Vector256.Create(*a2++);
                    var bv0 = b[0];
                    c00 = Fma.MultiplyAdd(bv0, av0, c00);
                    c10 = Fma.MultiplyAdd(bv0, av1, c10);
                    c20 = Fma.MultiplyAdd(bv0, av2, c20);
                    var bv1 = b[1];
                    c01 = Fma.MultiplyAdd(bv1, av0, c01);
                    c11 = Fma.MultiplyAdd(bv1, av1, c11);
                    c21 = Fma.MultiplyAdd(bv1, av2, c21);
                    var bv2 = b[2];
                    c02 = Fma.MultiplyAdd(bv2, av0, c02);
                    c12 = Fma.MultiplyAdd(bv2, av1, c12);
                    c22 = Fma.MultiplyAdd(bv2, av2, c22);
                    var bv3 = b[3];
                    c03 = Fma.MultiplyAdd(bv3, av0, c03);
                    c13 = Fma.MultiplyAdd(bv3, av1, c13);
                    c23 = Fma.MultiplyAdd(bv3, av2, c23);
                    bp += 32;
                }
                cp0[0] = c00;
                cp0[1] = c01;
                cp0[2] = c02;
                cp0[3] = c03;
                cp1[0] = c10;
                cp1[1] = c11;
                cp1[2] = c12;
                cp1[3] = c13;
                cp2[0] = c20;
                cp2[1] = c21;
                cp2[2] = c22;
                cp2[3] = c23;
            }
        }
    }
}
