using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

public partial class MathOps
{
    // Accumulate six rows over two adjacent 32-column packed panels. Each
    // output keeps the existing ascending reduction/FMA order; large-row
    // dispatch is bounded separately from the unchanged eight-row tail.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe bool TryPackedAvx512SixRows(int m, int n, int k, float* x, float* packed, float* dest)
    {
        if (!Avx512F.IsSupported || !Fma.IsSupported || m < 256 || m % 8 != 0 ||
            n < 512 || k < 512 || k % 64 != 0) return false;
        // Groups of 24 leave only zero, eight, or sixteen rows for wide tails.
        int bulk = m / 24 * 24;
        for (int column = 0; column < k; column += 64)
        {
            float* panel0 = packed + column * n;
            float* panel1 = panel0 + 32 * n;
            for (int row = 0; row < bulk; row += 6)
            {
                float* cGroup = dest + row * k + column;
                float* a0 = x + (row + 0) * n;
                float* a1 = x + (row + 1) * n;
                float* a2 = x + (row + 2) * n;
                float* a3 = x + (row + 3) * n;
                float* a4 = x + (row + 4) * n;
                float* a5 = x + (row + 5) * n;
                Vector512<float> c00 = ((Vector512<float>*)(cGroup + 0 * k))[0];
                Vector512<float> c01 = ((Vector512<float>*)(cGroup + 0 * k))[1];
                Vector512<float> c02 = ((Vector512<float>*)(cGroup + 0 * k))[2];
                Vector512<float> c03 = ((Vector512<float>*)(cGroup + 0 * k))[3];
                Vector512<float> c10 = ((Vector512<float>*)(cGroup + 1 * k))[0];
                Vector512<float> c11 = ((Vector512<float>*)(cGroup + 1 * k))[1];
                Vector512<float> c12 = ((Vector512<float>*)(cGroup + 1 * k))[2];
                Vector512<float> c13 = ((Vector512<float>*)(cGroup + 1 * k))[3];
                Vector512<float> c20 = ((Vector512<float>*)(cGroup + 2 * k))[0];
                Vector512<float> c21 = ((Vector512<float>*)(cGroup + 2 * k))[1];
                Vector512<float> c22 = ((Vector512<float>*)(cGroup + 2 * k))[2];
                Vector512<float> c23 = ((Vector512<float>*)(cGroup + 2 * k))[3];
                Vector512<float> c30 = ((Vector512<float>*)(cGroup + 3 * k))[0];
                Vector512<float> c31 = ((Vector512<float>*)(cGroup + 3 * k))[1];
                Vector512<float> c32 = ((Vector512<float>*)(cGroup + 3 * k))[2];
                Vector512<float> c33 = ((Vector512<float>*)(cGroup + 3 * k))[3];
                Vector512<float> c40 = ((Vector512<float>*)(cGroup + 4 * k))[0];
                Vector512<float> c41 = ((Vector512<float>*)(cGroup + 4 * k))[1];
                Vector512<float> c42 = ((Vector512<float>*)(cGroup + 4 * k))[2];
                Vector512<float> c43 = ((Vector512<float>*)(cGroup + 4 * k))[3];
                Vector512<float> c50 = ((Vector512<float>*)(cGroup + 5 * k))[0];
                Vector512<float> c51 = ((Vector512<float>*)(cGroup + 5 * k))[1];
                Vector512<float> c52 = ((Vector512<float>*)(cGroup + 5 * k))[2];
                Vector512<float> c53 = ((Vector512<float>*)(cGroup + 5 * k))[3];
                for (int reduction = 0; reduction < n; reduction++)
                {
                    var b0 = ((Vector512<float>*)(panel0 + reduction * 32))[0];
                    var b1 = ((Vector512<float>*)(panel0 + reduction * 32))[1];
                    var b2 = ((Vector512<float>*)(panel1 + reduction * 32))[0];
                    var b3 = ((Vector512<float>*)(panel1 + reduction * 32))[1];
                    Vector512<float> a;
                    a = Vector512.Create(a0[reduction]);
                    c00 = Vector512.FusedMultiplyAdd(a, b0, c00);
                    c01 = Vector512.FusedMultiplyAdd(a, b1, c01);
                    c02 = Vector512.FusedMultiplyAdd(a, b2, c02);
                    c03 = Vector512.FusedMultiplyAdd(a, b3, c03);
                    a = Vector512.Create(a1[reduction]);
                    c10 = Vector512.FusedMultiplyAdd(a, b0, c10);
                    c11 = Vector512.FusedMultiplyAdd(a, b1, c11);
                    c12 = Vector512.FusedMultiplyAdd(a, b2, c12);
                    c13 = Vector512.FusedMultiplyAdd(a, b3, c13);
                    a = Vector512.Create(a2[reduction]);
                    c20 = Vector512.FusedMultiplyAdd(a, b0, c20);
                    c21 = Vector512.FusedMultiplyAdd(a, b1, c21);
                    c22 = Vector512.FusedMultiplyAdd(a, b2, c22);
                    c23 = Vector512.FusedMultiplyAdd(a, b3, c23);
                    a = Vector512.Create(a3[reduction]);
                    c30 = Vector512.FusedMultiplyAdd(a, b0, c30);
                    c31 = Vector512.FusedMultiplyAdd(a, b1, c31);
                    c32 = Vector512.FusedMultiplyAdd(a, b2, c32);
                    c33 = Vector512.FusedMultiplyAdd(a, b3, c33);
                    a = Vector512.Create(a4[reduction]);
                    c40 = Vector512.FusedMultiplyAdd(a, b0, c40);
                    c41 = Vector512.FusedMultiplyAdd(a, b1, c41);
                    c42 = Vector512.FusedMultiplyAdd(a, b2, c42);
                    c43 = Vector512.FusedMultiplyAdd(a, b3, c43);
                    a = Vector512.Create(a5[reduction]);
                    c50 = Vector512.FusedMultiplyAdd(a, b0, c50);
                    c51 = Vector512.FusedMultiplyAdd(a, b1, c51);
                    c52 = Vector512.FusedMultiplyAdd(a, b2, c52);
                    c53 = Vector512.FusedMultiplyAdd(a, b3, c53);
                }
                ((Vector512<float>*)(cGroup + 0 * k))[0] = c00;
                ((Vector512<float>*)(cGroup + 0 * k))[1] = c01;
                ((Vector512<float>*)(cGroup + 0 * k))[2] = c02;
                ((Vector512<float>*)(cGroup + 0 * k))[3] = c03;
                ((Vector512<float>*)(cGroup + 1 * k))[0] = c10;
                ((Vector512<float>*)(cGroup + 1 * k))[1] = c11;
                ((Vector512<float>*)(cGroup + 1 * k))[2] = c12;
                ((Vector512<float>*)(cGroup + 1 * k))[3] = c13;
                ((Vector512<float>*)(cGroup + 2 * k))[0] = c20;
                ((Vector512<float>*)(cGroup + 2 * k))[1] = c21;
                ((Vector512<float>*)(cGroup + 2 * k))[2] = c22;
                ((Vector512<float>*)(cGroup + 2 * k))[3] = c23;
                ((Vector512<float>*)(cGroup + 3 * k))[0] = c30;
                ((Vector512<float>*)(cGroup + 3 * k))[1] = c31;
                ((Vector512<float>*)(cGroup + 3 * k))[2] = c32;
                ((Vector512<float>*)(cGroup + 3 * k))[3] = c33;
                ((Vector512<float>*)(cGroup + 4 * k))[0] = c40;
                ((Vector512<float>*)(cGroup + 4 * k))[1] = c41;
                ((Vector512<float>*)(cGroup + 4 * k))[2] = c42;
                ((Vector512<float>*)(cGroup + 4 * k))[3] = c43;
                ((Vector512<float>*)(cGroup + 5 * k))[0] = c50;
                ((Vector512<float>*)(cGroup + 5 * k))[1] = c51;
                ((Vector512<float>*)(cGroup + 5 * k))[2] = c52;
                ((Vector512<float>*)(cGroup + 5 * k))[3] = c53;
            }
        }
        if (bulk < m)
            for (int column = 0; column < k; column += 32)
                PackedTile8(m - bulk, n, x + bulk * n, packed + column * n, dest + bulk * k, k, column);
        return true;
    }
}
