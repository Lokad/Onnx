using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

public partial class MathOps
{
    // A fixed 512-wide reduction lets the JIT encode row offsets directly.
    // Limit dispatch to measured 128-row shapes; accumulation/FMA order and
    // the eight-row tail are identical to the existing packed path.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe bool TryPackedAvx512Fixed512Rows(int m, int n, int k, float* x, float* packed, float* dest)
    {
        if (!Avx512F.IsSupported || !Fma.IsSupported || m != 128 || n != 512 || k is not (512 or 2048 or 3072 or 4096 or 12288)) return false;
        for (int col = 0; col < k; col += 32)
        {
            float* panel = packed + col * 512;
            for (int row = 0; row < 120; row += 12)
            {
                float* cg = dest + row * k + col;
                float* ar = x + row * 512;
                Vector512<float> c0_0 = ((Vector512<float>*)(cg + 0 * k))[0];
                Vector512<float> c0_1 = ((Vector512<float>*)(cg + 0 * k))[1];
                Vector512<float> c1_0 = ((Vector512<float>*)(cg + 1 * k))[0];
                Vector512<float> c1_1 = ((Vector512<float>*)(cg + 1 * k))[1];
                Vector512<float> c2_0 = ((Vector512<float>*)(cg + 2 * k))[0];
                Vector512<float> c2_1 = ((Vector512<float>*)(cg + 2 * k))[1];
                Vector512<float> c3_0 = ((Vector512<float>*)(cg + 3 * k))[0];
                Vector512<float> c3_1 = ((Vector512<float>*)(cg + 3 * k))[1];
                Vector512<float> c4_0 = ((Vector512<float>*)(cg + 4 * k))[0];
                Vector512<float> c4_1 = ((Vector512<float>*)(cg + 4 * k))[1];
                Vector512<float> c5_0 = ((Vector512<float>*)(cg + 5 * k))[0];
                Vector512<float> c5_1 = ((Vector512<float>*)(cg + 5 * k))[1];
                Vector512<float> c6_0 = ((Vector512<float>*)(cg + 6 * k))[0];
                Vector512<float> c6_1 = ((Vector512<float>*)(cg + 6 * k))[1];
                Vector512<float> c7_0 = ((Vector512<float>*)(cg + 7 * k))[0];
                Vector512<float> c7_1 = ((Vector512<float>*)(cg + 7 * k))[1];
                Vector512<float> c8_0 = ((Vector512<float>*)(cg + 8 * k))[0];
                Vector512<float> c8_1 = ((Vector512<float>*)(cg + 8 * k))[1];
                Vector512<float> c9_0 = ((Vector512<float>*)(cg + 9 * k))[0];
                Vector512<float> c9_1 = ((Vector512<float>*)(cg + 9 * k))[1];
                Vector512<float> c10_0 = ((Vector512<float>*)(cg + 10 * k))[0];
                Vector512<float> c10_1 = ((Vector512<float>*)(cg + 10 * k))[1];
                Vector512<float> c11_0 = ((Vector512<float>*)(cg + 11 * k))[0];
                Vector512<float> c11_1 = ((Vector512<float>*)(cg + 11 * k))[1];
                float* bp = panel;
                for (int q = 0; q < 512; q++)
                {
                    var b0 = ((Vector512<float>*)bp)[0]; var b1 = ((Vector512<float>*)bp)[1];
                    Vector512<float> a;
                    a = Vector512.Create(ar[0]);
                    c0_0 = Vector512.FusedMultiplyAdd(a, b0, c0_0);
                    c0_1 = Vector512.FusedMultiplyAdd(a, b1, c0_1);
                    a = Vector512.Create(ar[512]);
                    c1_0 = Vector512.FusedMultiplyAdd(a, b0, c1_0);
                    c1_1 = Vector512.FusedMultiplyAdd(a, b1, c1_1);
                    a = Vector512.Create(ar[1024]);
                    c2_0 = Vector512.FusedMultiplyAdd(a, b0, c2_0);
                    c2_1 = Vector512.FusedMultiplyAdd(a, b1, c2_1);
                    a = Vector512.Create(ar[1536]);
                    c3_0 = Vector512.FusedMultiplyAdd(a, b0, c3_0);
                    c3_1 = Vector512.FusedMultiplyAdd(a, b1, c3_1);
                    a = Vector512.Create(ar[2048]);
                    c4_0 = Vector512.FusedMultiplyAdd(a, b0, c4_0);
                    c4_1 = Vector512.FusedMultiplyAdd(a, b1, c4_1);
                    a = Vector512.Create(ar[2560]);
                    c5_0 = Vector512.FusedMultiplyAdd(a, b0, c5_0);
                    c5_1 = Vector512.FusedMultiplyAdd(a, b1, c5_1);
                    a = Vector512.Create(ar[3072]);
                    c6_0 = Vector512.FusedMultiplyAdd(a, b0, c6_0);
                    c6_1 = Vector512.FusedMultiplyAdd(a, b1, c6_1);
                    a = Vector512.Create(ar[3584]);
                    c7_0 = Vector512.FusedMultiplyAdd(a, b0, c7_0);
                    c7_1 = Vector512.FusedMultiplyAdd(a, b1, c7_1);
                    a = Vector512.Create(ar[4096]);
                    c8_0 = Vector512.FusedMultiplyAdd(a, b0, c8_0);
                    c8_1 = Vector512.FusedMultiplyAdd(a, b1, c8_1);
                    a = Vector512.Create(ar[4608]);
                    c9_0 = Vector512.FusedMultiplyAdd(a, b0, c9_0);
                    c9_1 = Vector512.FusedMultiplyAdd(a, b1, c9_1);
                    a = Vector512.Create(ar[5120]);
                    c10_0 = Vector512.FusedMultiplyAdd(a, b0, c10_0);
                    c10_1 = Vector512.FusedMultiplyAdd(a, b1, c10_1);
                    a = Vector512.Create(ar[5632]);
                    c11_0 = Vector512.FusedMultiplyAdd(a, b0, c11_0);
                    c11_1 = Vector512.FusedMultiplyAdd(a, b1, c11_1);
                    ar++; bp += 32;
                }
                ((Vector512<float>*)(cg + 0 * k))[0] = c0_0;
                ((Vector512<float>*)(cg + 0 * k))[1] = c0_1;
                ((Vector512<float>*)(cg + 1 * k))[0] = c1_0;
                ((Vector512<float>*)(cg + 1 * k))[1] = c1_1;
                ((Vector512<float>*)(cg + 2 * k))[0] = c2_0;
                ((Vector512<float>*)(cg + 2 * k))[1] = c2_1;
                ((Vector512<float>*)(cg + 3 * k))[0] = c3_0;
                ((Vector512<float>*)(cg + 3 * k))[1] = c3_1;
                ((Vector512<float>*)(cg + 4 * k))[0] = c4_0;
                ((Vector512<float>*)(cg + 4 * k))[1] = c4_1;
                ((Vector512<float>*)(cg + 5 * k))[0] = c5_0;
                ((Vector512<float>*)(cg + 5 * k))[1] = c5_1;
                ((Vector512<float>*)(cg + 6 * k))[0] = c6_0;
                ((Vector512<float>*)(cg + 6 * k))[1] = c6_1;
                ((Vector512<float>*)(cg + 7 * k))[0] = c7_0;
                ((Vector512<float>*)(cg + 7 * k))[1] = c7_1;
                ((Vector512<float>*)(cg + 8 * k))[0] = c8_0;
                ((Vector512<float>*)(cg + 8 * k))[1] = c8_1;
                ((Vector512<float>*)(cg + 9 * k))[0] = c9_0;
                ((Vector512<float>*)(cg + 9 * k))[1] = c9_1;
                ((Vector512<float>*)(cg + 10 * k))[0] = c10_0;
                ((Vector512<float>*)(cg + 10 * k))[1] = c10_1;
                ((Vector512<float>*)(cg + 11 * k))[0] = c11_0;
                ((Vector512<float>*)(cg + 11 * k))[1] = c11_1;
            }
            PackedTile8(8, 512, x + 120 * 512, panel, dest + 120 * k, k, col);
        }
        return true;
    }
}
