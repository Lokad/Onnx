using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

public partial class MathOps
{
    interface IFixedReduction { static abstract int Width { get; } }
    readonly struct Reduction1024 : IFixedReduction { public static int Width => 1024; }
    readonly struct Reduction2048 : IFixedReduction { public static int Width => 2048; }

    internal static unsafe bool TryPackedAvx512FixedWideRows(int m, int n, int k, float* x, float* packed, float* dest)
    {
        if (!Avx512F.IsSupported || !Fma.IsSupported || m != 128 || k != 512) return false;
        if (n == 1024) PackedFixedWideRows<Reduction1024>(x, packed, dest);
        else if (n == 2048) PackedFixedWideRows<Reduction2048>(x, packed, dest);
        else return false;
        return true;
    }

    // Struct specialization turns the reduction width and row displacements
    // into constants, while keeping one arithmetic implementation.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static unsafe void PackedFixedWideRows<TReduction>(float* x, float* packed, float* dest)
        where TReduction : struct, IFixedReduction
    {
        int reduction = TReduction.Width;
        for (int col = 0; col < 512; col += 32)
        {
            float* panel = packed + col * reduction;
            for (int row = 0; row < 120; row += 12)
            {
                float* cg = dest + row * 512 + col;
                float* ar = x + row * reduction;
                Vector512<float> c0_0 = ((Vector512<float>*)(cg + 0 * 512))[0];
                Vector512<float> c0_1 = ((Vector512<float>*)(cg + 0 * 512))[1];
                Vector512<float> c1_0 = ((Vector512<float>*)(cg + 1 * 512))[0];
                Vector512<float> c1_1 = ((Vector512<float>*)(cg + 1 * 512))[1];
                Vector512<float> c2_0 = ((Vector512<float>*)(cg + 2 * 512))[0];
                Vector512<float> c2_1 = ((Vector512<float>*)(cg + 2 * 512))[1];
                Vector512<float> c3_0 = ((Vector512<float>*)(cg + 3 * 512))[0];
                Vector512<float> c3_1 = ((Vector512<float>*)(cg + 3 * 512))[1];
                Vector512<float> c4_0 = ((Vector512<float>*)(cg + 4 * 512))[0];
                Vector512<float> c4_1 = ((Vector512<float>*)(cg + 4 * 512))[1];
                Vector512<float> c5_0 = ((Vector512<float>*)(cg + 5 * 512))[0];
                Vector512<float> c5_1 = ((Vector512<float>*)(cg + 5 * 512))[1];
                Vector512<float> c6_0 = ((Vector512<float>*)(cg + 6 * 512))[0];
                Vector512<float> c6_1 = ((Vector512<float>*)(cg + 6 * 512))[1];
                Vector512<float> c7_0 = ((Vector512<float>*)(cg + 7 * 512))[0];
                Vector512<float> c7_1 = ((Vector512<float>*)(cg + 7 * 512))[1];
                Vector512<float> c8_0 = ((Vector512<float>*)(cg + 8 * 512))[0];
                Vector512<float> c8_1 = ((Vector512<float>*)(cg + 8 * 512))[1];
                Vector512<float> c9_0 = ((Vector512<float>*)(cg + 9 * 512))[0];
                Vector512<float> c9_1 = ((Vector512<float>*)(cg + 9 * 512))[1];
                Vector512<float> c10_0 = ((Vector512<float>*)(cg + 10 * 512))[0];
                Vector512<float> c10_1 = ((Vector512<float>*)(cg + 10 * 512))[1];
                Vector512<float> c11_0 = ((Vector512<float>*)(cg + 11 * 512))[0];
                Vector512<float> c11_1 = ((Vector512<float>*)(cg + 11 * 512))[1];
                float* bp = panel;
                for (int q = 0; q < reduction; q++)
                {
                    var b0 = ((Vector512<float>*)bp)[0]; var b1 = ((Vector512<float>*)bp)[1];
                    Vector512<float> a;
                    a = Vector512.Create(ar[0 * reduction]);
                    c0_0 = Vector512.FusedMultiplyAdd(a, b0, c0_0);
                    c0_1 = Vector512.FusedMultiplyAdd(a, b1, c0_1);
                    a = Vector512.Create(ar[1 * reduction]);
                    c1_0 = Vector512.FusedMultiplyAdd(a, b0, c1_0);
                    c1_1 = Vector512.FusedMultiplyAdd(a, b1, c1_1);
                    a = Vector512.Create(ar[2 * reduction]);
                    c2_0 = Vector512.FusedMultiplyAdd(a, b0, c2_0);
                    c2_1 = Vector512.FusedMultiplyAdd(a, b1, c2_1);
                    a = Vector512.Create(ar[3 * reduction]);
                    c3_0 = Vector512.FusedMultiplyAdd(a, b0, c3_0);
                    c3_1 = Vector512.FusedMultiplyAdd(a, b1, c3_1);
                    a = Vector512.Create(ar[4 * reduction]);
                    c4_0 = Vector512.FusedMultiplyAdd(a, b0, c4_0);
                    c4_1 = Vector512.FusedMultiplyAdd(a, b1, c4_1);
                    a = Vector512.Create(ar[5 * reduction]);
                    c5_0 = Vector512.FusedMultiplyAdd(a, b0, c5_0);
                    c5_1 = Vector512.FusedMultiplyAdd(a, b1, c5_1);
                    a = Vector512.Create(ar[6 * reduction]);
                    c6_0 = Vector512.FusedMultiplyAdd(a, b0, c6_0);
                    c6_1 = Vector512.FusedMultiplyAdd(a, b1, c6_1);
                    a = Vector512.Create(ar[7 * reduction]);
                    c7_0 = Vector512.FusedMultiplyAdd(a, b0, c7_0);
                    c7_1 = Vector512.FusedMultiplyAdd(a, b1, c7_1);
                    a = Vector512.Create(ar[8 * reduction]);
                    c8_0 = Vector512.FusedMultiplyAdd(a, b0, c8_0);
                    c8_1 = Vector512.FusedMultiplyAdd(a, b1, c8_1);
                    a = Vector512.Create(ar[9 * reduction]);
                    c9_0 = Vector512.FusedMultiplyAdd(a, b0, c9_0);
                    c9_1 = Vector512.FusedMultiplyAdd(a, b1, c9_1);
                    a = Vector512.Create(ar[10 * reduction]);
                    c10_0 = Vector512.FusedMultiplyAdd(a, b0, c10_0);
                    c10_1 = Vector512.FusedMultiplyAdd(a, b1, c10_1);
                    a = Vector512.Create(ar[11 * reduction]);
                    c11_0 = Vector512.FusedMultiplyAdd(a, b0, c11_0);
                    c11_1 = Vector512.FusedMultiplyAdd(a, b1, c11_1);
                    ar++; bp += 32;
                }
                ((Vector512<float>*)(cg + 0 * 512))[0] = c0_0;
                ((Vector512<float>*)(cg + 0 * 512))[1] = c0_1;
                ((Vector512<float>*)(cg + 1 * 512))[0] = c1_0;
                ((Vector512<float>*)(cg + 1 * 512))[1] = c1_1;
                ((Vector512<float>*)(cg + 2 * 512))[0] = c2_0;
                ((Vector512<float>*)(cg + 2 * 512))[1] = c2_1;
                ((Vector512<float>*)(cg + 3 * 512))[0] = c3_0;
                ((Vector512<float>*)(cg + 3 * 512))[1] = c3_1;
                ((Vector512<float>*)(cg + 4 * 512))[0] = c4_0;
                ((Vector512<float>*)(cg + 4 * 512))[1] = c4_1;
                ((Vector512<float>*)(cg + 5 * 512))[0] = c5_0;
                ((Vector512<float>*)(cg + 5 * 512))[1] = c5_1;
                ((Vector512<float>*)(cg + 6 * 512))[0] = c6_0;
                ((Vector512<float>*)(cg + 6 * 512))[1] = c6_1;
                ((Vector512<float>*)(cg + 7 * 512))[0] = c7_0;
                ((Vector512<float>*)(cg + 7 * 512))[1] = c7_1;
                ((Vector512<float>*)(cg + 8 * 512))[0] = c8_0;
                ((Vector512<float>*)(cg + 8 * 512))[1] = c8_1;
                ((Vector512<float>*)(cg + 9 * 512))[0] = c9_0;
                ((Vector512<float>*)(cg + 9 * 512))[1] = c9_1;
                ((Vector512<float>*)(cg + 10 * 512))[0] = c10_0;
                ((Vector512<float>*)(cg + 10 * 512))[1] = c10_1;
                ((Vector512<float>*)(cg + 11 * 512))[0] = c11_0;
                ((Vector512<float>*)(cg + 11 * 512))[1] = c11_1;
            }
            PackedTile8(8, reduction, x + 120 * reduction, panel, dest + 120 * 512, 512, col);
        }
    }
}
