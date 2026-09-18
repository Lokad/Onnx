using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

// Panel traversal adapted from the voice branch at 2b1138f. The narrow leaves
// below copy master's existing accumulation loops, including operand order.
// No overwrite, packing layout, row partition or column-tail changes.
public partial class MathOps
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe bool TryPackedAvx512Panels(int m, int n, int k, float* x, float* packed, float* dest)
    {
        if (!Avx512F.IsSupported || !Fma.IsSupported || m < 8 || n <= 0 || k <= 0 || k % 32 != 0)
            return false;
        // Restrict the first experiment to weights larger than the target's L2.
        if ((long)n * k <= 1024 * 1024 / sizeof(float)) return false;
        int main = m / 12 * 12;
        int rest = m - main;
        if (rest == 1) { main -= 12; rest = 13; }
        if (rest == 4 && main >= 12) { main -= 12; rest = 16; }
        else if (rest == 2 && main >= 12) { main -= 12; rest = 14; }
        int eights = 0;
        while (rest >= 8 && rest != 9) { eights += 8; rest -= 8; }
        int threes = 0, twos = 0;
        if (rest % 3 == 0) threes = rest;
        else if (rest % 2 == 0) twos = rest;
        else { threes = 3; twos = rest - 3; }
        int sweeps = (main > 0 ? 1 : 0) + (eights > 0 ? 1 : 0) + (threes > 0 ? 1 : 0) + (twos > 0 ? 1 : 0);
        if (sweeps < 2) return false;
        for (int column = 0; column < k; column += 32)
        {
            float* panel = packed + column * n;
            if (main > 0) PackedTile12(main, n, x, panel, dest, k, column);
            int row = main;
            if (eights > 0) PackedTile8(eights, n, x + row * n, panel, dest + row * k, k, column);
            row += eights;
            if (threes > 0) PackedPanel3(threes, n, x + row * n, panel, dest + row * k, k, column);
            row += threes;
            if (twos > 0) PackedPanel2(twos, n, x + row * n, panel, dest + row * k, k, column);
        }
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe void PackedPanel2(int M, int N, float* A, float* panel, float* C, int K, int kb)
    {
        int panelWidth = 4 * Vector256<float>.Count;
        for (int i = 0; i < M; i += 2)
        {
            float* a0 = A + i * N;
            float* a1 = a0 + N;
            float* b = panel;
            var Cpv1 = (Vector256<float>*)(C + i * K + kb);
            var Cpv2 = (Vector256<float>*)(C + (i + 1) * K + kb);
            Vector256<float> c00 = Cpv1[0];
            Vector256<float> c01 = Cpv1[1];
            Vector256<float> c02 = Cpv1[2];
            Vector256<float> c03 = Cpv1[3];
            Vector256<float> c10 = Cpv2[0];
            Vector256<float> c11 = Cpv2[1];
            Vector256<float> c12 = Cpv2[2];
            Vector256<float> c13 = Cpv2[3];
            for (int j = 0; j < N; ++j)
            {
                var Bpv = (Vector256<float>*)b;
                Vector256<float> bv0 = Bpv[0];
                Vector256<float> bv1 = Bpv[1];
                Vector256<float> bv2 = Bpv[2];
                Vector256<float> bv3 = Bpv[3];
                var av1 = Vector256.Create(*a0); a0 += 1;
                var av2 = Vector256.Create(*a1); a1 += 1;
                c00 = Fma.MultiplyAdd(bv0, av1, c00);
                c10 = Fma.MultiplyAdd(bv0, av2, c10);
                c01 = Fma.MultiplyAdd(bv1, av1, c01);
                c11 = Fma.MultiplyAdd(bv1, av2, c11);
                c02 = Fma.MultiplyAdd(bv2, av1, c02);
                c12 = Fma.MultiplyAdd(bv2, av2, c12);
                c03 = Fma.MultiplyAdd(bv3, av1, c03);
                c13 = Fma.MultiplyAdd(bv3, av2, c13);
                b += panelWidth;
            }
            Cpv1[0] = c00;
            Cpv1[1] = c01;
            Cpv1[2] = c02;
            Cpv1[3] = c03;
            Cpv2[0] = c10;
            Cpv2[1] = c11;
            Cpv2[2] = c12;
            Cpv2[3] = c13;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe void PackedPanel3(int M, int N, float* A, float* panel, float* C, int K, int kb)
    {
        for (int i = 0; i < M; i += 3)
        {
            var Ap1 = A + i * N;
            var Ap2 = Ap1 + N;
            var Ap3 = Ap2 + N;
            var Cpv1 = (Vector256<float>*)(C + i * K + kb);
            var Cpv2 = (Vector256<float>*)(C + (i + 1) * K + kb);
            var Cpv3 = (Vector256<float>*)(C + (i + 2) * K + kb);
            Vector256<float> c00 = Cpv1[0];
            Vector256<float> c01 = Cpv1[1];
            Vector256<float> c02 = Cpv1[2];
            Vector256<float> c03 = Cpv1[3];
            Vector256<float> c10 = Cpv2[0];
            Vector256<float> c11 = Cpv2[1];
            Vector256<float> c12 = Cpv2[2];
            Vector256<float> c13 = Cpv2[3];
            Vector256<float> c20 = Cpv3[0];
            Vector256<float> c21 = Cpv3[1];
            Vector256<float> c22 = Cpv3[2];
            Vector256<float> c23 = Cpv3[3];
            for (int j = 0; j < N; ++j)
            {
                var av1 = Vector256.Create(Ap1[j]);
                var av2 = Vector256.Create(Ap2[j]);
                var av3 = Vector256.Create(Ap3[j]);
                var Bpv = (Vector256<float>*)(panel + j * (4 * Vector256<float>.Count));
                c00 = Fma.MultiplyAdd(Bpv[0], av1, c00);
                c10 = Fma.MultiplyAdd(Bpv[0], av2, c10);
                c20 = Fma.MultiplyAdd(Bpv[0], av3, c20);
                c01 = Fma.MultiplyAdd(Bpv[1], av1, c01);
                c11 = Fma.MultiplyAdd(Bpv[1], av2, c11);
                c21 = Fma.MultiplyAdd(Bpv[1], av3, c21);
                c02 = Fma.MultiplyAdd(Bpv[2], av1, c02);
                c12 = Fma.MultiplyAdd(Bpv[2], av2, c12);
                c22 = Fma.MultiplyAdd(Bpv[2], av3, c22);
                c03 = Fma.MultiplyAdd(Bpv[3], av1, c03);
                c13 = Fma.MultiplyAdd(Bpv[3], av2, c13);
                c23 = Fma.MultiplyAdd(Bpv[3], av3, c23);
            }
            Cpv1[0] = c00;
            Cpv1[1] = c01;
            Cpv1[2] = c02;
            Cpv1[3] = c03;
            Cpv2[0] = c10;
            Cpv2[1] = c11;
            Cpv2[2] = c12;
            Cpv2[3] = c13;
            Cpv3[0] = c20;
            Cpv3[1] = c21;
            Cpv3[2] = c22;
            Cpv3[3] = c23;
        }
    }
}
