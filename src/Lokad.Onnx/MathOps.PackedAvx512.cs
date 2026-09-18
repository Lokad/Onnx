using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

// Adapted from feature/voice-model-cpu-benchmarks at 2b1138fe6f5d085e3749f6867d1603b1131ff029.
// This experiment imports only the full 32-column, 8/12-row accumulation tiles.
// Existing destination clearing, E94 column tails and AVX2 fallbacks stay intact.
// Enable explicitly with LOKAD_ONNX_PACKED_AVX512_ROWS=1 before process startup.
public partial class MathOps
{
    internal static unsafe bool TryPackedAvx512Rows(int m, int n, int k, float* x, float* packed, float* dest)
    {
        if (!Avx512F.IsSupported || !Fma.IsSupported || m < 8 || n <= 0 || k <= 0 || k % 32 != 0)
            return false;
        int main = m / 12 * 12;
        int rest = m - main;
        if (rest == 1) { main -= 12; rest = 13; }
        if (rest == 4 && main >= 12) { main -= 12; rest = 16; }
        else if (rest == 2 && main >= 12) { main -= 12; rest = 14; }
        if (main > 0)
            for (int column = 0; column < k; column += 32)
                PackedTile12(main, n, x, packed + column * n, dest, k, column);
        x += main * n;
        dest += main * k;
        int eights = 0;
        while (rest >= 8 && rest != 9) { eights += 8; rest -= 8; }
        if (eights > 0)
            for (int column = 0; column < k; column += 32)
                PackedTile8(eights, n, x, packed + column * n, dest, k, column);
        x += eights * n;
        dest += eights * k;
        if (rest == 0) return true;
        if (rest % 3 == 0)
            mm_unsafe_vectorized_intrinsics_3x4packed(rest, n, k, x, packed, dest);
        else if (rest % 2 == 0)
            mm_unsafe_vectorized_intrinsics_2x4packed_bump(rest, n, k, x, packed, dest);
        else
        {
            mm_unsafe_vectorized_intrinsics_3x4packed(3, n, k, x, packed, dest);
            mm_unsafe_vectorized_intrinsics_2x4packed_bump(rest - 3, n, k, x + 3 * n, packed, dest + 3 * k);
        }
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe void PackedTile12(int M,
                              int N,
                              float* A,
                              float* panel,
                              float* C,
                              int K,
                              int kb)
    {
            for (int i = 0; i < M; i += 12)
            {
                var Ap1 = A + i * N;
                var Ap2 = Ap1 + N;
                var Ap3 = Ap2 + N;
                var Ap4 = Ap3 + N;
                var Ap5 = Ap4 + N;
                var Ap6 = Ap5 + N;
                var Ap7 = Ap6 + N;
                var Ap8 = Ap7 + N;
                var Ap9 = Ap8 + N;
                var ApA = Ap9 + N;
                var ApB = ApA + N;
                var ApC = ApB + N;
                // C addresses recompute at the cold load/store points (8-row
                // style) so the twelve C-row pointers do not stay live
                // across the reduction loop and spill the A-row bases.
                float* cGroup = C + i * K + kb;
                Vector512<float> c00 = ((Vector512<float>*)(cGroup))[0];
                Vector512<float> c01 = ((Vector512<float>*)(cGroup))[1];
                Vector512<float> c10 = ((Vector512<float>*)(cGroup + 1 * K))[0];
                Vector512<float> c11 = ((Vector512<float>*)(cGroup + 1 * K))[1];
                Vector512<float> c20 = ((Vector512<float>*)(cGroup + 2 * K))[0];
                Vector512<float> c21 = ((Vector512<float>*)(cGroup + 2 * K))[1];
                Vector512<float> c30 = ((Vector512<float>*)(cGroup + 3 * K))[0];
                Vector512<float> c31 = ((Vector512<float>*)(cGroup + 3 * K))[1];
                Vector512<float> c40 = ((Vector512<float>*)(cGroup + 4 * K))[0];
                Vector512<float> c41 = ((Vector512<float>*)(cGroup + 4 * K))[1];
                Vector512<float> c50 = ((Vector512<float>*)(cGroup + 5 * K))[0];
                Vector512<float> c51 = ((Vector512<float>*)(cGroup + 5 * K))[1];
                Vector512<float> c70 = ((Vector512<float>*)(cGroup + 6 * K))[0];
                Vector512<float> c71 = ((Vector512<float>*)(cGroup + 6 * K))[1];
                Vector512<float> c80 = ((Vector512<float>*)(cGroup + 7 * K))[0];
                Vector512<float> c81 = ((Vector512<float>*)(cGroup + 7 * K))[1];
                Vector512<float> c90 = ((Vector512<float>*)(cGroup + 8 * K))[0];
                Vector512<float> c91 = ((Vector512<float>*)(cGroup + 8 * K))[1];
                Vector512<float> cA0 = ((Vector512<float>*)(cGroup + 9 * K))[0];
                Vector512<float> cA1 = ((Vector512<float>*)(cGroup + 9 * K))[1];
                Vector512<float> cB0 = ((Vector512<float>*)(cGroup + 10 * K))[0];
                Vector512<float> cB1 = ((Vector512<float>*)(cGroup + 10 * K))[1];
                Vector512<float> cC0 = ((Vector512<float>*)(cGroup + 11 * K))[0];
                Vector512<float> cC1 = ((Vector512<float>*)(cGroup + 11 * K))[1];
                for (int j = 0; j < N; ++j)
                {
                    var Bpv = (Vector512<float>*)(panel + j * (2 * Vector512<float>.Count));
                    var b0 = Bpv[0];
                    var b1 = Bpv[1];
                    var aa = Vector512.Create(Ap1[j]);
                    c00 = Avx512F.FusedMultiplyAdd(aa, b0, c00);
                    c01 = Avx512F.FusedMultiplyAdd(aa, b1, c01);
                    aa = Vector512.Create(Ap2[j]);
                    c10 = Avx512F.FusedMultiplyAdd(aa, b0, c10);
                    c11 = Avx512F.FusedMultiplyAdd(aa, b1, c11);
                    aa = Vector512.Create(Ap3[j]);
                    c20 = Avx512F.FusedMultiplyAdd(aa, b0, c20);
                    c21 = Avx512F.FusedMultiplyAdd(aa, b1, c21);
                    aa = Vector512.Create(Ap4[j]);
                    c30 = Avx512F.FusedMultiplyAdd(aa, b0, c30);
                    c31 = Avx512F.FusedMultiplyAdd(aa, b1, c31);
                    aa = Vector512.Create(Ap5[j]);
                    c40 = Avx512F.FusedMultiplyAdd(aa, b0, c40);
                    c41 = Avx512F.FusedMultiplyAdd(aa, b1, c41);
                    aa = Vector512.Create(Ap6[j]);
                    c50 = Avx512F.FusedMultiplyAdd(aa, b0, c50);
                    c51 = Avx512F.FusedMultiplyAdd(aa, b1, c51);
                    aa = Vector512.Create(Ap7[j]);
                    c70 = Avx512F.FusedMultiplyAdd(aa, b0, c70);
                    c71 = Avx512F.FusedMultiplyAdd(aa, b1, c71);
                    aa = Vector512.Create(Ap8[j]);
                    c80 = Avx512F.FusedMultiplyAdd(aa, b0, c80);
                    c81 = Avx512F.FusedMultiplyAdd(aa, b1, c81);
                    aa = Vector512.Create(Ap9[j]);
                    c90 = Avx512F.FusedMultiplyAdd(aa, b0, c90);
                    c91 = Avx512F.FusedMultiplyAdd(aa, b1, c91);
                    aa = Vector512.Create(ApA[j]);
                    cA0 = Avx512F.FusedMultiplyAdd(aa, b0, cA0);
                    cA1 = Avx512F.FusedMultiplyAdd(aa, b1, cA1);
                    aa = Vector512.Create(ApB[j]);
                    cB0 = Avx512F.FusedMultiplyAdd(aa, b0, cB0);
                    cB1 = Avx512F.FusedMultiplyAdd(aa, b1, cB1);
                    aa = Vector512.Create(ApC[j]);
                    cC0 = Avx512F.FusedMultiplyAdd(aa, b0, cC0);
                    cC1 = Avx512F.FusedMultiplyAdd(aa, b1, cC1);
                }
                ((Vector512<float>*)(cGroup))[0] = c00;
                ((Vector512<float>*)(cGroup))[1] = c01;
                ((Vector512<float>*)(cGroup + 1 * K))[0] = c10;
                ((Vector512<float>*)(cGroup + 1 * K))[1] = c11;
                ((Vector512<float>*)(cGroup + 2 * K))[0] = c20;
                ((Vector512<float>*)(cGroup + 2 * K))[1] = c21;
                ((Vector512<float>*)(cGroup + 3 * K))[0] = c30;
                ((Vector512<float>*)(cGroup + 3 * K))[1] = c31;
                ((Vector512<float>*)(cGroup + 4 * K))[0] = c40;
                ((Vector512<float>*)(cGroup + 4 * K))[1] = c41;
                ((Vector512<float>*)(cGroup + 5 * K))[0] = c50;
                ((Vector512<float>*)(cGroup + 5 * K))[1] = c51;
                ((Vector512<float>*)(cGroup + 6 * K))[0] = c70;
                ((Vector512<float>*)(cGroup + 6 * K))[1] = c71;
                ((Vector512<float>*)(cGroup + 7 * K))[0] = c80;
                ((Vector512<float>*)(cGroup + 7 * K))[1] = c81;
                ((Vector512<float>*)(cGroup + 8 * K))[0] = c90;
                ((Vector512<float>*)(cGroup + 8 * K))[1] = c91;
                ((Vector512<float>*)(cGroup + 9 * K))[0] = cA0;
                ((Vector512<float>*)(cGroup + 9 * K))[1] = cA1;
                ((Vector512<float>*)(cGroup + 10 * K))[0] = cB0;
                ((Vector512<float>*)(cGroup + 10 * K))[1] = cB1;
                ((Vector512<float>*)(cGroup + 11 * K))[0] = cC0;
                ((Vector512<float>*)(cGroup + 11 * K))[1] = cC1;
            }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe void PackedTile8(int M,
                              int N,
                              float* A,
                              float* panel,
                              float* C,
                              int K,
                              int kb)
    {
            for (int i = 0; i < M; i += 8)
            {
                // Hoisted A-row bases: the eight row pointers stay live in
                // registers across the reduction loop (verified in the M4
                // disassembly), replacing per-element address recomputation.
                // C-group addresses still recompute; they are loop-cold.
                float* aGroup = A + i * N;
                float* cGroup = C + i * K + kb;
                float* a0 = aGroup + 0 * N;
                float* a1 = aGroup + 1 * N;
                float* a2 = aGroup + 2 * N;
                float* a3 = aGroup + 3 * N;
                float* a4 = aGroup + 4 * N;
                float* a5 = aGroup + 5 * N;
                float* a6 = aGroup + 6 * N;
                float* a7 = aGroup + 7 * N;
                Vector512<float> c00 = ((Vector512<float>*)(cGroup))[0];
                Vector512<float> c01 = ((Vector512<float>*)(cGroup))[1];
                Vector512<float> c10 = ((Vector512<float>*)(cGroup + 1 * K))[0];
                Vector512<float> c11 = ((Vector512<float>*)(cGroup + 1 * K))[1];
                Vector512<float> c20 = ((Vector512<float>*)(cGroup + 2 * K))[0];
                Vector512<float> c21 = ((Vector512<float>*)(cGroup + 2 * K))[1];
                Vector512<float> c30 = ((Vector512<float>*)(cGroup + 3 * K))[0];
                Vector512<float> c31 = ((Vector512<float>*)(cGroup + 3 * K))[1];
                Vector512<float> c40 = ((Vector512<float>*)(cGroup + 4 * K))[0];
                Vector512<float> c41 = ((Vector512<float>*)(cGroup + 4 * K))[1];
                Vector512<float> c50 = ((Vector512<float>*)(cGroup + 5 * K))[0];
                Vector512<float> c51 = ((Vector512<float>*)(cGroup + 5 * K))[1];
                Vector512<float> c60 = ((Vector512<float>*)(cGroup + 6 * K))[0];
                Vector512<float> c61 = ((Vector512<float>*)(cGroup + 6 * K))[1];
                Vector512<float> c70 = ((Vector512<float>*)(cGroup + 7 * K))[0];
                Vector512<float> c71 = ((Vector512<float>*)(cGroup + 7 * K))[1];
                for (int j = 0; j < N; ++j)
                {
                    var Bpv = (Vector512<float>*)(panel + j * (2 * Vector512<float>.Count));
                    var b0 = Bpv[0];
                    var b1 = Bpv[1];
                    var aa = Vector512.Create(a0[j]);
                    c00 = Avx512F.FusedMultiplyAdd(aa, b0, c00);
                    c01 = Avx512F.FusedMultiplyAdd(aa, b1, c01);
                    aa = Vector512.Create(a1[j]);
                    c10 = Avx512F.FusedMultiplyAdd(aa, b0, c10);
                    c11 = Avx512F.FusedMultiplyAdd(aa, b1, c11);
                    aa = Vector512.Create(a2[j]);
                    c20 = Avx512F.FusedMultiplyAdd(aa, b0, c20);
                    c21 = Avx512F.FusedMultiplyAdd(aa, b1, c21);
                    aa = Vector512.Create(a3[j]);
                    c30 = Avx512F.FusedMultiplyAdd(aa, b0, c30);
                    c31 = Avx512F.FusedMultiplyAdd(aa, b1, c31);
                    aa = Vector512.Create(a4[j]);
                    c40 = Avx512F.FusedMultiplyAdd(aa, b0, c40);
                    c41 = Avx512F.FusedMultiplyAdd(aa, b1, c41);
                    aa = Vector512.Create(a5[j]);
                    c50 = Avx512F.FusedMultiplyAdd(aa, b0, c50);
                    c51 = Avx512F.FusedMultiplyAdd(aa, b1, c51);
                    aa = Vector512.Create(a6[j]);
                    c60 = Avx512F.FusedMultiplyAdd(aa, b0, c60);
                    c61 = Avx512F.FusedMultiplyAdd(aa, b1, c61);
                    aa = Vector512.Create(a7[j]);
                    c70 = Avx512F.FusedMultiplyAdd(aa, b0, c70);
                    c71 = Avx512F.FusedMultiplyAdd(aa, b1, c71);
                }
                ((Vector512<float>*)(cGroup))[0] = c00;
                ((Vector512<float>*)(cGroup))[1] = c01;
                ((Vector512<float>*)(cGroup + 1 * K))[0] = c10;
                ((Vector512<float>*)(cGroup + 1 * K))[1] = c11;
                ((Vector512<float>*)(cGroup + 2 * K))[0] = c20;
                ((Vector512<float>*)(cGroup + 2 * K))[1] = c21;
                ((Vector512<float>*)(cGroup + 3 * K))[0] = c30;
                ((Vector512<float>*)(cGroup + 3 * K))[1] = c31;
                ((Vector512<float>*)(cGroup + 4 * K))[0] = c40;
                ((Vector512<float>*)(cGroup + 4 * K))[1] = c41;
                ((Vector512<float>*)(cGroup + 5 * K))[0] = c50;
                ((Vector512<float>*)(cGroup + 5 * K))[1] = c51;
                ((Vector512<float>*)(cGroup + 6 * K))[0] = c60;
                ((Vector512<float>*)(cGroup + 6 * K))[1] = c61;
                ((Vector512<float>*)(cGroup + 7 * K))[0] = c70;
                ((Vector512<float>*)(cGroup + 7 * K))[1] = c71;
            }
    }
}
