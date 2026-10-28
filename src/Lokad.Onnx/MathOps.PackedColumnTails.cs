using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

public partial class MathOps
{
    // The compact tail retains its original row stride. Eight output rows
    // share each B vector without splitting or reordering any reduction.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe void PackedColumnTailEightRows(int rows, int reduction,
        int outputStride, int packedStride, float* a, float* b, float* c)
    {
        for (int row = 0; row < rows; row += 8)
        {
            float* a0 = a + row * reduction;
            float* a1 = a0 + reduction;
            float* a2 = a1 + reduction;
            float* a3 = a2 + reduction;
            float* a4 = a3 + reduction;
            float* a5 = a4 + reduction;
            float* a6 = a5 + reduction;
            float* a7 = a6 + reduction;
            float* output = c + row * outputStride;
            var c0 = *(Vector256<float>*)(output);
            var c1 = *(Vector256<float>*)(output + outputStride);
            var c2 = *(Vector256<float>*)(output + 2 * outputStride);
            var c3 = *(Vector256<float>*)(output + 3 * outputStride);
            var c4 = *(Vector256<float>*)(output + 4 * outputStride);
            var c5 = *(Vector256<float>*)(output + 5 * outputStride);
            var c6 = *(Vector256<float>*)(output + 6 * outputStride);
            var c7 = *(Vector256<float>*)(output + 7 * outputStride);
            for (int j = 0; j < reduction; j++)
            {
                var bv = *(Vector256<float>*)(b + j * packedStride);
                c0 = Fma.MultiplyAdd(bv, Vector256.Create(a0[j]), c0);
                c1 = Fma.MultiplyAdd(bv, Vector256.Create(a1[j]), c1);
                c2 = Fma.MultiplyAdd(bv, Vector256.Create(a2[j]), c2);
                c3 = Fma.MultiplyAdd(bv, Vector256.Create(a3[j]), c3);
                c4 = Fma.MultiplyAdd(bv, Vector256.Create(a4[j]), c4);
                c5 = Fma.MultiplyAdd(bv, Vector256.Create(a5[j]), c5);
                c6 = Fma.MultiplyAdd(bv, Vector256.Create(a6[j]), c6);
                c7 = Fma.MultiplyAdd(bv, Vector256.Create(a7[j]), c7);
            }
            *(Vector256<float>*)(output) = c0;
            *(Vector256<float>*)(output + outputStride) = c1;
            *(Vector256<float>*)(output + 2 * outputStride) = c2;
            *(Vector256<float>*)(output + 3 * outputStride) = c3;
            *(Vector256<float>*)(output + 4 * outputStride) = c4;
            *(Vector256<float>*)(output + 5 * outputStride) = c5;
            *(Vector256<float>*)(output + 6 * outputStride) = c6;
            *(Vector256<float>*)(output + 7 * outputStride) = c7;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe void PackedColumnMaskedEightRows(int rows, int reduction,
        int outputStride, int packedStride, int columns, float* a, float* b, float* c)
    {
        var mask = Vector256.Create(columns > 0 ? -1 : 0, columns > 1 ? -1 : 0,
            columns > 2 ? -1 : 0, columns > 3 ? -1 : 0, columns > 4 ? -1 : 0,
            columns > 5 ? -1 : 0, columns > 6 ? -1 : 0, 0);
        for (int row = 0; row < rows; row += 8)
        {
            float* a0 = a + row * reduction;
            float* a1 = a0 + reduction;
            float* a2 = a1 + reduction;
            float* a3 = a2 + reduction;
            float* a4 = a3 + reduction;
            float* a5 = a4 + reduction;
            float* a6 = a5 + reduction;
            float* a7 = a6 + reduction;
            float* output = c + row * outputStride;
            var c0 = Avx2.MaskLoad((int*)output, mask).AsSingle();
            var c1 = Avx2.MaskLoad((int*)(output + outputStride), mask).AsSingle();
            var c2 = Avx2.MaskLoad((int*)(output + 2 * outputStride), mask).AsSingle();
            var c3 = Avx2.MaskLoad((int*)(output + 3 * outputStride), mask).AsSingle();
            var c4 = Avx2.MaskLoad((int*)(output + 4 * outputStride), mask).AsSingle();
            var c5 = Avx2.MaskLoad((int*)(output + 5 * outputStride), mask).AsSingle();
            var c6 = Avx2.MaskLoad((int*)(output + 6 * outputStride), mask).AsSingle();
            var c7 = Avx2.MaskLoad((int*)(output + 7 * outputStride), mask).AsSingle();
            for (int j = 0; j < reduction; j++)
            {
                var bv = Avx2.MaskLoad((int*)(b + j * packedStride), mask).AsSingle();
                // Preserve the original A*B then C+product operations, never FMA.
                c0 = c0 + Vector256.Create(a0[j]) * bv;
                c1 = c1 + Vector256.Create(a1[j]) * bv;
                c2 = c2 + Vector256.Create(a2[j]) * bv;
                c3 = c3 + Vector256.Create(a3[j]) * bv;
                c4 = c4 + Vector256.Create(a4[j]) * bv;
                c5 = c5 + Vector256.Create(a5[j]) * bv;
                c6 = c6 + Vector256.Create(a6[j]) * bv;
                c7 = c7 + Vector256.Create(a7[j]) * bv;
            }
            Avx2.MaskStore((int*)output, mask, c0.AsInt32());
            Avx2.MaskStore((int*)(output + outputStride), mask, c1.AsInt32());
            Avx2.MaskStore((int*)(output + 2 * outputStride), mask, c2.AsInt32());
            Avx2.MaskStore((int*)(output + 3 * outputStride), mask, c3.AsInt32());
            Avx2.MaskStore((int*)(output + 4 * outputStride), mask, c4.AsInt32());
            Avx2.MaskStore((int*)(output + 5 * outputStride), mask, c5.AsInt32());
            Avx2.MaskStore((int*)(output + 6 * outputStride), mask, c6.AsInt32());
            Avx2.MaskStore((int*)(output + 7 * outputStride), mask, c7.AsInt32());
        }
    }
    // The compact column tail is shared with the wide full-panel consumer.
    // M is even. Complete eight-column lanes use FMA; the final masked lanes
    // retain their separate multiply/add order from the original two-row route.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static unsafe void PackedColumnTailsEvenRows(int M, int N, int K, float* A, float* P, float* C)
    {
        int tiles = K / 32, blocked = tiles * 32;
        int rem = K - blocked;
        if (rem > 0)
        {
            float* T = P + tiles * N * (4 * Vector256<float>.Count);
            int rv = rem / Vector256<float>.Count;
            int sharedRows = M >= 64 && N >= 64 ? M - M % 8 : 0;
            for (int tt = 0; tt < rv; tt++)
            {
                if (sharedRows > 0)
                    PackedColumnTailEightRows(sharedRows, N, K, rem, A,
                        T + tt * Vector256<float>.Count, C + blocked + tt * Vector256<float>.Count);
                for (int i = sharedRows; i < M; i += 2)
                {
                    var Ap1 = A + i * N;
                    var Ap2 = Ap1 + N;
                    var rC1 = (Vector256<float>*)(C + i * K + blocked);
                    var rC2 = (Vector256<float>*)(C + (i + 1) * K + blocked);
                    Vector256<float> c1 = rC1[tt];
                    Vector256<float> c2 = rC2[tt];
                    for (int j = 0; j < N; ++j)
                    {
                        var Bpv = (Vector256<float>*)(T + j * rem + tt * Vector256<float>.Count);
                        c1 = Fma.MultiplyAdd(Bpv[0], Vector256.Create(Ap1[j]), c1);
                        c2 = Fma.MultiplyAdd(Bpv[0], Vector256.Create(Ap2[j]), c2);
                    }
                    rC1[tt] = c1;
                    rC2[tt] = c2;
                }
            }
            int vcols = rv * Vector256<float>.Count;
            // Narrow tail (fewer than 8 columns) accumulates in registers
            // instead of read-modify-writing C on every reduction step: the
            // per-element chain keeps the exact j-ascending mul-then-add order
            // of the loop it replaces, so results agree bit-wise while each C
            // row is loaded once and stored once.
            int tail = rem - vcols;
            int maskedRows = tail > 0 && Avx2.IsSupported ? sharedRows : 0;
            if (maskedRows > 0)
                PackedColumnMaskedEightRows(maskedRows, N, K, rem, tail, A,
                    T + vcols, C + blocked + vcols);
            if (tail > 0)
            for (int i = maskedRows; i < M; i += 2)
            {
                var Ap1 = A + i * N;
                var Ap2 = Ap1 + N;
                var Cp1 = C + i * K + blocked + vcols;
                var Cp2 = Cp1 + K;
                // E94: masked-vector narrow tail. Same j-ascending mul-then-add
                // order per element as the scalar switch it replaces (MULPS then
                // ADDPS, never FMA), so results agree bit-wise. Integer maskmov
                // moves the bits (fault-suppressed, discarded by masked store);
                // arithmetic stays float. Scalar fallback preserves order.
                if (Avx2.IsSupported)
                {
                    Vector256<int> tmask = Vector256.Create(tail > 0 ? -1 : 0, tail > 1 ? -1 : 0, tail > 2 ? -1 : 0, tail > 3 ? -1 : 0, tail > 4 ? -1 : 0, tail > 5 ? -1 : 0, tail > 6 ? -1 : 0, 0);
                    Vector256<int> ci1 = Avx2.MaskLoad((int*)Cp1, tmask);
                    Vector256<int> ci2 = Avx2.MaskLoad((int*)Cp2, tmask);
                    Vector256<float> c1 = Unsafe.As<Vector256<int>, Vector256<float>>(ref ci1);
                    Vector256<float> c2 = Unsafe.As<Vector256<int>, Vector256<float>>(ref ci2);
                    for (int j = 0; j < N; ++j)
                    {
                        Vector256<int> bi = Avx2.MaskLoad((int*)(T + j * rem + vcols), tmask);
                        Vector256<float> bv = Unsafe.As<Vector256<int>, Vector256<float>>(ref bi);
                        var av1 = Vector256.Create(Ap1[j]);
                        var av2 = Vector256.Create(Ap2[j]);
                        c1 = c1 + av1 * bv;
                        c2 = c2 + av2 * bv;
                    }
                    Vector256<int> co1 = Unsafe.As<Vector256<float>, Vector256<int>>(ref c1);
                    Vector256<int> co2 = Unsafe.As<Vector256<float>, Vector256<int>>(ref c2);
                    Avx2.MaskStore((int*)Cp1, tmask, co1);
                    Avx2.MaskStore((int*)Cp2, tmask, co2);
                }
                else
                {
                    for (int j = 0; j < N; ++j)
                    {
                        float a1 = Ap1[j];
                        float a2 = Ap2[j];
                        var t = T + j * rem + vcols;
                        for (int k = 0; k < tail; k++) { Cp1[k] += a1 * t[k]; Cp2[k] += a2 * t[k]; }
                    }
                }
            }
        }
    }
}
