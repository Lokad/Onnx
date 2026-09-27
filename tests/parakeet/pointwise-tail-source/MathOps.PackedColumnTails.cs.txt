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
}
