using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

internal static unsafe partial class ConvBlockedSpatial
{
    // Fixed 128/256-channel F(4,3) reductions: short ascending FMAs, then a
    // balanced binary sum. Every scratch slot is assigned before its carry reads it.
    [SkipLocalsInit]
    static void MultiplyWinograd4Pairwise512(float* v, float* u, float* p, int c, int m)
    {
        const int lanes = 16;
        Vector512<float>* partial = stackalloc Vector512<float>[6 * WinogradBatch * 2];
        for (int component = 0; component < 36; component++)
        for (int oc = 0; oc < m; oc += 2 * lanes)
        {
            bool second = oc + lanes < m;
            for (int first = 0; first < c; first += 8)
            {
                var a0 = Vector512<float>.Zero; var b0 = a0;
                var a1 = Vector512<float>.Zero; var b1 = a1;
                var a2 = Vector512<float>.Zero; var b2 = a2;
                var a3 = Vector512<float>.Zero; var b3 = a3;
                var a4 = Vector512<float>.Zero; var b4 = a4;
                var a5 = Vector512<float>.Zero; var b5 = a5;
                var a6 = Vector512<float>.Zero; var b6 = a6;
                var a7 = Vector512<float>.Zero; var b7 = a7;
                float* weights = u + component * c * m + oc * c + first * lanes;
                float* inputs = v + (component * WinogradInputChannels(c, 36) + first) * WinogradBatch;
                for (int ic = 0; ic < 8; ic++)
                {
                    var weight = *(Vector512<float>*)weights;
                    var weight1 = second ? *(Vector512<float>*)(weights + c * lanes) : Vector512<float>.Zero;
                    var x0 = Vector512.Create(inputs[0]);
                    a0 = Avx512F.FusedMultiplyAdd(x0, weight, a0);
                    var x1 = Vector512.Create(inputs[1]);
                    a1 = Avx512F.FusedMultiplyAdd(x1, weight, a1);
                    var x2 = Vector512.Create(inputs[2]);
                    a2 = Avx512F.FusedMultiplyAdd(x2, weight, a2);
                    var x3 = Vector512.Create(inputs[3]);
                    a3 = Avx512F.FusedMultiplyAdd(x3, weight, a3);
                    var x4 = Vector512.Create(inputs[4]);
                    a4 = Avx512F.FusedMultiplyAdd(x4, weight, a4);
                    var x5 = Vector512.Create(inputs[5]);
                    a5 = Avx512F.FusedMultiplyAdd(x5, weight, a5);
                    var x6 = Vector512.Create(inputs[6]);
                    a6 = Avx512F.FusedMultiplyAdd(x6, weight, a6);
                    var x7 = Vector512.Create(inputs[7]);
                    a7 = Avx512F.FusedMultiplyAdd(x7, weight, a7);
                    if (second)
                    {
                        b0 = Avx512F.FusedMultiplyAdd(x0, weight1, b0);
                        b1 = Avx512F.FusedMultiplyAdd(x1, weight1, b1);
                        b2 = Avx512F.FusedMultiplyAdd(x2, weight1, b2);
                        b3 = Avx512F.FusedMultiplyAdd(x3, weight1, b3);
                        b4 = Avx512F.FusedMultiplyAdd(x4, weight1, b4);
                        b5 = Avx512F.FusedMultiplyAdd(x5, weight1, b5);
                        b6 = Avx512F.FusedMultiplyAdd(x6, weight1, b6);
                        b7 = Avx512F.FusedMultiplyAdd(x7, weight1, b7);
                    }
                    weights += lanes; inputs += WinogradBatch;
                }
                int level = 0;
                for (int group = first / 8; (group & 1) != 0; group >>= 1, level++)
                {
                    Vector512<float>* carry = partial + level * WinogradBatch * 2;
                    a0 = carry[0] + a0;
                    a1 = carry[1] + a1;
                    a2 = carry[2] + a2;
                    a3 = carry[3] + a3;
                    a4 = carry[4] + a4;
                    a5 = carry[5] + a5;
                    a6 = carry[6] + a6;
                    a7 = carry[7] + a7;
                    if (second)
                    {
                        b0 = carry[WinogradBatch + 0] + b0;
                        b1 = carry[WinogradBatch + 1] + b1;
                        b2 = carry[WinogradBatch + 2] + b2;
                        b3 = carry[WinogradBatch + 3] + b3;
                        b4 = carry[WinogradBatch + 4] + b4;
                        b5 = carry[WinogradBatch + 5] + b5;
                        b6 = carry[WinogradBatch + 6] + b6;
                        b7 = carry[WinogradBatch + 7] + b7;
                    }
                }
                Vector512<float>* pending = partial + level * WinogradBatch * 2;
                pending[0] = a0;
                pending[1] = a1;
                pending[2] = a2;
                pending[3] = a3;
                pending[4] = a4;
                pending[5] = a5;
                pending[6] = a6;
                pending[7] = a7;
                if (second)
                {
                    pending[WinogradBatch + 0] = b0;
                    pending[WinogradBatch + 1] = b1;
                    pending[WinogradBatch + 2] = b2;
                    pending[WinogradBatch + 3] = b3;
                    pending[WinogradBatch + 4] = b4;
                    pending[WinogradBatch + 5] = b5;
                    pending[WinogradBatch + 6] = b6;
                    pending[WinogradBatch + 7] = b7;
                }
            }
            int finalLevel = c == 128 ? 4 : 5;
            Vector512<float>* total = partial + finalLevel * WinogradBatch * 2;
            float* output = p + (component * m + oc) * WinogradBatch;
            *(Vector512<float>*)(output + 0 * lanes) = total[0];
            *(Vector512<float>*)(output + 1 * lanes) = total[1];
            *(Vector512<float>*)(output + 2 * lanes) = total[2];
            *(Vector512<float>*)(output + 3 * lanes) = total[3];
            *(Vector512<float>*)(output + 4 * lanes) = total[4];
            *(Vector512<float>*)(output + 5 * lanes) = total[5];
            *(Vector512<float>*)(output + 6 * lanes) = total[6];
            *(Vector512<float>*)(output + 7 * lanes) = total[7];
            if (second)
            {
                *(Vector512<float>*)(output + WinogradBatch * lanes + 0 * lanes) = total[WinogradBatch + 0];
                *(Vector512<float>*)(output + WinogradBatch * lanes + 1 * lanes) = total[WinogradBatch + 1];
                *(Vector512<float>*)(output + WinogradBatch * lanes + 2 * lanes) = total[WinogradBatch + 2];
                *(Vector512<float>*)(output + WinogradBatch * lanes + 3 * lanes) = total[WinogradBatch + 3];
                *(Vector512<float>*)(output + WinogradBatch * lanes + 4 * lanes) = total[WinogradBatch + 4];
                *(Vector512<float>*)(output + WinogradBatch * lanes + 5 * lanes) = total[WinogradBatch + 5];
                *(Vector512<float>*)(output + WinogradBatch * lanes + 6 * lanes) = total[WinogradBatch + 6];
                *(Vector512<float>*)(output + WinogradBatch * lanes + 7 * lanes) = total[WinogradBatch + 7];
            }
        }
    }
}
