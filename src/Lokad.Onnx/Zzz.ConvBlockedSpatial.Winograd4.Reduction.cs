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
        Vector512<float>* partial = stackalloc Vector512<float>[6 * WinogradBatch];
        for (int component = 0; component < 36; component++)
        for (int oc = 0; oc < m; oc += lanes)
        {
            for (int first = 0; first < c; first += 8)
            {
                var a0 = Vector512<float>.Zero;
                var a1 = Vector512<float>.Zero;
                var a2 = Vector512<float>.Zero;
                var a3 = Vector512<float>.Zero;
                var a4 = Vector512<float>.Zero;
                var a5 = Vector512<float>.Zero;
                var a6 = Vector512<float>.Zero;
                var a7 = Vector512<float>.Zero;
                float* weights = u + (component * c + first) * m + oc;
                float* inputs = v + (component * c + first) * WinogradBatch;
                for (int ic = 0; ic < 8; ic++)
                {
                    var weight = *(Vector512<float>*)weights;
                    a0 = Avx512F.FusedMultiplyAdd(Vector512.Create(inputs[0]), weight, a0);
                    a1 = Avx512F.FusedMultiplyAdd(Vector512.Create(inputs[1]), weight, a1);
                    a2 = Avx512F.FusedMultiplyAdd(Vector512.Create(inputs[2]), weight, a2);
                    a3 = Avx512F.FusedMultiplyAdd(Vector512.Create(inputs[3]), weight, a3);
                    a4 = Avx512F.FusedMultiplyAdd(Vector512.Create(inputs[4]), weight, a4);
                    a5 = Avx512F.FusedMultiplyAdd(Vector512.Create(inputs[5]), weight, a5);
                    a6 = Avx512F.FusedMultiplyAdd(Vector512.Create(inputs[6]), weight, a6);
                    a7 = Avx512F.FusedMultiplyAdd(Vector512.Create(inputs[7]), weight, a7);
                    weights += m; inputs += WinogradBatch;
                }
                int level = 0;
                for (int group = first / 8; (group & 1) != 0; group >>= 1, level++)
                {
                    a0 = partial[level * WinogradBatch + 0] + a0;
                    a1 = partial[level * WinogradBatch + 1] + a1;
                    a2 = partial[level * WinogradBatch + 2] + a2;
                    a3 = partial[level * WinogradBatch + 3] + a3;
                    a4 = partial[level * WinogradBatch + 4] + a4;
                    a5 = partial[level * WinogradBatch + 5] + a5;
                    a6 = partial[level * WinogradBatch + 6] + a6;
                    a7 = partial[level * WinogradBatch + 7] + a7;
                }
                partial[level * WinogradBatch + 0] = a0;
                partial[level * WinogradBatch + 1] = a1;
                partial[level * WinogradBatch + 2] = a2;
                partial[level * WinogradBatch + 3] = a3;
                partial[level * WinogradBatch + 4] = a4;
                partial[level * WinogradBatch + 5] = a5;
                partial[level * WinogradBatch + 6] = a6;
                partial[level * WinogradBatch + 7] = a7;
            }
            int finalLevel = c == 128 ? 4 : 5;
            float* output = p + (component * m + oc) * WinogradBatch;
            *(Vector512<float>*)(output + 0 * lanes) = partial[finalLevel * WinogradBatch + 0];
            *(Vector512<float>*)(output + 1 * lanes) = partial[finalLevel * WinogradBatch + 1];
            *(Vector512<float>*)(output + 2 * lanes) = partial[finalLevel * WinogradBatch + 2];
            *(Vector512<float>*)(output + 3 * lanes) = partial[finalLevel * WinogradBatch + 3];
            *(Vector512<float>*)(output + 4 * lanes) = partial[finalLevel * WinogradBatch + 4];
            *(Vector512<float>*)(output + 5 * lanes) = partial[finalLevel * WinogradBatch + 5];
            *(Vector512<float>*)(output + 6 * lanes) = partial[finalLevel * WinogradBatch + 6];
            *(Vector512<float>*)(output + 7 * lanes) = partial[finalLevel * WinogradBatch + 7];
        }
    }
}
