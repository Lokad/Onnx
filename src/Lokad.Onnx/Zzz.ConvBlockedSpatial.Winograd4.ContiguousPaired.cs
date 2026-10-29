#if NET11_0_OR_GREATER
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

internal static unsafe partial class ConvBlockedSpatial
{
    // The existing contiguous reader supplies eight tiles per channel; each
    // vector half keeps one channel's original arithmetic and border masks.
    [SkipLocalsInit]
    static void TransformWinograd4InputContiguousPaired(ReadOnlySpan<float> input, Span<float> transformed,
        int c, int h, int w, int top, int left)
    {
        bool border = top < 0 || top + 5 >= h || left < 0 || left + 35 >= w;
        int base0 = Math.Clamp(left, 0, w - 32), base1 = Math.Clamp(left + 4, 0, w - 32);
        var tiles = Vector256.Create(0, 4, 8, 12, 16, 20, 24, 28);
        var order = Vector512.Create(tiles + Vector256.Create(left - base0), Vector256<int>.Zero);
        var order4 = Vector512.Create(tiles + Vector256.Create(left + 4 - base1), Vector256<int>.Zero);
        var order1 = order + Vector512.Create(1); var order2 = order + Vector512.Create(2);
        var order3 = order + Vector512.Create(3); var order5 = order4 + Vector512.Create(1);
        Vector512<int>* masks = stackalloc Vector512<int>[6];
        if (border)
            for (int x = 0; x < 6; x++)
            {
                var positions = tiles + Vector256.Create(left + x);
                var mask = Vector256.GreaterThanOrEqual(positions, Vector256<int>.Zero)
                    & Vector256.LessThan(positions, Vector256.Create(w));
                masks[x] = Vector512.Create(mask, mask);
            }
        Vector512<float>* rows = stackalloc Vector512<float>[36];
        Vector512<float>* temporary = stackalloc Vector512<float>[36];
        int inputChannels = WinogradInputChannels(c, 36);
        fixed (float* source = input, destination = transformed)
        for (int ic = 0; ic < c; ic += 2)
        {
            for (int y = 0; y < 6; y++)
            {
                Vector512<float>* rowBuffer = rows + y * 6;
                if (border && (uint)(top + y) >= (uint)h)
                {
                    for (int x = 0; x < 6; x++) rowBuffer[x] = Vector512<float>.Zero;
                    continue;
                }
                float* row = source + ic * h * w + (top + y) * w;
                float* next = row + h * w;
                var a = Avx512F.LoadVector512(row + base0); var b = Avx512F.LoadVector512(row + base0 + 16);
                var a1 = Avx512F.LoadVector512(next + base0); var b1 = Avx512F.LoadVector512(next + base0 + 16);
                rowBuffer[0] = Vector512.Create(Avx512F.PermuteVar16x32x2(a, order, b).GetLower(),
                    Avx512F.PermuteVar16x32x2(a1, order, b1).GetLower());
                rowBuffer[1] = Vector512.Create(Avx512F.PermuteVar16x32x2(a, order1, b).GetLower(),
                    Avx512F.PermuteVar16x32x2(a1, order1, b1).GetLower());
                rowBuffer[2] = Vector512.Create(Avx512F.PermuteVar16x32x2(a, order2, b).GetLower(),
                    Avx512F.PermuteVar16x32x2(a1, order2, b1).GetLower());
                rowBuffer[3] = Vector512.Create(Avx512F.PermuteVar16x32x2(a, order3, b).GetLower(),
                    Avx512F.PermuteVar16x32x2(a1, order3, b1).GetLower());
                a = Avx512F.LoadVector512(row + base1); b = Avx512F.LoadVector512(row + base1 + 16);
                a1 = Avx512F.LoadVector512(next + base1); b1 = Avx512F.LoadVector512(next + base1 + 16);
                rowBuffer[4] = Vector512.Create(Avx512F.PermuteVar16x32x2(a, order4, b).GetLower(),
                    Avx512F.PermuteVar16x32x2(a1, order4, b1).GetLower());
                rowBuffer[5] = Vector512.Create(Avx512F.PermuteVar16x32x2(a, order5, b).GetLower(),
                    Avx512F.PermuteVar16x32x2(a1, order5, b1).GetLower());
                if (border)
                    for (int x = 0; x < 6; x++) rowBuffer[x] = (rowBuffer[x].AsInt32() & masks[x]).AsSingle();
            }
            for (int y = 0; y < 6; y++) InputWinograd4Paired(rows + y * 6, 1, temporary + y * 6, 1);
            for (int x = 0; x < 6; x++) InputWinograd4Paired(temporary + x, 6, rows + x, 6);
            for (int component = 0; component < 36; component++)
                *(Vector512<float>*)(destination + (component * inputChannels + ic) * 8) = rows[component];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void InputWinograd4Paired(Vector512<float>* input, int stride, Vector512<float>* output, int outputStride)
    {
        var a = input[0]; var b = input[stride]; var c = input[2 * stride];
        var d = input[3 * stride]; var e = input[4 * stride]; var f = input[5 * stride];
        output[0] = (4 * a - 5 * c) + e;
        output[outputStride] = (d + e) - 4 * (b + c);
        output[2 * outputStride] = (e - d) + 4 * (b - c);
        output[3 * outputStride] = 2 * (d - b) + (e - c);
        output[4 * outputStride] = 2 * (b - d) + (e - c);
        output[5 * outputStride] = (4 * b - 5 * d) + f;
    }
}
#endif
