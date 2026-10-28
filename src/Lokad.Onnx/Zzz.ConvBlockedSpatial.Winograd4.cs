using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

// Opt-in F(4,3), with bounded reductions and direct fallback.
internal static unsafe partial class ConvBlockedSpatial
{
    [SkipLocalsInit]
    internal static float[]? PrepareWinograd4(ReadOnlySpan<float> weights, int c, int m, int lanes)
    {
        Geometry(c, m, 1, 1, 1, lanes);
        if (c > 64 && (lanes != 16 || c is not (128 or 256))) return null;
        if (weights.Length != checked(c * m * 9)) throw new ArgumentException("Weight extent");
        if (!Finite(weights)) return null;
        // Preserve direct fallback for magnitudes that can overflow the original
        // seven-term float transform. Precision changes here must not broaden it.
        foreach (float value in weights) if (MathF.Abs(value) > float.MaxValue / 8) return null;
        var result = new float[checked(36 * c * m)];
        Span<double> rows = stackalloc double[18];
        Span<double> column = stackalloc double[6];
        for (int oc = 0; oc < m; oc++)
        for (int ic = 0; ic < c; ic++)
        {
            var g = weights.Slice((oc * c + ic) * 9, 9);
            for (int y = 0; y < 3; y++) WeightWinograd4(g[y * 3], g[y * 3 + 1], g[y * 3 + 2], rows.Slice(y * 6, 6));
            for (int x = 0; x < 6; x++)
            {
                WeightWinograd4(rows[x], rows[6 + x], rows[12 + x], column);
                for (int y = 0; y < 6; y++) result[((y * 6 + x) * c + ic) * m + oc] = (float)column[y];
            }
        }
        return Finite(result) ? result : null;
    }

    static void WeightWinograd4(double a, double b, double c, Span<double> output)
    {
        output[0] = a * .25;
        output[1] = ((a + b) + c) * (-1.0 / 6);
        output[2] = ((a - b) + c) * (-1.0 / 6);
        output[3] = ((a + 2 * b) + 4 * c) * (1.0 / 24);
        output[4] = ((a - 2 * b) + 4 * c) * (1.0 / 24);
        output[5] = c;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void InputWinograd4(Vector256<float>* input, int stride, Vector256<float>* output, int outputStride)
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

    [SkipLocalsInit]
    static void TransformWinograd4Input(ReadOnlySpan<float> input, Span<float> transformed,
        int c, int h, int w, int tileWidth, int first, int count)
    {
        int top = first / tileWidth * 4 - 1, left = first % tileWidth * 4 - 1;
        if (Avx512F.IsSupported && count == WinogradBatch && first / tileWidth == (first + 7) / tileWidth
            && top >= 0 && top + 5 < h && left >= 0 && left + 35 < w)
        {
            TransformWinograd4InputContiguous(input, transformed, c, h, w, top, left);
            return;
        }
        int* indices = stackalloc int[36 * WinogradBatch];
        int* masks = stackalloc int[36 * WinogradBatch];
        Vector256<float>* rows = stackalloc Vector256<float>[36];
        Vector256<float>* temporary = stackalloc Vector256<float>[36];
        for (int tile = 0; tile < WinogradBatch; tile++)
        {
            int tileTop = (first + tile) / tileWidth * 4 - 1;
            int tileLeft = (first + tile) % tileWidth * 4 - 1;
            for (int y = 0; y < 6; y++)
            for (int x = 0; x < 6; x++)
            {
                int slot = (y * 6 + x) * WinogradBatch + tile;
                bool valid = tile < count && (uint)(tileTop + y) < (uint)h && (uint)(tileLeft + x) < (uint)w;
                indices[slot] = valid ? (tileTop + y) * w + tileLeft + x : 0;
                masks[slot] = valid ? -1 : 0;
            }
        }
        fixed (float* source = input, destination = transformed)
        for (int ic = 0; ic < c; ic++)
        {
            float* channel = source + ic * h * w;
            for (int component = 0; component < 36; component++)
                rows[component] = Avx2.GatherMaskVector256(Vector256<float>.Zero, channel,
                    *(Vector256<int>*)(indices + component * 8), (*(Vector256<int>*)(masks + component * 8)).AsSingle(), 4);
            for (int y = 0; y < 6; y++) InputWinograd4(rows + y * 6, 1, temporary + y * 6, 1);
            for (int x = 0; x < 6; x++) InputWinograd4(temporary + x, 6, rows + x, 6);
            for (int component = 0; component < 36; component++)
                *(Vector256<float>*)(destination + (component * c + ic) * 8) = rows[component];
        }
    }

    [SkipLocalsInit]
    static void TransformWinograd4InputContiguous(ReadOnlySpan<float> input, Span<float> transformed,
        int c, int h, int w, int top, int left)
    {
        // Two load pairs cover columns left through left+35, proved by the caller.
        var order = Vector512.Create(Vector256.Create(0, 4, 8, 12, 16, 20, 24, 28), Vector256<int>.Zero);
        var one = Vector512.Create(1); var two = Vector512.Create(2); var three = Vector512.Create(3);
        Vector256<float>* rows = stackalloc Vector256<float>[36];
        Vector256<float>* temporary = stackalloc Vector256<float>[36];
        fixed (float* source = input, destination = transformed)
        for (int ic = 0; ic < c; ic++)
        {
            for (int y = 0; y < 6; y++)
            {
                float* start = source + ic * h * w + (top + y) * w + left;
                var a = Avx512F.LoadVector512(start); var b = Avx512F.LoadVector512(start + 16);
                rows[y * 6] = Avx512F.PermuteVar16x32x2(a, order, b).GetLower();
                rows[y * 6 + 1] = Avx512F.PermuteVar16x32x2(a, order + one, b).GetLower();
                rows[y * 6 + 2] = Avx512F.PermuteVar16x32x2(a, order + two, b).GetLower();
                rows[y * 6 + 3] = Avx512F.PermuteVar16x32x2(a, order + three, b).GetLower();
                a = Avx512F.LoadVector512(start + 4); b = Avx512F.LoadVector512(start + 20);
                rows[y * 6 + 4] = Avx512F.PermuteVar16x32x2(a, order, b).GetLower();
                rows[y * 6 + 5] = Avx512F.PermuteVar16x32x2(a, order + one, b).GetLower();
            }
            for (int y = 0; y < 6; y++) InputWinograd4(rows + y * 6, 1, temporary + y * 6, 1);
            for (int x = 0; x < 6; x++) InputWinograd4(temporary + x, 6, rows + x, 6);
            for (int component = 0; component < 36; component++)
                *(Vector256<float>*)(destination + (component * c + ic) * 8) = rows[component];
        }
    }

    [SkipLocalsInit]
    static void OutputWinograd4_256(float* p, float* output, int m, int h, int w, int tileWidth, int first, int count)
    {
        const int lanes = 8;
        Vector256<float>* temporary = stackalloc Vector256<float>[24];
        Vector256<float>* rows = stackalloc Vector256<float>[16];
        int step = m * WinogradBatch;
        for (int oc = 0; oc < m; oc += lanes)
        for (int tile = 0; tile < count; tile++)
        {
            float* src = p + oc * WinogradBatch + tile * lanes;
            for (int x = 0; x < 6; x++)
                OutputWinograd4_256((Vector256<float>*)(src + x * step), step * 6 / lanes, temporary + x, 6);
            for (int y = 0; y < 4; y++) OutputWinograd4_256(temporary + y * 6, 1, rows + y * 4, 1);
            int top = (first + tile) / tileWidth * 4, left = (first + tile) % tileWidth * 4;
            for (int y = 0; y < 4 && top + y < h; y++)
            for (int x = 0; x < 4 && left + x < w; x++)
                *(Vector256<float>*)(output + oc * h * w + ((top + y) * w + left + x) * lanes) = rows[y * 4 + x];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void OutputWinograd4_256(Vector256<float>* input, int stride, Vector256<float>* output, int outputStride)
    {
        var a = input[0]; var b = input[stride]; var c = input[2 * stride];
        var d = input[3 * stride]; var e = input[4 * stride]; var f = input[5 * stride];
        output[0] = ((a + b) + c) + (d + e);
        output[outputStride] = (b - c) + 2 * (d - e);
        output[2 * outputStride] = (b + c) + 4 * (d + e);
        output[3 * outputStride] = ((b - c) + 8 * (d - e)) + f;
    }

    [SkipLocalsInit]
    static void OutputWinograd4_512(float* p, float* output, int m, int h, int w, int tileWidth, int first, int count)
    {
        const int lanes = 16;
        Vector512<float>* temporary = stackalloc Vector512<float>[24];
        Vector512<float>* rows = stackalloc Vector512<float>[16];
        int step = m * WinogradBatch;
        for (int oc = 0; oc < m; oc += lanes)
        for (int tile = 0; tile < count; tile++)
        {
            float* src = p + oc * WinogradBatch + tile * lanes;
            for (int x = 0; x < 6; x++)
                OutputWinograd4_512((Vector512<float>*)(src + x * step), step * 6 / lanes, temporary + x, 6);
            for (int y = 0; y < 4; y++) OutputWinograd4_512(temporary + y * 6, 1, rows + y * 4, 1);
            int top = (first + tile) / tileWidth * 4, left = (first + tile) % tileWidth * 4;
            for (int y = 0; y < 4 && top + y < h; y++)
            for (int x = 0; x < 4 && left + x < w; x++)
                *(Vector512<float>*)(output + oc * h * w + ((top + y) * w + left + x) * lanes) = rows[y * 4 + x];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static void OutputWinograd4_512(Vector512<float>* input, int stride, Vector512<float>* output, int outputStride)
    {
        var a = input[0]; var b = input[stride]; var c = input[2 * stride];
        var d = input[3 * stride]; var e = input[4 * stride]; var f = input[5 * stride];
        output[0] = ((a + b) + c) + (d + e);
        output[outputStride] = (b - c) + 2 * (d - e);
        output[2 * outputStride] = (b + c) + 4 * (d + e);
        output[3 * outputStride] = ((b - c) + 8 * (d - e)) + f;
    }
}
