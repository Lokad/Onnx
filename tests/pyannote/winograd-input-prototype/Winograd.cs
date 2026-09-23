using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

// Isolated numerical prototype. No graph or product dispatch references this file.
internal static unsafe partial class ConvBlockedSpatial
{
    internal const int WinogradBatch = 8;

    internal static float[]? PrepareWinograd(ReadOnlySpan<float> weights, int c, int m, int lanes)
    {
        Geometry(c, m, 1, 1, 1, lanes);
        if (weights.Length != checked(c * m * 9)) throw new ArgumentException("Weight extent");
        if (!Finite(weights)) return null;
        var result = new float[checked(16 * c * m)];
        Span<float> row = stackalloc float[12];
        for (int oc = 0; oc < m; oc++)
        for (int ic = 0; ic < c; ic++)
        {
            var g = weights.Slice((oc * c + ic) * 9, 9);
            for (int r = 0; r < 3; r++)
            {
                float a = g[r * 3], b = g[r * 3 + 1], d = g[r * 3 + 2];
                row[r * 4] = a;
                row[r * 4 + 1] = ((a + b) + d) * .5f;
                row[r * 4 + 2] = ((a - b) + d) * .5f;
                row[r * 4 + 3] = d;
            }
            for (int col = 0; col < 4; col++)
            {
                float a = row[col], b = row[4 + col], d = row[8 + col];
                result[(col * c + ic) * m + oc] = a;
                result[((4 + col) * c + ic) * m + oc] = ((a + b) + d) * .5f;
                result[((8 + col) * c + ic) * m + oc] = ((a - b) + d) * .5f;
                result[((12 + col) * c + ic) * m + oc] = d;
            }
        }
        return Finite(result) ? result : null;
    }

    internal static bool PlanWinograd(int c, int m, int h, int w, out int input, out int products, out int output)
    {
        input = products = output = 0;
        if (c < 1 || m < 1 || h < 1 || w < 1) return false;
        try
        {
            long a = checked(16L * c * WinogradBatch), b = checked(16L * m * WinogradBatch);
            long d = checked((long)m * h * w);
            if (checked(a + b + d) > 64L * 1024 * 1024 / sizeof(float)) return false;
            input = (int)a; products = (int)b; output = (int)d;
            return true;
        }
        catch (OverflowException) { return false; }
    }

    internal static bool ExecuteWinograd(ReadOnlySpan<float> input, ReadOnlySpan<float> prepared,
        ReadOnlySpan<float> bias, ReadOnlySpan<float> residual, Span<float> destination,
        Span<float> transformed, Span<float> products, Span<float> blocked,
        int c, int m, int h, int w, int lanes, bool relu)
    {
        Geometry(c, m, h, w, 1, lanes);
        if (!PlanWinograd(c, m, h, w, out int ni, out int np, out int no)) return false;
        if (input.Length != checked(c * h * w) || prepared.Length != checked(16 * c * m)
            || destination.Length != no || (bias.Length != 0 && bias.Length != m)
            || (residual.Length != 0 && residual.Length != no)
            || transformed.Length < ni || products.Length < np || blocked.Length < no)
            throw new ArgumentException("Winograd buffer extent");
        transformed = transformed.Slice(0, ni); products = products.Slice(0, np); blocked = blocked.Slice(0, no);
        if (input.Overlaps(destination) || prepared.Overlaps(destination) || bias.Overlaps(destination) || residual.Overlaps(destination)
            || input.Overlaps(transformed) || prepared.Overlaps(transformed) || bias.Overlaps(transformed) || residual.Overlaps(transformed)
            || input.Overlaps(products) || prepared.Overlaps(products) || bias.Overlaps(products) || residual.Overlaps(products)
            || input.Overlaps(blocked) || prepared.Overlaps(blocked) || bias.Overlaps(blocked) || residual.Overlaps(blocked)
            || transformed.Overlaps(products) || transformed.Overlaps(blocked) || transformed.Overlaps(destination)
            || products.Overlaps(blocked) || products.Overlaps(destination) || blocked.Overlaps(destination))
            throw new ArgumentException("Winograd overlapping buffers");
        if (lanes == 16 && !Avx512F.IsSupported || lanes == 8 && (!Avx2.IsSupported || !Fma.IsSupported))
            throw new PlatformNotSupportedException();
        if (!Finite(input) || !Finite(prepared) || !Finite(bias) || !Finite(residual)) return false;
        int tileWidth = (w + 1) / 2, tiles = checked(((h + 1) / 2) * tileWidth);
        for (int first = 0; first < tiles; first += WinogradBatch)
        {
            int count = Math.Min(WinogradBatch, tiles - first);
            TransformWinogradInput(input, transformed, c, h, w, tileWidth, first, count);
            if (!Finite(transformed)) return false;
            fixed (float* v = transformed, u = prepared, p = products, dst = blocked)
            {
                if (lanes == 16) MultiplyWinograd512(v, u, p, c, m);
                else MultiplyWinograd256(v, u, p, c, m);
                if (!Finite(products)) return false;
                if (lanes == 16) OutputWinograd512(p, dst, m, h, w, tileWidth, first, count);
                else OutputWinograd256(p, dst, m, h, w, tileWidth, first, count);
            }
        }
        // Never expose a partial result when intermediate arithmetic overflows.
        if (!Finite(blocked) || !EpilogueRange(blocked) || !EpilogueRange(bias) || !EpilogueRange(residual)) return false;
        UnpackEpilogue(blocked, destination, bias, residual, m, h * w, lanes, relu);
        return true;
    }

    static bool EpilogueRange(ReadOnlySpan<float> values)
    {
        const float limit = float.MaxValue / 4;
        foreach (float value in values) if (MathF.Abs(value) > limit) return false;
        return true;
    }

    static void TransformWinogradInput(ReadOnlySpan<float> input, Span<float> transformed,
        int c, int h, int w, int tileWidth, int first, int count)
    {
        // Each vector lane is a tile. Coordinates and masks are reused for every channel.
        int* indices = stackalloc int[16 * WinogradBatch];
        int* masks = stackalloc int[16 * WinogradBatch];
        Vector256<float>* row = stackalloc Vector256<float>[16];
        for (int tile = 0; tile < WinogradBatch; tile++)
        {
            int top = (first + tile) / tileWidth * 2 - 1;
            int left = (first + tile) % tileWidth * 2 - 1;
            for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
            {
                int slot = (y * 4 + x) * WinogradBatch + tile;
                bool valid = tile < count && (uint)(top + y) < (uint)h && (uint)(left + x) < (uint)w;
                indices[slot] = valid ? (top + y) * w + left + x : 0;
                masks[slot] = valid ? -1 : 0;
            }
        }
        fixed (float* source = input, destination = transformed)
        for (int ic = 0; ic < c; ic++)
        {
            float* channel = source + ic * h * w;
            for (int y = 0; y < 4; y++)
            {
                int offset = y * 4 * WinogradBatch;
                var a = Avx2.GatherMaskVector256(Vector256<float>.Zero, channel,
                    Avx.LoadVector256(indices + offset), Avx.LoadVector256((float*)(masks + offset)), 4);
                var b = Avx2.GatherMaskVector256(Vector256<float>.Zero, channel,
                    Avx.LoadVector256(indices + offset + 8), Avx.LoadVector256((float*)(masks + offset + 8)), 4);
                var d = Avx2.GatherMaskVector256(Vector256<float>.Zero, channel,
                    Avx.LoadVector256(indices + offset + 16), Avx.LoadVector256((float*)(masks + offset + 16)), 4);
                var e = Avx2.GatherMaskVector256(Vector256<float>.Zero, channel,
                    Avx.LoadVector256(indices + offset + 24), Avx.LoadVector256((float*)(masks + offset + 24)), 4);
                row[y * 4] = Avx.Subtract(a, d);
                row[y * 4 + 1] = Avx.Add(b, d);
                row[y * 4 + 2] = Avx.Subtract(d, b);
                row[y * 4 + 3] = Avx.Subtract(b, e);
            }
            for (int col = 0; col < 4; col++)
            {
                var a = row[col]; var b = row[4 + col]; var d = row[8 + col]; var e = row[12 + col];
                Avx.Store(destination + (col * c + ic) * WinogradBatch, Avx.Subtract(a, d));
                Avx.Store(destination + ((4 + col) * c + ic) * WinogradBatch, Avx.Add(b, d));
                Avx.Store(destination + ((8 + col) * c + ic) * WinogradBatch, Avx.Subtract(d, b));
                Avx.Store(destination + ((12 + col) * c + ic) * WinogradBatch, Avx.Subtract(b, e));
            }
        }
    }
}
