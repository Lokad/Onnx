namespace Lokad.Onnx;

using System;
using System.Runtime.Intrinsics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Blocked-channel 3x3 convolution kernel (M5): NCHWc16 blocked layout with
/// sixteen output channels per vector, eight spatial positions per interior
/// tile, and bounds-checked vector border positions. Reduction order
/// (channel blocks, taps, channels) reassociates the im2col GEMM order, so
/// agreement with production Conv2D is checked at a tolerance gate, never
/// bit identity. Covers float32 group-1 stride-1 pad-1 kernels with channel
/// counts divisible by sixteen and batch one; everything else declines to
/// the generic path. Filter blocking is per call here; prepared blocking
/// belongs to the persistent-region stage.
/// </summary>
public static class MathOpsConvBlocked
{
    const int CB = 16;
    const int T = 8;

    public static bool TryConvBlocked3x3S1P1(
        Tensor<float> input,
        Tensor<float> weight,
        Tensor<float>? bias,
        int n,
        int c,
        int h,
        int w,
        int m,
        MathOps.PadInfo pad,
        TensorExecutionOptions options,
        bool fuseRelu,
        out Tensor<float>? output)
    {
        output = null;
        if (!Avx512F.IsSupported) return false;
        if (!options.UseSimd || !options.UseIntrinsics) return false;
        if (n != 1 || c < CB || m < CB || c % CB != 0 || m % CB != 0) return false;
        if (h < 1 || w < 1) return false;
        if (pad.top != 1 || pad.left != 1 || pad.bottom != 1 || pad.right != 1) return false;
        var xd = input.ToDenseTensor();
        var wd = weight.ToDenseTensor();
        if (xd.Length < c * h * w || wd.Length < m * c * 9) return false;
        float[]? ba = bias?.ToArray();
        if (ba is not null && ba.Length != m) return false;
        int cbN = c / CB;
        int mbN = m / CB;
        var xs = xd.Buffer.Span;
        var ws = wd.Buffer.Span;
        var actB = new float[cbN * h * w * CB];
        for (int cc = 0; cc < c; cc++)
        {
            int srcBase = cc * h * w;
            int dstBase = cc / CB * h * w * CB + cc % CB;
            for (int i = 0; i < h * w; i++) actB[dstBase + i * CB] = xs[srcBase + i];
        }
        var filtB = new float[mbN * cbN * 9 * CB * CB];
        for (int mb = 0; mb < mbN; mb++)
            for (int cb = 0; cb < cbN; cb++)
                for (int k9 = 0; k9 < 9; k9++)
                    for (int cc = 0; cc < CB; cc++)
                        for (int mm = 0; mm < CB; mm++)
                            filtB[((((mb * cbN + cb) * 9 + k9) * CB) + cc) * CB + mm]
                                = ws[((mb * CB + mm) * c + cb * CB + cc) * 9 + k9];
        var outB = new float[mbN * h * w * CB];
        unsafe
        {
            fixed (float* ab = actB, fb = filtB, db = outB)
                BlockedKernel3x3(ab, fb, db, cbN, mbN, h, w);
        }
        var ys = new float[m * h * w];
        for (int mb = 0; mb < mbN; mb++)
            for (int mm = 0; mm < CB; mm++)
            {
                float bi = ba is null ? 0f : ba[mb * CB + mm];
                int srcBase = (mb * h * w) * CB + mm;
                int dstBase = (mb * CB + mm) * h * w;
                for (int i = 0; i < h * w; i++)
                {
                    float v = outB[srcBase + i * CB] + bi;
                    ys[dstBase + i] = fuseRelu && v < 0f ? 0f : v;
                }
            }
        output = new DenseTensor<float>(new Memory<float>(ys), new[] { n, m, h, w });
        return true;
    }

    static unsafe void BlockedKernel3x3(float* act, float* filt, float* dst, int cbN, int mbN, int h, int w)
    {
        for (int mb = 0; mb < mbN; mb++)
            for (int y = 0; y < h; y++)
            {
                bool yIn = (uint)(y - 1) < (uint)(h - 2);
                int x = 0;
                if (w < 2)
                {
                    for (; x < w; x++) BorderPos(act, filt, dst, mb, cbN, y, x, h, w);
                    continue;
                }
                if (!yIn && h >= 2)
                {
                    // Edge row with room for neighbors on one side: the six
                    // valid taps run vectorized; x borders and remainders
                    // keep the checked path (two rows per layer at most).
                    BorderPos(act, filt, dst, mb, cbN, y, 0, h, w);
                    for (x = 1; x + T <= w - 1; x += T) InteriorTileYEdge(act, filt, dst, mb, cbN, y, x, h, w, y == 0);
                    for (; x < w; x++) BorderPos(act, filt, dst, mb, cbN, y, x, h, w);
                    continue;
                }
                if (!yIn)
                {
                    for (; x < w; x++) BorderPos(act, filt, dst, mb, cbN, y, x, h, w);
                    continue;
                }
                BorderPos(act, filt, dst, mb, cbN, y, 0, h, w);
                for (x = 1; x + T <= w - 1; x += T) InteriorTile(act, filt, dst, mb, cbN, y, x, h, w);
                if (x + 4 <= w - 1)
                {
                    InteriorTileT4(act, filt, dst, mb, cbN, y, x, h, w);
                    x += 4;
                }
                for (; x < w; x++) BorderPos(act, filt, dst, mb, cbN, y, x, h, w);
            }
    }

    /// <summary>Four-position interior tile: same nest as <see cref="InteriorTile"/> with four accumulators, covering middle remainders at interior rate instead of the scalar border path. Only y-interior rows with x+3 inside the map may call it.</summary>
    static unsafe void InteriorTileT4(float* act, float* filt, float* dst, int mb, int cbN, int y, int x, int h, int w)
    {
        var a0 = Vector512<float>.Zero; var a1 = Vector512<float>.Zero;
        var a2 = Vector512<float>.Zero; var a3 = Vector512<float>.Zero;
        for (int cb = 0; cb < cbN; cb++)
            for (int k9 = 0; k9 < 9; k9++)
            {
                int dy = k9 / 3 - 1, dx = k9 % 3 - 1;
                float* fbase = filt + (((mb * cbN + cb) * 9 + k9) * CB) * CB;
                float* abase = act + ((cb * h + y + dy) * w + x + dx) * CB;
                for (int cc = 0; cc < CB; cc++)
                {
                    var fv = Vector512.Load(fbase + cc * CB);
                    a0 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[0 * CB + cc]), a0);
                    a1 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[1 * CB + cc]), a1);
                    a2 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[2 * CB + cc]), a2);
                    a3 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[3 * CB + cc]), a3);
                }
            }
        float* db = dst + ((mb * h + y) * w + x) * CB;
        a0.Store(db + 0 * CB); a1.Store(db + 1 * CB);
        a2.Store(db + 2 * CB); a3.Store(db + 3 * CB);
    }

    /// <summary>Eight-position edge-row tile: the six valid taps of a y-border row with no per-tap bounds checks. Top rows use taps 3-8 (dy 0..1), bottom rows taps 0-5 (dy -1..0). Only x-interior tiles of the matching edge row may call it.</summary>
    static unsafe void InteriorTileYEdge(float* act, float* filt, float* dst, int mb, int cbN, int y, int x, int h, int w, bool topEdge)
    {
        int kLo = topEdge ? 3 : 0;
        int kHi = topEdge ? 9 : 6;
        var a0 = Vector512<float>.Zero; var a1 = Vector512<float>.Zero;
        var a2 = Vector512<float>.Zero; var a3 = Vector512<float>.Zero;
        var a4 = Vector512<float>.Zero; var a5 = Vector512<float>.Zero;
        var a6 = Vector512<float>.Zero; var a7 = Vector512<float>.Zero;
        for (int cb = 0; cb < cbN; cb++)
            for (int k9 = kLo; k9 < kHi; k9++)
            {
                int dy = k9 / 3 - 1, dx = k9 % 3 - 1;
                float* fbase = filt + (((mb * cbN + cb) * 9 + k9) * CB) * CB;
                float* abase = act + ((cb * h + y + dy) * w + x + dx) * CB;
                for (int cc = 0; cc < CB; cc++)
                {
                    var fv = Vector512.Load(fbase + cc * CB);
                    a0 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[0 * CB + cc]), a0);
                    a1 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[1 * CB + cc]), a1);
                    a2 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[2 * CB + cc]), a2);
                    a3 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[3 * CB + cc]), a3);
                    a4 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[4 * CB + cc]), a4);
                    a5 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[5 * CB + cc]), a5);
                    a6 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[6 * CB + cc]), a6);
                    a7 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[7 * CB + cc]), a7);
                }
            }
        float* db = dst + ((mb * h + y) * w + x) * CB;
        a0.Store(db + 0 * CB); a1.Store(db + 1 * CB);
        a2.Store(db + 2 * CB); a3.Store(db + 3 * CB);
        a4.Store(db + 4 * CB); a5.Store(db + 5 * CB);
        a6.Store(db + 6 * CB); a7.Store(db + 7 * CB);
    }

    static unsafe void InteriorTile(float* act, float* filt, float* dst, int mb, int cbN, int y, int x, int h, int w)
    {
        var a0 = Vector512<float>.Zero; var a1 = Vector512<float>.Zero;
        var a2 = Vector512<float>.Zero; var a3 = Vector512<float>.Zero;
        var a4 = Vector512<float>.Zero; var a5 = Vector512<float>.Zero;
        var a6 = Vector512<float>.Zero; var a7 = Vector512<float>.Zero;
        for (int cb = 0; cb < cbN; cb++)
            for (int k9 = 0; k9 < 9; k9++)
            {
                int dy = k9 / 3 - 1, dx = k9 % 3 - 1;
                float* fbase = filt + (((mb * cbN + cb) * 9 + k9) * CB) * CB;
                float* abase = act + ((cb * h + y + dy) * w + x + dx) * CB;
                for (int cc = 0; cc < CB; cc++)
                {
                    var fv = Vector512.Load(fbase + cc * CB);
                    a0 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[0 * CB + cc]), a0);
                    a1 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[1 * CB + cc]), a1);
                    a2 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[2 * CB + cc]), a2);
                    a3 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[3 * CB + cc]), a3);
                    a4 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[4 * CB + cc]), a4);
                    a5 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[5 * CB + cc]), a5);
                    a6 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[6 * CB + cc]), a6);
                    a7 = Avx512F.FusedMultiplyAdd(fv, Vector512.Create(abase[7 * CB + cc]), a7);
                }
            }
        float* db = dst + ((mb * h + y) * w + x) * CB;
        a0.Store(db + 0 * CB); a1.Store(db + 1 * CB);
        a2.Store(db + 2 * CB); a3.Store(db + 3 * CB);
        a4.Store(db + 4 * CB); a5.Store(db + 5 * CB);
        a6.Store(db + 6 * CB); a7.Store(db + 7 * CB);
    }

    static unsafe void BorderPos(float* act, float* filt, float* dst, int mb, int cbN, int y, int x, int h, int w)
    {
        var sum = Vector512<float>.Zero;
        for (int cb = 0; cb < cbN; cb++)
            for (int tap = 0; tap < 9; tap++)
            {
                int iy = y + tap / 3 - 1, ix = x + tap % 3 - 1;
                if ((uint)iy >= (uint)h || (uint)ix >= (uint)w) continue;
                float* av = act + ((cb * h + iy) * w + ix) * CB;
                float* fv = filt + (((mb * cbN + cb) * 9 + tap) * CB) * CB;
                for (int cc = 0; cc < CB; cc++)
                    sum = Avx512F.FusedMultiplyAdd(Vector512.Load(fv + cc * CB), Vector512.Create(av[cc]), sum);
            }
        sum.Store(dst + ((mb * h + y) * w + x) * CB);
    }

    /// <summary>Elementwise add over blocked buffers (region epilogue piece).</summary>
    /// <remarks>Bit-identical to scalar Add on the same values: one rounding per element, no reassociation. The destination may alias either input.</remarks>
    public static void BlockedAdd(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> destination, TensorExecutionOptions? options)
    {
        if (left.Length != right.Length || destination.Length < left.Length)
            throw new ArgumentException("Blocked add requires matching input lengths and a large enough destination.");
        int n = left.Length;
        if ((options?.UseSimd ?? true) && (options?.UseIntrinsics ?? true))
        {
            if (Avx512F.IsSupported)
            {
                int full = n & ~(Vector512<float>.Count - 1);
                int i = 0;
                for (; i < full; i += Vector512<float>.Count)
                    (Vector512.LoadUnsafe(ref MemoryMarshal.GetReference(left), (nuint)i)
                        + Vector512.LoadUnsafe(ref MemoryMarshal.GetReference(right), (nuint)i))
                        .StoreUnsafe(ref MemoryMarshal.GetReference(destination), (nuint)i);
                for (; i < n; i++) destination[i] = left[i] + right[i];
                return;
            }
            if (Avx.IsSupported)
            {
                int full = n & ~(Vector256<float>.Count - 1);
                int i = 0;
                for (; i < full; i += Vector256<float>.Count)
                    (Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(left), (nuint)i)
                        + Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(right), (nuint)i))
                        .StoreUnsafe(ref MemoryMarshal.GetReference(destination), (nuint)i);
                for (; i < n; i++) destination[i] = left[i] + right[i];
                return;
            }
        }
        for (int i = 0; i < n; i++) destination[i] = left[i] + right[i];
    }

    /// <summary>In-place Relu over a blocked buffer (region epilogue piece).</summary>
    /// <remarks>Exact production edge semantics: negatives map to +0 while signed zero and NaN are preserved (only LessThan selects the zero replacement). Bit-identical to production Relu on the same values.</remarks>
    public static void BlockedRelu(Span<float> data, TensorExecutionOptions? options)
    {
        int n = data.Length;
        if ((options?.UseSimd ?? true) && (options?.UseIntrinsics ?? true))
        {
            if (Avx512F.IsSupported)
            {
                var zero = Vector512<float>.Zero;
                int full = n & ~(Vector512<float>.Count - 1);
                int i = 0;
                for (; i < full; i += Vector512<float>.Count)
                {
                    var v = Vector512.LoadUnsafe(ref MemoryMarshal.GetReference(data), (nuint)i);
                    Vector512.ConditionalSelect(Vector512.LessThan(v, zero), zero, v)
                        .StoreUnsafe(ref MemoryMarshal.GetReference(data), (nuint)i);
                }
                for (; i < n; i++)
                {
                    float v = data[i];
                    data[i] = v <= 0f ? (v == 0f ? v : 0f) : v;
                }
                return;
            }
            if (Avx.IsSupported)
            {
                var zero = Vector256<float>.Zero;
                int full = n & ~(Vector256<float>.Count - 1);
                int i = 0;
                for (; i < full; i += Vector256<float>.Count)
                {
                    var v = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(data), (nuint)i);
                    Vector256.ConditionalSelect(Vector256.LessThan(v, zero), zero, v)
                        .StoreUnsafe(ref MemoryMarshal.GetReference(data), (nuint)i);
                }
                for (; i < n; i++)
                {
                    float v = data[i];
                    data[i] = v <= 0f ? (v == 0f ? v : 0f) : v;
                }
                return;
            }
        }
        for (int i = 0; i < n; i++)
        {
            float v = data[i];
            data[i] = v <= 0f ? (v == 0f ? v : 0f) : v;
        }
    }

}
