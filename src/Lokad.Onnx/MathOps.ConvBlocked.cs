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
        var actB = new float[cbN * h * w * CB];
        BlockInput(xd.Buffer.Span, actB, c, h, w);
        var filtB = PackBlockedFilter(wd.Buffer.Span, m, c);
        var outB = new float[mbN * h * w * CB];
        BlockedConvCore(actB, filtB, outB, cbN, mbN, h, w);
        var ys = new float[m * h * w];
        UnblockOutput(outB, ys, ba, m, h, w, fuseRelu);
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
                if (!yIn || w < 2)
                {
                    for (; x < w; x++) BorderPos(act, filt, dst, mb, cbN, y, x, h, w);
                    continue;
                }
                BorderPos(act, filt, dst, mb, cbN, y, 0, h, w);
                for (x = 1; x + T <= w - 1; x += T) InteriorTile(act, filt, dst, mb, cbN, y, x, h, w);
                for (; x < w; x++) BorderPos(act, filt, dst, mb, cbN, y, x, h, w);
            }
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


    /// <summary>Adds a per-channel bias over a blocked buffer (region epilogue piece).</summary>
    /// <remarks>Blocked layout groups sixteen lanes of channel mb*16+mm at every spatial position, so one contiguous sixteen-wide bias load serves a whole channel block across all spatial positions. Bit-identical to scalar bias addition. The destination may alias the data input. Spatial is the H-by-W position count; channels come from the bias length.</remarks>
    public static void BlockedBiasAdd(ReadOnlySpan<float> data, ReadOnlySpan<float> bias, Span<float> destination, int spatial, TensorExecutionOptions? options)
    {
        int m = bias.Length;
        if (m % CB != 0 || spatial <= 0) throw new ArgumentException("Blocked bias add needs a multiple-of-16 channel count and a positive spatial extent.");
        if (data.Length != m * spatial || destination.Length < data.Length)
            throw new ArgumentException("Blocked bias add requires data covering every channel and a large enough destination.");
        if ((options?.UseSimd ?? true) && (options?.UseIntrinsics ?? true))
        {
            if (Avx512F.IsSupported)
            {
                for (int mb = 0; mb < m / CB; mb++)
                {
                    var vb = Vector512.LoadUnsafe(ref MemoryMarshal.GetReference(bias), (nuint)(mb * CB));
                    int base_ = mb * spatial * CB;
                    for (int s = 0; s < spatial; s++)
                    {
                        int at = base_ + s * CB;
                        (Vector512.LoadUnsafe(ref MemoryMarshal.GetReference(data), (nuint)at) + vb)
                            .StoreUnsafe(ref MemoryMarshal.GetReference(destination), (nuint)at);
                    }
                }
                return;
            }
            if (Avx.IsSupported)
            {
                for (int mb = 0; mb < m / CB; mb++)
                {
                    var vb0 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(bias), (nuint)(mb * CB));
                    var vb1 = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(bias), (nuint)(mb * CB + 8));
                    int base_ = mb * spatial * CB;
                    for (int s = 0; s < spatial; s++)
                    {
                        int at = base_ + s * CB;
                        (Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(data), (nuint)at) + vb0)
                            .StoreUnsafe(ref MemoryMarshal.GetReference(destination), (nuint)at);
                        (Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(data), (nuint)(at + 8)) + vb1)
                            .StoreUnsafe(ref MemoryMarshal.GetReference(destination), (nuint)(at + 8));
                    }
                }
                return;
            }
        }
        for (int mb = 0; mb < m / CB; mb++)
            for (int s = 0; s < spatial; s++)
                for (int mm = 0; mm < CB; mm++)
                    destination[(mb * spatial + s) * CB + mm] = data[(mb * spatial + s) * CB + mm] + bias[mb * CB + mm];
    }


    /// <summary>Blocks one NCHW channel plane set into NCHWc16 layout.</summary>
    internal static void BlockInput(ReadOnlySpan<float> xs, Span<float> dst, int c, int h, int w)
    {
        for (int cc = 0; cc < c; cc++)
        {
            int srcBase = cc * h * w;
            int dstBase = cc / CB * h * w * CB + cc % CB;
            for (int i = 0; i < h * w; i++) dst[dstBase + i * CB] = xs[srcBase + i];
        }
    }

    /// <summary>Packs one [M,C,3,3] filter set into the blocked micro-tile layout with output lanes innermost.</summary>
    internal static float[] PackBlockedFilter(ReadOnlySpan<float> ws, int m, int c)
    {
        int cbN = c / CB;
        int mbN = m / CB;
        var filtB = new float[mbN * cbN * 9 * CB * CB];
        for (int mb = 0; mb < mbN; mb++)
            for (int cb = 0; cb < cbN; cb++)
                for (int k9 = 0; k9 < 9; k9++)
                    for (int cc = 0; cc < CB; cc++)
                        for (int mm = 0; mm < CB; mm++)
                            filtB[((((mb * cbN + cb) * 9 + k9) * CB) + cc) * CB + mm]
                                = ws[((mb * CB + mm) * c + cb * CB + cc) * 9 + k9];
        return filtB;
    }

    /// <summary>Runs the vector blocked kernel over pre-blocked buffers; the caller must have gated SIMD execution.</summary>
    internal static void BlockedConvCore(float[] actB, float[] filtB, float[] outB, int cbN, int mbN, int h, int w)
    {
        unsafe
        {
            fixed (float* ab = actB, fb = filtB, db = outB)
                BlockedKernel3x3(ab, fb, db, cbN, mbN, h, w);
        }
    }

    /// <summary>Runs the scalar blocked kernel over pre-blocked buffers.</summary>
    /// <remarks>Same traversal and per-lane summation order as the vector kernel, with single-rounding fused multiply-add throughout, so both lanes agree bit for bit.</remarks>
    internal static void BlockedConvCoreScalar(float[] actB, float[] filtB, float[] outB, int cbN, int mbN, int h, int w)
    {
        unsafe
        {
            fixed (float* act = actB, filt = filtB, dst = outB)
            {
                for (int mb = 0; mb < mbN; mb++)
                    for (int y = 0; y < h; y++)
                        for (int x = 0; x < w; x++)
                            for (int mm = 0; mm < CB; mm++)
                            {
                                float acc = 0f;
                                for (int cb = 0; cb < cbN; cb++)
                                    for (int k9 = 0; k9 < 9; k9++)
                                    {
                                        int iy = y + k9 / 3 - 1, ix = x + k9 % 3 - 1;
                                        if ((uint)iy >= (uint)h || (uint)ix >= (uint)w) continue;
                                        float* av = act + ((cb * h + iy) * w + ix) * CB;
                                        float* fv = filt + (((mb * cbN + cb) * 9 + k9) * CB) * CB;
                                        for (int cc = 0; cc < CB; cc++)
                                            acc = MathF.FusedMultiplyAdd(fv[cc * CB + mm], av[cc], acc);
                                    }
                                dst[((mb * h + y) * w + x) * CB + mm] = acc;
                            }
            }
        }
    }

    /// <summary>Runs the blocked kernel over pre-blocked buffers, vector when the options and ISA allow, scalar otherwise.</summary>
    internal static void RunBlockedConv(float[] actB, float[] filtB, float[] outB, int cbN, int mbN, int h, int w, TensorExecutionOptions? options)
    {
        if ((options?.UseSimd ?? true) && (options?.UseIntrinsics ?? true) && Avx512F.IsSupported)
            BlockedConvCore(actB, filtB, outB, cbN, mbN, h, w);
        else
            BlockedConvCoreScalar(actB, filtB, outB, cbN, mbN, h, w);
    }

    /// <summary>Converts one blocked output to NCHW, applying bias and an optional Relu epilogue.</summary>
    internal static void UnblockOutput(ReadOnlySpan<float> outB, Span<float> ys, float[]? bias, int m, int h, int w, bool fuseRelu)
    {
        int mbN = m / CB;
        for (int mb = 0; mb < mbN; mb++)
            for (int mm = 0; mm < CB; mm++)
            {
                float bi = bias is null ? 0f : bias[mb * CB + mm];
                int srcBase = (mb * h * w) * CB + mm;
                int dstBase = (mb * CB + mm) * h * w;
                for (int i = 0; i < h * w; i++)
                {
                    float v = outB[srcBase + i * CB] + bi;
                    ys[dstBase + i] = fuseRelu && v < 0f ? 0f : v;
                }
            }
    }

}
