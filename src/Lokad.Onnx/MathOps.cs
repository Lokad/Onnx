using System;
using System.Numerics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace Lokad.Onnx;

public class MathOps
{
    public struct PadInfo
    {
        public int h;
        public int w;
        public int top;
        public int left;
        public int right;
        public int bottom;
    }

    public enum PadType
    {
        Valid,
        SameUpper,
        SameLower,
        Value
    }

    public enum ResizeMode
    {
        Nearest,
        Linear,
        Cubic
    }

    public enum ResizeCoordinateTransformation
    {
        HalfPixel,
        AlignCorners,
        Asymmetric
    }

    public enum ResizeNearestMode
    {
        Floor,
        Ceil,
        RoundPreferFloor,
        RoundPreferCeil
    }

    public struct Conv2DOutputInfo
    {
        public PadInfo PadInfo;
        public int[] Shape;
    }

    // Integer division for output geometry. Validated callers pass positive
    // strides; both helpers stay exact for every numerator, including
    // shrunken dims, where truncating division alone would round the wrong way.
    static int CeilDivPositive(int numerator, int stride)
    {
        int quotient = numerator / stride;
        int remainder = numerator % stride;
        if (remainder == 0) return quotient;
        return numerator > 0 ? quotient + 1 : quotient;
    }

    static int FloorDivPositive(int numerator, int stride)
    {
        int quotient = numerator / stride;
        int remainder = numerator % stride;
        if (remainder == 0 || numerator > 0) return quotient;
        return quotient - 1;
    }

    // Splits total padding around one axis: the odd cell goes to the end,
    // except for SAME_LOWER where it goes to the start (ONNX auto_pad).
    static (int start, int end) SplitPad(int total, bool extraAtStart)
    {
        int start = extraAtStart ? (total + 1) / 2 : total / 2;
        return (start, total - start);
    }

    // Output geometry per ONNX auto_pad mode: VALID pads nothing, SAME modes
    // preserve ceil(input/stride) outputs distributing the deficit, and VALUE
    // pads uniformly, delegating to the explicit formula below.
    public static Conv2DOutputInfo GetConv2DOutputInfo(PadType pad, int inHeight, int inWidth, int strideHeight, int strideWidth, int filterHeight, int filterWidth, int? padValue)
    {
        var padInfo = new PadInfo();
        int outHeight = 0;
        int outWidth = 0;
        switch (pad)
        {
            case PadType.Valid:
                outHeight = CeilDivPositive(inHeight - filterHeight + 1, strideHeight);
                outWidth = CeilDivPositive(inWidth - filterWidth + 1, strideWidth);
                break;

            case PadType.SameUpper:
            case PadType.SameLower:
                bool extraAtStart = pad == PadType.SameLower;
                outHeight = CeilDivPositive(inHeight, strideHeight);
                outWidth = CeilDivPositive(inWidth, strideWidth);
                int padH = Math.Max(0, (outHeight - 1) * strideHeight + filterHeight - inHeight);
                int padW = Math.Max(0, (outWidth - 1) * strideWidth + filterWidth - inWidth);
                var (top, bottom) = SplitPad(padH, extraAtStart);
                var (left, right) = SplitPad(padW, extraAtStart);
                padInfo = new PadInfo { top = top, bottom = bottom, left = left, right = right, h = padH, w = padW };
                break;

            case PadType.Value:
                if (padValue == null) throw new ArgumentNullException(nameof(padValue));
                padInfo = new PadInfo { top = padValue.Value, bottom = padValue.Value, left = padValue.Value, right = padValue.Value, h = 2 * padValue.Value, w = 2 * padValue.Value };
                var outShape = GetConv2DOutputShape(new int[] { inHeight, inWidth }, filterHeight, filterWidth, strideHeight, strideWidth, padInfo.h, padInfo.w);
                outHeight = outShape[0];
                outWidth = outShape[1];
                break;
        }
        return new Conv2DOutputInfo { PadInfo = padInfo, Shape = new int[] { outHeight, outWidth } };
    }

    // Explicit-pads output geometry per the ONNX formula
    // floor((input + pad_total - kernel) / stride) + 1 per axis.
    public static int[] GetConv2DOutputShape(int[] inputShape, int kernelHeight, int kernelWidth, int strideY, int strideX, int padY, int padX)
    {
        return new int[]
        {
            FloorDivPositive(inputShape[0] - kernelHeight + padY, strideY) + 1,
            FloorDivPositive(inputShape[1] - kernelWidth + padX, strideX) + 1,
        };
    }

    public static int GetConv2DEffectiveFilterSize(int filterSize, int dilation) => dilation <= 1 ? filterSize : filterSize + (filterSize - 1) * (dilation - 1);

    /// <summary>
    /// Matrix multiplication.
    /// </summary>
    /// <param name="M">A rows.</param>
    /// <param name="N">A columns.</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="B">Right matrix.</param>
    /// <param name="C">Result matrix.</param>
    public unsafe static void mm(int M,
                          int N,
                          int K,
                          int* A,
                          int* B,
                          int* C)
    {
        for (int i = 0; i < M; ++i)
        {
            var rowA = A + i * N;
            var rowC = C + i * K;
            for (int k = 0; k < K; ++k)
            {
                var total = rowC[k];
                for (int j = 0; j < N; ++j)
                {
                    total += rowA[j] * B[j * K + k];
                }
                rowC[k] = total;
            }
        }
    }

    /// <summary>
    /// Matrix multiplication.
    /// </summary>
    /// <param name="M">A rows.</param>
    /// <param name="N">A columns.</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="B">Right matrix.</param>
    /// <param name="C">Result matrix.</param>
    public unsafe static void mm(int M,
                          int N,
                          int K,
                          double* A,
                          double* B,
                          double* C)
    {
        for (int i = 0; i < M; ++i)
        {
            var rowA = A + i * N;
            var rowC = C + i * K;
            for (int k = 0; k < K; ++k)
            {
                var total = rowC[k];
                for (int j = 0; j < N; ++j)
                {
                    total += rowA[j] * B[j * K + k];
                }
                rowC[k] = total;
            }
        }
    }

    /// <summary>
    /// Matrix multiplication.
    /// </summary>
    /// <param name="M">A rows.</param>
    /// <param name="N">A columns.</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="B">Right matrix.</param>
    /// <param name="C">Result matrix.</param>
    public unsafe static void mm(int M,
                          int N,
                          int K,
                          float* A,
                          float* B,
                          float* C)
    {
        for (int i = 0; i < M; ++i)
        {
            var rowA = A + i * N;
            var rowC = C + i * K;
            for (int k = 0; k < K; ++k)
            {
                var total = rowC[k];
                for (int j = 0; j < N; ++j)
                {
                    total += rowA[j] * B[j * K + k];
                }
                rowC[k] = total;
            }
        }
    }

    /// <summary>
    /// Matrix multiplication.
    /// </summary>
    /// <param name="M">A rows.</param>
    /// <param name="N">A columns.</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="B">Right matrix.</param>
    /// <param name="C">Result matrix.</param>
    public unsafe static void mm_unsafe_vectorized(int M,
                          int N,
                          int K,
                          int* A,
                          int* B,
                          int* C)
    {
        var v = Vector<int>.Count;
        for (int i = 0; i < M; i++)
        {
            var Ap = A + i * N;
            var Cp = C + i * K;
            for (int j = 0; j < N; ++j)
            {
                var a = Ap[j];
                var Bp = B + j * K;
                var av = new Vector<int>(a);
                int ceiling = (K / Vector<int>.Count) * Vector<int>.Count;
                var Bkv = MemoryMarshal.Cast<int, Vector<int>>(new Span<int>(Bp, K));
                var Ckv = MemoryMarshal.Cast<int, Vector<int>>(new Span<int>(Cp, K));
                for (int k = 0; k < Bkv.Length; k++)
                {
                    Ckv[k] = Ckv[k] + av * Bkv[k];
                }
                for (int k = ceiling; k < K; k++)
                {
                    Cp[k] += a * Bp[k];
                }
            }
        }
    }

    /// <summary>
    /// Matrix multiplication.
    /// </summary>
    /// <param name="M">A rows.</param>
    /// <param name="N">A columns.</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="B">Right matrix.</param>
    /// <param name="C">Result matrix.</param>
    public unsafe static void mm_unsafe_vectorized(int M,
                          int N,
                          int K,
                          float* A,
                          float* B,
                          float* C)
    {
        var v = Vector<float>.Count;
        for (int i = 0; i < M; i++)
        {
            var Ap = A + i * N;
            var Cp = C + i * K;
            for (int j = 0; j < N; ++j)
            {
                var a = Ap[j];
                var Bp = B + j * K;
                var av = new Vector<float>(a);
                int ceiling = (K / Vector<float>.Count) * Vector<float>.Count;
                var Bkv = MemoryMarshal.Cast<float, Vector<float>>(new Span<float>(Bp, K));
                var Ckv = MemoryMarshal.Cast<float, Vector<float>>(new Span<float>(Cp, K));
                for (int k = 0; k < Bkv.Length; k++)
                {
                    Ckv[k] = Ckv[k] + av * Bkv[k];
                }
                for (int k = ceiling; k < K; k++)
                {
                    Cp[k] += a * Bp[k];
                }
            }
        }
    }

    /// <summary>
    /// Matrix multiplication.
    /// </summary>
    /// <param name="M">A rows.</param>
    /// <param name="N">A columns.</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="B">Right matrix.</param>
    /// <param name="C">Result matrix.</param>
    public unsafe static void mm_unsafe_vectorized(int M,
                          int N,
                          int K,
                          double* A,
                          double* B,
                          double* C)
    {
        var v = Vector<double>.Count;
        for (int i = 0; i < M; i++)
        {
            var Ap = A + i * N;
            var Cp = C + i * K;
            for (int j = 0; j < N; ++j)
            {
                var a = Ap[j];
                var Bp = B + j * K;
                var av = new Vector<double>(a);
                int ceiling = (K / Vector<double>.Count) * Vector<double>.Count;
                var Bkv = MemoryMarshal.Cast<double, Vector<double>>(new Span<double>(Bp, K));
                var Ckv = MemoryMarshal.Cast<double, Vector<double>>(new Span<double>(Cp, K));
                for (int k = 0; k < Bkv.Length; k++)
                {
                    Ckv[k] = Ckv[k] + av * Bkv[k];
                }
                for (int k = ceiling; k < K; k++)
                {
                    Cp[k] += a * Bp[k];
                }
            }
        }
    }


    /// <summary>
    /// Matrix multiplication.
    /// </summary>
    /// <param name="M">A rows.</param>
    /// <param name="N">A columns.</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="B">Right matrix.</param>
    /// <param name="C">Result matrix.</param>
    public unsafe static void mm_unsafe_vectorized_intrinsics(int M,
                          int N,
                          int K,
                          float* A,
                          float* B,
                          float* C)
    {

        for (int i = 0; i < M; i++)
        {
            var Ap = A + i * N;
            var Cp = C + i * K;
            for (int j = 0; j < N; ++j)
            {
                var a = Ap[j];
                var Bp = B + j * K;
                var av = Vector256.Create(a);
                int ceiling = (K / Vector256<float>.Count) * Vector256<float>.Count;
                var Bkv = MemoryMarshal.Cast<float, Vector256<float>>(new Span<float>(Bp, K));
                var Ckv = MemoryMarshal.Cast<float, Vector256<float>>(new Span<float>(Cp, K));
                for (int k = 0; k < Bkv.Length; k++)
                {
                    Ckv[k] = Fma.MultiplyAdd(Bkv[k], av, Ckv[k]);
                }
                for (int k = ceiling; k < K; k++)
                {
                    Cp[k] += a * Bp[k];
                }
            }
        }
    }

    /// <summary>
    /// Matrix multiplication.
    /// </summary>
    /// <param name="M">A rows.</param>
    /// <param name="N">A columns.</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="B">Right matrix.</param>
    /// <param name="C">Result matrix.</param>
    public unsafe static void mm_unsafe_vectorized_intrinsics_2x4(int M,
                          int N,
                          int K,
                          float* A,
                          float* B,
                          float* C)
    {

        // Optimizations: 
        //
        //  - Process the output two lines at a time (to halve the number of loads from B),
        //    and assume that M is even to avoid having dedicated code for a last odd line.
        //
        //  - Manually 4x unroll the inner loop, and assume that K is a multiple of 32 
        //    to avoid having dedicated code for the remainder of lines.
        //
        //  - Use pointers and casts instead of span, in order to avoid overhead from bounds
        //    checking. The #if DEBUG lines are here to protect against out-of-bounds in 
        //    debug mode.

        if (M % 2 != 0)
            throw new ArgumentException(nameof(M));

        // The unrolled body covers the largest multiple of 32 columns; the
        // vector-then-scalar tail below covers the remainder in the same
        // ascending order, so results match the non-unrolled kernel bitwise.
        int blocked = K - (K % (4 * Vector256<float>.Count));

#if DEBUG
        var Aend = A + M * N;
        var Bend = B + N * K;
        var Cend = C + M * K;
#endif

        for (int i = 0; i < M; i += 2)
        {
            var Ap1 = A + i * N;
            var Ap2 = Ap1 + N;

            var Cp1 = C + i * K;
            var Cp2 = Cp1 + K;

            for (int j = 0; j < N; ++j)
            {
#if DEBUG
                Debug.Assert(Ap1 + j < Aend);
                Debug.Assert(Ap2 + j < Aend);
#endif
                var av1 = Vector256.Create(Ap1[j]);
                var av2 = Vector256.Create(Ap2[j]);

                var Bp = B + j * K;

                var Bpv = (Vector256<float>*)Bp;
                var Cpv1 = (Vector256<float>*)Cp1;
                var Cpv2 = (Vector256<float>*)Cp2;
                var Bepv = (Vector256<float>*)(Bp + blocked);

#if DEBUG
                Debug.Assert(Bepv <= Bend);
#endif

                while (Bpv < Bepv)
                {
#if DEBUG
                    Debug.Assert(Cpv1 + 3 < Cend);
                    Debug.Assert(Cpv2 + 3 < Cend);
#endif
                    Vector256<float> bv;

                    bv = Bpv[0];
                    Cpv1[0] = Fma.MultiplyAdd(bv, av1, Cpv1[0]);
                    Cpv2[0] = Fma.MultiplyAdd(bv, av2, Cpv2[0]);

                    bv = Bpv[1];
                    Cpv1[1] = Fma.MultiplyAdd(bv, av1, Cpv1[1]);
                    Cpv2[1] = Fma.MultiplyAdd(bv, av2, Cpv2[1]);

                    bv = Bpv[2];
                    Cpv1[2] = Fma.MultiplyAdd(bv, av1, Cpv1[2]);
                    Cpv2[2] = Fma.MultiplyAdd(bv, av2, Cpv2[2]);

                    bv = Bpv[3];
                    Cpv1[3] = Fma.MultiplyAdd(bv, av1, Cpv1[3]);
                    Cpv2[3] = Fma.MultiplyAdd(bv, av2, Cpv2[3]);

                    Bpv += 4;
                    Cpv1 += 4;
                    Cpv2 += 4;
                }
                int rem = K - blocked;
                if (rem > 0)
                {
                    float a1 = Ap1[j];
                    float a2 = Ap2[j];
                    var rB = (Vector256<float>*)(Bp + blocked);
                    var rC1 = (Vector256<float>*)(Cp1 + blocked);
                    var rC2 = (Vector256<float>*)(Cp2 + blocked);
                    int rv = rem / Vector256<float>.Count;
                    for (int t = 0; t < rv; t++)
                    {
                        rC1[t] = Fma.MultiplyAdd(rB[t], av1, rC1[t]);
                        rC2[t] = Fma.MultiplyAdd(rB[t], av2, rC2[t]);
                    }
                    for (int k = blocked + rv * Vector256<float>.Count; k < K; k++)
                    {
                        Cp1[k] += a1 * Bp[k];
                        Cp2[k] += a2 * Bp[k];
                    }
                }
            }
        }
    }

    /// <summary>
    /// Matrix multiplication with register-tiled output accumulation.
    /// </summary>
    /// <param name="M">A rows (must be even).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="B">Right matrix.</param>
    /// <param name="C">Result matrix.</param>
    /// <remarks>
    /// Same 2-row by 32-column blocking as the unrolled kernel above, but each
    /// tile accumulates in locals across the reduction axis with one load and
    /// one store per tile instead of per step. Full tiles and the vector tail
    /// keep the per-element operation order of the unrolled kernel, and the
    /// scalar tail uses the same reduction-major order, so results agree with
    /// it bit-wise on every shape (see kernel agreement tests).
    /// </remarks>
    public unsafe static void mm_unsafe_vectorized_intrinsics_2x4tiled(int M,
                          int N,
                          int K,
                          float* A,
                          float* B,
                          float* C)
    {
        if (M % 2 != 0)
            throw new ArgumentException(nameof(M));

        int blocked = K - (K % (4 * Vector256<float>.Count));

        for (int i = 0; i < M; i += 2)
        {
            var Ap1 = A + i * N;
            var Ap2 = Ap1 + N;

            var Cp1 = C + i * K;
            var Cp2 = Cp1 + K;

            for (int kb = 0; kb < blocked; kb += 4 * Vector256<float>.Count)
            {
                var Cpv1 = (Vector256<float>*)(Cp1 + kb);
                var Cpv2 = (Vector256<float>*)(Cp2 + kb);
                Vector256<float> c00 = Cpv1[0];
                Vector256<float> c01 = Cpv1[1];
                Vector256<float> c02 = Cpv1[2];
                Vector256<float> c03 = Cpv1[3];
                Vector256<float> c10 = Cpv2[0];
                Vector256<float> c11 = Cpv2[1];
                Vector256<float> c12 = Cpv2[2];
                Vector256<float> c13 = Cpv2[3];
                for (int j = 0; j < N; ++j)
                {
                    var av1 = Vector256.Create(Ap1[j]);
                    var av2 = Vector256.Create(Ap2[j]);
                    var Bpv = (Vector256<float>*)(B + j * K + kb);
                    c00 = Fma.MultiplyAdd(Bpv[0], av1, c00);
                    c10 = Fma.MultiplyAdd(Bpv[0], av2, c10);
                    c01 = Fma.MultiplyAdd(Bpv[1], av1, c01);
                    c11 = Fma.MultiplyAdd(Bpv[1], av2, c11);
                    c02 = Fma.MultiplyAdd(Bpv[2], av1, c02);
                    c12 = Fma.MultiplyAdd(Bpv[2], av2, c12);
                    c03 = Fma.MultiplyAdd(Bpv[3], av1, c03);
                    c13 = Fma.MultiplyAdd(Bpv[3], av2, c13);
                }
                Cpv1[0] = c00;
                Cpv1[1] = c01;
                Cpv1[2] = c02;
                Cpv1[3] = c03;
                Cpv2[0] = c10;
                Cpv2[1] = c11;
                Cpv2[2] = c12;
                Cpv2[3] = c13;
            }
            int rem = K - blocked;
            if (rem > 0)
            {
                var rC1 = (Vector256<float>*)(Cp1 + blocked);
                var rC2 = (Vector256<float>*)(Cp2 + blocked);
                int rv = rem / Vector256<float>.Count;
                for (int t = 0; t < rv; t++)
                {
                    Vector256<float> c1 = rC1[t];
                    Vector256<float> c2 = rC2[t];
                    for (int j = 0; j < N; ++j)
                    {
                        var Bpv = (Vector256<float>*)(B + j * K + blocked);
                        c1 = Fma.MultiplyAdd(Bpv[t], Vector256.Create(Ap1[j]), c1);
                        c2 = Fma.MultiplyAdd(Bpv[t], Vector256.Create(Ap2[j]), c2);
                    }
                    rC1[t] = c1;
                    rC2[t] = c2;
                }
                // Scalar tail keeps the reduction-major order of the
                // unrolled kernel (accumulate into the destination per step)
                // so results agree with it bit-wise on every shape.
                for (int j = 0; j < N; ++j)
                {
                    float a1 = Ap1[j];
                    float a2 = Ap2[j];
                    var Brow = B + j * K;
                    for (int k = blocked + rv * Vector256<float>.Count; k < K; k++)
                    {
                        Cp1[k] += a1 * Brow[k];
                        Cp2[k] += a2 * Brow[k];
                    }
                }
            }
        }
    }

    /// <summary>
    /// Packs B into 32-column panels laid out contiguously, with any tail
    /// columns appended row-major. Panel relocation is exact, so a kernel
    /// reading panels computes the same per-element order as the tiled kernel.
    /// </summary>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="B">Right matrix, row-major.</param>
    /// <param name="P">Destination with room for N times K floats.</param>
    public unsafe static void PackPanelsB(int N,
                          int K,
                          float* B,
                          float* P)
    {
        int panel = 4 * Vector256<float>.Count;
        int blocked = K - (K % panel);
        bool wide512 = Avx512F.IsSupported;
        bool wide256 = Avx.IsSupported;
        for (int kb = 0; kb < blocked; kb += panel)
        {
            float* dst = P + (kb / panel) * N * panel;
            for (int j = 0; j < N; j++)
            {
                float* src = B + j * K + kb;
                float* d = dst + j * panel;
                if (wide512)
                {
                    ((Vector512<float>*)d)[0] = ((Vector512<float>*)src)[0];
                    ((Vector512<float>*)d)[1] = ((Vector512<float>*)src)[1];
                }
                else if (wide256)
                {
                    ((Vector256<float>*)d)[0] = ((Vector256<float>*)src)[0];
                    ((Vector256<float>*)d)[1] = ((Vector256<float>*)src)[1];
                    ((Vector256<float>*)d)[2] = ((Vector256<float>*)src)[2];
                    ((Vector256<float>*)d)[3] = ((Vector256<float>*)src)[3];
                }
                else
                {
                    for (int kk = 0; kk < panel; kk++) d[kk] = src[kk];
                }
            }
        }
        int rem = K - blocked;
        float* tail = P + (blocked / (4 * Vector256<float>.Count)) * N * (4 * Vector256<float>.Count);
        for (int j = 0; j < N; j++)
        {
            float* src = B + j * K + blocked;
            float* dst = tail + j * rem;
            for (int k = 0; k < rem; k++) dst[k] = src[k];
        }
    }

    /// <summary>
    /// Register-tiled matrix multiplication reading panel-packed B.
    /// </summary>
    /// <param name="M">A rows (must be even).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="P">Panel-packed right matrix from PackPanelsB.</param>
    /// <param name="C">Result matrix.</param>
    /// <remarks>
    /// Same tiles and per-element order as the tiled kernel, so results agree
    /// with it bit-wise; panels only replace strided row jumps with sequential
    /// reads. Tails read the appended row-major tail block in the same order.
    /// </remarks>
    /// <param name="overwrite">True when C holds uninitialized data that every element computation overwrites; false accumulates onto the existing contents, which the caller must have zeroed.</param>
    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe static void mm_unsafe_vectorized_intrinsics_2x4packed(int M,
                          int N,
                          int K,
                          float* A,
                          float* P,
                          float* C,
                          bool overwrite)
    {
        if (M % 2 != 0)
            throw new ArgumentException(nameof(M));

        int blocked = K - (K % (4 * Vector256<float>.Count));
        int tiles = blocked / (4 * Vector256<float>.Count);

        // Kb panels lead so each panel is fetched once and stays L1/L2 resident
        // across every row group; per-element FMA order matches the row-led nest
        // exactly, so results agree bit-wise with the tiled kernel.
        for (int tb = 0; tb < tiles; tb++)
        {
            int kb = tb * (4 * Vector256<float>.Count);
            float* panel = P + tb * N * (4 * Vector256<float>.Count);
            mm_2x4packed_tile(M, N, A, panel, C, K, kb, overwrite);
        }
        int rem = K - blocked;
        if (rem > 0)
        {
            float* T = P + tiles * N * (4 * Vector256<float>.Count);
            int rv = rem / Vector256<float>.Count;
            for (int tt = 0; tt < rv; tt++)
            {
                for (int i = 0; i < M; i += 2)
                {
                    var Ap1 = A + i * N;
                    var Ap2 = Ap1 + N;
                    var rC1 = (Vector256<float>*)(C + i * K + blocked);
                    var rC2 = (Vector256<float>*)(C + (i + 1) * K + blocked);
                    Vector256<float> c1 = overwrite ? Vector256<float>.Zero : rC1[tt];
                    Vector256<float> c2 = overwrite ? Vector256<float>.Zero : rC2[tt];
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
            if (tail > 0)
            for (int i = 0; i < M; i += 2)
            {
                var Ap1 = A + i * N;
                var Ap2 = Ap1 + N;
                var Cp1 = C + i * K + blocked + vcols;
                var Cp2 = Cp1 + K;
                switch (tail)
                {
                    case 1:
                    {
                        float c1 = overwrite ? 0f : Cp1[0], c2 = overwrite ? 0f : Cp2[0];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c1 += Ap1[j] * t[0]; c2 += Ap2[j] * t[0]; }
                        Cp1[0] = c1; Cp2[0] = c2;
                        break;
                    }
                    case 2:
                    {
                        float c10 = overwrite ? 0f : Cp1[0], c11 = overwrite ? 0f : Cp1[1], c20 = overwrite ? 0f : Cp2[0], c21 = overwrite ? 0f : Cp2[1];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; }
                        Cp1[0] = c10; Cp1[1] = c11; Cp2[0] = c20; Cp2[1] = c21;
                        break;
                    }
                    case 3:
                    {
                        float c10 = overwrite ? 0f : Cp1[0], c11 = overwrite ? 0f : Cp1[1], c12 = overwrite ? 0f : Cp1[2], c20 = overwrite ? 0f : Cp2[0], c21 = overwrite ? 0f : Cp2[1], c22 = overwrite ? 0f : Cp2[2];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; }
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22;
                        break;
                    }
                    case 4:
                    {
                        float c10 = overwrite ? 0f : Cp1[0], c11 = overwrite ? 0f : Cp1[1], c12 = overwrite ? 0f : Cp1[2], c13 = overwrite ? 0f : Cp1[3];
                        float c20 = overwrite ? 0f : Cp2[0], c21 = overwrite ? 0f : Cp2[1], c22 = overwrite ? 0f : Cp2[2], c23 = overwrite ? 0f : Cp2[3];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; }
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23;
                        break;
                    }
                    case 5:
                    {
                        float c10 = overwrite ? 0f : Cp1[0], c11 = overwrite ? 0f : Cp1[1], c12 = overwrite ? 0f : Cp1[2], c13 = overwrite ? 0f : Cp1[3], c14 = overwrite ? 0f : Cp1[4];
                        float c20 = overwrite ? 0f : Cp2[0], c21 = overwrite ? 0f : Cp2[1], c22 = overwrite ? 0f : Cp2[2], c23 = overwrite ? 0f : Cp2[3], c24 = overwrite ? 0f : Cp2[4];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; }
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13; Cp1[4] = c14;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp2[4] = c24;
                        break;
                    }
                    case 6:
                    {
                        float c10 = overwrite ? 0f : Cp1[0], c11 = overwrite ? 0f : Cp1[1], c12 = overwrite ? 0f : Cp1[2], c13 = overwrite ? 0f : Cp1[3], c14 = overwrite ? 0f : Cp1[4], c15 = overwrite ? 0f : Cp1[5];
                        float c20 = overwrite ? 0f : Cp2[0], c21 = overwrite ? 0f : Cp2[1], c22 = overwrite ? 0f : Cp2[2], c23 = overwrite ? 0f : Cp2[3], c24 = overwrite ? 0f : Cp2[4], c25 = overwrite ? 0f : Cp2[5];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; c15 += Ap1[j] * t[5]; c25 += Ap2[j] * t[5]; }
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13; Cp1[4] = c14; Cp1[5] = c15;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp2[4] = c24; Cp2[5] = c25;
                        break;
                    }
                    default:
                    {
                        float c10 = overwrite ? 0f : Cp1[0], c11 = overwrite ? 0f : Cp1[1], c12 = overwrite ? 0f : Cp1[2], c13 = overwrite ? 0f : Cp1[3], c14 = overwrite ? 0f : Cp1[4], c15 = overwrite ? 0f : Cp1[5], c16 = overwrite ? 0f : Cp1[6];
                        float c20 = overwrite ? 0f : Cp2[0], c21 = overwrite ? 0f : Cp2[1], c22 = overwrite ? 0f : Cp2[2], c23 = overwrite ? 0f : Cp2[3], c24 = overwrite ? 0f : Cp2[4], c25 = overwrite ? 0f : Cp2[5], c26 = overwrite ? 0f : Cp2[6];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; c15 += Ap1[j] * t[5]; c25 += Ap2[j] * t[5]; c16 += Ap1[j] * t[6]; c26 += Ap2[j] * t[6]; }
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13; Cp1[4] = c14; Cp1[5] = c15; Cp1[6] = c16;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp2[4] = c24; Cp2[5] = c25; Cp2[6] = c26;
                        break;
                    }
                }
            }
        }
    }
    /// <summary>
    /// One full 32-column tile of the 2-row packed nest across all of
    /// its row groups. Extracted for the tiled row-group composer;
    /// the full method calls it per tile in the same order,
    /// so results match bit for bit.
    /// </summary>
    /// <param name="M">A rows (must be even).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="panel">One packed panel tile.</param>
    /// <param name="C">Result matrix.</param>
    /// <param name="K">B columns.</param>
    /// <param name="kb">Leading output column of this tile.</param>
    /// <param name="overwrite">True when C holds uninitialized data that every element computation overwrites; false accumulates onto the existing contents, which the caller must have zeroed.</param>
    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe static void mm_2x4packed_tile(int M,
                              int N,
                              float* A,
                              float* panel,
                              float* C,
                              int K,
                              int kb,
                              bool overwrite)
    {
            for (int i = 0; i < M; i += 2)
            {
                var Ap1 = A + i * N;
                var Ap2 = Ap1 + N;
                var Cpv1 = (Vector256<float>*)(C + i * K + kb);
                var Cpv2 = (Vector256<float>*)(C + (i + 1) * K + kb);
                Vector256<float> c00 = overwrite ? Vector256<float>.Zero : Cpv1[0];
                Vector256<float> c01 = overwrite ? Vector256<float>.Zero : Cpv1[1];
                Vector256<float> c02 = overwrite ? Vector256<float>.Zero : Cpv1[2];
                Vector256<float> c03 = overwrite ? Vector256<float>.Zero : Cpv1[3];
                Vector256<float> c10 = overwrite ? Vector256<float>.Zero : Cpv2[0];
                Vector256<float> c11 = overwrite ? Vector256<float>.Zero : Cpv2[1];
                Vector256<float> c12 = overwrite ? Vector256<float>.Zero : Cpv2[2];
                Vector256<float> c13 = overwrite ? Vector256<float>.Zero : Cpv2[3];
                for (int j = 0; j < N; ++j)
                {
                    var av1 = Vector256.Create(Ap1[j]);
                    var av2 = Vector256.Create(Ap2[j]);
                    var Bpv = (Vector256<float>*)(panel + j * (4 * Vector256<float>.Count));
                    c00 = Fma.MultiplyAdd(Bpv[0], av1, c00);
                    c10 = Fma.MultiplyAdd(Bpv[0], av2, c10);
                    c01 = Fma.MultiplyAdd(Bpv[1], av1, c01);
                    c11 = Fma.MultiplyAdd(Bpv[1], av2, c11);
                    c02 = Fma.MultiplyAdd(Bpv[2], av1, c02);
                    c12 = Fma.MultiplyAdd(Bpv[2], av2, c12);
                    c03 = Fma.MultiplyAdd(Bpv[3], av1, c03);
                    c13 = Fma.MultiplyAdd(Bpv[3], av2, c13);
                }
                Cpv1[0] = c00;
                Cpv1[1] = c01;
                Cpv1[2] = c02;
                Cpv1[3] = c03;
                Cpv2[0] = c10;
                Cpv2[1] = c11;
                Cpv2[2] = c12;
                Cpv2[3] = c13;
            }
    }
    /// <summary>
    /// Register-tiled matrix multiplication reading panel-packed B, three
    /// rows per group.
    /// </summary>
    /// <param name="M">A rows (must be a multiple of 3).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="P">Panel-packed right matrix from PackPanelsB.</param>
    /// <param name="C">Result matrix.</param>
    /// <remarks>
    /// Same panels and per-element order as the 2-row packed kernel, so results
    /// agree with it bit-wise; three rows share each B vector, cutting B loads
    /// per FMA at the same broadcast rate. Tails read the appended row-major
    /// tail block in the same order.
    /// </remarks>
    /// <param name="overwrite">True when C holds uninitialized data that every element computation overwrites; false accumulates onto the existing contents, which the caller must have zeroed.</param>
    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe static void mm_unsafe_vectorized_intrinsics_3x4packed(int M,
                          int N,
                          int K,
                          float* A,
                          float* P,
                          float* C,
                          bool overwrite)
    {
        if (M % 3 != 0)
            throw new ArgumentException(nameof(M));

        int blocked = K - (K % (4 * Vector256<float>.Count));
        int tiles = blocked / (4 * Vector256<float>.Count);

        // Kb panels lead so each panel is fetched once and stays L1/L2 resident
        // across every row group; per-element FMA order matches the 2-row nest
        // exactly, so results agree bit-wise with the other packed kernels.
        for (int tb = 0; tb < tiles; tb++)
        {
            int kb = tb * (4 * Vector256<float>.Count);
            float* panel = P + tb * N * (4 * Vector256<float>.Count);
            mm_3x4packed_tile(M, N, A, panel, C, K, kb, overwrite);
        }
        int rem = K - blocked;
        if (rem > 0)
        {
            float* T = P + tiles * N * (4 * Vector256<float>.Count);
            int rv = rem / Vector256<float>.Count;
            for (int tt = 0; tt < rv; tt++)
            {
                for (int i = 0; i < M; i += 3)
                {
                    var Ap1 = A + i * N;
                    var Ap2 = Ap1 + N;
                    var Ap3 = Ap2 + N;
                    var rC1 = (Vector256<float>*)(C + i * K + blocked);
                    var rC2 = (Vector256<float>*)(C + (i + 1) * K + blocked);
                    var rC3 = (Vector256<float>*)(C + (i + 2) * K + blocked);
                    Vector256<float> c1 = overwrite ? Vector256<float>.Zero : rC1[tt];
                    Vector256<float> c2 = overwrite ? Vector256<float>.Zero : rC2[tt];
                    Vector256<float> c3 = overwrite ? Vector256<float>.Zero : rC3[tt];
                    for (int j = 0; j < N; ++j)
                    {
                        var Bpv = (Vector256<float>*)(T + j * rem + tt * Vector256<float>.Count);
                        c1 = Fma.MultiplyAdd(Bpv[0], Vector256.Create(Ap1[j]), c1);
                        c2 = Fma.MultiplyAdd(Bpv[0], Vector256.Create(Ap2[j]), c2);
                        c3 = Fma.MultiplyAdd(Bpv[0], Vector256.Create(Ap3[j]), c3);
                    }
                    rC1[tt] = c1;
                    rC2[tt] = c2;
                    rC3[tt] = c3;
                }
            }
            int vcols = rv * Vector256<float>.Count;
            // Narrow tail (fewer than 8 columns) accumulates in registers
            // instead of read-modify-writing C on every reduction step: the
            // per-element chain keeps the exact j-ascending mul-then-add order
            // of the loop it replaces, so results agree bit-wise while each C
            // row is loaded once and stored once.
            int tail = rem - vcols;
            if (tail > 0)
            for (int i = 0; i < M; i += 3)
            {
                var Ap1 = A + i * N;
                var Ap2 = Ap1 + N;
                var Ap3 = Ap2 + N;
                var Cp1 = C + i * K + blocked + vcols;
                var Cp2 = Cp1 + K;
                var Cp3 = Cp2 + K;
                switch (tail)
                {
                    case 1:
                    {
                        float c1 = overwrite ? 0f : Cp1[0], c2 = overwrite ? 0f : Cp2[0], c3 = overwrite ? 0f : Cp3[0];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c1 += Ap1[j] * t[0]; c2 += Ap2[j] * t[0]; c3 += Ap3[j] * t[0]; }
                        Cp1[0] = c1; Cp2[0] = c2; Cp3[0] = c3;
                        break;
                    }
                    case 2:
                    {
                        float c10 = overwrite ? 0f : Cp1[0], c11 = overwrite ? 0f : Cp1[1], c20 = overwrite ? 0f : Cp2[0], c21 = overwrite ? 0f : Cp2[1], c30 = overwrite ? 0f : Cp3[0], c31 = overwrite ? 0f : Cp3[1];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; }
                        Cp1[0] = c10; Cp1[1] = c11; Cp2[0] = c20; Cp2[1] = c21; Cp3[0] = c30; Cp3[1] = c31;
                        break;
                    }
                    case 3:
                    {
                        float c10 = overwrite ? 0f : Cp1[0], c11 = overwrite ? 0f : Cp1[1], c12 = overwrite ? 0f : Cp1[2], c20 = overwrite ? 0f : Cp2[0], c21 = overwrite ? 0f : Cp2[1], c22 = overwrite ? 0f : Cp2[2], c30 = overwrite ? 0f : Cp3[0], c31 = overwrite ? 0f : Cp3[1], c32 = overwrite ? 0f : Cp3[2];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; }
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32;
                        break;
                    }
                    case 4:
                    {
                        float c10 = overwrite ? 0f : Cp1[0], c11 = overwrite ? 0f : Cp1[1], c12 = overwrite ? 0f : Cp1[2], c13 = overwrite ? 0f : Cp1[3];
                        float c20 = overwrite ? 0f : Cp2[0], c21 = overwrite ? 0f : Cp2[1], c22 = overwrite ? 0f : Cp2[2], c23 = overwrite ? 0f : Cp2[3];
                        float c30 = overwrite ? 0f : Cp3[0], c31 = overwrite ? 0f : Cp3[1], c32 = overwrite ? 0f : Cp3[2], c33 = overwrite ? 0f : Cp3[3];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c33 += Ap3[j] * t[3]; }
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23;
                        Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32; Cp3[3] = c33;
                        break;
                    }
                    case 5:
                    {
                        float c10 = overwrite ? 0f : Cp1[0], c11 = overwrite ? 0f : Cp1[1], c12 = overwrite ? 0f : Cp1[2], c13 = overwrite ? 0f : Cp1[3], c14 = overwrite ? 0f : Cp1[4];
                        float c20 = overwrite ? 0f : Cp2[0], c21 = overwrite ? 0f : Cp2[1], c22 = overwrite ? 0f : Cp2[2], c23 = overwrite ? 0f : Cp2[3], c24 = overwrite ? 0f : Cp2[4];
                        float c30 = overwrite ? 0f : Cp3[0], c31 = overwrite ? 0f : Cp3[1], c32 = overwrite ? 0f : Cp3[2], c33 = overwrite ? 0f : Cp3[3], c34 = overwrite ? 0f : Cp3[4];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c33 += Ap3[j] * t[3]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; c34 += Ap3[j] * t[4]; }
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13; Cp1[4] = c14;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp2[4] = c24;
                        Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32; Cp3[3] = c33; Cp3[4] = c34;
                        break;
                    }
                    case 6:
                    {
                        float c10 = overwrite ? 0f : Cp1[0], c11 = overwrite ? 0f : Cp1[1], c12 = overwrite ? 0f : Cp1[2], c13 = overwrite ? 0f : Cp1[3], c14 = overwrite ? 0f : Cp1[4], c15 = overwrite ? 0f : Cp1[5];
                        float c20 = overwrite ? 0f : Cp2[0], c21 = overwrite ? 0f : Cp2[1], c22 = overwrite ? 0f : Cp2[2], c23 = overwrite ? 0f : Cp2[3], c24 = overwrite ? 0f : Cp2[4], c25 = overwrite ? 0f : Cp2[5];
                        float c30 = overwrite ? 0f : Cp3[0], c31 = overwrite ? 0f : Cp3[1], c32 = overwrite ? 0f : Cp3[2], c33 = overwrite ? 0f : Cp3[3], c34 = overwrite ? 0f : Cp3[4], c35 = overwrite ? 0f : Cp3[5];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c33 += Ap3[j] * t[3]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; c34 += Ap3[j] * t[4]; c15 += Ap1[j] * t[5]; c25 += Ap2[j] * t[5]; c35 += Ap3[j] * t[5]; }
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13; Cp1[4] = c14; Cp1[5] = c15;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp2[4] = c24; Cp2[5] = c25;
                        Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32; Cp3[3] = c33; Cp3[4] = c34; Cp3[5] = c35;
                        break;
                    }
                    default:
                    {
                        float c10 = overwrite ? 0f : Cp1[0], c11 = overwrite ? 0f : Cp1[1], c12 = overwrite ? 0f : Cp1[2], c13 = overwrite ? 0f : Cp1[3], c14 = overwrite ? 0f : Cp1[4], c15 = overwrite ? 0f : Cp1[5], c16 = overwrite ? 0f : Cp1[6];
                        float c20 = overwrite ? 0f : Cp2[0], c21 = overwrite ? 0f : Cp2[1], c22 = overwrite ? 0f : Cp2[2], c23 = overwrite ? 0f : Cp2[3], c24 = overwrite ? 0f : Cp2[4], c25 = overwrite ? 0f : Cp2[5], c26 = overwrite ? 0f : Cp2[6];
                        float c30 = overwrite ? 0f : Cp3[0], c31 = overwrite ? 0f : Cp3[1], c32 = overwrite ? 0f : Cp3[2], c33 = overwrite ? 0f : Cp3[3], c34 = overwrite ? 0f : Cp3[4], c35 = overwrite ? 0f : Cp3[5], c36 = overwrite ? 0f : Cp3[6];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c33 += Ap3[j] * t[3]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; c34 += Ap3[j] * t[4]; c15 += Ap1[j] * t[5]; c25 += Ap2[j] * t[5]; c35 += Ap3[j] * t[5]; c16 += Ap1[j] * t[6]; c26 += Ap2[j] * t[6]; c36 += Ap3[j] * t[6]; }
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13; Cp1[4] = c14; Cp1[5] = c15; Cp1[6] = c16;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp2[4] = c24; Cp2[5] = c25; Cp2[6] = c26;
                        Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32; Cp3[3] = c33; Cp3[4] = c34; Cp3[5] = c35; Cp3[6] = c36;
                        break;
                    }
                }
            }
        }
    }
    /// <summary>
    /// One full 32-column tile of the 3-row packed nest across all of
    /// its row groups. Extracted for the tiled row-group composer;
    /// the full method calls it per tile in the same order,
    /// so results match bit for bit.
    /// </summary>
    /// <param name="M">A rows (must be a multiple of 3).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="panel">One packed panel tile.</param>
    /// <param name="C">Result matrix.</param>
    /// <param name="K">B columns.</param>
    /// <param name="kb">Leading output column of this tile.</param>
    /// <param name="overwrite">True when C holds uninitialized data that every element computation overwrites; false accumulates onto the existing contents, which the caller must have zeroed.</param>
    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe static void mm_3x4packed_tile(int M,
                              int N,
                              float* A,
                              float* panel,
                              float* C,
                              int K,
                              int kb,
                              bool overwrite)
    {
            for (int i = 0; i < M; i += 3)
            {
                var Ap1 = A + i * N;
                var Ap2 = Ap1 + N;
                var Ap3 = Ap2 + N;
                var Cpv1 = (Vector256<float>*)(C + i * K + kb);
                var Cpv2 = (Vector256<float>*)(C + (i + 1) * K + kb);
                var Cpv3 = (Vector256<float>*)(C + (i + 2) * K + kb);
                Vector256<float> c00 = overwrite ? Vector256<float>.Zero : Cpv1[0];
                Vector256<float> c01 = overwrite ? Vector256<float>.Zero : Cpv1[1];
                Vector256<float> c02 = overwrite ? Vector256<float>.Zero : Cpv1[2];
                Vector256<float> c03 = overwrite ? Vector256<float>.Zero : Cpv1[3];
                Vector256<float> c10 = overwrite ? Vector256<float>.Zero : Cpv2[0];
                Vector256<float> c11 = overwrite ? Vector256<float>.Zero : Cpv2[1];
                Vector256<float> c12 = overwrite ? Vector256<float>.Zero : Cpv2[2];
                Vector256<float> c13 = overwrite ? Vector256<float>.Zero : Cpv2[3];
                Vector256<float> c20 = overwrite ? Vector256<float>.Zero : Cpv3[0];
                Vector256<float> c21 = overwrite ? Vector256<float>.Zero : Cpv3[1];
                Vector256<float> c22 = overwrite ? Vector256<float>.Zero : Cpv3[2];
                Vector256<float> c23 = overwrite ? Vector256<float>.Zero : Cpv3[3];
                for (int j = 0; j < N; ++j)
                {
                    var av1 = Vector256.Create(Ap1[j]);
                    var av2 = Vector256.Create(Ap2[j]);
                    var av3 = Vector256.Create(Ap3[j]);
                    var Bpv = (Vector256<float>*)(panel + j * (4 * Vector256<float>.Count));
                    c00 = Fma.MultiplyAdd(Bpv[0], av1, c00);
                    c10 = Fma.MultiplyAdd(Bpv[0], av2, c10);
                    c20 = Fma.MultiplyAdd(Bpv[0], av3, c20);
                    c01 = Fma.MultiplyAdd(Bpv[1], av1, c01);
                    c11 = Fma.MultiplyAdd(Bpv[1], av2, c11);
                    c21 = Fma.MultiplyAdd(Bpv[1], av3, c21);
                    c02 = Fma.MultiplyAdd(Bpv[2], av1, c02);
                    c12 = Fma.MultiplyAdd(Bpv[2], av2, c12);
                    c22 = Fma.MultiplyAdd(Bpv[2], av3, c22);
                    c03 = Fma.MultiplyAdd(Bpv[3], av1, c03);
                    c13 = Fma.MultiplyAdd(Bpv[3], av2, c13);
                    c23 = Fma.MultiplyAdd(Bpv[3], av3, c23);
                }
                Cpv1[0] = c00;
                Cpv1[1] = c01;
                Cpv1[2] = c02;
                Cpv1[3] = c03;
                Cpv2[0] = c10;
                Cpv2[1] = c11;
                Cpv2[2] = c12;
                Cpv2[3] = c13;
                Cpv3[0] = c20;
                Cpv3[1] = c21;
                Cpv3[2] = c22;
                Cpv3[3] = c23;
            }
    }


    /// <summary>
    /// Matrix multiplication.
    /// </summary>
    /// <param name="M">A rows.</param>
    /// <param name="N">A columns.</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="B">Right matrix.</param>
    /// <param name="C">Result matrix.</param>
    /// <summary>
    /// Register-tiled matrix multiplication reading panel-packed B, six
    /// rows per group with masked column tails. Panels, tails, and the
    /// per-element FMA order match the 2/3-row packed nests exactly, so
    /// results agree bit-wise with them; 6-way B sharing halves panel
    /// re-reads against the 2-row nest at the same 14-register budget.
    /// </summary>
    /// <param name="M">A rows (must be a multiple of 6).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="P">Panel-packed right matrix from PackPanelsB.</param>
    /// <param name="C">Result matrix.</param>
    /// <param name="overwrite">True when C holds uninitialized data that every element computation overwrites; false accumulates onto the existing contents, which the caller must have zeroed.</param>
    /// <param name="overwrite">True when C holds uninitialized data that every element computation overwrites; false accumulates onto the existing contents, which the caller must have zeroed.</param>
    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe static void mm_unsafe_vectorized_avx512_6x32packed(int M,
                          int N,
                          int K,
                          float* A,
                          float* P,
                          float* C,
                          bool overwrite)
    {
        if (M % 6 != 0)
            throw new ArgumentException(nameof(M));

        int blocked = K - (K % (2 * Vector512<float>.Count));
        int tiles = blocked / (2 * Vector512<float>.Count);

        for (int tb = 0; tb < tiles; tb++)
        {
            int kb = tb * (2 * Vector512<float>.Count);
            float* panel = P + tb * N * (2 * Vector512<float>.Count);
            for (int i = 0; i < M; i += 6)
            {
                var Ap1 = A + i * N;
                var Ap2 = Ap1 + N;
                var Ap3 = Ap2 + N;
                var Ap4 = Ap3 + N;
                var Ap5 = Ap4 + N;
                var Ap6 = Ap5 + N;
                var Cp1 = (Vector512<float>*)(C + i * K + kb);
                var Cp2 = (Vector512<float>*)(C + (i + 1) * K + kb);
                var Cp3 = (Vector512<float>*)(C + (i + 2) * K + kb);
                var Cp4 = (Vector512<float>*)(C + (i + 3) * K + kb);
                var Cp5 = (Vector512<float>*)(C + (i + 4) * K + kb);
                var Cp6 = (Vector512<float>*)(C + (i + 5) * K + kb);
                Vector512<float> c00 = overwrite ? Vector512<float>.Zero : Cp1[0];
                Vector512<float> c01 = overwrite ? Vector512<float>.Zero : Cp1[1];
                Vector512<float> c10 = overwrite ? Vector512<float>.Zero : Cp2[0];
                Vector512<float> c11 = overwrite ? Vector512<float>.Zero : Cp2[1];
                Vector512<float> c20 = overwrite ? Vector512<float>.Zero : Cp3[0];
                Vector512<float> c21 = overwrite ? Vector512<float>.Zero : Cp3[1];
                Vector512<float> c30 = overwrite ? Vector512<float>.Zero : Cp4[0];
                Vector512<float> c31 = overwrite ? Vector512<float>.Zero : Cp4[1];
                Vector512<float> c40 = overwrite ? Vector512<float>.Zero : Cp5[0];
                Vector512<float> c41 = overwrite ? Vector512<float>.Zero : Cp5[1];
                Vector512<float> c50 = overwrite ? Vector512<float>.Zero : Cp6[0];
                Vector512<float> c51 = overwrite ? Vector512<float>.Zero : Cp6[1];
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
                }
                Cp1[0] = c00;
                Cp1[1] = c01;
                Cp2[0] = c10;
                Cp2[1] = c11;
                Cp3[0] = c20;
                Cp3[1] = c21;
                Cp4[0] = c30;
                Cp4[1] = c31;
                Cp5[0] = c40;
                Cp5[1] = c41;
                Cp6[0] = c50;
                Cp6[1] = c51;
            }
        }
        int rem = K - blocked;
        if (rem > 0)
        {
            float* T = P + tiles * N * (2 * Vector512<float>.Count);
            int rv = rem / Vector512<float>.Count;
            for (int tt = 0; tt < rv; tt++)
            {
                for (int i = 0; i < M; i += 6)
                {
                    var Ap1 = A + i * N;
                    var Ap2 = Ap1 + N;
                    var Ap3 = Ap2 + N;
                    var Ap4 = Ap3 + N;
                    var Ap5 = Ap4 + N;
                    var Ap6 = Ap5 + N;
                    var rC1 = (Vector512<float>*)(C + i * K + blocked);
                    var rC2 = (Vector512<float>*)(C + (i + 1) * K + blocked);
                    var rC3 = (Vector512<float>*)(C + (i + 2) * K + blocked);
                    var rC4 = (Vector512<float>*)(C + (i + 3) * K + blocked);
                    var rC5 = (Vector512<float>*)(C + (i + 4) * K + blocked);
                    var rC6 = (Vector512<float>*)(C + (i + 5) * K + blocked);
                    Vector512<float> c1 = overwrite ? Vector512<float>.Zero : rC1[tt];
                    Vector512<float> c2 = overwrite ? Vector512<float>.Zero : rC2[tt];
                    Vector512<float> c3 = overwrite ? Vector512<float>.Zero : rC3[tt];
                    Vector512<float> c4 = overwrite ? Vector512<float>.Zero : rC4[tt];
                    Vector512<float> c5 = overwrite ? Vector512<float>.Zero : rC5[tt];
                    Vector512<float> c6 = overwrite ? Vector512<float>.Zero : rC6[tt];
                    for (int j = 0; j < N; ++j)
                    {
                        var Bpv = (Vector512<float>*)(T + j * rem + tt * Vector512<float>.Count);
                        c1 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(Ap1[j]), c1);
                        c2 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(Ap2[j]), c2);
                        c3 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(Ap3[j]), c3);
                        c4 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(Ap4[j]), c4);
                        c5 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(Ap5[j]), c5);
                        c6 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(Ap6[j]), c6);
                    }
                    rC1[tt] = c1;
                    rC2[tt] = c2;
                    rC3[tt] = c3;
                    rC4[tt] = c4;
                    rC5[tt] = c5;
                    rC6[tt] = c6;
                }
            }
            int vcols = rv * Vector512<float>.Count;
            // Masked narrow tail: fault-suppressed masked loads keep the
            // vector width into the row-major tail without a scalar loop,
            // and masked stores never touch columns past the edge. Sign-set
            // lanes select memory, matching blend semantics on both sides.
            int rem2 = rem - vcols;
            if (rem2 > 0)
            {
                float* mbuf = stackalloc float[Vector512<float>.Count];
                for (int q = 0; q < Vector512<float>.Count; q++) mbuf[q] = q < rem2 ? -1f : 0f;
                var vmask = Vector512.Load(mbuf);
                for (int i = 0; i < M; i += 6)
                {
                    var mC1 = C + i * K + blocked + vcols;
                    var mC2 = C + (i + 1) * K + blocked + vcols;
                    var mC3 = C + (i + 2) * K + blocked + vcols;
                    var mC4 = C + (i + 3) * K + blocked + vcols;
                    var mC5 = C + (i + 4) * K + blocked + vcols;
                    var mC6 = C + (i + 5) * K + blocked + vcols;
                    Vector512<float> d1 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(mC1, vmask, Vector512<float>.Zero);
                    Vector512<float> d2 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(mC2, vmask, Vector512<float>.Zero);
                    Vector512<float> d3 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(mC3, vmask, Vector512<float>.Zero);
                    Vector512<float> d4 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(mC4, vmask, Vector512<float>.Zero);
                    Vector512<float> d5 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(mC5, vmask, Vector512<float>.Zero);
                    Vector512<float> d6 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(mC6, vmask, Vector512<float>.Zero);
                    for (int j = 0; j < N; ++j)
                    {
                        var Bm = T + j * rem + vcols;
                        var bv = Avx512F.MaskLoad(Bm, vmask, Vector512<float>.Zero);
                        d1 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((A + i * N)[j]), d1);
                        d2 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((A + (i + 1) * N)[j]), d2);
                        d3 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((A + (i + 2) * N)[j]), d3);
                        d4 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((A + (i + 3) * N)[j]), d4);
                        d5 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((A + (i + 4) * N)[j]), d5);
                        d6 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((A + (i + 5) * N)[j]), d6);
                    }
                    Avx512F.MaskStore(mC1, vmask, d1);
                    Avx512F.MaskStore(mC2, vmask, d2);
                    Avx512F.MaskStore(mC3, vmask, d3);
                    Avx512F.MaskStore(mC4, vmask, d4);
                    Avx512F.MaskStore(mC5, vmask, d5);
                    Avx512F.MaskStore(mC6, vmask, d6);
                }
            }
        }
    }
    /// <summary>
    /// Register-tiled matrix multiplication reading panel-packed B, twelve
    /// rows per group with masked column tails. Panels, tails, and the
    /// per-element FMA order match the 2/3/6-row packed nests exactly, so
    /// results agree bit-wise with them; 12-way B sharing halves panel
    /// re-reads against the 6-row nest within the 32-register file.
    /// </summary>
    /// <param name="M">A rows (must be a multiple of 12).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="P">Panel-packed right matrix from PackPanelsB.</param>
    /// <param name="C">Result matrix.</param>
    /// <param name="overwrite">True when C holds uninitialized data that every element computation overwrites; false accumulates onto the existing contents, which the caller must have zeroed.</param>
    /// <param name="overwrite">True when C holds uninitialized data that every element computation overwrites; false accumulates onto the existing contents, which the caller must have zeroed.</param>
    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe static void mm_unsafe_vectorized_avx512_12x32packed(int M,
                          int N,
                          int K,
                          float* A,
                          float* P,
                          float* C,
                          bool overwrite)
    {
        if (M % 12 != 0)
            throw new ArgumentException(nameof(M));

        int blocked = K - (K % (2 * Vector512<float>.Count));
        int tiles = blocked / (2 * Vector512<float>.Count);

        for (int tb = 0; tb < tiles; tb++)
        {
            int kb = tb * (2 * Vector512<float>.Count);
            float* panel = P + tb * N * (2 * Vector512<float>.Count);
            if (PackedSweepPrefetch && Sse.IsSupported) mm_avx512_12x32packed_tile_pref(M, N, A, panel, C, K, kb, overwrite);
            else mm_avx512_12x32packed_tile(M, N, A, panel, C, K, kb, overwrite);
        }
        int rem = K - blocked;
        if (rem > 0)
            mm_avx512_12x32packed_col_tail(M, N, K, A, P, C, blocked, tiles, rem, overwrite);
    }
    // B3 P1 prototype switch (see PLAN.md): route 12-row packed tiles through the prefetch variant.
    // Off by default; probes and tests toggle it. Values are unaffected either way.
    public static bool PackedSweepPrefetch;

    /// <summary>Prefetch distance in j-iterations for the P1 prototype (tuned in mirrors).</summary>
    public const int PackedSweepPrefetchDistance = 8;

    /// <summary>
    /// B3 P1 prototype: 12-row packed tile with software prefetch on the B sweep (see PLAN.md).
    /// Same arithmetic and order as mm_avx512_12x32packed_tile, hence bitwise-identical; delete or promote on mirror verdict.
    /// </summary>
    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe static void mm_avx512_12x32packed_tile_pref(int M,
                              int N,
                              float* A,
                              float* panel,
                              float* C,
                              int K,
                              int kb,
                              bool overwrite)
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
                Vector512<float> c00 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup))[0];
                Vector512<float> c01 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup))[1];
                Vector512<float> c10 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 1 * K))[0];
                Vector512<float> c11 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 1 * K))[1];
                Vector512<float> c20 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 2 * K))[0];
                Vector512<float> c21 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 2 * K))[1];
                Vector512<float> c30 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 3 * K))[0];
                Vector512<float> c31 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 3 * K))[1];
                Vector512<float> c40 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 4 * K))[0];
                Vector512<float> c41 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 4 * K))[1];
                Vector512<float> c50 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 5 * K))[0];
                Vector512<float> c51 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 5 * K))[1];
                Vector512<float> c70 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 6 * K))[0];
                Vector512<float> c71 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 6 * K))[1];
                Vector512<float> c80 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 7 * K))[0];
                Vector512<float> c81 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 7 * K))[1];
                Vector512<float> c90 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 8 * K))[0];
                Vector512<float> c91 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 8 * K))[1];
                Vector512<float> cA0 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 9 * K))[0];
                Vector512<float> cA1 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 9 * K))[1];
                Vector512<float> cB0 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 10 * K))[0];
                Vector512<float> cB1 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 10 * K))[1];
                Vector512<float> cC0 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 11 * K))[0];
                Vector512<float> cC1 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 11 * K))[1];
                for (int j = 0; j < N; ++j)
                {
                    // B3 P1 prototype: software prefetch ahead on the packed B sweep (MLAS pattern).
                    // Hint only: values are unaffected, so agreement with the plain lane is bitwise.
                    int jp = j + PackedSweepPrefetchDistance;
                    if (Sse.IsSupported && (uint)jp < (uint)N)
                    {
                        Sse.Prefetch0(panel + jp * (2 * Vector512<float>.Count));
                        Sse.Prefetch0(panel + jp * (2 * Vector512<float>.Count) + 16);
                    }
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
    /// <summary>
    /// One full 32-column tile of the 12-row AVX512 packed nest across all of
    /// its row groups. Extracted so the tiled row-group composer can chain
    /// widths per tile; the full method calls it per tile in the same order,
    /// so results match bit for bit.
    /// </summary>
    /// <param name="M">A rows (must be a multiple of 12).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="panel">One packed panel tile.</param>
    /// <param name="C">Result matrix.</param>
    /// <param name="K">B columns.</param>
    /// <param name="kb">Leading output column of this tile.</param>
    /// <param name="overwrite">True when C holds uninitialized data that every element computation overwrites; false accumulates onto the existing contents, which the caller must have zeroed.</param>
    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe static void mm_avx512_12x32packed_tile(int M,
                              int N,
                              float* A,
                              float* panel,
                              float* C,
                              int K,
                              int kb,
                              bool overwrite)
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
                Vector512<float> c00 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup))[0];
                Vector512<float> c01 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup))[1];
                Vector512<float> c10 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 1 * K))[0];
                Vector512<float> c11 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 1 * K))[1];
                Vector512<float> c20 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 2 * K))[0];
                Vector512<float> c21 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 2 * K))[1];
                Vector512<float> c30 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 3 * K))[0];
                Vector512<float> c31 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 3 * K))[1];
                Vector512<float> c40 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 4 * K))[0];
                Vector512<float> c41 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 4 * K))[1];
                Vector512<float> c50 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 5 * K))[0];
                Vector512<float> c51 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 5 * K))[1];
                Vector512<float> c70 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 6 * K))[0];
                Vector512<float> c71 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 6 * K))[1];
                Vector512<float> c80 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 7 * K))[0];
                Vector512<float> c81 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 7 * K))[1];
                Vector512<float> c90 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 8 * K))[0];
                Vector512<float> c91 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 8 * K))[1];
                Vector512<float> cA0 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 9 * K))[0];
                Vector512<float> cA1 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 9 * K))[1];
                Vector512<float> cB0 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 10 * K))[0];
                Vector512<float> cB1 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 10 * K))[1];
                Vector512<float> cC0 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 11 * K))[0];
                Vector512<float> cC1 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 11 * K))[1];
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
    /// <summary>
    /// Column-tail half of the 12-row AVX512 packed nest: full-vector
    /// remainders first, then one masked remainder. Split from the main
    /// nest so the hot reduction loop stays compact for the JIT; takes
    /// the same operands plus the precomputed tile geometry.
    /// </summary>
    /// <param name="M">A rows (must be a multiple of 12).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="P">Panel-packed right matrix from PackPanelsB.</param>
    /// <param name="C">Result matrix.</param>
    /// <param name="blocked">Leading output columns covered by full tiles.</param>
    /// <param name="tiles">Count of full 32-column tiles.</param>
    /// <param name="rem">Trailing output columns (must be positive).</param>
    /// <param name="overwrite">True when C holds uninitialized data that every element computation overwrites; false accumulates onto the existing contents, which the caller must have zeroed.</param>
    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe static void mm_avx512_12x32packed_col_tail(int M,
                              int N,
                              int K,
                              float* A,
                              float* P,
                              float* C,
                              int blocked,
                              int tiles,
                              int rem,
                              bool overwrite)
    {
        float* T = P + tiles * N * (2 * Vector512<float>.Count);
        int rv = rem / Vector512<float>.Count;
        for (int tt = 0; tt < rv; tt++)
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
                var rC1 = (Vector512<float>*)(C + i * K + blocked);
                var rC2 = (Vector512<float>*)(C + (i + 1) * K + blocked);
                var rC3 = (Vector512<float>*)(C + (i + 2) * K + blocked);
                var rC4 = (Vector512<float>*)(C + (i + 3) * K + blocked);
                var rC5 = (Vector512<float>*)(C + (i + 4) * K + blocked);
                var rC6 = (Vector512<float>*)(C + (i + 5) * K + blocked);
                var rC7 = (Vector512<float>*)(C + (i + 6) * K + blocked);
                var rC8 = (Vector512<float>*)(C + (i + 7) * K + blocked);
                var rC9 = (Vector512<float>*)(C + (i + 8) * K + blocked);
                var rCA = (Vector512<float>*)(C + (i + 9) * K + blocked);
                var rCB = (Vector512<float>*)(C + (i + 10) * K + blocked);
                var rCC = (Vector512<float>*)(C + (i + 11) * K + blocked);
                Vector512<float> c1 = overwrite ? Vector512<float>.Zero : rC1[tt];
                Vector512<float> c2 = overwrite ? Vector512<float>.Zero : rC2[tt];
                Vector512<float> c3 = overwrite ? Vector512<float>.Zero : rC3[tt];
                Vector512<float> c4 = overwrite ? Vector512<float>.Zero : rC4[tt];
                Vector512<float> c5 = overwrite ? Vector512<float>.Zero : rC5[tt];
                Vector512<float> c6 = overwrite ? Vector512<float>.Zero : rC6[tt];
                Vector512<float> c7 = overwrite ? Vector512<float>.Zero : rC7[tt];
                Vector512<float> c8 = overwrite ? Vector512<float>.Zero : rC8[tt];
                Vector512<float> c9 = overwrite ? Vector512<float>.Zero : rC9[tt];
                Vector512<float> cA = overwrite ? Vector512<float>.Zero : rCA[tt];
                Vector512<float> cB = overwrite ? Vector512<float>.Zero : rCB[tt];
                Vector512<float> cC = overwrite ? Vector512<float>.Zero : rCC[tt];
                for (int j = 0; j < N; ++j)
                {
                    var Bpv = (Vector512<float>*)(T + j * rem + tt * Vector512<float>.Count);
                    c1 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(Ap1[j]), c1);
                    c2 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(Ap2[j]), c2);
                    c3 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(Ap3[j]), c3);
                    c4 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(Ap4[j]), c4);
                    c5 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(Ap5[j]), c5);
                    c6 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(Ap6[j]), c6);
                    c7 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(Ap7[j]), c7);
                    c8 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(Ap8[j]), c8);
                    c9 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(Ap9[j]), c9);
                    cA = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(ApA[j]), cA);
                    cB = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(ApB[j]), cB);
                    cC = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(ApC[j]), cC);
                }
                rC1[tt] = c1;
                rC2[tt] = c2;
                rC3[tt] = c3;
                rC4[tt] = c4;
                rC5[tt] = c5;
                rC6[tt] = c6;
                rC7[tt] = c7;
                rC8[tt] = c8;
                rC9[tt] = c9;
                rCA[tt] = cA;
                rCB[tt] = cB;
                rCC[tt] = cC;
            }
        }
        int vcols = rv * Vector512<float>.Count;
        int rem2 = rem - vcols;
        if (rem2 > 0)
        {
            float* mbuf = stackalloc float[Vector512<float>.Count];
            for (int q = 0; q < Vector512<float>.Count; q++) mbuf[q] = q < rem2 ? -1f : 0f;
            var vmask = Vector512.Load(mbuf);
            for (int i = 0; i < M; i += 12)
            {
                var mC1 = C + i * K + blocked + vcols;
                var mC2 = C + (i + 1) * K + blocked + vcols;
                var mC3 = C + (i + 2) * K + blocked + vcols;
                var mC4 = C + (i + 3) * K + blocked + vcols;
                var mC5 = C + (i + 4) * K + blocked + vcols;
                var mC6 = C + (i + 5) * K + blocked + vcols;
                var mC7 = C + (i + 6) * K + blocked + vcols;
                var mC8 = C + (i + 7) * K + blocked + vcols;
                var mC9 = C + (i + 8) * K + blocked + vcols;
                var mCA = C + (i + 9) * K + blocked + vcols;
                var mCB = C + (i + 10) * K + blocked + vcols;
                var mCC = C + (i + 11) * K + blocked + vcols;
                Vector512<float> d1 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(mC1, vmask, Vector512<float>.Zero);
                Vector512<float> d2 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(mC2, vmask, Vector512<float>.Zero);
                Vector512<float> d3 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(mC3, vmask, Vector512<float>.Zero);
                Vector512<float> d4 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(mC4, vmask, Vector512<float>.Zero);
                Vector512<float> d5 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(mC5, vmask, Vector512<float>.Zero);
                Vector512<float> d6 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(mC6, vmask, Vector512<float>.Zero);
                Vector512<float> d7 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(mC7, vmask, Vector512<float>.Zero);
                Vector512<float> d8 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(mC8, vmask, Vector512<float>.Zero);
                Vector512<float> d9 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(mC9, vmask, Vector512<float>.Zero);
                Vector512<float> dA = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(mCA, vmask, Vector512<float>.Zero);
                Vector512<float> dB = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(mCB, vmask, Vector512<float>.Zero);
                Vector512<float> dC = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(mCC, vmask, Vector512<float>.Zero);
                for (int j = 0; j < N; ++j)
                {
                    var Bm = T + j * rem + vcols;
                    var bv = Avx512F.MaskLoad(Bm, vmask, Vector512<float>.Zero);
                    d1 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((A + i * N)[j]), d1);
                    d2 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((A + (i + 1) * N)[j]), d2);
                    d3 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((A + (i + 2) * N)[j]), d3);
                    d4 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((A + (i + 3) * N)[j]), d4);
                    d5 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((A + (i + 4) * N)[j]), d5);
                    d6 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((A + (i + 5) * N)[j]), d6);
                    d7 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((A + (i + 6) * N)[j]), d7);
                    d8 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((A + (i + 7) * N)[j]), d8);
                    d9 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((A + (i + 8) * N)[j]), d9);
                    dA = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((A + (i + 9) * N)[j]), dA);
                    dB = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((A + (i + 10) * N)[j]), dB);
                    dC = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((A + (i + 11) * N)[j]), dC);
                }
                Avx512F.MaskStore(mC1, vmask, d1);
                Avx512F.MaskStore(mC2, vmask, d2);
                Avx512F.MaskStore(mC3, vmask, d3);
                Avx512F.MaskStore(mC4, vmask, d4);
                Avx512F.MaskStore(mC5, vmask, d5);
                Avx512F.MaskStore(mC6, vmask, d6);
                Avx512F.MaskStore(mC7, vmask, d7);
                Avx512F.MaskStore(mC8, vmask, d8);
                Avx512F.MaskStore(mC9, vmask, d9);
                Avx512F.MaskStore(mCA, vmask, dA);
                Avx512F.MaskStore(mCB, vmask, dB);
                Avx512F.MaskStore(mCC, vmask, dC);
            }
        }
    }
    /// <summary>
    /// Register-tiled matrix multiplication reading panel-packed B, eight
    /// rows per group with masked column tails. Panels, tails, and the
    /// per-element FMA order match the 2/3/6/12-row packed nests exactly, so
    /// results agree bit-wise with them; 8-way B sharing absorbs remainders
    /// that would otherwise stream panels through narrow 2-row tails.
    /// </summary>
    /// <param name="M">A rows (must be a multiple of 8).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="P">Panel-packed right matrix from PackPanelsB.</param>
    /// <param name="C">Result matrix.</param>
    /// <param name="overwrite">True when C holds uninitialized data that every element computation overwrites; false accumulates onto the existing contents, which the caller must have zeroed.</param>
    /// <param name="overwrite">True when C holds uninitialized data that every element computation overwrites; false accumulates onto the existing contents, which the caller must have zeroed.</param>
    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe static void mm_unsafe_vectorized_avx512_8x32packed(int M,
                          int N,
                          int K,
                          float* A,
                          float* P,
                          float* C,
                          bool overwrite)
    {
        if (M % 8 != 0)
            throw new ArgumentException(nameof(M));

        int blocked = K - (K % (2 * Vector512<float>.Count));
        int tiles = blocked / (2 * Vector512<float>.Count);

        for (int tb = 0; tb < tiles; tb++)
        {
            int kb = tb * (2 * Vector512<float>.Count);
            float* panel = P + tb * N * (2 * Vector512<float>.Count);
            mm_avx512_8x32packed_tile(M, N, A, panel, C, K, kb, overwrite);
        }
        int rem = K - blocked;
        if (rem > 0)
            mm_avx512_8x32packed_col_tail(M, N, K, A, P, C, blocked, tiles, rem, overwrite);
    }
    /// <summary>
    /// One full 32-column tile of the 8-row AVX512 packed nest across all of
    /// its row groups. Extracted so the tiled row-group composer can chain
    /// widths per tile; the full method calls it per tile in the same order,
    /// so results match bit for bit.
    /// </summary>
    /// <param name="M">A rows (must be a multiple of 8).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="panel">One packed panel tile.</param>
    /// <param name="C">Result matrix.</param>
    /// <param name="K">B columns.</param>
    /// <param name="kb">Leading output column of this tile.</param>
    /// <param name="overwrite">True when C holds uninitialized data that every element computation overwrites; false accumulates onto the existing contents, which the caller must have zeroed.</param>
    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe static void mm_avx512_8x32packed_tile(int M,
                              int N,
                              float* A,
                              float* panel,
                              float* C,
                              int K,
                              int kb,
                              bool overwrite)
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
                Vector512<float> c00 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup))[0];
                Vector512<float> c01 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup))[1];
                Vector512<float> c10 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 1 * K))[0];
                Vector512<float> c11 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 1 * K))[1];
                Vector512<float> c20 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 2 * K))[0];
                Vector512<float> c21 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 2 * K))[1];
                Vector512<float> c30 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 3 * K))[0];
                Vector512<float> c31 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 3 * K))[1];
                Vector512<float> c40 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 4 * K))[0];
                Vector512<float> c41 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 4 * K))[1];
                Vector512<float> c50 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 5 * K))[0];
                Vector512<float> c51 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 5 * K))[1];
                Vector512<float> c60 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 6 * K))[0];
                Vector512<float> c61 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 6 * K))[1];
                Vector512<float> c70 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 7 * K))[0];
                Vector512<float> c71 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 7 * K))[1];
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
    /// <summary>
    /// Column-tail half of the 8-row AVX512 packed nest: full-vector
    /// remainders first, then one masked remainder. Split from the main
    /// nest so the hot reduction loop stays compact for the JIT; takes
    /// the same operands plus the precomputed tile geometry.
    /// </summary>
    /// <param name="M">A rows (must be a multiple of 8).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="P">Panel-packed right matrix from PackPanelsB.</param>
    /// <param name="C">Result matrix.</param>
    /// <param name="blocked">Leading output columns covered by full tiles.</param>
    /// <param name="tiles">Count of full 32-column tiles.</param>
    /// <param name="rem">Trailing output columns (must be positive).</param>
    /// <param name="overwrite">True when C holds uninitialized data that every element computation overwrites; false accumulates onto the existing contents, which the caller must have zeroed.</param>
    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe static void mm_avx512_8x32packed_col_tail(int M,
                              int N,
                              int K,
                              float* A,
                              float* P,
                              float* C,
                              int blocked,
                              int tiles,
                              int rem,
                              bool overwrite)
    {
        float* T = P + tiles * N * (2 * Vector512<float>.Count);
        int rv = rem / Vector512<float>.Count;
        for (int tt = 0; tt < rv; tt++)
        {
            for (int i = 0; i < M; i += 8)
            {
                float* aGroup = A + i * N;
                float* cGroup = C + i * K + blocked;
                Vector512<float> c0 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup))[tt];
                Vector512<float> c1 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 1 * K))[tt];
                Vector512<float> c2 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 2 * K))[tt];
                Vector512<float> c3 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 3 * K))[tt];
                Vector512<float> c4 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 4 * K))[tt];
                Vector512<float> c5 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 5 * K))[tt];
                Vector512<float> c6 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 6 * K))[tt];
                Vector512<float> c7 = overwrite ? Vector512<float>.Zero : ((Vector512<float>*)(cGroup + 7 * K))[tt];
                for (int j = 0; j < N; ++j)
                {
                    var Bpv = (Vector512<float>*)(T + j * rem + tt * Vector512<float>.Count);
                    c0 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create(aGroup[j]), c0);
                    c1 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create((aGroup + 1 * N)[j]), c1);
                    c2 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create((aGroup + 2 * N)[j]), c2);
                    c3 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create((aGroup + 3 * N)[j]), c3);
                    c4 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create((aGroup + 4 * N)[j]), c4);
                    c5 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create((aGroup + 5 * N)[j]), c5);
                    c6 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create((aGroup + 6 * N)[j]), c6);
                    c7 = Avx512F.FusedMultiplyAdd(Bpv[0], Vector512.Create((aGroup + 7 * N)[j]), c7);
                }
                ((Vector512<float>*)(cGroup))[tt] = c0;
                ((Vector512<float>*)(cGroup + 1 * K))[tt] = c1;
                ((Vector512<float>*)(cGroup + 2 * K))[tt] = c2;
                ((Vector512<float>*)(cGroup + 3 * K))[tt] = c3;
                ((Vector512<float>*)(cGroup + 4 * K))[tt] = c4;
                ((Vector512<float>*)(cGroup + 5 * K))[tt] = c5;
                ((Vector512<float>*)(cGroup + 6 * K))[tt] = c6;
                ((Vector512<float>*)(cGroup + 7 * K))[tt] = c7;
            }
        }
        int vcols = rv * Vector512<float>.Count;
        int rem2 = rem - vcols;
        if (rem2 > 0)
        {
            float* mbuf = stackalloc float[Vector512<float>.Count];
            for (int q = 0; q < Vector512<float>.Count; q++) mbuf[q] = q < rem2 ? -1f : 0f;
            var vmask = Vector512.Load(mbuf);
            for (int i = 0; i < M; i += 8)
            {
                float* aGroup = A + i * N;
                float* cGroup = C + i * K + blocked + vcols;
                Vector512<float> d0 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(cGroup, vmask, Vector512<float>.Zero);
                Vector512<float> d1 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(cGroup + 1 * K, vmask, Vector512<float>.Zero);
                Vector512<float> d2 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(cGroup + 2 * K, vmask, Vector512<float>.Zero);
                Vector512<float> d3 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(cGroup + 3 * K, vmask, Vector512<float>.Zero);
                Vector512<float> d4 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(cGroup + 4 * K, vmask, Vector512<float>.Zero);
                Vector512<float> d5 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(cGroup + 5 * K, vmask, Vector512<float>.Zero);
                Vector512<float> d6 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(cGroup + 6 * K, vmask, Vector512<float>.Zero);
                Vector512<float> d7 = overwrite ? Vector512<float>.Zero : Avx512F.MaskLoad(cGroup + 7 * K, vmask, Vector512<float>.Zero);
                for (int j = 0; j < N; ++j)
                {
                    var Bm = T + j * rem + vcols;
                    var bv = Avx512F.MaskLoad(Bm, vmask, Vector512<float>.Zero);
                    d0 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create(aGroup[j]), d0);
                    d1 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((aGroup + 1 * N)[j]), d1);
                    d2 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((aGroup + 2 * N)[j]), d2);
                    d3 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((aGroup + 3 * N)[j]), d3);
                    d4 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((aGroup + 4 * N)[j]), d4);
                    d5 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((aGroup + 5 * N)[j]), d5);
                    d6 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((aGroup + 6 * N)[j]), d6);
                    d7 = Avx512F.FusedMultiplyAdd(bv, Vector512.Create((aGroup + 7 * N)[j]), d7);
                }
                Avx512F.MaskStore(cGroup, vmask, d0);
                Avx512F.MaskStore(cGroup + 1 * K, vmask, d1);
                Avx512F.MaskStore(cGroup + 2 * K, vmask, d2);
                Avx512F.MaskStore(cGroup + 3 * K, vmask, d3);
                Avx512F.MaskStore(cGroup + 4 * K, vmask, d4);
                Avx512F.MaskStore(cGroup + 5 * K, vmask, d5);
                Avx512F.MaskStore(cGroup + 6 * K, vmask, d6);
                Avx512F.MaskStore(cGroup + 7 * K, vmask, d7);
            }
        }
    }
    public unsafe static void mm_unsafe_vectorized_intrinsics(int M,
                          int N,
                          int K,
                          double* A,
                          double* B,
                          double* C)
    {

        for (int i = 0; i < M; i++)
        {
            var Ap = A + i * N;
            var Cp = C + i * K;
            for (int j = 0; j < N; ++j)
            {
                var a = Ap[j];
                var Bp = B + j * K;
                var av = Vector256.Create(a);
                int ceiling = (K / Vector256<double>.Count) * Vector256<double>.Count;
                var Bkv = MemoryMarshal.Cast<double, Vector256<double>>(new Span<double>(Bp, K));
                var Ckv = MemoryMarshal.Cast<double, Vector256<double>>(new Span<double>(Cp, K));
                for (int k = 0; k < Bkv.Length; k++)
                {
                    Ckv[k] = Fma.MultiplyAdd(Bkv[k], av, Ckv[k]);
                }
                for (int k = ceiling; k < K; k++)
                {
                    Cp[k] += a * Bp[k];
                }
            }
        }
    }
    /// <summary>
    /// Image to column conversion.
    /// </summary>
    /// <param name="src">Source data.</param>
    /// <param name="srcC">Input channels.</param>
    /// <param name="srcH">Input height.</param>
    /// <param name="srcW">Input width.</param>
    /// <param name="kernelY">Kernel height.</param>
    /// <param name="kernelX">Kernel width.</param>
    /// <param name="dilationY">Dilation of the kernel by height.</param>
    /// <param name="dilationX">Dilation of the kernel by width.</param>
    /// <param name="strideY">Stride of the convolution by height.</param>
    /// <param name="strideX">Stride of the convolution by width.</param>
    /// <param name="padY">Zero padding at the top (begin height).</param>
    /// <param name="padX">Zero padding at the left (begin width).</param>
    /// <param name="padH">Zero padding at the bottom (end height).</param>
    /// <param name="padW">Zero padding at the right (end width).</param>
    /// <param name="buf">Buffer.</param>
    /// <summary>
    /// Dot product of two float spans over original row-major rows, so short
    /// projections need no transposed copy. Uses 8-wide AVX256 FMA
    /// accumulation with a scalar tail when intrinsics are enabled and
    /// supported; the portable scalar loop otherwise. Vector accumulation
    /// reassociates the sum, so results agree with the scalar order within
    /// float rounding, not bit for bit.
    /// </summary>
    public static float RowDot(ReadOnlySpan<float> x, ReadOnlySpan<float> y, TensorExecutionOptions? options)
    {
        int n = Math.Min(x.Length, y.Length);
        if (n == 0) return 0f;
        if ((options?.UseSimd ?? true) && (options?.UseIntrinsics ?? true) && Avx.IsSupported && Fma.IsSupported)
        {
            ref float xr = ref MemoryMarshal.GetReference(x);
            ref float yr = ref MemoryMarshal.GetReference(y);
            var acc = Vector256<float>.Zero;
            int i = 0;
            int full = n & ~7;
            for (; i < full; i += 8)
                acc = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref xr, (nuint)i), Vector256.LoadUnsafe(ref yr, (nuint)i), acc);
            float sum = Vector256.Sum(acc);
            for (; i < n; i++) sum += x[i] * y[i];
            return sum;
        }
        float total = 0f;
        for (int i = 0; i < n; i++) total += x[i] * y[i];
        return total;
    }
    /// <summary>
    /// Four dot products sharing one left span, for LSTM gate projections:
    /// each dot keeps RowDot's exact per-element order (vector chunks then
    /// scalar tail, portable scalar loop otherwise), so results agree
    /// bit-wise with four RowDot calls while paying one call setup and
    /// streaming the shared span once.
    /// </summary>
    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void RowDot4(ReadOnlySpan<float> x, ReadOnlySpan<float> y0, ReadOnlySpan<float> y1, ReadOnlySpan<float> y2, ReadOnlySpan<float> y3, out float d0, out float d1, out float d2, out float d3, TensorExecutionOptions? options)
    {
        int n = Math.Min(Math.Min(x.Length, y0.Length), Math.Min(Math.Min(y1.Length, y2.Length), y3.Length));
        d0 = 0f; d1 = 0f; d2 = 0f; d3 = 0f;
        if (n == 0) return;
        if ((options?.UseSimd ?? true) && (options?.UseIntrinsics ?? true) && Avx.IsSupported && Fma.IsSupported)
        {
            ref float xr = ref MemoryMarshal.GetReference(x);
            ref float r0 = ref MemoryMarshal.GetReference(y0);
            ref float r1 = ref MemoryMarshal.GetReference(y1);
            ref float r2 = ref MemoryMarshal.GetReference(y2);
            ref float r3 = ref MemoryMarshal.GetReference(y3);
            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;
            var acc2 = Vector256<float>.Zero;
            var acc3 = Vector256<float>.Zero;
            int i = 0;
            int full = n & ~7;
            for (; i < full; i += 8)
            {
                var xv = Vector256.LoadUnsafe(ref xr, (nuint)i);
                acc0 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref r0, (nuint)i), xv, acc0);
                acc1 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref r1, (nuint)i), xv, acc1);
                acc2 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref r2, (nuint)i), xv, acc2);
                acc3 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref r3, (nuint)i), xv, acc3);
            }
            float s0 = Vector256.Sum(acc0);
            float s1 = Vector256.Sum(acc1);
            float s2 = Vector256.Sum(acc2);
            float s3 = Vector256.Sum(acc3);
            for (; i < n; i++)
            {
                s0 += x[i] * y0[i];
                s1 += x[i] * y1[i];
                s2 += x[i] * y2[i];
                s3 += x[i] * y3[i];
            }
            d0 = s0; d1 = s1; d2 = s2; d3 = s3;
            return;
        }
        float t0 = 0f, t1 = 0f, t2 = 0f, t3 = 0f;
        for (int i = 0; i < n; i++)
        {
            t0 += x[i] * y0[i];
            t1 += x[i] * y1[i];
            t2 += x[i] * y2[i];
            t3 += x[i] * y3[i];
        }
        d0 = t0; d1 = t1; d2 = t2; d3 = t3;
    }

    /// <summary>
    /// Matrix-vector product over a prepared [K,O] panel (for example a
    /// transposed LSTM direction slice): y[o] = sum_k x[k]*P[k*O+o], or
    /// accumulated onto y when requested (ORT GEMM beta=1 semantics).
    /// SIMD lanes cover whole outputs, so no horizontal sum is needed;
    /// 128-wide AVX512 blocks precede 64-wide AVX256 blocks with a scalar
    /// tail, and the portable scalar loop covers the rest. Every lane sums
    /// each output in k order with fused multiply-add, so all paths agree
    /// bit for bit; the double-oracle tests pin the absolute contract.
    /// </summary>
    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void MatVecPanel(ReadOnlySpan<float> x, ReadOnlySpan<float> panel, Span<float> y, int outputs, int k, bool accumulate, TensorExecutionOptions? options)
    {
        int o = Math.Min(outputs, y.Length);
        int kk = Math.Min(k, x.Length);
        if (o > 0 && kk > 0) o = Math.Min(o, panel.Length / kk);
        if (o <= 0 || kk <= 0)
        {
            if (!accumulate)
            {
                int fill = Math.Min(outputs, y.Length);
                for (int i = 0; i < fill; i++) y[i] = 0f;
            }
            return;
        }
        if (!((options?.UseSimd ?? true) && (options?.UseIntrinsics ?? true) && Avx.IsSupported && Fma.IsSupported))
        {
            MatVecPanelScalar(x, panel, y, o, kk, accumulate);
            return;
        }
        ref float xr = ref MemoryMarshal.GetReference(x);
        ref float pr = ref MemoryMarshal.GetReference(panel);
        ref float yr = ref MemoryMarshal.GetReference(y);
        int base_ = 0;
        if (Avx512F.IsSupported)
        {
            const int B = 128;
            for (; base_ + B <= o; base_ += B)
            {
                var a0 = accumulate ? Vector512.LoadUnsafe(ref yr, (nuint)base_) : Vector512<float>.Zero;
                var a1 = accumulate ? Vector512.LoadUnsafe(ref yr, (nuint)(base_ + 16)) : Vector512<float>.Zero;
                var a2 = accumulate ? Vector512.LoadUnsafe(ref yr, (nuint)(base_ + 32)) : Vector512<float>.Zero;
                var a3 = accumulate ? Vector512.LoadUnsafe(ref yr, (nuint)(base_ + 48)) : Vector512<float>.Zero;
                var a4 = accumulate ? Vector512.LoadUnsafe(ref yr, (nuint)(base_ + 64)) : Vector512<float>.Zero;
                var a5 = accumulate ? Vector512.LoadUnsafe(ref yr, (nuint)(base_ + 80)) : Vector512<float>.Zero;
                var a6 = accumulate ? Vector512.LoadUnsafe(ref yr, (nuint)(base_ + 96)) : Vector512<float>.Zero;
                var a7 = accumulate ? Vector512.LoadUnsafe(ref yr, (nuint)(base_ + 112)) : Vector512<float>.Zero;
                for (int j = 0; j < kk; j++)
                {
                    var b = Vector512.Create(x[j]);
                    nuint row = (nuint)(j * o + base_);
                    a0 = Avx512F.FusedMultiplyAdd(b, Vector512.LoadUnsafe(ref pr, row), a0);
                    a1 = Avx512F.FusedMultiplyAdd(b, Vector512.LoadUnsafe(ref pr, row + 16), a1);
                    a2 = Avx512F.FusedMultiplyAdd(b, Vector512.LoadUnsafe(ref pr, row + 32), a2);
                    a3 = Avx512F.FusedMultiplyAdd(b, Vector512.LoadUnsafe(ref pr, row + 48), a3);
                    a4 = Avx512F.FusedMultiplyAdd(b, Vector512.LoadUnsafe(ref pr, row + 64), a4);
                    a5 = Avx512F.FusedMultiplyAdd(b, Vector512.LoadUnsafe(ref pr, row + 80), a5);
                    a6 = Avx512F.FusedMultiplyAdd(b, Vector512.LoadUnsafe(ref pr, row + 96), a6);
                    a7 = Avx512F.FusedMultiplyAdd(b, Vector512.LoadUnsafe(ref pr, row + 112), a7);
                }
                a0.StoreUnsafe(ref yr, (nuint)base_);
                a1.StoreUnsafe(ref yr, (nuint)(base_ + 16));
                a2.StoreUnsafe(ref yr, (nuint)(base_ + 32));
                a3.StoreUnsafe(ref yr, (nuint)(base_ + 48));
                a4.StoreUnsafe(ref yr, (nuint)(base_ + 64));
                a5.StoreUnsafe(ref yr, (nuint)(base_ + 80));
                a6.StoreUnsafe(ref yr, (nuint)(base_ + 96));
                a7.StoreUnsafe(ref yr, (nuint)(base_ + 112));
            }
        }
        {
            const int B = 64;
            for (; base_ + B <= o; base_ += B)
            {
                var a0 = accumulate ? Vector256.LoadUnsafe(ref yr, (nuint)base_) : Vector256<float>.Zero;
                var a1 = accumulate ? Vector256.LoadUnsafe(ref yr, (nuint)(base_ + 8)) : Vector256<float>.Zero;
                var a2 = accumulate ? Vector256.LoadUnsafe(ref yr, (nuint)(base_ + 16)) : Vector256<float>.Zero;
                var a3 = accumulate ? Vector256.LoadUnsafe(ref yr, (nuint)(base_ + 24)) : Vector256<float>.Zero;
                var a4 = accumulate ? Vector256.LoadUnsafe(ref yr, (nuint)(base_ + 32)) : Vector256<float>.Zero;
                var a5 = accumulate ? Vector256.LoadUnsafe(ref yr, (nuint)(base_ + 40)) : Vector256<float>.Zero;
                var a6 = accumulate ? Vector256.LoadUnsafe(ref yr, (nuint)(base_ + 48)) : Vector256<float>.Zero;
                var a7 = accumulate ? Vector256.LoadUnsafe(ref yr, (nuint)(base_ + 56)) : Vector256<float>.Zero;
                for (int j = 0; j < kk; j++)
                {
                    var b = Vector256.Create(x[j]);
                    nuint row = (nuint)(j * o + base_);
                    a0 = Fma.MultiplyAdd(b, Vector256.LoadUnsafe(ref pr, row), a0);
                    a1 = Fma.MultiplyAdd(b, Vector256.LoadUnsafe(ref pr, row + 8), a1);
                    a2 = Fma.MultiplyAdd(b, Vector256.LoadUnsafe(ref pr, row + 16), a2);
                    a3 = Fma.MultiplyAdd(b, Vector256.LoadUnsafe(ref pr, row + 24), a3);
                    a4 = Fma.MultiplyAdd(b, Vector256.LoadUnsafe(ref pr, row + 32), a4);
                    a5 = Fma.MultiplyAdd(b, Vector256.LoadUnsafe(ref pr, row + 40), a5);
                    a6 = Fma.MultiplyAdd(b, Vector256.LoadUnsafe(ref pr, row + 48), a6);
                    a7 = Fma.MultiplyAdd(b, Vector256.LoadUnsafe(ref pr, row + 56), a7);
                }
                a0.StoreUnsafe(ref yr, (nuint)base_);
                a1.StoreUnsafe(ref yr, (nuint)(base_ + 8));
                a2.StoreUnsafe(ref yr, (nuint)(base_ + 16));
                a3.StoreUnsafe(ref yr, (nuint)(base_ + 24));
                a4.StoreUnsafe(ref yr, (nuint)(base_ + 32));
                a5.StoreUnsafe(ref yr, (nuint)(base_ + 40));
                a6.StoreUnsafe(ref yr, (nuint)(base_ + 48));
                a7.StoreUnsafe(ref yr, (nuint)(base_ + 56));
            }
        }
        for (int t = base_; t < o; t++)
        {
            float s = accumulate ? y[t] : 0f;
            for (int j = 0; j < kk; j++) s = MathF.FusedMultiplyAdd(x[j], panel[j * o + t], s);
            y[t] = s;
        }
    }

    static void MatVecPanelScalar(ReadOnlySpan<float> x, ReadOnlySpan<float> panel, Span<float> y, int o, int kk, bool accumulate)
    {
        // Single-rounding FMA like the vector lanes, so every hardware and
        // options path agrees bit for bit.
        for (int t = 0; t < o; t++)
        {
            float s = accumulate ? y[t] : 0f;
            for (int j = 0; j < kk; j++) s = MathF.FusedMultiplyAdd(x[j], panel[j * o + t], s);
            y[t] = s;
        }
    }

    /// <summary>
    /// Vector single-precision exp, Cephes-style range reduction with a
    /// degree-5 polynomial: matches scalar MathF.Exp within a few ulps over
    /// the finite range (validated at 1e-6 relative, never bit-identical).
    /// Out-of-range inputs saturate (+Inf above ~88, +0 below ~-88, NaN
    /// propagates) exactly like the scalar edges the sigmoid/tanh wrappers
    /// depend on. Requires AVX/FMA/AVX2 (checked by callers).
    /// </summary>
    static Vector256<float> ExpVector256(Vector256<float> v)
    {
        var log2e = Vector256.Create(1.44269504088896341f);
        var c1 = Vector256.Create(-0.693359375f);
        var c2 = Vector256.Create(2.12194440e-4f);
        var p0 = Vector256.Create(1.9875691500E-4f);
        var p1 = Vector256.Create(1.3981999507E-3f);
        var p2 = Vector256.Create(8.3334519073E-3f);
        var p3 = Vector256.Create(4.1665795894E-2f);
        var p4 = Vector256.Create(1.6666665459E-1f);
        var p5 = Vector256.Create(5.0000001201E-1f);
        var n = Avx.ConvertToVector256Int32(Avx.Floor(Fma.MultiplyAdd(v, log2e, Vector256.Create(0.5f))));
        var r = Fma.MultiplyAdd(Avx.ConvertToVector256Single(n), c1, v);
        r = Fma.MultiplyAdd(Avx.ConvertToVector256Single(n), c2, r);
        var z = r * r;
        var y = Fma.MultiplyAdd(p0, r, p1);
        y = Fma.MultiplyAdd(y, r, p2);
        y = Fma.MultiplyAdd(y, r, p3);
        y = Fma.MultiplyAdd(y, r, p4);
        y = Fma.MultiplyAdd(y, r, p5);
        y = Fma.MultiplyAdd(y, z, r);
        y = y + Vector256<float>.One;
        var scale = Avx2.ShiftLeftLogical(n + Vector256.Create(127), 23).AsSingle();
        var scaled = y * scale;
        var over = Avx.CompareGreaterThan(v, Vector256.Create(88.3762626647949f));
        var under = Avx.CompareLessThan(v, Vector256.Create(-88.0f));
        var nan = Avx.CompareUnordered(v, v);
        var result = Avx.BlendVariable(scaled, Vector256.Create(float.PositiveInfinity), over);
        result = Avx.BlendVariable(result, Vector256<float>.Zero, under);
        result = Avx.BlendVariable(result, v, nan);
        return result;
    }

    /// <summary>
    /// Sigmoid over a span: 1/(1+exp(-x)) with the vector core, scalar libm
    /// tail. Matches the scalar LSTM/elementwise formula within float
    /// rounding (validated at 1e-6, never bit-identical).
    /// </summary>
    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SigmoidSpan(ReadOnlySpan<float> xs, Span<float> ys)
    {
        int n = Math.Min(xs.Length, ys.Length);
        if (!(Avx.IsSupported && Avx2.IsSupported && Fma.IsSupported))
        {
            for (int j = 0; j < n; j++) ys[j] = 1f / (1f + MathF.Exp(-xs[j]));
            return;
        }
        int i = 0;
        int full = n & ~7;
        unsafe
        {
            fixed (float* s = xs, d = ys)
            {
                for (; i < full; i += 8)
                {
                    var v = *(Vector256<float>*)(s + i);
                    var e = ExpVector256(Vector256<float>.Zero - v);
                    *(Vector256<float>*)(d + i) = Vector256<float>.One / (e + Vector256<float>.One);
                }
            }
        }
        for (; i < n; i++) ys[i] = 1f / (1f + MathF.Exp(-xs[i]));
    }

    /// <summary>
    /// Tanh over a span via odd symmetry around a non-overflowing exp:
    /// sign(x)*(1-2/(exp(2|x|)+1)) for |x|<=10, sign otherwise. Matches
    /// scalar MathF.Tanh within float rounding (validated at 1e-6, never
    /// bit-identical).
    /// </summary>
    // Tier0-stuck leaf (see PLAN qdA2): force Tier1; few benchmark calls never trip promotion.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void TanhSpan(ReadOnlySpan<float> xs, Span<float> ys)
    {
        int n = Math.Min(xs.Length, ys.Length);
        if (!(Avx.IsSupported && Avx2.IsSupported && Fma.IsSupported))
        {
            for (int j = 0; j < n; j++) ys[j] = MathF.Tanh(xs[j]);
            return;
        }
        int i = 0;
        int full = n & ~7;
        unsafe
        {
            fixed (float* s = xs, d = ys)
            {
                var ten = Vector256.Create(10f);
                var two = Vector256.Create(2f);
                for (; i < full; i += 8)
                {
                    var v = *(Vector256<float>*)(s + i);
                    var sign = Vector256.Create(1f);
                    var neg = Avx.CompareLessThan(v, Vector256<float>.Zero); // Note: -0.0 yields +0.0 here (== -0.0, no downstream effect).
                    var a = Avx.BlendVariable(v, Vector256<float>.Zero - v, neg);
                    sign = Avx.BlendVariable(sign, Vector256.Create(-1f), neg);
                    var e = ExpVector256(a + a);
                    var core = Vector256<float>.One - two / (e + Vector256<float>.One);
                    var big = Avx.CompareGreaterThan(a, ten);
                    *(Vector256<float>*)(d + i) = Avx.BlendVariable(sign * core, sign, big);
                }
            }
        }
        for (; i < n; i++) ys[i] = MathF.Tanh(xs[i]);
    }

    /// <summary>
    /// Swish over a span: x/(1+exp(-x)) with the vector core, scalar libm
    /// tail. Matches the unfused Sigmoid-times-x chain within float rounding
    /// (validated at 1e-6, never bit-identical).
    /// </summary>
    public static void SwishSpan(ReadOnlySpan<float> xs, Span<float> ys)
    {
        int n = Math.Min(xs.Length, ys.Length);
        if (!(Avx.IsSupported && Avx2.IsSupported && Fma.IsSupported))
        {
            for (int j = 0; j < n; j++) ys[j] = xs[j] / (1f + MathF.Exp(-xs[j]));
            return;
        }
        int i = 0;
        int full = n & ~7;
        unsafe
        {
            fixed (float* s = xs, d = ys)
            {
                for (; i < full; i += 8)
                {
                    var v = *(Vector256<float>*)(s + i);
                    var e = ExpVector256(Vector256<float>.Zero - v);
                    var g = Vector256<float>.One / (e + Vector256<float>.One);
                    *(Vector256<float>*)(d + i) = v * g;
                }
            }
        }
        for (; i < n; i++) ys[i] = xs[i] / (1f + MathF.Exp(-xs[i]));
    }

    /// <summary>
    /// Gated multiply over spans: plain[i]*sigmoid(gated[i]) with the vector
    /// core, scalar libm tail. Covers GLU halves and (with the same span on
    /// both sides) Swish; matches the unfused chain within float rounding
    /// (validated at 1e-6, never bit-identical).
    /// </summary>
    public static void SigmoidMulSpan(ReadOnlySpan<float> plain, ReadOnlySpan<float> gated, Span<float> ys)
    {
        int n = Math.Min(Math.Min(plain.Length, gated.Length), ys.Length);
        if (!(Avx.IsSupported && Avx2.IsSupported && Fma.IsSupported))
        {
            for (int j = 0; j < n; j++) ys[j] = plain[j] * (1f / (1f + MathF.Exp(-gated[j])));
            return;
        }
        int i = 0;
        int full = n & ~7;
        unsafe
        {
            fixed (float* p = plain, g = gated, d = ys)
            {
                for (; i < full; i += 8)
                {
                    var v = *(Vector256<float>*)(p + i);
                    var u = *(Vector256<float>*)(g + i);
                    var e = ExpVector256(Vector256<float>.Zero - u);
                    *(Vector256<float>*)(d + i) = v * (Vector256<float>.One / (e + Vector256<float>.One));
                }
            }
        }
        for (; i < n; i++) ys[i] = plain[i] * (1f / (1f + MathF.Exp(-gated[i])));
    }
    public static unsafe void Im2col(float* src,
                              int srcC,
                              int srcH,
                              int srcW,
                              int kernelY,
                              int kernelX,
                              int dilationY,
                              int dilationX,
                              int strideY,
                              int strideX,
                              int padY,
                              int padX,
                              int padH,
                              int padW,
                              float* buf)
    {
        int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
        int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
        for (int sc = 0; sc < srcC; ++sc)
        {
            for (int ky = 0; ky < kernelY; ++ky)
            {
                int row0 = ky * dilationY - padY;
                for (int kx = 0; kx < kernelX; ++kx)
                {
                    int col0 = kx * dilationX - padX;
                    for (int dy = 0; dy < dstH; ++dy)
                    {
                        int sy = row0 + dy * strideY;
                        var line = (uint)sy < (uint)srcH ? src + (sc * srcH + sy) * srcW : null;
                        for (int dx = 0; dx < dstW; ++dx, ++buf)
                        {
                            int sx = col0 + dx * strideX;
                            *buf = (line != null && (uint)sx < (uint)srcW) ? line[sx] : 0;
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Image to column conversion restricted to a contiguous output-column
    /// range. Column c holds output position (c / dstW, c % dstW), and the
    /// buffer lays out as [srcC*kernelY*kernelX, colCount] row-major with
    /// the same values the full Im2col patch holds at those columns. The
    /// shared dispatcher may still select different vectorized kernels per
    /// block shape, so tiled results agree with the single-pass path within
    /// float rounding (validated at the 1e-4 gate), not bit for bit.
    /// </summary>
    /// <param name="src">Source data.</param>
    /// <param name="srcC">Input channels.</param>
    /// <param name="srcH">Input height.</param>
    /// <param name="srcW">Input width.</param>
    /// <param name="kernelY">Kernel height.</param>
    /// <param name="kernelX">Kernel width.</param>
    /// <param name="dilationY">Dilation of the kernel by height.</param>
    /// <param name="dilationX">Dilation of the kernel by width.</param>
    /// <param name="strideY">Stride of the convolution by height.</param>
    /// <param name="strideX">Stride of the convolution by width.</param>
    /// <param name="padY">Zero padding at the top (begin height).</param>
    /// <param name="padX">Zero padding at the left (begin width).</param>
    /// <param name="padH">Zero padding at the bottom (end height).</param>
    /// <param name="padW">Zero padding at the right (end width).</param>
    /// <param name="dstW">Full output width; columns index dy * dstW + dx.</param>
    /// <param name="colStart">First output column to convert.</param>
    /// <param name="colCount">Number of output columns to convert.</param>
    /// <param name="buf">Buffer.</param>
    public static unsafe void Im2colRange(float* src,
                              int srcC,
                              int srcH,
                              int srcW,
                              int kernelY,
                              int kernelX,
                              int dilationY,
                              int dilationX,
                              int strideY,
                              int strideX,
                              int padY,
                              int padX,
                              int padH,
                              int padW,
                              int dstW,
                              int colStart,
                              int colCount,
                              float* buf)
    {
        int dyFirst = colStart / dstW;
        int dyLast = (colStart + colCount - 1) / dstW;
        // Pure copies, so vectorization is bit-identical; gate on hardware
        // only (no options reach this helper). At stride-one width with a
        // valid source row, the in-bounds run copies contiguously with
        // scalar zero borders; everything else keeps the scalar loop.
        // Taps share one source row: kernel-x sits innermost so each
        // (channel, tap-row, output-row) resolves its input line once for
        // all tap columns instead of once per patch row.
        bool vector = Avx.IsSupported;
        for (int sc = 0; sc < srcC; ++sc)
        {
            for (int ky = 0; ky < kernelY; ++ky)
            {
                int row0 = ky * dilationY - padY;
                for (int dy = dyFirst; dy <= dyLast; ++dy)
                {
                    int sy = row0 + dy * strideY;
                    var line = (uint)sy < (uint)srcH ? src + (sc * srcH + sy) * srcW : null;
                    int dxLo = dy == dyFirst ? colStart - dy * dstW : 0;
                    int dxHi = dy == dyLast ? colStart + colCount - dy * dstW : dstW;
                    for (int kx = 0; kx < kernelX; ++kx)
                    {
                        int col0 = kx * dilationX - padX;
                        float* row = buf + ((sc * kernelY + ky) * kernelX + kx) * colCount - colStart;
                        if (vector && strideX == 1 && line != null)
                        {
                            int vxLo = dxLo > -col0 ? dxLo : -col0;
                            int vxHi = dxHi < srcW - col0 ? dxHi : srcW - col0;
                            int i = dxLo;
                            for (; i < vxLo; i++) row[dy * dstW + i] = 0f;
                            int vecEnd = vxLo + ((vxHi - vxLo) & ~7);
                            for (; i < vecEnd; i += 8)
                                *(Vector256<float>*)(row + dy * dstW + i) = *(Vector256<float>*)(line + col0 + i);
                            for (; i < vxHi; i++) row[dy * dstW + i] = line[col0 + i];
                            for (; i < dxHi; i++) row[dy * dstW + i] = 0f;
                        }
                        else if (strideX == 2 && dilationX == 1 && line != null && (Avx512F.IsSupported || Avx2.IsSupported))
                        {
                            // Stride-two width gather: output dx reads line[col0 + 2 * dx],
                            // so each vector of outputs sits in twice as many contiguous
                            // inputs. Vector loads plus one deinterleave shuffle replace
                            // the scalar strided loads with their per-element bounds
                            // checks. The same input values land in the same patch slots,
                            // so the change is a pure copy reorder and stays bit-identical;
                            // the scalar head/tail cover padding and short rows exactly
                            // like the fallback below.
                            int loadLo = col0 >= 0 ? 0 : ((-col0 + 1) / 2);
                            float* dst = row + dy * dstW;
                            int i = dxLo;
                            if (Avx512F.IsSupported)
                            {
                                // 16 outputs per step from 32 contiguous inputs: a per-lane
                                // shuffle pairs each 128-bit lane, then a cross-lane permute
                                // orders the deinterleaved halves.
                                int loadHi = (srcW - 32 - col0) >= 0 ? ((srcW - 32 - col0) / 2 + 1) : -1;
                                int vLo = dxLo > loadLo ? dxLo : loadLo;
                                int vHi = dxHi < loadHi ? dxHi : loadHi;
                                var deint16 = Vector512.Create(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
                                for (; i < vLo; i++)
                                {
                                    int sx = col0 + i * 2;
                                    dst[i] = (uint)sx < (uint)srcW ? line[sx] : 0;
                                }
                                int vecEnd = vLo + ((vHi - vLo) & ~15);
                                for (; i < vecEnd; i += 16)
                                {
                                    float* s = line + col0 + i * 2;
                                    Vector512<float> t = Avx512F.Shuffle(*(Vector512<float>*)s, *(Vector512<float>*)(s + 16), (byte)0x88);
                                    *(Vector512<float>*)(dst + i) = Avx512F.PermuteVar16x32(t, deint16);
                                }
                                for (; i < dxHi; i++)
                                {
                                    int sx = col0 + i * 2;
                                    dst[i] = (uint)sx < (uint)srcW ? line[sx] : 0;
                                }
                            }
                            else
                            {
                                // 8 outputs per step from 16 contiguous inputs.
                                int loadHi = (srcW - 16 - col0) >= 0 ? ((srcW - 16 - col0) / 2 + 1) : -1;
                                int vLo = dxLo > loadLo ? dxLo : loadLo;
                                int vHi = dxHi < loadHi ? dxHi : loadHi;
                                var deint8 = Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7);
                                for (; i < vLo; i++)
                                {
                                    int sx = col0 + i * 2;
                                    dst[i] = (uint)sx < (uint)srcW ? line[sx] : 0;
                                }
                                int vecEnd = vLo + ((vHi - vLo) & ~7);
                                for (; i < vecEnd; i += 8)
                                {
                                    float* s = line + col0 + i * 2;
                                    Vector256<float> t = Avx.Shuffle(*(Vector256<float>*)s, *(Vector256<float>*)(s + 8), (byte)0x88);
                                    *(Vector256<float>*)(dst + i) = Avx2.PermuteVar8x32(t, deint8);
                                }
                                for (; i < dxHi; i++)
                                {
                                    int sx = col0 + i * 2;
                                    dst[i] = (uint)sx < (uint)srcW ? line[sx] : 0;
                                }
                            }
                        }
                        else
                        {
                            for (int dx = dxLo; dx < dxHi; ++dx)
                            {
                                int sx = col0 + dx * strideX;
                                row[dy * dstW + dx] = (line != null && (uint)sx < (uint)srcW) ? line[sx] : 0;
                            }
                        }
                    }
                }
            }
        }
    }

    public static unsafe void Im2col(double* src,
                             int srcC,
                             int srcH,
                             int srcW,
                             int kernelY,
                             int kernelX,
                             int dilationY,
                             int dilationX,
                             int strideY,
                             int strideX,
                             int padY,
                             int padX,
                             int padH,
                             int padW,
                             double* buf)
    {
        int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
        int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
        for (int sc = 0; sc < srcC; ++sc)
        {
            for (int ky = 0; ky < kernelY; ++ky)
            {
                int row0 = ky * dilationY - padY;
                for (int kx = 0; kx < kernelX; ++kx)
                {
                    int col0 = kx * dilationX - padX;
                    for (int dy = 0; dy < dstH; ++dy)
                    {
                        int sy = row0 + dy * strideY;
                        var line = (uint)sy < (uint)srcH ? src + (sc * srcH + sy) * srcW : null;
                        for (int dx = 0; dx < dstW; ++dx, ++buf)
                        {
                            int sx = col0 + dx * strideX;
                            *buf = (line != null && (uint)sx < (uint)srcW) ? line[sx] : 0;
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Im2Col-based implementation of two-dimensional convolution.
    /// </summary>
    /// <param name="src">Source data.</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="srcC">Input channels.</param>
    /// <param name="srcH">Input height.</param>
    /// <param name="srcW">Input width.</param>
    /// <param name="kernelY">Kernel height.</param>
    /// <param name="kernelX">Kernel width.</param>
    /// <param name="dilationY">Dilation of the kernel by height.</param>
    /// <param name="dilationX">Dilation of the kernel by width.</param>
    /// <param name="strideY">Stride of the convolution by height.</param>
    /// <param name="strideX">Stride of the convolution by width.</param>
    /// <param name="padY">Zero padding at the top (begin height).</param>
    /// <param name="padX">Zero padding at the left (begin width).</param>
    /// <param name="padH">Zero padding at the bottom (end height).</param>
    /// <param name="padW">Zero padding at the right (end width).</param>
    /// <param name="group">Convolution groups. If group=srcC=dstC, convolution is depthwise separable.</param>
    /// <param name="weight">Weights (kernels).</param>
    /// <param name="bias">Bias.</param>
    /// <param name="dst">Destination memory.</param>
    /// <param name="dstC">Output channels.</param>
    public static unsafe void Conv2D(float* src,
                                    int batch,
                                    int srcC,
                                    int srcH,
                                    int srcW,
                                    int kernelY,
                                    int kernelX,
                                    int dilationY,
                                    int dilationX,
                                    int strideY,
                                    int strideX,
                                    int padY,
                                    int padX,
                                    int padH,
                                    int padW,
                                    int group,
                                    float* weight,
                                    float* dst,
                                    int dstC,
                                    float* bias)
    {
        /// <summary>
        /// Matrix multiplication.
        /// </summary>
        /// <param name="M">A rows.</param>
        /// <param name="N">A columns.</param>
        /// <param name="K">B columns.</param>
        /// <param name="A">Left matrix.</param>
        /// <param name="B">Right matrix.</param>
        /// <param name="C">Result matrix.</param>
        unsafe void _mm(int M,
                              int N,
                              int K,
                              float* A,
                              float* B,
                              float* C)
        {
            for (int i = 0; i < M; i++)
            {
                var Cp = C + i * N;
                var Ap = A + i * K;
                for (int j = 0; j < N; ++j)
                {
                    Cp[j] = 0;
                }
                for (int k = 0; k < K; ++k)
                {
                    var a = Ap[k];
                    var Bp = B + k * N;
                    for (int j = 0; j < N; ++j)
                    {
                        Cp[j] += a * Bp[j];
                    }
                }
            }
        }

        int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
        int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
        int M = dstC / group;
        int N = dstH * dstW;
        int K = srcC * kernelY * kernelX / group;
        var buf = (float*)Marshal.AllocCoTaskMem(srcC * kernelY * kernelX * dstH * dstW * sizeof(float));
        try
        {
            for (int b = 0; b < batch; ++b)
            {
                Im2col(src, srcC, srcH, srcW, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, buf);
                for (int g = 0; g < group; ++g)
                {
                    _mm(M, N, K, weight + M * K * g, buf + N * K * g, dst + M * N * g);
                }

                if (bias != null)
                {
                    for (int i = 0; i < dstC; ++i)
                    {
                        var pdst = dst + i * N;
                        for (int j = 0; j < N; ++j)
                        {
                            pdst[j] += bias[i];
                        }
                    }
                }
                src += srcC * srcH * srcW;
                dst += dstC * dstH * dstW;
            }
        }
        finally { Marshal.FreeCoTaskMem((IntPtr)buf); }
    }


    /// <summary>
    /// Im2Col-based implementation of two-dimensional convolution.
    /// </summary>
    /// <param name="src">Source data.</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="srcC">Input channels.</param>
    /// <param name="srcH">Input height.</param>
    /// <param name="srcW">Input width.</param>
    /// <param name="kernelY">Kernel height.</param>
    /// <param name="kernelX">Kernel width.</param>
    /// <param name="dilationY">Dilation of the kernel by height.</param>
    /// <param name="dilationX">Dilation of the kernel by width.</param>
    /// <param name="strideY">Stride of the convolution by height.</param>
    /// <param name="strideX">Stride of the convolution by width.</param>
    /// <param name="padY">Zero padding at the top (begin height).</param>
    /// <param name="padX">Zero padding at the left (begin width).</param>
    /// <param name="padH">Zero padding at the bottom (end height).</param>
    /// <param name="padW">Zero padding at the right (end width).</param>
    /// <param name="group">Convolution groups. If group=srcC=dstC, convolution is depthwise separable.</param>
    /// <param name="weight">Weights (kernels).</param>
    /// <param name="bias">Bias.</param>
    /// <param name="dst">Destination memory.</param>
    /// <param name="dstC">Output channels.</param
    public static unsafe void Conv2D(double* src,
                                    int batch,
                                    int srcC,
                                    int srcH,
                                    int srcW,
                                    int kernelY,
                                    int kernelX,
                                    int dilationY,
                                    int dilationX,
                                    int strideY,
                                    int strideX,
                                    int padY,
                                    int padX,
                                    int padH,
                                    int padW,
                                    int group,
                                    double* weight,
                                    double* bias,
                                    double* dst,
                                    int dstC)
    {
        /// <summary>
        /// Matrix multiplication.
        /// </summary>
        /// <param name="M">A rows.</param>
        /// <param name="N">A columns.</param>
        /// <param name="K">B columns.</param>
        /// <param name="A">Left matrix.</param>
        /// <param name="B">Right matrix.</param>
        /// <param name="C">Result matrix.</param>
        unsafe void _mm(int M,
                              int N,
                              int K,
                              double* A,
                              double* B,
                              double* C)
        {
            for (int i = 0; i < M; i++)
            {
                var Cp = C + i * N;
                var Ap = A + i * K;
                for (int j = 0; j < N; ++j)
                {
                    Cp[j] = 0;
                }
                for (int k = 0; k < K; ++k)
                {
                    var a = Ap[k];
                    var Bp = B + k * N;
                    for (int j = 0; j < N; ++j)
                    {
                        Cp[j] += a * Bp[j];
                    }
                }
            }
        }

        int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
        int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
        int M = dstC / group;
        int N = dstH * dstW;
        int K = srcC * kernelY * kernelX / group;
        var buf = (double*)Marshal.AllocCoTaskMem(srcC * kernelY * kernelX * dstH * dstW * sizeof(double));
        try
        {
            for (int b = 0; b < batch; ++b)
            {
                Im2col(src, srcC, srcH, srcW, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, buf);
                for (int g = 0; g < group; ++g)
                {
                    _mm(M, N, K, weight + M * K * g, buf + N * K * g, dst + M * N * g);
                }
                if (bias != null)
                {
                    for (int i = 0; i < dstC; ++i)
                    {
                        var pdst = dst + i * N;
                        for (int j = 0; j < N; ++j)
                        {
                            pdst[j] += bias[i];
                        }
                    }
                }
                src += srcC * srcH * srcW;
                dst += dstC * dstH * dstW;
            }
        }
        finally { Marshal.FreeCoTaskMem((IntPtr)buf); }
    }

    // Independently implemented from Abramowitz and Stegun, Handbook of Mathematical
    // Functions, formula 7.1.26 (public-domain U.S. government work): for x >= 0,
    // erf(x) = 1 - (a1*t + a2*t^2 + ... + a5*t^5) * exp(-x^2) with t = 1/(1+p*x),
    // and erf is odd, so erf(-x) = -erf(x). The coefficients below are the published
    // mathematical constants of that formula; the descending table plus Horner loop
    // and the early-return odd symmetry are this implementation's own expression.
    private const float ErfStegunP = 0.3275911f;
    private static readonly float[] ErfStegunTableDesc = new float[] { 1.061405429f, -1.453152027f, 1.421413741f, -0.284496736f, 0.254829592f };
    public static float Erf(float x)
    {
        // A&S 7.1.26 coefficients sum to 0.999999999, leaving residue at the
        // origin in wider arithmetic; erf(0) is exactly 0 by definition,
        // and erf is odd, so the signed zero itself is returned (ORT 1.29:
        // erf(+0)=+0, erf(-0)=-0).
        if (x == 0) return x;
        if (x < 0)
        {
            return -ErfMagnitude(-x);
        }
        return ErfMagnitude(Math.Abs(x));
    }
    private static float ErfMagnitude(float ax)
    {
        float t = 1.0f / (1.0f + ErfStegunP * ax);
        float poly = ErfStegunTableDesc[0];
        for (int i = 1; i < ErfStegunTableDesc.Length; i++)
        {
            poly = poly * t + ErfStegunTableDesc[i];
        }
        return 1.0f - poly * t * MathF.Exp(-ax * ax);
    }

    /// <summary>Vectorized error function; MLAS rational approximation (split polynomial plus embedded exponential), FMA evaluation.</summary>
    /// <remarks>Max absolute error 6.2e-08 against 50-digit truth, measured on a dense sweep plus split-boundary values (the Abramowitz-Stegun core it replaces measured 2.7e-07). NaN in, NaN out; infinities saturate to +-1.</remarks>
    public static Vector<float> ErfVector(Vector<float> v)
    {
        var negZero = new Vector<float>(-0.0f);
        var signBits = Vector.BitwiseAnd(v, negZero);
        var ax = Vector.BitwiseAnd(Vector.OnesComplement(negZero), v);
        ax = Vector.ConditionalSelect(Vector.GreaterThan(ax, new Vector<float>(3.925f)), new Vector<float>(3.925f), ax);
        var sq = ax * ax;
        var rs = new Vector<float>(-5.99104969e-4f);
        rs = Vector.FusedMultiplyAdd(rs, sq, new Vector<float>(4.99339588e-3f));
        rs = Vector.FusedMultiplyAdd(rs, sq, new Vector<float>(-2.67667342e-2f));
        rs = Vector.FusedMultiplyAdd(rs, sq, new Vector<float>(1.12818025e-1f));
        rs = Vector.FusedMultiplyAdd(rs, sq, new Vector<float>(-3.76124859e-1f));
        rs = Vector.FusedMultiplyAdd(rs, sq, new Vector<float>(1.28379151e-1f));
        rs = Vector.FusedMultiplyAdd(rs, ax, ax);
        var big = Vector.GreaterThan(ax, new Vector<float>(0.921875f));
        rs = Vector.ConditionalSelect(big, Vector<float>.Zero, rs);
        var ab = Vector.ConditionalSelect(big, ax, Vector<float>.Zero);
        var rb = new Vector<float>(1.72948930e-5f);
        rb = Vector.FusedMultiplyAdd(rb, ab, new Vector<float>(-3.83208680e-4f));
        rb = Vector.FusedMultiplyAdd(rb, ab, new Vector<float>(3.88393435e-3f));
        rb = Vector.FusedMultiplyAdd(rb, ab, new Vector<float>(-2.42545605e-2f));
        rb = Vector.FusedMultiplyAdd(rb, ab, new Vector<float>(1.06777847e-1f));
        rb = Vector.FusedMultiplyAdd(rb, ab, new Vector<float>(6.34846687e-1f));
        rb = Vector.FusedMultiplyAdd(rb, ab, new Vector<float>(1.28717512e-1f));
        rb = Vector.FusedMultiplyAdd(rb, ab, ab);
        var t = Vector<float>.Zero - rb;
        t = Vector.ConditionalSelect(Vector.LessThan(t, new Vector<float>(-88.3762626647949f)), new Vector<float>(-88.3762626647949f), t);
        var r = Vector.FusedMultiplyAdd(new Vector<float>(1.44269504088896341f), t, new Vector<float>(12582912.0f));
        r = r - new Vector<float>(12582912.0f);
        var fx = Vector.FusedMultiplyAdd(r, new Vector<float>(-6.93145752e-1f), t);
        fx = Vector.FusedMultiplyAdd(r, new Vector<float>(-1.42860677e-6f), fx);
        var y = new Vector<float>(1.38319808e-3f);
        y = Vector.FusedMultiplyAdd(y, fx, new Vector<float>(8.37550033e-3f));
        y = Vector.FusedMultiplyAdd(y, fx, new Vector<float>(4.16689515e-2f));
        y = Vector.FusedMultiplyAdd(y, fx, new Vector<float>(1.66664466e-1f));
        y = Vector.FusedMultiplyAdd(y, fx, new Vector<float>(4.99999851e-1f));
        y = Vector.FusedMultiplyAdd(y, fx, Vector<float>.One);
        y = Vector.FusedMultiplyAdd(y, fx, Vector<float>.One);
        var ri = Vector.ConvertToInt32(r);
        ri = Vector.Min(Vector.Max(ri, new Vector<int>(-126)), new Vector<int>(127));
        y = y * Vector.AsVectorSingle(Vector.ShiftLeft(ri + new Vector<int>(127), 23));
        y = Vector<float>.One - y;
        y = Vector.BitwiseOr(rs, y);
        return Vector.BitwiseOr(y, signBits);
    }

    /// <summary>Vectorized base-e exponential; Taylor degree-7 over a Cody-Waite reduced argument with FMA Horner evaluation.</summary>
    /// <remarks>Relative error is order 1e-7 against MathF.Exp on finite inputs. NaN in, NaN out; large positives overflow to +Infinity and large negatives underflow toward zero like the scalar path.</remarks>
    public static Vector<float> ExpVector(Vector<float> v)
    {
        var isFinite = Vector.Equals(v, v);
        var x = Vector.Min(Vector.Max(v, new Vector<float>(-88.722839f)), new Vector<float>(88.722839f));
        var scaled = x * new Vector<float>(1.44269504088896341f);
        var shifted = Vector.ConditionalSelect(Vector.GreaterThanOrEqual(scaled, Vector<float>.Zero), scaled + new Vector<float>(0.5f), scaled - new Vector<float>(0.5f));
        var n = Vector.ConvertToInt32(shifted);
        var clamped = Vector.Min(Vector.Max(n, new Vector<int>(-126)), new Vector<int>(127));
        var nf = Vector.ConvertToSingle(clamped);
        var r = Vector.FusedMultiplyAdd(nf, new Vector<float>(-0.693359375f), x);
        r = Vector.FusedMultiplyAdd(nf, new Vector<float>(2.12194440e-4f), r);
        var p = new Vector<float>(1f / 5040f);
        p = Vector.FusedMultiplyAdd(p, r, new Vector<float>(1f / 720f));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<float>(1f / 120f));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<float>(1f / 24f));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<float>(1f / 6f));
        p = Vector.FusedMultiplyAdd(p, r, new Vector<float>(0.5f));
        p = Vector.FusedMultiplyAdd(p, r, Vector<float>.One);
        p = Vector.FusedMultiplyAdd(p, r, Vector<float>.One);
        var scale = Vector.AsVectorSingle(Vector.ShiftLeft(clamped + new Vector<int>(127), 23));
        var y = p * scale;
        y = Vector.ConditionalSelect(Vector.GreaterThan(v, new Vector<float>(88.722839f)), new Vector<float>(float.PositiveInfinity), y);
        y = Vector.ConditionalSelect(Vector.LessThan(v, new Vector<float>(-88.722839f)), Vector<float>.Zero, y);
        return Vector.ConditionalSelect(isFinite, y, new Vector<float>(float.NaN));
    }
    // Same Abramowitz and Stegun 7.1.26 derivation as the float overload above,
    // in double precision with its own table and magnitude helper.
    private const double ErfStegunPDouble = 0.3275911;
    private static readonly double[] ErfStegunTableDescDouble = new double[] { 1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592 };
    public static double Erf(double x)
    {
        // Same signed-zero origin guard as the float overload above:
        // erf is odd, so erf(-0) is -0 (no ORT double kernel; exact).
        if (x == 0) return x;
        if (x < 0)
        {
            return -ErfMagnitudeDouble(-x);
        }
        return ErfMagnitudeDouble(Math.Abs(x));
    }
    private static double ErfMagnitudeDouble(double ax)
    {
        double t = 1.0 / (1.0 + ErfStegunPDouble * ax);
        double poly = ErfStegunTableDescDouble[0];
        for (int i = 1; i < ErfStegunTableDescDouble.Length; i++)
        {
            poly = poly * t + ErfStegunTableDescDouble[i];
        }
        return 1.0 - poly * t * Math.Exp(-ax * ax);
    }


}
