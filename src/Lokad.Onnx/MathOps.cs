using System;
using System.Numerics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Diagnostics;

namespace Lokad.Onnx;

public partial class MathOps
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
    /// Single-row K-blocked GEMV-style kernel (P06 decode path). The classic
    /// 1-row kernel read-modify-writes the whole C row per reduction step,
    /// which triples traffic once C outgrows L1 (logits: 200KB rows). This
    /// keeps one 32-column chunk resident in registers across the full
    /// reduction and stores once, so B streams once and C is written once.
    /// Per-element arithmetic matches the 1-row kernel bit-wise: FMA chains
    /// in j-ascending order over [0, ceiling), scalar mul-add over the same
    /// tail, with the 32-column blocking boundaries aligned to the 8-wide
    /// vector lanes (32 is a multiple of 8).
    /// </summary>
    /// <param name="M">A rows (must be 1).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix (one row).</param>
    /// <param name="B">Right matrix, row-major.</param>
    /// <param name="C">Result row (accumulated, like the 1-row kernel).</param>
    /// </summary>
    public unsafe static void mm_m1_kblocked(int M,
                          int N,
                          int K,
                          float* A,
                          float* B,
                          float* C)
    {
        if (M != 1)
            throw new ArgumentException(nameof(M));
        const int Chunk = 32; // 4 AVX vectors; Vector256<float>.Count is not a C# const
        int blocked = K - (K % Chunk);
        for (int kb = 0; kb < blocked; kb += Chunk)
        {
            var Cpv = (Vector256<float>*)(C + kb);
            Vector256<float> c0 = Cpv[0];
            Vector256<float> c1 = Cpv[1];
            Vector256<float> c2 = Cpv[2];
            Vector256<float> c3 = Cpv[3];
            for (int j = 0; j < N; ++j)
            {
                var av = Vector256.Create(A[j]);
                var Bpv = (Vector256<float>*)(B + j * K + kb);
                c0 = Fma.MultiplyAdd(Bpv[0], av, c0);
                c1 = Fma.MultiplyAdd(Bpv[1], av, c1);
                c2 = Fma.MultiplyAdd(Bpv[2], av, c2);
                c3 = Fma.MultiplyAdd(Bpv[3], av, c3);
            }
            Cpv[0] = c0;
            Cpv[1] = c1;
            Cpv[2] = c2;
            Cpv[3] = c3;
        }
        int ceiling = (K / Vector256<float>.Count) * Vector256<float>.Count;
        for (int k = blocked; k < ceiling; k += Vector256<float>.Count)
        {
            Vector256<float> c = *(Vector256<float>*)(C + k);
            for (int j = 0; j < N; ++j)
            {
                var Bpv = (Vector256<float>*)(B + j * K + k);
                c = Fma.MultiplyAdd(Bpv[0], Vector256.Create(A[j]), c);
            }
            *(Vector256<float>*)(C + k) = c;
        }
        for (int k = ceiling; k < K; k++)
        {
            for (int j = 0; j < N; ++j) C[k] += A[j] * B[j * K + k];
        }
    }

    /// <summary>
    /// Register-tiled matrix multiplication with epilogue scaling (E5-1 M2):
    /// identical tiles, FMA order and tails to
    /// mm_unsafe_vectorized_intrinsics_2x4tiled, except every stored element
    /// is scaled once by alpha on its single write-back. The destination must
    /// be zeroed on entry (same accumulate contract as the tiled kernel); each
    /// element is written exactly once across main, rem and tail regions, so
    /// no scaling is ever applied twice or missed. Scalar tail accumulates per
    /// column in registers in the same j-ascending order, then scales once.
    /// </summary>
    /// <param name="M">A rows (must be even).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="B">Right matrix.</param>
    /// <param name="C">Zeroed result matrix.</param>
    /// <param name="alpha">Epilogue scale applied once per stored element.</param>
    public unsafe static void mm_unsafe_vectorized_intrinsics_2x4tiled_alpha(int M,
                          int N,
                          int K,
                          float* A,
                          float* B,
                          float* C,
                          float alpha)
    {
        if (M % 2 != 0)
            throw new ArgumentException(nameof(M));

        var alphaVec = Vector256.Create(alpha);
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
                Cpv1[0] = c00 * alphaVec;
                Cpv1[1] = c01 * alphaVec;
                Cpv1[2] = c02 * alphaVec;
                Cpv1[3] = c03 * alphaVec;
                Cpv2[0] = c10 * alphaVec;
                Cpv2[1] = c11 * alphaVec;
                Cpv2[2] = c12 * alphaVec;
                Cpv2[3] = c13 * alphaVec;
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
                    rC1[t] = c1 * alphaVec;
                    rC2[t] = c2 * alphaVec;
                }
                // Scalar tail: same j-ascending per-column accumulation as the
                // tiled kernel, register-held, scaled once on the single store.
                for (int k = blocked + rv * Vector256<float>.Count; k < K; k++)
                {
                    float acc1 = Cp1[k];
                    float acc2 = Cp2[k];
                    for (int j = 0; j < N; ++j)
                    {
                        acc1 += Ap1[j] * (B + j * K)[k];
                        acc2 += Ap2[j] * (B + j * K)[k];
                    }
                    Cp1[k] = alpha * acc1;
                    Cp2[k] = alpha * acc2;
                }
            }
        }
    }


    /// <summary>
    /// Register-tiled matrix multiplication over 6-row groups with 16-column
    /// halves (E5-2 M1): the same per-element j-ascending FMA chains as
    /// mm_unsafe_vectorized_intrinsics_2x4tiled, but each B vector is loaded
    /// once per 6-row group and reused across all six rows through one live
    /// broadcast temporary per vector, so B streams M/6 times instead of M/2.
    /// Twelve accumulators (6 rows by 2 vectors) fit the sixteen AVX2 registers
    /// alongside the B vectors and one broadcast; 6x32 halves would spill.
    /// M must be a multiple of 6; dispatch covers other even counts with the
    /// 2-row kernel. Vector coverage matches the tiled kernel exactly (every
    /// k below the 8-wide ceiling runs FMA chains, the scalar tail keeps the
    /// legacy per-step order), so results agree with it bit-wise on every shape.
    /// </summary>
    /// <param name="M">A rows (must be a multiple of 6).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="B">Right matrix.</param>
    /// <param name="C">Result matrix (accumulated, like the tiled kernel).</param>
    public unsafe static void mm_unsafe_vectorized_intrinsics_6x2tiled(int M,
                          int N,
                          int K,
                          float* A,
                          float* B,
                          float* C)
    {
        if (M % 6 != 0)
            throw new ArgumentException(nameof(M));

        int half = 2 * Vector256<float>.Count;
        int blocked = K - (K % half);

        for (int i = 0; i < M; i += 6)
        {
            var Ap0 = A + (i + 0) * N;
            var Ap1 = A + (i + 1) * N;
            var Ap2 = A + (i + 2) * N;
            var Ap3 = A + (i + 3) * N;
            var Ap4 = A + (i + 4) * N;
            var Ap5 = A + (i + 5) * N;

            var Cp0 = C + (i + 0) * K;
            var Cp1 = C + (i + 1) * K;
            var Cp2 = C + (i + 2) * K;
            var Cp3 = C + (i + 3) * K;
            var Cp4 = C + (i + 4) * K;
            var Cp5 = C + (i + 5) * K;

            for (int kb = 0; kb < blocked; kb += half)
            {
                var Q0 = (Vector256<float>*)(Cp0 + kb);
                var Q1 = (Vector256<float>*)(Cp1 + kb);
                var Q2 = (Vector256<float>*)(Cp2 + kb);
                var Q3 = (Vector256<float>*)(Cp3 + kb);
                var Q4 = (Vector256<float>*)(Cp4 + kb);
                var Q5 = (Vector256<float>*)(Cp5 + kb);
                Vector256<float> c00 = Q0[0];
                Vector256<float> c01 = Q0[1];
                Vector256<float> c10 = Q1[0];
                Vector256<float> c11 = Q1[1];
                Vector256<float> c20 = Q2[0];
                Vector256<float> c21 = Q2[1];
                Vector256<float> c30 = Q3[0];
                Vector256<float> c31 = Q3[1];
                Vector256<float> c40 = Q4[0];
                Vector256<float> c41 = Q4[1];
                Vector256<float> c50 = Q5[0];
                Vector256<float> c51 = Q5[1];
                for (int j = 0; j < N; ++j)
                {
                    var Bpv = (Vector256<float>*)(B + j * K + kb);
                    Vector256<float> bv0 = Bpv[0];
                    Vector256<float> bv1 = Bpv[1];
                    var av0 = Vector256.Create(Ap0[j]);
                    c00 = Fma.MultiplyAdd(bv0, av0, c00);
                    c01 = Fma.MultiplyAdd(bv1, av0, c01);
                    var av1 = Vector256.Create(Ap1[j]);
                    c10 = Fma.MultiplyAdd(bv0, av1, c10);
                    c11 = Fma.MultiplyAdd(bv1, av1, c11);
                    var av2 = Vector256.Create(Ap2[j]);
                    c20 = Fma.MultiplyAdd(bv0, av2, c20);
                    c21 = Fma.MultiplyAdd(bv1, av2, c21);
                    var av3 = Vector256.Create(Ap3[j]);
                    c30 = Fma.MultiplyAdd(bv0, av3, c30);
                    c31 = Fma.MultiplyAdd(bv1, av3, c31);
                    var av4 = Vector256.Create(Ap4[j]);
                    c40 = Fma.MultiplyAdd(bv0, av4, c40);
                    c41 = Fma.MultiplyAdd(bv1, av4, c41);
                    var av5 = Vector256.Create(Ap5[j]);
                    c50 = Fma.MultiplyAdd(bv0, av5, c50);
                    c51 = Fma.MultiplyAdd(bv1, av5, c51);
                }
                Q0[0] = c00;
                Q0[1] = c01;
                Q1[0] = c10;
                Q1[1] = c11;
                Q2[0] = c20;
                Q2[1] = c21;
                Q3[0] = c30;
                Q3[1] = c31;
                Q4[0] = c40;
                Q4[1] = c41;
                Q5[0] = c50;
                Q5[1] = c51;
            }
            int rem = K - blocked;
            if (rem > 0)
            {
                int rv = rem / Vector256<float>.Count;
                for (int t = 0; t < rv; t++)
                {
                    Vector256<float> c0 = ((Vector256<float>*)(Cp0 + blocked))[t];
                    Vector256<float> c1 = ((Vector256<float>*)(Cp1 + blocked))[t];
                    Vector256<float> c2 = ((Vector256<float>*)(Cp2 + blocked))[t];
                    Vector256<float> c3 = ((Vector256<float>*)(Cp3 + blocked))[t];
                    Vector256<float> c4 = ((Vector256<float>*)(Cp4 + blocked))[t];
                    Vector256<float> c5 = ((Vector256<float>*)(Cp5 + blocked))[t];
                    for (int j = 0; j < N; ++j)
                    {
                        var Bpv = (Vector256<float>*)(B + j * K + blocked);
                        Vector256<float> bv = Bpv[t];
                        c0 = Fma.MultiplyAdd(bv, Vector256.Create(Ap0[j]), c0);
                        c1 = Fma.MultiplyAdd(bv, Vector256.Create(Ap1[j]), c1);
                        c2 = Fma.MultiplyAdd(bv, Vector256.Create(Ap2[j]), c2);
                        c3 = Fma.MultiplyAdd(bv, Vector256.Create(Ap3[j]), c3);
                        c4 = Fma.MultiplyAdd(bv, Vector256.Create(Ap4[j]), c4);
                        c5 = Fma.MultiplyAdd(bv, Vector256.Create(Ap5[j]), c5);
                    }
                    ((Vector256<float>*)(Cp0 + blocked))[t] = c0;
                    ((Vector256<float>*)(Cp1 + blocked))[t] = c1;
                    ((Vector256<float>*)(Cp2 + blocked))[t] = c2;
                    ((Vector256<float>*)(Cp3 + blocked))[t] = c3;
                    ((Vector256<float>*)(Cp4 + blocked))[t] = c4;
                    ((Vector256<float>*)(Cp5 + blocked))[t] = c5;
                }
                // Scalar tail keeps the reduction-major order of the
                // unrolled kernel (accumulate into the destination per step)
                // so results agree with it bit-wise on every shape.
                for (int j = 0; j < N; ++j)
                {
                    float a0 = Ap0[j];
                    float a1 = Ap1[j];
                    float a2 = Ap2[j];
                    float a3 = Ap3[j];
                    float a4 = Ap4[j];
                    float a5 = Ap5[j];
                    var Brow = B + j * K;
                    for (int k = blocked + rv * Vector256<float>.Count; k < K; k++)
                    {
                        Cp0[k] += a0 * Brow[k];
                        Cp1[k] += a1 * Brow[k];
                        Cp2[k] += a2 * Brow[k];
                        Cp3[k] += a3 * Brow[k];
                        Cp4[k] += a4 * Brow[k];
                        Cp5[k] += a5 * Brow[k];
                    }
                }
            }
        }
    }

    /// <summary>
    /// Register-tiled 6-row multiplication with epilogue scaling (E5-2 M1):
    /// identical tiles, FMA order and tails to
    /// mm_unsafe_vectorized_intrinsics_6x2tiled, except every stored element
    /// is scaled once by alpha on its single write-back. The destination must
    /// be zeroed on entry (same accumulate contract as the 2-row alpha twin);
    /// each element is written exactly once across main, rem and tail regions,
    /// so no scaling is ever applied twice or missed. Scalar tail accumulates
    /// per column in registers in the same j-ascending order, then scales once.
    /// </summary>
    /// <param name="M">A rows (must be a multiple of 6).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="B">Right matrix.</param>
    /// <param name="C">Zeroed result matrix.</param>
    /// <param name="alpha">Epilogue scale applied once per stored element.</param>
    public unsafe static void mm_unsafe_vectorized_intrinsics_6x2tiled_alpha(int M,
                          int N,
                          int K,
                          float* A,
                          float* B,
                          float* C,
                          float alpha)
    {
        if (M % 6 != 0)
            throw new ArgumentException(nameof(M));

        var alphaVec = Vector256.Create(alpha);
        int half = 2 * Vector256<float>.Count;
        int blocked = K - (K % half);

        for (int i = 0; i < M; i += 6)
        {
            var Ap0 = A + (i + 0) * N;
            var Ap1 = A + (i + 1) * N;
            var Ap2 = A + (i + 2) * N;
            var Ap3 = A + (i + 3) * N;
            var Ap4 = A + (i + 4) * N;
            var Ap5 = A + (i + 5) * N;

            var Cp0 = C + (i + 0) * K;
            var Cp1 = C + (i + 1) * K;
            var Cp2 = C + (i + 2) * K;
            var Cp3 = C + (i + 3) * K;
            var Cp4 = C + (i + 4) * K;
            var Cp5 = C + (i + 5) * K;

            for (int kb = 0; kb < blocked; kb += half)
            {
                var Q0 = (Vector256<float>*)(Cp0 + kb);
                var Q1 = (Vector256<float>*)(Cp1 + kb);
                var Q2 = (Vector256<float>*)(Cp2 + kb);
                var Q3 = (Vector256<float>*)(Cp3 + kb);
                var Q4 = (Vector256<float>*)(Cp4 + kb);
                var Q5 = (Vector256<float>*)(Cp5 + kb);
                Vector256<float> c00 = Q0[0];
                Vector256<float> c01 = Q0[1];
                Vector256<float> c10 = Q1[0];
                Vector256<float> c11 = Q1[1];
                Vector256<float> c20 = Q2[0];
                Vector256<float> c21 = Q2[1];
                Vector256<float> c30 = Q3[0];
                Vector256<float> c31 = Q3[1];
                Vector256<float> c40 = Q4[0];
                Vector256<float> c41 = Q4[1];
                Vector256<float> c50 = Q5[0];
                Vector256<float> c51 = Q5[1];
                for (int j = 0; j < N; ++j)
                {
                    var Bpv = (Vector256<float>*)(B + j * K + kb);
                    Vector256<float> bv0 = Bpv[0];
                    Vector256<float> bv1 = Bpv[1];
                    var av0 = Vector256.Create(Ap0[j]);
                    c00 = Fma.MultiplyAdd(bv0, av0, c00);
                    c01 = Fma.MultiplyAdd(bv1, av0, c01);
                    var av1 = Vector256.Create(Ap1[j]);
                    c10 = Fma.MultiplyAdd(bv0, av1, c10);
                    c11 = Fma.MultiplyAdd(bv1, av1, c11);
                    var av2 = Vector256.Create(Ap2[j]);
                    c20 = Fma.MultiplyAdd(bv0, av2, c20);
                    c21 = Fma.MultiplyAdd(bv1, av2, c21);
                    var av3 = Vector256.Create(Ap3[j]);
                    c30 = Fma.MultiplyAdd(bv0, av3, c30);
                    c31 = Fma.MultiplyAdd(bv1, av3, c31);
                    var av4 = Vector256.Create(Ap4[j]);
                    c40 = Fma.MultiplyAdd(bv0, av4, c40);
                    c41 = Fma.MultiplyAdd(bv1, av4, c41);
                    var av5 = Vector256.Create(Ap5[j]);
                    c50 = Fma.MultiplyAdd(bv0, av5, c50);
                    c51 = Fma.MultiplyAdd(bv1, av5, c51);
                }
                Q0[0] = c00 * alphaVec;
                Q0[1] = c01 * alphaVec;
                Q1[0] = c10 * alphaVec;
                Q1[1] = c11 * alphaVec;
                Q2[0] = c20 * alphaVec;
                Q2[1] = c21 * alphaVec;
                Q3[0] = c30 * alphaVec;
                Q3[1] = c31 * alphaVec;
                Q4[0] = c40 * alphaVec;
                Q4[1] = c41 * alphaVec;
                Q5[0] = c50 * alphaVec;
                Q5[1] = c51 * alphaVec;
            }
            int rem = K - blocked;
            if (rem > 0)
            {
                int rv = rem / Vector256<float>.Count;
                for (int t = 0; t < rv; t++)
                {
                    Vector256<float> c0 = ((Vector256<float>*)(Cp0 + blocked))[t];
                    Vector256<float> c1 = ((Vector256<float>*)(Cp1 + blocked))[t];
                    Vector256<float> c2 = ((Vector256<float>*)(Cp2 + blocked))[t];
                    Vector256<float> c3 = ((Vector256<float>*)(Cp3 + blocked))[t];
                    Vector256<float> c4 = ((Vector256<float>*)(Cp4 + blocked))[t];
                    Vector256<float> c5 = ((Vector256<float>*)(Cp5 + blocked))[t];
                    for (int j = 0; j < N; ++j)
                    {
                        var Bpv = (Vector256<float>*)(B + j * K + blocked);
                        Vector256<float> bv = Bpv[t];
                        c0 = Fma.MultiplyAdd(bv, Vector256.Create(Ap0[j]), c0);
                        c1 = Fma.MultiplyAdd(bv, Vector256.Create(Ap1[j]), c1);
                        c2 = Fma.MultiplyAdd(bv, Vector256.Create(Ap2[j]), c2);
                        c3 = Fma.MultiplyAdd(bv, Vector256.Create(Ap3[j]), c3);
                        c4 = Fma.MultiplyAdd(bv, Vector256.Create(Ap4[j]), c4);
                        c5 = Fma.MultiplyAdd(bv, Vector256.Create(Ap5[j]), c5);
                    }
                    ((Vector256<float>*)(Cp0 + blocked))[t] = c0 * alphaVec;
                    ((Vector256<float>*)(Cp1 + blocked))[t] = c1 * alphaVec;
                    ((Vector256<float>*)(Cp2 + blocked))[t] = c2 * alphaVec;
                    ((Vector256<float>*)(Cp3 + blocked))[t] = c3 * alphaVec;
                    ((Vector256<float>*)(Cp4 + blocked))[t] = c4 * alphaVec;
                    ((Vector256<float>*)(Cp5 + blocked))[t] = c5 * alphaVec;
                }
                // Scalar tail: same j-ascending per-column accumulation as the
                // 2-row alpha twin, register-held, scaled once on the single store.
                for (int k = blocked + rv * Vector256<float>.Count; k < K; k++)
                {
                    float acc0 = Cp0[k];
                    float acc1 = Cp1[k];
                    float acc2 = Cp2[k];
                    float acc3 = Cp3[k];
                    float acc4 = Cp4[k];
                    float acc5 = Cp5[k];
                    for (int j = 0; j < N; ++j)
                    {
                        acc0 += Ap0[j] * (B + j * K)[k];
                        acc1 += Ap1[j] * (B + j * K)[k];
                        acc2 += Ap2[j] * (B + j * K)[k];
                        acc3 += Ap3[j] * (B + j * K)[k];
                        acc4 += Ap4[j] * (B + j * K)[k];
                        acc5 += Ap5[j] * (B + j * K)[k];
                    }
                    Cp0[k] = alpha * acc0;
                    Cp1[k] = alpha * acc1;
                    Cp2[k] = alpha * acc2;
                    Cp3[k] = alpha * acc3;
                    Cp4[k] = alpha * acc4;
                    Cp5[k] = alpha * acc5;
                }
            }
        }
    }


    /// <summary>
    /// Register-tiled matrix multiplication over 6-row groups with 32-column
    /// AVX-512 halves (E5-2 M2): the M1 6x16 block widened to Vector512, so
    /// each step loads 2 B vectors once per group and reuses them across all
    /// six rows through one live broadcast temporary per vector (12 FMAs per
    /// 2 loads, 15 live vectors of 32 architectural zmm). Sub-32 columns fall
    /// back to 256-bit lanes with the exact M1 boundaries, so every k below
    /// the 8-wide ceiling runs FMA chains and the scalar tail keeps the legacy
    /// per-step order: results agree with the tiled kernel bit-wise on every
    /// shape. Requires Avx512F (Avx2 for the tail lanes); callers check
    /// capability, like the other raw kernels. M must be a multiple of 6.
    /// </summary>
    /// <param name="M">A rows (must be a multiple of 6).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="B">Right matrix.</param>
    /// <param name="C">Result matrix (accumulated, like the tiled kernel).</param>
    public unsafe static void mm_unsafe_vectorized_intrinsics_6x2tiled512(int M,
                          int N,
                          int K,
                          float* A,
                          float* B,
                          float* C)
    {
        if (M % 6 != 0)
            throw new ArgumentException(nameof(M));

        int word = Vector512<float>.Count;
        int blocked = K - (K % (2 * word));
        int mid = blocked + ((K - blocked) / word) * word;

        for (int i = 0; i < M; i += 6)
        {

            var Ap0 = A + (i + 0) * N;
            var Ap1 = A + (i + 1) * N;
            var Ap2 = A + (i + 2) * N;
            var Ap3 = A + (i + 3) * N;
            var Ap4 = A + (i + 4) * N;
            var Ap5 = A + (i + 5) * N;

            var Cp0 = C + (i + 0) * K;
            var Cp1 = C + (i + 1) * K;
            var Cp2 = C + (i + 2) * K;
            var Cp3 = C + (i + 3) * K;
            var Cp4 = C + (i + 4) * K;
            var Cp5 = C + (i + 5) * K;

            for (int kb = 0; kb < blocked; kb += 2 * word)
            {
                var Q0 = (Vector512<float>*)(Cp0 + kb);
                var Q1 = (Vector512<float>*)(Cp1 + kb);
                var Q2 = (Vector512<float>*)(Cp2 + kb);
                var Q3 = (Vector512<float>*)(Cp3 + kb);
                var Q4 = (Vector512<float>*)(Cp4 + kb);
                var Q5 = (Vector512<float>*)(Cp5 + kb);
                Vector512<float> c00 = Q0[0];
                Vector512<float> c01 = Q0[1];
                Vector512<float> c10 = Q1[0];
                Vector512<float> c11 = Q1[1];
                Vector512<float> c20 = Q2[0];
                Vector512<float> c21 = Q2[1];
                Vector512<float> c30 = Q3[0];
                Vector512<float> c31 = Q3[1];
                Vector512<float> c40 = Q4[0];
                Vector512<float> c41 = Q4[1];
                Vector512<float> c50 = Q5[0];
                Vector512<float> c51 = Q5[1];
                for (int j = 0; j < N; ++j)
                {
                    var Bpv = (Vector512<float>*)(B + j * K + kb);
                    Vector512<float> bv0 = Bpv[0];
                    Vector512<float> bv1 = Bpv[1];
                    var av0 = Vector512.Create(Ap0[j]);
                    c00 = Avx512F.FusedMultiplyAdd(bv0, av0, c00);
                    c01 = Avx512F.FusedMultiplyAdd(bv1, av0, c01);
                    var av1 = Vector512.Create(Ap1[j]);
                    c10 = Avx512F.FusedMultiplyAdd(bv0, av1, c10);
                    c11 = Avx512F.FusedMultiplyAdd(bv1, av1, c11);
                    var av2 = Vector512.Create(Ap2[j]);
                    c20 = Avx512F.FusedMultiplyAdd(bv0, av2, c20);
                    c21 = Avx512F.FusedMultiplyAdd(bv1, av2, c21);
                    var av3 = Vector512.Create(Ap3[j]);
                    c30 = Avx512F.FusedMultiplyAdd(bv0, av3, c30);
                    c31 = Avx512F.FusedMultiplyAdd(bv1, av3, c31);
                    var av4 = Vector512.Create(Ap4[j]);
                    c40 = Avx512F.FusedMultiplyAdd(bv0, av4, c40);
                    c41 = Avx512F.FusedMultiplyAdd(bv1, av4, c41);
                    var av5 = Vector512.Create(Ap5[j]);
                    c50 = Avx512F.FusedMultiplyAdd(bv0, av5, c50);
                    c51 = Avx512F.FusedMultiplyAdd(bv1, av5, c51);
                }
                Q0[0] = c00;
                Q0[1] = c01;
                Q1[0] = c10;
                Q1[1] = c11;
                Q2[0] = c20;
                Q2[1] = c21;
                Q3[0] = c30;
                Q3[1] = c31;
                Q4[0] = c40;
                Q4[1] = c41;
                Q5[0] = c50;
                Q5[1] = c51;
            }

            for (int kb = blocked; kb < mid; kb += word)
            {
                var R0 = (Vector256<float>*)(Cp0 + kb);
                var R1 = (Vector256<float>*)(Cp1 + kb);
                var R2 = (Vector256<float>*)(Cp2 + kb);
                var R3 = (Vector256<float>*)(Cp3 + kb);
                var R4 = (Vector256<float>*)(Cp4 + kb);
                var R5 = (Vector256<float>*)(Cp5 + kb);
                Vector256<float> d00 = R0[0];
                Vector256<float> d01 = R0[1];
                Vector256<float> d10 = R1[0];
                Vector256<float> d11 = R1[1];
                Vector256<float> d20 = R2[0];
                Vector256<float> d21 = R2[1];
                Vector256<float> d30 = R3[0];
                Vector256<float> d31 = R3[1];
                Vector256<float> d40 = R4[0];
                Vector256<float> d41 = R4[1];
                Vector256<float> d50 = R5[0];
                Vector256<float> d51 = R5[1];
                for (int j = 0; j < N; ++j)
                {
                    var Bpv = (Vector256<float>*)(B + j * K + kb);
                    Vector256<float> bv0 = Bpv[0];
                    Vector256<float> bv1 = Bpv[1];
                    var av0 = Vector256.Create(Ap0[j]);
                    d00 = Fma.MultiplyAdd(bv0, av0, d00);
                    d01 = Fma.MultiplyAdd(bv1, av0, d01);
                    var av1 = Vector256.Create(Ap1[j]);
                    d10 = Fma.MultiplyAdd(bv0, av1, d10);
                    d11 = Fma.MultiplyAdd(bv1, av1, d11);
                    var av2 = Vector256.Create(Ap2[j]);
                    d20 = Fma.MultiplyAdd(bv0, av2, d20);
                    d21 = Fma.MultiplyAdd(bv1, av2, d21);
                    var av3 = Vector256.Create(Ap3[j]);
                    d30 = Fma.MultiplyAdd(bv0, av3, d30);
                    d31 = Fma.MultiplyAdd(bv1, av3, d31);
                    var av4 = Vector256.Create(Ap4[j]);
                    d40 = Fma.MultiplyAdd(bv0, av4, d40);
                    d41 = Fma.MultiplyAdd(bv1, av4, d41);
                    var av5 = Vector256.Create(Ap5[j]);
                    d50 = Fma.MultiplyAdd(bv0, av5, d50);
                    d51 = Fma.MultiplyAdd(bv1, av5, d51);
                }
                R0[0] = d00;
                R0[1] = d01;
                R1[0] = d10;
                R1[1] = d11;
                R2[0] = d20;
                R2[1] = d21;
                R3[0] = d30;
                R3[1] = d31;
                R4[0] = d40;
                R4[1] = d41;
                R5[0] = d50;
                R5[1] = d51;
            }

            int rem = K - mid;
            if (rem > 0)
            {
                int rv = rem / Vector256<float>.Count;
                for (int t = 0; t < rv; t++)
                {
                    Vector256<float> c0 = ((Vector256<float>*)(Cp0 + mid))[t];
                    Vector256<float> c1 = ((Vector256<float>*)(Cp1 + mid))[t];
                    Vector256<float> c2 = ((Vector256<float>*)(Cp2 + mid))[t];
                    Vector256<float> c3 = ((Vector256<float>*)(Cp3 + mid))[t];
                    Vector256<float> c4 = ((Vector256<float>*)(Cp4 + mid))[t];
                    Vector256<float> c5 = ((Vector256<float>*)(Cp5 + mid))[t];
                    for (int j = 0; j < N; ++j)
                    {
                        var Bpv = (Vector256<float>*)(B + j * K + mid);
                        Vector256<float> bv = Bpv[t];
                        c0 = Fma.MultiplyAdd(bv, Vector256.Create(Ap0[j]), c0);
                        c1 = Fma.MultiplyAdd(bv, Vector256.Create(Ap1[j]), c1);
                        c2 = Fma.MultiplyAdd(bv, Vector256.Create(Ap2[j]), c2);
                        c3 = Fma.MultiplyAdd(bv, Vector256.Create(Ap3[j]), c3);
                        c4 = Fma.MultiplyAdd(bv, Vector256.Create(Ap4[j]), c4);
                        c5 = Fma.MultiplyAdd(bv, Vector256.Create(Ap5[j]), c5);
                    }
                    ((Vector256<float>*)(Cp0 + mid))[t] = c0;
                    ((Vector256<float>*)(Cp1 + mid))[t] = c1;
                    ((Vector256<float>*)(Cp2 + mid))[t] = c2;
                    ((Vector256<float>*)(Cp3 + mid))[t] = c3;
                    ((Vector256<float>*)(Cp4 + mid))[t] = c4;
                    ((Vector256<float>*)(Cp5 + mid))[t] = c5;
                }
                // Scalar tail keeps the reduction-major order of the
                // unrolled kernel (accumulate into the destination per step)
                // so results agree with it bit-wise on every shape.
                for (int j = 0; j < N; ++j)
                {
                    float a0 = Ap0[j];
                    float a1 = Ap1[j];
                    float a2 = Ap2[j];
                    float a3 = Ap3[j];
                    float a4 = Ap4[j];
                    float a5 = Ap5[j];
                    var Brow = B + j * K;
                    for (int k = mid + rv * Vector256<float>.Count; k < K; k++)
                    {
                        Cp0[k] += a0 * Brow[k];
                        Cp1[k] += a1 * Brow[k];
                        Cp2[k] += a2 * Brow[k];
                        Cp3[k] += a3 * Brow[k];
                        Cp4[k] += a4 * Brow[k];
                        Cp5[k] += a5 * Brow[k];
                    }
                }
            }

        }
    }

    /// <summary>
    /// Register-tiled 6-row AVX-512 multiplication with epilogue scaling
    /// (E5-2 M2): identical tiles, FMA order and tails to
    /// mm_unsafe_vectorized_intrinsics_6x2tiled512, except every stored element
    /// is scaled once by alpha on its single write-back. The destination must
    /// be zeroed on entry (same accumulate contract as the 2-row alpha twin);
    /// each element is written exactly once across main, half, rem and tail
    /// regions, so no scaling is ever applied twice or missed.
    /// </summary>
    /// <param name="M">A rows (must be a multiple of 6).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="B">Right matrix.</param>
    /// <param name="C">Zeroed result matrix.</param>
    /// <param name="alpha">Epilogue scale applied once per stored element.</param>
    public unsafe static void mm_unsafe_vectorized_intrinsics_6x2tiled512_alpha(int M,
                          int N,
                          int K,
                          float* A,
                          float* B,
                          float* C,
                          float alpha)
    {
        if (M % 6 != 0)
            throw new ArgumentException(nameof(M));

        var alphaVec = Vector512.Create(alpha);
        var alphaVec256 = Vector256.Create(alpha);
        int word = Vector512<float>.Count;
        int blocked = K - (K % (2 * word));
        int mid = blocked + ((K - blocked) / word) * word;

        for (int i = 0; i < M; i += 6)
        {

            var Ap0 = A + (i + 0) * N;
            var Ap1 = A + (i + 1) * N;
            var Ap2 = A + (i + 2) * N;
            var Ap3 = A + (i + 3) * N;
            var Ap4 = A + (i + 4) * N;
            var Ap5 = A + (i + 5) * N;

            var Cp0 = C + (i + 0) * K;
            var Cp1 = C + (i + 1) * K;
            var Cp2 = C + (i + 2) * K;
            var Cp3 = C + (i + 3) * K;
            var Cp4 = C + (i + 4) * K;
            var Cp5 = C + (i + 5) * K;

            for (int kb = 0; kb < blocked; kb += 2 * word)
            {
                var Q0 = (Vector512<float>*)(Cp0 + kb);
                var Q1 = (Vector512<float>*)(Cp1 + kb);
                var Q2 = (Vector512<float>*)(Cp2 + kb);
                var Q3 = (Vector512<float>*)(Cp3 + kb);
                var Q4 = (Vector512<float>*)(Cp4 + kb);
                var Q5 = (Vector512<float>*)(Cp5 + kb);
                Vector512<float> c00 = Q0[0];
                Vector512<float> c01 = Q0[1];
                Vector512<float> c10 = Q1[0];
                Vector512<float> c11 = Q1[1];
                Vector512<float> c20 = Q2[0];
                Vector512<float> c21 = Q2[1];
                Vector512<float> c30 = Q3[0];
                Vector512<float> c31 = Q3[1];
                Vector512<float> c40 = Q4[0];
                Vector512<float> c41 = Q4[1];
                Vector512<float> c50 = Q5[0];
                Vector512<float> c51 = Q5[1];
                for (int j = 0; j < N; ++j)
                {
                    var Bpv = (Vector512<float>*)(B + j * K + kb);
                    Vector512<float> bv0 = Bpv[0];
                    Vector512<float> bv1 = Bpv[1];
                    var av0 = Vector512.Create(Ap0[j]);
                    c00 = Avx512F.FusedMultiplyAdd(bv0, av0, c00);
                    c01 = Avx512F.FusedMultiplyAdd(bv1, av0, c01);
                    var av1 = Vector512.Create(Ap1[j]);
                    c10 = Avx512F.FusedMultiplyAdd(bv0, av1, c10);
                    c11 = Avx512F.FusedMultiplyAdd(bv1, av1, c11);
                    var av2 = Vector512.Create(Ap2[j]);
                    c20 = Avx512F.FusedMultiplyAdd(bv0, av2, c20);
                    c21 = Avx512F.FusedMultiplyAdd(bv1, av2, c21);
                    var av3 = Vector512.Create(Ap3[j]);
                    c30 = Avx512F.FusedMultiplyAdd(bv0, av3, c30);
                    c31 = Avx512F.FusedMultiplyAdd(bv1, av3, c31);
                    var av4 = Vector512.Create(Ap4[j]);
                    c40 = Avx512F.FusedMultiplyAdd(bv0, av4, c40);
                    c41 = Avx512F.FusedMultiplyAdd(bv1, av4, c41);
                    var av5 = Vector512.Create(Ap5[j]);
                    c50 = Avx512F.FusedMultiplyAdd(bv0, av5, c50);
                    c51 = Avx512F.FusedMultiplyAdd(bv1, av5, c51);
                }
                Q0[0] = c00 * alphaVec;
                Q0[1] = c01 * alphaVec;
                Q1[0] = c10 * alphaVec;
                Q1[1] = c11 * alphaVec;
                Q2[0] = c20 * alphaVec;
                Q2[1] = c21 * alphaVec;
                Q3[0] = c30 * alphaVec;
                Q3[1] = c31 * alphaVec;
                Q4[0] = c40 * alphaVec;
                Q4[1] = c41 * alphaVec;
                Q5[0] = c50 * alphaVec;
                Q5[1] = c51 * alphaVec;
            }

            for (int kb = blocked; kb < mid; kb += word)
            {
                var R0 = (Vector256<float>*)(Cp0 + kb);
                var R1 = (Vector256<float>*)(Cp1 + kb);
                var R2 = (Vector256<float>*)(Cp2 + kb);
                var R3 = (Vector256<float>*)(Cp3 + kb);
                var R4 = (Vector256<float>*)(Cp4 + kb);
                var R5 = (Vector256<float>*)(Cp5 + kb);
                Vector256<float> d00 = R0[0];
                Vector256<float> d01 = R0[1];
                Vector256<float> d10 = R1[0];
                Vector256<float> d11 = R1[1];
                Vector256<float> d20 = R2[0];
                Vector256<float> d21 = R2[1];
                Vector256<float> d30 = R3[0];
                Vector256<float> d31 = R3[1];
                Vector256<float> d40 = R4[0];
                Vector256<float> d41 = R4[1];
                Vector256<float> d50 = R5[0];
                Vector256<float> d51 = R5[1];
                for (int j = 0; j < N; ++j)
                {
                    var Bpv = (Vector256<float>*)(B + j * K + kb);
                    Vector256<float> bv0 = Bpv[0];
                    Vector256<float> bv1 = Bpv[1];
                    var av0 = Vector256.Create(Ap0[j]);
                    d00 = Fma.MultiplyAdd(bv0, av0, d00);
                    d01 = Fma.MultiplyAdd(bv1, av0, d01);
                    var av1 = Vector256.Create(Ap1[j]);
                    d10 = Fma.MultiplyAdd(bv0, av1, d10);
                    d11 = Fma.MultiplyAdd(bv1, av1, d11);
                    var av2 = Vector256.Create(Ap2[j]);
                    d20 = Fma.MultiplyAdd(bv0, av2, d20);
                    d21 = Fma.MultiplyAdd(bv1, av2, d21);
                    var av3 = Vector256.Create(Ap3[j]);
                    d30 = Fma.MultiplyAdd(bv0, av3, d30);
                    d31 = Fma.MultiplyAdd(bv1, av3, d31);
                    var av4 = Vector256.Create(Ap4[j]);
                    d40 = Fma.MultiplyAdd(bv0, av4, d40);
                    d41 = Fma.MultiplyAdd(bv1, av4, d41);
                    var av5 = Vector256.Create(Ap5[j]);
                    d50 = Fma.MultiplyAdd(bv0, av5, d50);
                    d51 = Fma.MultiplyAdd(bv1, av5, d51);
                }
                R0[0] = d00 * alphaVec256;
                R0[1] = d01 * alphaVec256;
                R1[0] = d10 * alphaVec256;
                R1[1] = d11 * alphaVec256;
                R2[0] = d20 * alphaVec256;
                R2[1] = d21 * alphaVec256;
                R3[0] = d30 * alphaVec256;
                R3[1] = d31 * alphaVec256;
                R4[0] = d40 * alphaVec256;
                R4[1] = d41 * alphaVec256;
                R5[0] = d50 * alphaVec256;
                R5[1] = d51 * alphaVec256;
            }

            int rem = K - mid;
            if (rem > 0)
            {
                int rv = rem / Vector256<float>.Count;
                for (int t = 0; t < rv; t++)
                {
                    Vector256<float> c0 = ((Vector256<float>*)(Cp0 + mid))[t];
                    Vector256<float> c1 = ((Vector256<float>*)(Cp1 + mid))[t];
                    Vector256<float> c2 = ((Vector256<float>*)(Cp2 + mid))[t];
                    Vector256<float> c3 = ((Vector256<float>*)(Cp3 + mid))[t];
                    Vector256<float> c4 = ((Vector256<float>*)(Cp4 + mid))[t];
                    Vector256<float> c5 = ((Vector256<float>*)(Cp5 + mid))[t];
                    for (int j = 0; j < N; ++j)
                    {
                        var Bpv = (Vector256<float>*)(B + j * K + mid);
                        Vector256<float> bv = Bpv[t];
                        c0 = Fma.MultiplyAdd(bv, Vector256.Create(Ap0[j]), c0);
                        c1 = Fma.MultiplyAdd(bv, Vector256.Create(Ap1[j]), c1);
                        c2 = Fma.MultiplyAdd(bv, Vector256.Create(Ap2[j]), c2);
                        c3 = Fma.MultiplyAdd(bv, Vector256.Create(Ap3[j]), c3);
                        c4 = Fma.MultiplyAdd(bv, Vector256.Create(Ap4[j]), c4);
                        c5 = Fma.MultiplyAdd(bv, Vector256.Create(Ap5[j]), c5);
                    }
                    ((Vector256<float>*)(Cp0 + mid))[t] = c0 * alphaVec256;
                    ((Vector256<float>*)(Cp1 + mid))[t] = c1 * alphaVec256;
                    ((Vector256<float>*)(Cp2 + mid))[t] = c2 * alphaVec256;
                    ((Vector256<float>*)(Cp3 + mid))[t] = c3 * alphaVec256;
                    ((Vector256<float>*)(Cp4 + mid))[t] = c4 * alphaVec256;
                    ((Vector256<float>*)(Cp5 + mid))[t] = c5 * alphaVec256;
                }
                // Scalar tail: same j-ascending per-column accumulation as the
                // 2-row alpha twin, register-held, scaled once on the single store.
                for (int k = mid + rv * Vector256<float>.Count; k < K; k++)
                {
                    float acc0 = Cp0[k];
                    float acc1 = Cp1[k];
                    float acc2 = Cp2[k];
                    float acc3 = Cp3[k];
                    float acc4 = Cp4[k];
                    float acc5 = Cp5[k];
                    for (int j = 0; j < N; ++j)
                    {
                        acc0 += Ap0[j] * (B + j * K)[k];
                        acc1 += Ap1[j] * (B + j * K)[k];
                        acc2 += Ap2[j] * (B + j * K)[k];
                        acc3 += Ap3[j] * (B + j * K)[k];
                        acc4 += Ap4[j] * (B + j * K)[k];
                        acc5 += Ap5[j] * (B + j * K)[k];
                    }
                    Cp0[k] = alpha * acc0;
                    Cp1[k] = alpha * acc1;
                    Cp2[k] = alpha * acc2;
                    Cp3[k] = alpha * acc3;
                    Cp4[k] = alpha * acc4;
                    Cp5[k] = alpha * acc5;
                }
            }

        }
    }


    /// <summary>
    /// 8x8-shuffle transpose of the last two axes (E5-3 M2): transposes every
    /// [S,D] face of a standard row-major [B,H,S,D] buffer into [B,H,D,S].
    /// Full 8x8 tiles move through registers (8 loads, 8 unpacks, 8 shuffles,
    /// 8 lane-crosses, 8 stores) instead of 64 strided scalar pairs; partial
    /// edge tiles keep the scalar indices of the TransposeInto fast path, so
    /// results agree with it bit-wise on every shape, including NaN payloads
    /// and signed zero (pure movement, no arithmetic). Requires Avx; callers
    /// check capability, like the other raw kernels.
    /// </summary>
    /// <param name="dimB">Batch count.</param>
    /// <param name="dimH">Head count.</param>
    /// <param name="dimS">Rows per face.</param>
    /// <param name="dimD">Columns per face.</param>
    /// <param name="src">Source [B,H,S,D] buffer.</param>
    /// <param name="dst">Destination [B,H,D,S] buffer.</param>
    public unsafe static void transpose_unsafe_shuffle8x8_lastTwoAxes(int dimB, int dimH, int dimS, int dimD, float* src, float* dst)
    {
        const int Tile = 8;
        for (int b = 0; b < dimB; b++)
            for (int h = 0; h < dimH; h++)
            {
                float* sBase = src + ((b * dimH + h) * dimS) * dimD;
                float* dBase = dst + ((b * dimH + h) * dimD) * dimS;
                for (int i = 0; i < dimS; i += Tile)
                    for (int j = 0; j < dimD; j += Tile)
                    {
                        if (i + Tile <= dimS && j + Tile <= dimD)
                        {
                            var r0 = *(Vector256<float>*)(sBase + (i + 0) * dimD + j);
                            var r1 = *(Vector256<float>*)(sBase + (i + 1) * dimD + j);
                            var r2 = *(Vector256<float>*)(sBase + (i + 2) * dimD + j);
                            var r3 = *(Vector256<float>*)(sBase + (i + 3) * dimD + j);
                            var r4 = *(Vector256<float>*)(sBase + (i + 4) * dimD + j);
                            var r5 = *(Vector256<float>*)(sBase + (i + 5) * dimD + j);
                            var r6 = *(Vector256<float>*)(sBase + (i + 6) * dimD + j);
                            var r7 = *(Vector256<float>*)(sBase + (i + 7) * dimD + j);
                            var t0 = Avx.UnpackLow(r0, r1);
                            var t1 = Avx.UnpackHigh(r0, r1);
                            var t2 = Avx.UnpackLow(r2, r3);
                            var t3 = Avx.UnpackHigh(r2, r3);
                            var t4 = Avx.UnpackLow(r4, r5);
                            var t5 = Avx.UnpackHigh(r4, r5);
                            var t6 = Avx.UnpackLow(r6, r7);
                            var t7 = Avx.UnpackHigh(r6, r7);
                            var e0 = Avx.Shuffle(t0, t2, 0x44);
                            var e1 = Avx.Shuffle(t0, t2, 0xEE);
                            var e2 = Avx.Shuffle(t1, t3, 0x44);
                            var e3 = Avx.Shuffle(t1, t3, 0xEE);
                            var e4 = Avx.Shuffle(t4, t6, 0x44);
                            var e5 = Avx.Shuffle(t4, t6, 0xEE);
                            var e6 = Avx.Shuffle(t5, t7, 0x44);
                            var e7 = Avx.Shuffle(t5, t7, 0xEE);
                            *(Vector256<float>*)(dBase + (j + 0) * dimS + i) = Avx.Permute2x128(e0, e4, 0x20);
                            *(Vector256<float>*)(dBase + (j + 1) * dimS + i) = Avx.Permute2x128(e1, e5, 0x20);
                            *(Vector256<float>*)(dBase + (j + 2) * dimS + i) = Avx.Permute2x128(e2, e6, 0x20);
                            *(Vector256<float>*)(dBase + (j + 3) * dimS + i) = Avx.Permute2x128(e3, e7, 0x20);
                            *(Vector256<float>*)(dBase + (j + 4) * dimS + i) = Avx.Permute2x128(e0, e4, 0x31);
                            *(Vector256<float>*)(dBase + (j + 5) * dimS + i) = Avx.Permute2x128(e1, e5, 0x31);
                            *(Vector256<float>*)(dBase + (j + 6) * dimS + i) = Avx.Permute2x128(e2, e6, 0x31);
                            *(Vector256<float>*)(dBase + (j + 7) * dimS + i) = Avx.Permute2x128(e3, e7, 0x31);
                        }
                        else
                        {
                            for (int ii = i; ii < System.Math.Min(i + Tile, dimS); ii++)
                                for (int jj = j; jj < System.Math.Min(j + Tile, dimD); jj++)
                                    dBase[jj * dimS + ii] = sBase[ii * dimD + jj];
                        }
                    }
            }
    }


    /// <summary>
    /// Head-merge face transpose (E5-3 M2): transposes every [H,S,D] region of
    /// a standard row-major [B,H,S,D] buffer into [B,S,D,H], matching the
    /// TransposeInto (0,2,3,1) fast path element for element. Source elements
    /// stay strided (one cache line per head), but each destination run of 8
    /// heads moves with a single vector store built from two 4-lane creates,
    /// with scalar stores only for the H tail; no arithmetic is performed, so
    /// results agree bit-wise on every shape, including NaN payloads and
    /// signed zero. No SIMD ISA is required (lane creates lower to moves).
    /// </summary>
    /// <param name="dimB">Batch count.</param>
    /// <param name="dimS">Tokens per face.</param>
    /// <param name="dimH">Heads per face.</param>
    /// <param name="dimD">Columns per face.</param>
    /// <param name="src">Source [B,H,S,D] buffer.</param>
    /// <param name="dst">Destination [B,S,D,H] buffer.</param>
    public unsafe static void transpose_unsafe_vector8_headMerge(int dimB, int dimS, int dimH, int dimD, float* src, float* dst)
    {
        int vec = Vector256<float>.Count;
        for (int b = 0; b < dimB; b++)
            for (int i = 0; i < dimS; i++)
                for (int jj = 0; jj < dimD; jj++)
                {
                    float* dRun = dst + (((b * dimS + i) * dimD + jj) * dimH);
                    int k = 0;
                    for (; k + vec <= dimH; k += vec)
                    {
                        float* sK = src + ((b * dimH + k) * dimS + i) * dimD + jj;
                        var lo = Vector128.Create(sK[0 * dimS * dimD], sK[1 * dimS * dimD], sK[2 * dimS * dimD], sK[3 * dimS * dimD]);
                        var hi = Vector128.Create(sK[4 * dimS * dimD], sK[5 * dimS * dimD], sK[6 * dimS * dimD], sK[7 * dimS * dimD]);
                        *(Vector256<float>*)(dRun + k) = Vector256.Create(lo, hi);
                    }
                    for (; k < dimH; k++)
                        dRun[k] = src[(((b * dimH + k) * dimS + i) * dimD) + jj];
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
        int blocked = K - (K % (4 * Vector256<float>.Count));
        for (int kb = 0; kb < blocked; kb += 4 * Vector256<float>.Count)
        {
            float* dst = P + (kb / (4 * Vector256<float>.Count)) * N * (4 * Vector256<float>.Count);
            for (int j = 0; j < N; j++)
            {
                float* src = B + j * K + kb;
                float* d = dst + j * (4 * Vector256<float>.Count);
                for (int kk = 0; kk < 4 * Vector256<float>.Count; kk++) d[kk] = src[kk];
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
    public unsafe static void mm_unsafe_vectorized_intrinsics_2x4packed(int M,
                          int N,
                          int K,
                          float* A,
                          float* P,
                          float* C)
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
            for (int i = 0; i < M; i += 2)
            {
                var Ap1 = A + i * N;
                var Ap2 = Ap1 + N;
                var Cpv1 = (Vector256<float>*)(C + i * K + kb);
                var Cpv2 = (Vector256<float>*)(C + (i + 1) * K + kb);
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
                        float c1 = Cp1[0], c2 = Cp2[0];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c1 += Ap1[j] * t[0]; c2 += Ap2[j] * t[0]; }
                        Cp1[0] = c1; Cp2[0] = c2;
                        break;
                    }
                    case 2:
                    {
                        float c10 = Cp1[0], c11 = Cp1[1], c20 = Cp2[0], c21 = Cp2[1];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; }
                        Cp1[0] = c10; Cp1[1] = c11; Cp2[0] = c20; Cp2[1] = c21;
                        break;
                    }
                    case 3:
                    {
                        float c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; }
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22;
                        break;
                    }
                    case 4:
                    {
                        float c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; }
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23;
                        break;
                    }
                    case 5:
                    {
                        float c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3], c14 = Cp1[4];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3], c24 = Cp2[4];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; }
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13; Cp1[4] = c14;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp2[4] = c24;
                        break;
                    }
                    case 6:
                    {
                        float c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3], c14 = Cp1[4], c15 = Cp1[5];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3], c24 = Cp2[4], c25 = Cp2[5];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; c15 += Ap1[j] * t[5]; c25 += Ap2[j] * t[5]; }
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13; Cp1[4] = c14; Cp1[5] = c15;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp2[4] = c24; Cp2[5] = c25;
                        break;
                    }
                    default:
                    {
                        float c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3], c14 = Cp1[4], c15 = Cp1[5], c16 = Cp1[6];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3], c24 = Cp2[4], c25 = Cp2[5], c26 = Cp2[6];
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
    /// <summary>
    /// Register-tiled matrix multiplication reading panel-packed B, two
    /// rows per group with pointer-bump addressing (E65).
    /// </summary>
    /// <param name="M">A rows (must be even).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="P">Panel-packed right matrix from PackPanelsB.</param>
    /// <param name="C">Result matrix (accumulated, like the other packed kernels).</param>
    /// <remarks>
    /// Same panels and per-element order as the indexed 2-row packed kernel,
    /// so results agree with it bit-wise; the j-loop advances B by one panel
    /// and A by one element per step instead of recomputing indexed addresses.
    /// Tails are verbatim copies of the reference nest.
    /// </remarks>
    public unsafe static void mm_unsafe_vectorized_intrinsics_2x4packed_bump(int M,
                          int N,
                          int K,
                          float* A,
                          float* P,
                          float* C)
    {
        if (M % 2 != 0)
            throw new ArgumentException(nameof(M));

        int panelWidth = 4 * Vector256<float>.Count;
        int blocked = K - (K % panelWidth);
        int tiles = blocked / panelWidth;

        // Kb panels lead so each panel is fetched once and stays L1/L2 resident
        // across every row group, exactly like the reference nest.
        for (int tb = 0; tb < tiles; tb++)
        {
            int kb = tb * panelWidth;
            float* panel = P + tb * N * panelWidth;
            for (int i = 0; i < M; i += 2)
            {
                float* a0 = A + i * N;
                float* a1 = a0 + N;
                float* b = panel;
                var Cpv1 = (Vector256<float>*)(C + i * K + kb);
                var Cpv2 = (Vector256<float>*)(C + (i + 1) * K + kb);
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
                    var Bpv = (Vector256<float>*)b;
                    Vector256<float> bv0 = Bpv[0];
                    Vector256<float> bv1 = Bpv[1];
                    Vector256<float> bv2 = Bpv[2];
                    Vector256<float> bv3 = Bpv[3];
                    var av1 = Vector256.Create(*a0); a0 += 1;
                    var av2 = Vector256.Create(*a1); a1 += 1;
                    c00 = Fma.MultiplyAdd(bv0, av1, c00);
                    c10 = Fma.MultiplyAdd(bv0, av2, c10);
                    c01 = Fma.MultiplyAdd(bv1, av1, c01);
                    c11 = Fma.MultiplyAdd(bv1, av2, c11);
                    c02 = Fma.MultiplyAdd(bv2, av1, c02);
                    c12 = Fma.MultiplyAdd(bv2, av2, c12);
                    c03 = Fma.MultiplyAdd(bv3, av1, c03);
                    c13 = Fma.MultiplyAdd(bv3, av2, c13);
                    b += panelWidth;
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
            if (tail > 0)
            for (int i = 0; i < M; i += 2)
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
    public unsafe static void mm_unsafe_vectorized_intrinsics_3x4packed(int M,
                          int N,
                          int K,
                          float* A,
                          float* P,
                          float* C)
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
            for (int i = 0; i < M; i += 3)
            {
                var Ap1 = A + i * N;
                var Ap2 = Ap1 + N;
                var Ap3 = Ap2 + N;
                var Cpv1 = (Vector256<float>*)(C + i * K + kb);
                var Cpv2 = (Vector256<float>*)(C + (i + 1) * K + kb);
                var Cpv3 = (Vector256<float>*)(C + (i + 2) * K + kb);
                Vector256<float> c00 = Cpv1[0];
                Vector256<float> c01 = Cpv1[1];
                Vector256<float> c02 = Cpv1[2];
                Vector256<float> c03 = Cpv1[3];
                Vector256<float> c10 = Cpv2[0];
                Vector256<float> c11 = Cpv2[1];
                Vector256<float> c12 = Cpv2[2];
                Vector256<float> c13 = Cpv2[3];
                Vector256<float> c20 = Cpv3[0];
                Vector256<float> c21 = Cpv3[1];
                Vector256<float> c22 = Cpv3[2];
                Vector256<float> c23 = Cpv3[3];
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
                    Vector256<float> c1 = rC1[tt];
                    Vector256<float> c2 = rC2[tt];
                    Vector256<float> c3 = rC3[tt];
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
                    Vector256<int> ci3 = Avx2.MaskLoad((int*)Cp3, tmask);
                    Vector256<float> c1 = Unsafe.As<Vector256<int>, Vector256<float>>(ref ci1);
                    Vector256<float> c2 = Unsafe.As<Vector256<int>, Vector256<float>>(ref ci2);
                    Vector256<float> c3 = Unsafe.As<Vector256<int>, Vector256<float>>(ref ci3);
                    for (int j = 0; j < N; ++j)
                    {
                        Vector256<int> bi = Avx2.MaskLoad((int*)(T + j * rem + vcols), tmask);
                        Vector256<float> bv = Unsafe.As<Vector256<int>, Vector256<float>>(ref bi);
                        var av1 = Vector256.Create(Ap1[j]);
                        var av2 = Vector256.Create(Ap2[j]);
                        var av3 = Vector256.Create(Ap3[j]);
                        c1 = c1 + av1 * bv;
                        c2 = c2 + av2 * bv;
                        c3 = c3 + av3 * bv;
                    }
                    Vector256<int> co1 = Unsafe.As<Vector256<float>, Vector256<int>>(ref c1);
                    Vector256<int> co2 = Unsafe.As<Vector256<float>, Vector256<int>>(ref c2);
                    Vector256<int> co3 = Unsafe.As<Vector256<float>, Vector256<int>>(ref c3);
                    Avx2.MaskStore((int*)Cp1, tmask, co1);
                    Avx2.MaskStore((int*)Cp2, tmask, co2);
                    Avx2.MaskStore((int*)Cp3, tmask, co3);
                }
                else
                {
                    for (int j = 0; j < N; ++j)
                    {
                        float a1 = Ap1[j];
                        float a2 = Ap2[j];
                        float a3 = Ap3[j];
                        var t = T + j * rem + vcols;
                        for (int k = 0; k < tail; k++) { Cp1[k] += a1 * t[k]; Cp2[k] += a2 * t[k]; Cp3[k] += a3 * t[k]; }
                    }
                }
            }
        }
    }


    /// <summary>
    /// Register-tiled matrix multiplication reading panel-packed B, six
    /// rows per group over 16-column halves (E61).
    /// </summary>
    /// <param name="M">A rows (must be a multiple of 6).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="P">Panel-packed right matrix from PackPanelsB.</param>
    /// <param name="C">Result matrix (accumulated, like the other packed kernels).</param>
    /// <remarks>
    /// Same panels and per-element order as the 2-row packed kernel, so results
    /// agree with it bit-wise; six rows share each B vector, cutting B loads
    /// per FMA to a third of the 2-row rate at the same broadcast rate.
    /// Sixteen-column halves keep twelve accumulators live alongside the B
    /// vectors and one broadcast temporary within sixteen AVX2 registers.
    /// Tails read the appended row-major tail block in the same order.
    /// </remarks>
    public unsafe static void mm_unsafe_vectorized_intrinsics_6x4packed(int M,
                          int N,
                          int K,
                          float* A,
                          float* P,
                          float* C)
    {
        if (M % 6 != 0)
            throw new ArgumentException(nameof(M));

        int panelWidth = 4 * Vector256<float>.Count;
        int halfWidth = 2 * Vector256<float>.Count;
        int blocked = K - (K % panelWidth);
        int tiles = blocked / panelWidth;

        // Kb panels lead so each panel is fetched once and stays L1/L2 resident
        // across every row group; per-element FMA order matches the 2-row nest
        // exactly, so results agree bit-wise with the other packed kernels.
        for (int tb = 0; tb < tiles; tb++)
        {
            int kb = tb * panelWidth;
            float* panel = P + tb * N * panelWidth;
            for (int hh = 0; hh < 2; hh++)
            {
                int ko = kb + hh * halfWidth;
                for (int i = 0; i < M; i += 6)
                {
                    var Ap0 = A + i * N;
                    var Ap1 = Ap0 + N;
                    var Ap2 = Ap1 + N;
                    var Ap3 = Ap2 + N;
                    var Ap4 = Ap3 + N;
                    var Ap5 = Ap4 + N;
                    var Cp0 = (Vector256<float>*)(C + i * K + ko);
                    var Cp1 = (Vector256<float>*)(C + (i + 1) * K + ko);
                    var Cp2 = (Vector256<float>*)(C + (i + 2) * K + ko);
                    var Cp3 = (Vector256<float>*)(C + (i + 3) * K + ko);
                    var Cp4 = (Vector256<float>*)(C + (i + 4) * K + ko);
                    var Cp5 = (Vector256<float>*)(C + (i + 5) * K + ko);
                    Vector256<float> c00 = Cp0[0];
                    Vector256<float> c01 = Cp0[1];
                    Vector256<float> c10 = Cp1[0];
                    Vector256<float> c11 = Cp1[1];
                    Vector256<float> c20 = Cp2[0];
                    Vector256<float> c21 = Cp2[1];
                    Vector256<float> c30 = Cp3[0];
                    Vector256<float> c31 = Cp3[1];
                    Vector256<float> c40 = Cp4[0];
                    Vector256<float> c41 = Cp4[1];
                    Vector256<float> c50 = Cp5[0];
                    Vector256<float> c51 = Cp5[1];
                    for (int j = 0; j < N; ++j)
                    {
                        var Bpv = (Vector256<float>*)(panel + j * panelWidth + hh * halfWidth);
                        Vector256<float> bv0 = Bpv[0];
                        Vector256<float> bv1 = Bpv[1];
                        var av0 = Vector256.Create(Ap0[j]);
                        c00 = Fma.MultiplyAdd(bv0, av0, c00);
                        c01 = Fma.MultiplyAdd(bv1, av0, c01);
                        var av1 = Vector256.Create(Ap1[j]);
                        c10 = Fma.MultiplyAdd(bv0, av1, c10);
                        c11 = Fma.MultiplyAdd(bv1, av1, c11);
                        var av2 = Vector256.Create(Ap2[j]);
                        c20 = Fma.MultiplyAdd(bv0, av2, c20);
                        c21 = Fma.MultiplyAdd(bv1, av2, c21);
                        var av3 = Vector256.Create(Ap3[j]);
                        c30 = Fma.MultiplyAdd(bv0, av3, c30);
                        c31 = Fma.MultiplyAdd(bv1, av3, c31);
                        var av4 = Vector256.Create(Ap4[j]);
                        c40 = Fma.MultiplyAdd(bv0, av4, c40);
                        c41 = Fma.MultiplyAdd(bv1, av4, c41);
                        var av5 = Vector256.Create(Ap5[j]);
                        c50 = Fma.MultiplyAdd(bv0, av5, c50);
                        c51 = Fma.MultiplyAdd(bv1, av5, c51);
                    }
                    Cp0[0] = c00;
                    Cp0[1] = c01;
                    Cp1[0] = c10;
                    Cp1[1] = c11;
                    Cp2[0] = c20;
                    Cp2[1] = c21;
                    Cp3[0] = c30;
                    Cp3[1] = c31;
                    Cp4[0] = c40;
                    Cp4[1] = c41;
                    Cp5[0] = c50;
                    Cp5[1] = c51;
                }
            }
        }
        int rem = K - blocked;
        if (rem > 0)
        {
            float* T = P + tiles * N * panelWidth;
            int rv = rem / Vector256<float>.Count;
            for (int tt = 0; tt < rv; tt++)
            {
                for (int i = 0; i < M; i += 6)
                {
                    var Ap0 = A + i * N;
                    var Ap1 = Ap0 + N;
                    var Ap2 = Ap1 + N;
                    var Ap3 = Ap2 + N;
                    var Ap4 = Ap3 + N;
                    var Ap5 = Ap4 + N;
                    var rC0 = (Vector256<float>*)(C + i * K + blocked);
                    var rC1 = (Vector256<float>*)(C + (i + 1) * K + blocked);
                    var rC2 = (Vector256<float>*)(C + (i + 2) * K + blocked);
                    var rC3 = (Vector256<float>*)(C + (i + 3) * K + blocked);
                    var rC4 = (Vector256<float>*)(C + (i + 4) * K + blocked);
                    var rC5 = (Vector256<float>*)(C + (i + 5) * K + blocked);
                    Vector256<float> c0 = rC0[tt];
                    Vector256<float> c1 = rC1[tt];
                    Vector256<float> c2 = rC2[tt];
                    Vector256<float> c3 = rC3[tt];
                    Vector256<float> c4 = rC4[tt];
                    Vector256<float> c5 = rC5[tt];
                    for (int j = 0; j < N; ++j)
                    {
                        var Bpv = (Vector256<float>*)(T + j * rem + tt * Vector256<float>.Count);
                        var bv = Bpv[0];
                        c0 = Fma.MultiplyAdd(bv, Vector256.Create(Ap0[j]), c0);
                        c1 = Fma.MultiplyAdd(bv, Vector256.Create(Ap1[j]), c1);
                        c2 = Fma.MultiplyAdd(bv, Vector256.Create(Ap2[j]), c2);
                        c3 = Fma.MultiplyAdd(bv, Vector256.Create(Ap3[j]), c3);
                        c4 = Fma.MultiplyAdd(bv, Vector256.Create(Ap4[j]), c4);
                        c5 = Fma.MultiplyAdd(bv, Vector256.Create(Ap5[j]), c5);
                    }
                    rC0[tt] = c0;
                    rC1[tt] = c1;
                    rC2[tt] = c2;
                    rC3[tt] = c3;
                    rC4[tt] = c4;
                    rC5[tt] = c5;
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
            for (int i = 0; i < M; i += 6)
            {
                var Ap0 = A + i * N;
                var Ap1 = Ap0 + N;
                var Ap2 = Ap1 + N;
                var Ap3 = Ap2 + N;
                var Ap4 = Ap3 + N;
                var Ap5 = Ap4 + N;
                var Cp0 = C + i * K + blocked + vcols;
                var Cp1 = Cp0 + K;
                var Cp2 = Cp1 + K;
                var Cp3 = Cp2 + K;
                var Cp4 = Cp3 + K;
                var Cp5 = Cp4 + K;
                switch (tail)
                {
                    case 1:
                    {
                        float c0 = Cp0[0], c1 = Cp1[0], c2 = Cp2[0], c3 = Cp3[0], c4 = Cp4[0], c5 = Cp5[0];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c0 += Ap0[j] * t[0]; c1 += Ap1[j] * t[0]; c2 += Ap2[j] * t[0]; c3 += Ap3[j] * t[0]; c4 += Ap4[j] * t[0]; c5 += Ap5[j] * t[0]; }
                        Cp0[0] = c0; Cp1[0] = c1; Cp2[0] = c2; Cp3[0] = c3; Cp4[0] = c4; Cp5[0] = c5;
                        break;
                    }
                    case 2:
                    {
                        float c00 = Cp0[0], c01 = Cp0[1], c10 = Cp1[0], c11 = Cp1[1], c20 = Cp2[0], c21 = Cp2[1];
                        float c30 = Cp3[0], c31 = Cp3[1], c40 = Cp4[0], c41 = Cp4[1], c50 = Cp5[0], c51 = Cp5[1];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c00 += Ap0[j] * t[0]; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c40 += Ap4[j] * t[0]; c50 += Ap5[j] * t[0]; c01 += Ap0[j] * t[1]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c41 += Ap4[j] * t[1]; c51 += Ap5[j] * t[1]; }
                        Cp0[0] = c00; Cp0[1] = c01; Cp1[0] = c10; Cp1[1] = c11; Cp2[0] = c20; Cp2[1] = c21;
                        Cp3[0] = c30; Cp3[1] = c31; Cp4[0] = c40; Cp4[1] = c41; Cp5[0] = c50; Cp5[1] = c51;
                        break;
                    }
                    case 3:
                    {
                        float c00 = Cp0[0], c01 = Cp0[1], c02 = Cp0[2], c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c30 = Cp3[0], c31 = Cp3[1], c32 = Cp3[2];
                        float c40 = Cp4[0], c41 = Cp4[1], c42 = Cp4[2], c50 = Cp5[0], c51 = Cp5[1], c52 = Cp5[2];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c00 += Ap0[j] * t[0]; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c40 += Ap4[j] * t[0]; c50 += Ap5[j] * t[0]; c01 += Ap0[j] * t[1]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c41 += Ap4[j] * t[1]; c51 += Ap5[j] * t[1]; c02 += Ap0[j] * t[2]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; c42 += Ap4[j] * t[2]; c52 += Ap5[j] * t[2]; }
                        Cp0[0] = c00; Cp0[1] = c01; Cp0[2] = c02; Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32;
                        Cp4[0] = c40; Cp4[1] = c41; Cp4[2] = c42; Cp5[0] = c50; Cp5[1] = c51; Cp5[2] = c52;
                        break;
                    }
                    case 4:
                    {
                        float c00 = Cp0[0], c01 = Cp0[1], c02 = Cp0[2], c03 = Cp0[3], c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3], c30 = Cp3[0], c31 = Cp3[1], c32 = Cp3[2], c33 = Cp3[3];
                        float c40 = Cp4[0], c41 = Cp4[1], c42 = Cp4[2], c43 = Cp4[3], c50 = Cp5[0], c51 = Cp5[1], c52 = Cp5[2], c53 = Cp5[3];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c00 += Ap0[j] * t[0]; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c40 += Ap4[j] * t[0]; c50 += Ap5[j] * t[0]; c01 += Ap0[j] * t[1]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c41 += Ap4[j] * t[1]; c51 += Ap5[j] * t[1]; c02 += Ap0[j] * t[2]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; c42 += Ap4[j] * t[2]; c52 += Ap5[j] * t[2]; c03 += Ap0[j] * t[3]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c33 += Ap3[j] * t[3]; c43 += Ap4[j] * t[3]; c53 += Ap5[j] * t[3]; }
                        Cp0[0] = c00; Cp0[1] = c01; Cp0[2] = c02; Cp0[3] = c03; Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32; Cp3[3] = c33;
                        Cp4[0] = c40; Cp4[1] = c41; Cp4[2] = c42; Cp4[3] = c43; Cp5[0] = c50; Cp5[1] = c51; Cp5[2] = c52; Cp5[3] = c53;
                        break;
                    }
                    case 5:
                    {
                        float c00 = Cp0[0], c01 = Cp0[1], c02 = Cp0[2], c03 = Cp0[3], c04 = Cp0[4], c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3], c14 = Cp1[4];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3], c24 = Cp2[4], c30 = Cp3[0], c31 = Cp3[1], c32 = Cp3[2], c33 = Cp3[3], c34 = Cp3[4];
                        float c40 = Cp4[0], c41 = Cp4[1], c42 = Cp4[2], c43 = Cp4[3], c44 = Cp4[4], c50 = Cp5[0], c51 = Cp5[1], c52 = Cp5[2], c53 = Cp5[3], c54 = Cp5[4];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c00 += Ap0[j] * t[0]; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c40 += Ap4[j] * t[0]; c50 += Ap5[j] * t[0]; c01 += Ap0[j] * t[1]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c41 += Ap4[j] * t[1]; c51 += Ap5[j] * t[1]; c02 += Ap0[j] * t[2]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; c42 += Ap4[j] * t[2]; c52 += Ap5[j] * t[2]; c03 += Ap0[j] * t[3]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c33 += Ap3[j] * t[3]; c43 += Ap4[j] * t[3]; c53 += Ap5[j] * t[3]; c04 += Ap0[j] * t[4]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; c34 += Ap3[j] * t[4]; c44 += Ap4[j] * t[4]; c54 += Ap5[j] * t[4]; }
                        Cp0[0] = c00; Cp0[1] = c01; Cp0[2] = c02; Cp0[3] = c03; Cp0[4] = c04;
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13; Cp1[4] = c14;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp2[4] = c24;
                        Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32; Cp3[3] = c33; Cp3[4] = c34;
                        Cp4[0] = c40; Cp4[1] = c41; Cp4[2] = c42; Cp4[3] = c43; Cp4[4] = c44;
                        Cp5[0] = c50; Cp5[1] = c51; Cp5[2] = c52; Cp5[3] = c53; Cp5[4] = c54;
                        break;
                    }
                    case 6:
                    {
                        float c00 = Cp0[0], c01 = Cp0[1], c02 = Cp0[2], c03 = Cp0[3], c04 = Cp0[4], c05 = Cp0[5];
                        float c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3], c14 = Cp1[4], c15 = Cp1[5];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3], c24 = Cp2[4], c25 = Cp2[5];
                        float c30 = Cp3[0], c31 = Cp3[1], c32 = Cp3[2], c33 = Cp3[3], c34 = Cp3[4], c35 = Cp3[5];
                        float c40 = Cp4[0], c41 = Cp4[1], c42 = Cp4[2], c43 = Cp4[3], c44 = Cp4[4], c45 = Cp4[5];
                        float c50 = Cp5[0], c51 = Cp5[1], c52 = Cp5[2], c53 = Cp5[3], c54 = Cp5[4], c55 = Cp5[5];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c00 += Ap0[j] * t[0]; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c40 += Ap4[j] * t[0]; c50 += Ap5[j] * t[0]; c01 += Ap0[j] * t[1]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c41 += Ap4[j] * t[1]; c51 += Ap5[j] * t[1]; c02 += Ap0[j] * t[2]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; c42 += Ap4[j] * t[2]; c52 += Ap5[j] * t[2]; c03 += Ap0[j] * t[3]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c33 += Ap3[j] * t[3]; c43 += Ap4[j] * t[3]; c53 += Ap5[j] * t[3]; c04 += Ap0[j] * t[4]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; c34 += Ap3[j] * t[4]; c44 += Ap4[j] * t[4]; c54 += Ap5[j] * t[4]; c05 += Ap0[j] * t[5]; c15 += Ap1[j] * t[5]; c25 += Ap2[j] * t[5]; c35 += Ap3[j] * t[5]; c45 += Ap4[j] * t[5]; c55 += Ap5[j] * t[5]; }
                        Cp0[0] = c00; Cp0[1] = c01; Cp0[2] = c02; Cp0[3] = c03; Cp0[4] = c04; Cp0[5] = c05;
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13; Cp1[4] = c14; Cp1[5] = c15;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp2[4] = c24; Cp2[5] = c25;
                        Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32; Cp3[3] = c33; Cp3[4] = c34; Cp3[5] = c35;
                        Cp4[0] = c40; Cp4[1] = c41; Cp4[2] = c42; Cp4[3] = c43; Cp4[4] = c44; Cp4[5] = c45;
                        Cp5[0] = c50; Cp5[1] = c51; Cp5[2] = c52; Cp5[3] = c53; Cp5[4] = c54; Cp5[5] = c55;
                        break;
                    }
                    default:
                    {
                        float c00 = Cp0[0], c01 = Cp0[1], c02 = Cp0[2], c03 = Cp0[3], c04 = Cp0[4], c05 = Cp0[5], c06 = Cp0[6];
                        float c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3], c14 = Cp1[4], c15 = Cp1[5], c16 = Cp1[6];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3], c24 = Cp2[4], c25 = Cp2[5], c26 = Cp2[6];
                        float c30 = Cp3[0], c31 = Cp3[1], c32 = Cp3[2], c33 = Cp3[3], c34 = Cp3[4], c35 = Cp3[5], c36 = Cp3[6];
                        float c40 = Cp4[0], c41 = Cp4[1], c42 = Cp4[2], c43 = Cp4[3], c44 = Cp4[4], c45 = Cp4[5], c46 = Cp4[6];
                        float c50 = Cp5[0], c51 = Cp5[1], c52 = Cp5[2], c53 = Cp5[3], c54 = Cp5[4], c55 = Cp5[5], c56 = Cp5[6];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c00 += Ap0[j] * t[0]; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c40 += Ap4[j] * t[0]; c50 += Ap5[j] * t[0]; c01 += Ap0[j] * t[1]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c41 += Ap4[j] * t[1]; c51 += Ap5[j] * t[1]; c02 += Ap0[j] * t[2]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; c42 += Ap4[j] * t[2]; c52 += Ap5[j] * t[2]; c03 += Ap0[j] * t[3]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c33 += Ap3[j] * t[3]; c43 += Ap4[j] * t[3]; c53 += Ap5[j] * t[3]; c04 += Ap0[j] * t[4]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; c34 += Ap3[j] * t[4]; c44 += Ap4[j] * t[4]; c54 += Ap5[j] * t[4]; c05 += Ap0[j] * t[5]; c15 += Ap1[j] * t[5]; c25 += Ap2[j] * t[5]; c35 += Ap3[j] * t[5]; c45 += Ap4[j] * t[5]; c55 += Ap5[j] * t[5]; c06 += Ap0[j] * t[6]; c16 += Ap1[j] * t[6]; c26 += Ap2[j] * t[6]; c36 += Ap3[j] * t[6]; c46 += Ap4[j] * t[6]; c56 += Ap5[j] * t[6]; }
                        Cp0[0] = c00; Cp0[1] = c01; Cp0[2] = c02; Cp0[3] = c03; Cp0[4] = c04; Cp0[5] = c05; Cp0[6] = c06;
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13; Cp1[4] = c14; Cp1[5] = c15; Cp1[6] = c16;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp2[4] = c24; Cp2[5] = c25; Cp2[6] = c26;
                        Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32; Cp3[3] = c33; Cp3[4] = c34; Cp3[5] = c35; Cp3[6] = c36;
                        Cp4[0] = c40; Cp4[1] = c41; Cp4[2] = c42; Cp4[3] = c43; Cp4[4] = c44; Cp4[5] = c45; Cp4[6] = c46;
                        Cp5[0] = c50; Cp5[1] = c51; Cp5[2] = c52; Cp5[3] = c53; Cp5[4] = c54; Cp5[5] = c55; Cp5[6] = c56;
                        break;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Register-tiled matrix multiplication reading panel-packed B, four
    /// rows per group over 16-column halves (E63).
    /// </summary>
    /// <param name="M">A rows (must be a multiple of 4).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="P">Panel-packed right matrix from PackPanelsB.</param>
    /// <param name="C">Result matrix (accumulated, like the other packed kernels).</param>
    /// <remarks>
    /// Same panels and per-element order as the 2-row packed kernel, so results
    /// agree with it bit-wise; four rows share each B vector. Sixteen-column
    /// halves keep eight accumulators live alongside the B vectors and one
    /// broadcast temporary within sixteen AVX2 registers. Tails read the
    /// appended row-major tail block in the same order.
    /// </remarks>
    public unsafe static void mm_unsafe_vectorized_intrinsics_4x4packed(int M,
                          int N,
                          int K,
                          float* A,
                          float* P,
                          float* C)
    {
        if (M % 4 != 0)
            throw new ArgumentException(nameof(M));

        int panelWidth = 4 * Vector256<float>.Count;
        int halfWidth = 2 * Vector256<float>.Count;
        int blocked = K - (K % panelWidth);
        int tiles = blocked / panelWidth;

        // Kb panels lead so each panel is fetched once and stays L1/L2 resident
        // across every row group; per-element FMA order matches the 2-row nest
        // exactly, so results agree bit-wise with the other packed kernels.
        for (int tb = 0; tb < tiles; tb++)
        {
            int kb = tb * panelWidth;
            float* panel = P + tb * N * panelWidth;
            for (int hh = 0; hh < 2; hh++)
            {
                int ko = kb + hh * halfWidth;
                for (int i = 0; i < M; i += 4)
                {
                    var Ap0 = A + i * N;
                    var Ap1 = Ap0 + N;
                    var Ap2 = Ap1 + N;
                    var Ap3 = Ap2 + N;
                    var Cp0 = (Vector256<float>*)(C + i * K + ko);
                    var Cp1 = (Vector256<float>*)(C + (i + 1) * K + ko);
                    var Cp2 = (Vector256<float>*)(C + (i + 2) * K + ko);
                    var Cp3 = (Vector256<float>*)(C + (i + 3) * K + ko);
                    Vector256<float> c00 = Cp0[0];
                    Vector256<float> c01 = Cp0[1];
                    Vector256<float> c10 = Cp1[0];
                    Vector256<float> c11 = Cp1[1];
                    Vector256<float> c20 = Cp2[0];
                    Vector256<float> c21 = Cp2[1];
                    Vector256<float> c30 = Cp3[0];
                    Vector256<float> c31 = Cp3[1];
                    for (int j = 0; j < N; ++j)
                    {
                        var Bpv = (Vector256<float>*)(panel + j * panelWidth + hh * halfWidth);
                        Vector256<float> bv0 = Bpv[0];
                        Vector256<float> bv1 = Bpv[1];
                        var av0 = Vector256.Create(Ap0[j]);
                        c00 = Fma.MultiplyAdd(bv0, av0, c00);
                        c01 = Fma.MultiplyAdd(bv1, av0, c01);
                        var av1 = Vector256.Create(Ap1[j]);
                        c10 = Fma.MultiplyAdd(bv0, av1, c10);
                        c11 = Fma.MultiplyAdd(bv1, av1, c11);
                        var av2 = Vector256.Create(Ap2[j]);
                        c20 = Fma.MultiplyAdd(bv0, av2, c20);
                        c21 = Fma.MultiplyAdd(bv1, av2, c21);
                        var av3 = Vector256.Create(Ap3[j]);
                        c30 = Fma.MultiplyAdd(bv0, av3, c30);
                        c31 = Fma.MultiplyAdd(bv1, av3, c31);
                    }
                    Cp0[0] = c00;
                    Cp0[1] = c01;
                    Cp1[0] = c10;
                    Cp1[1] = c11;
                    Cp2[0] = c20;
                    Cp2[1] = c21;
                    Cp3[0] = c30;
                    Cp3[1] = c31;
                }
            }
        }
        int rem = K - blocked;
        if (rem > 0)
        {
            float* T = P + tiles * N * panelWidth;
            int rv = rem / Vector256<float>.Count;
            for (int tt = 0; tt < rv; tt++)
            {
                for (int i = 0; i < M; i += 4)
                {
                    var Ap0 = A + i * N;
                    var Ap1 = Ap0 + N;
                    var Ap2 = Ap1 + N;
                    var Ap3 = Ap2 + N;
                    var rC0 = (Vector256<float>*)(C + i * K + blocked);
                    var rC1 = (Vector256<float>*)(C + (i + 1) * K + blocked);
                    var rC2 = (Vector256<float>*)(C + (i + 2) * K + blocked);
                    var rC3 = (Vector256<float>*)(C + (i + 3) * K + blocked);
                    Vector256<float> c0 = rC0[tt];
                    Vector256<float> c1 = rC1[tt];
                    Vector256<float> c2 = rC2[tt];
                    Vector256<float> c3 = rC3[tt];
                    for (int j = 0; j < N; ++j)
                    {
                        var Bpv = (Vector256<float>*)(T + j * rem + tt * Vector256<float>.Count);
                        var bv = Bpv[0];
                        c0 = Fma.MultiplyAdd(bv, Vector256.Create(Ap0[j]), c0);
                        c1 = Fma.MultiplyAdd(bv, Vector256.Create(Ap1[j]), c1);
                        c2 = Fma.MultiplyAdd(bv, Vector256.Create(Ap2[j]), c2);
                        c3 = Fma.MultiplyAdd(bv, Vector256.Create(Ap3[j]), c3);
                    }
                    rC0[tt] = c0;
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
            for (int i = 0; i < M; i += 4)
            {
                var Ap0 = A + i * N;
                var Ap1 = Ap0 + N;
                var Ap2 = Ap1 + N;
                var Ap3 = Ap2 + N;
                var Cp0 = C + i * K + blocked + vcols;
                var Cp1 = Cp0 + K;
                var Cp2 = Cp1 + K;
                var Cp3 = Cp2 + K;
                switch (tail)
                {
                    case 1:
                    {
                        float c0 = Cp0[0], c1 = Cp1[0], c2 = Cp2[0], c3 = Cp3[0];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c0 += Ap0[j] * t[0]; c1 += Ap1[j] * t[0]; c2 += Ap2[j] * t[0]; c3 += Ap3[j] * t[0]; }
                        Cp0[0] = c0; Cp1[0] = c1; Cp2[0] = c2; Cp3[0] = c3;
                        break;
                    }
                    case 2:
                    {
                        float c00 = Cp0[0], c01 = Cp0[1], c10 = Cp1[0], c11 = Cp1[1], c20 = Cp2[0], c21 = Cp2[1], c30 = Cp3[0], c31 = Cp3[1];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c00 += Ap0[j] * t[0]; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c01 += Ap0[j] * t[1]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; }
                        Cp0[0] = c00; Cp0[1] = c01; Cp1[0] = c10; Cp1[1] = c11; Cp2[0] = c20; Cp2[1] = c21; Cp3[0] = c30; Cp3[1] = c31;
                        break;
                    }
                    case 3:
                    {
                        float c00 = Cp0[0], c01 = Cp0[1], c02 = Cp0[2], c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c30 = Cp3[0], c31 = Cp3[1], c32 = Cp3[2];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c00 += Ap0[j] * t[0]; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c01 += Ap0[j] * t[1]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c02 += Ap0[j] * t[2]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; }
                        Cp0[0] = c00; Cp0[1] = c01; Cp0[2] = c02; Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32;
                        break;
                    }
                    case 4:
                    {
                        float c00 = Cp0[0], c01 = Cp0[1], c02 = Cp0[2], c03 = Cp0[3], c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3], c30 = Cp3[0], c31 = Cp3[1], c32 = Cp3[2], c33 = Cp3[3];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c00 += Ap0[j] * t[0]; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c01 += Ap0[j] * t[1]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c02 += Ap0[j] * t[2]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; c03 += Ap0[j] * t[3]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c33 += Ap3[j] * t[3]; }
                        Cp0[0] = c00; Cp0[1] = c01; Cp0[2] = c02; Cp0[3] = c03; Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32; Cp3[3] = c33;
                        break;
                    }
                    case 5:
                    {
                        float c00 = Cp0[0], c01 = Cp0[1], c02 = Cp0[2], c03 = Cp0[3], c04 = Cp0[4], c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3], c14 = Cp1[4];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3], c24 = Cp2[4], c30 = Cp3[0], c31 = Cp3[1], c32 = Cp3[2], c33 = Cp3[3], c34 = Cp3[4];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c00 += Ap0[j] * t[0]; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c01 += Ap0[j] * t[1]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c02 += Ap0[j] * t[2]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; c03 += Ap0[j] * t[3]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c33 += Ap3[j] * t[3]; c04 += Ap0[j] * t[4]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; c34 += Ap3[j] * t[4]; }
                        Cp0[0] = c00; Cp0[1] = c01; Cp0[2] = c02; Cp0[3] = c03; Cp0[4] = c04;
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13; Cp1[4] = c14;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp2[4] = c24;
                        Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32; Cp3[3] = c33; Cp3[4] = c34;
                        break;
                    }
                    case 6:
                    {
                        float c00 = Cp0[0], c01 = Cp0[1], c02 = Cp0[2], c03 = Cp0[3], c04 = Cp0[4], c05 = Cp0[5];
                        float c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3], c14 = Cp1[4], c15 = Cp1[5];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3], c24 = Cp2[4], c25 = Cp2[5];
                        float c30 = Cp3[0], c31 = Cp3[1], c32 = Cp3[2], c33 = Cp3[3], c34 = Cp3[4], c35 = Cp3[5];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c00 += Ap0[j] * t[0]; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c01 += Ap0[j] * t[1]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c02 += Ap0[j] * t[2]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; c03 += Ap0[j] * t[3]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c33 += Ap3[j] * t[3]; c04 += Ap0[j] * t[4]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; c34 += Ap3[j] * t[4]; c05 += Ap0[j] * t[5]; c15 += Ap1[j] * t[5]; c25 += Ap2[j] * t[5]; c35 += Ap3[j] * t[5]; }
                        Cp0[0] = c00; Cp0[1] = c01; Cp0[2] = c02; Cp0[3] = c03; Cp0[4] = c04; Cp0[5] = c05;
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13; Cp1[4] = c14; Cp1[5] = c15;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp2[4] = c24; Cp2[5] = c25;
                        Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32; Cp3[3] = c33; Cp3[4] = c34; Cp3[5] = c35;
                        break;
                    }
                    default:
                    {
                        float c00 = Cp0[0], c01 = Cp0[1], c02 = Cp0[2], c03 = Cp0[3], c04 = Cp0[4], c05 = Cp0[5], c06 = Cp0[6];
                        float c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3], c14 = Cp1[4], c15 = Cp1[5], c16 = Cp1[6];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3], c24 = Cp2[4], c25 = Cp2[5], c26 = Cp2[6];
                        float c30 = Cp3[0], c31 = Cp3[1], c32 = Cp3[2], c33 = Cp3[3], c34 = Cp3[4], c35 = Cp3[5], c36 = Cp3[6];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c00 += Ap0[j] * t[0]; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c01 += Ap0[j] * t[1]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c02 += Ap0[j] * t[2]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; c03 += Ap0[j] * t[3]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c33 += Ap3[j] * t[3]; c04 += Ap0[j] * t[4]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; c34 += Ap3[j] * t[4]; c05 += Ap0[j] * t[5]; c15 += Ap1[j] * t[5]; c25 += Ap2[j] * t[5]; c35 += Ap3[j] * t[5]; c06 += Ap0[j] * t[6]; c16 += Ap1[j] * t[6]; c26 += Ap2[j] * t[6]; c36 += Ap3[j] * t[6]; }
                        Cp0[0] = c00; Cp0[1] = c01; Cp0[2] = c02; Cp0[3] = c03; Cp0[4] = c04; Cp0[5] = c05; Cp0[6] = c06;
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
    /// Register-tiled matrix multiplication reading panel-packed B, eight
    /// rows per group over 8-column quarters (E63).
    /// </summary>
    /// <param name="M">A rows (must be a multiple of 8).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="P">Panel-packed right matrix from PackPanelsB.</param>
    /// <param name="C">Result matrix (accumulated, like the other packed kernels).</param>
    /// <remarks>
    /// Same panels and per-element order as the 2-row packed kernel, so results
    /// agree with it bit-wise; eight rows share each B vector. Eight-column
    /// quarters keep eight accumulators live alongside the B vector and one
    /// broadcast temporary within sixteen AVX2 registers. Tails read the
    /// appended row-major tail block in the same order.
    /// </remarks>
    public unsafe static void mm_unsafe_vectorized_intrinsics_8x8packed(int M,
                          int N,
                          int K,
                          float* A,
                          float* P,
                          float* C)
    {
        if (M % 8 != 0)
            throw new ArgumentException(nameof(M));

        int panelWidth = 4 * Vector256<float>.Count;
        int quarterWidth = Vector256<float>.Count;
        int blocked = K - (K % panelWidth);
        int tiles = blocked / panelWidth;

        // Kb panels lead so each panel is fetched once and stays L1/L2 resident
        // across every row group; per-element FMA order matches the 2-row nest
        // exactly, so results agree bit-wise with the other packed kernels.
        for (int tb = 0; tb < tiles; tb++)
        {
            int kb = tb * panelWidth;
            float* panel = P + tb * N * panelWidth;
            for (int qq = 0; qq < 4; qq++)
            {
                int ko = kb + qq * quarterWidth;
                for (int i = 0; i < M; i += 8)
                {
                    var Ap0 = A + i * N;
                    var Ap1 = Ap0 + N;
                    var Ap2 = Ap1 + N;
                    var Ap3 = Ap2 + N;
                    var Ap4 = Ap3 + N;
                    var Ap5 = Ap4 + N;
                    var Ap6 = Ap5 + N;
                    var Ap7 = Ap6 + N;
                    var Cp0 = (Vector256<float>*)(C + i * K + ko);
                    var Cp1 = (Vector256<float>*)(C + (i + 1) * K + ko);
                    var Cp2 = (Vector256<float>*)(C + (i + 2) * K + ko);
                    var Cp3 = (Vector256<float>*)(C + (i + 3) * K + ko);
                    var Cp4 = (Vector256<float>*)(C + (i + 4) * K + ko);
                    var Cp5 = (Vector256<float>*)(C + (i + 5) * K + ko);
                    var Cp6 = (Vector256<float>*)(C + (i + 6) * K + ko);
                    var Cp7 = (Vector256<float>*)(C + (i + 7) * K + ko);
                    Vector256<float> c0 = Cp0[0];
                    Vector256<float> c1 = Cp1[0];
                    Vector256<float> c2 = Cp2[0];
                    Vector256<float> c3 = Cp3[0];
                    Vector256<float> c4 = Cp4[0];
                    Vector256<float> c5 = Cp5[0];
                    Vector256<float> c6 = Cp6[0];
                    Vector256<float> c7 = Cp7[0];
                    for (int j = 0; j < N; ++j)
                    {
                        var Bpv = (Vector256<float>*)(panel + j * panelWidth + qq * quarterWidth);
                        Vector256<float> bv = Bpv[0];
                        var av0 = Vector256.Create(Ap0[j]);
                        c0 = Fma.MultiplyAdd(bv, av0, c0);
                        var av1 = Vector256.Create(Ap1[j]);
                        c1 = Fma.MultiplyAdd(bv, av1, c1);
                        var av2 = Vector256.Create(Ap2[j]);
                        c2 = Fma.MultiplyAdd(bv, av2, c2);
                        var av3 = Vector256.Create(Ap3[j]);
                        c3 = Fma.MultiplyAdd(bv, av3, c3);
                        var av4 = Vector256.Create(Ap4[j]);
                        c4 = Fma.MultiplyAdd(bv, av4, c4);
                        var av5 = Vector256.Create(Ap5[j]);
                        c5 = Fma.MultiplyAdd(bv, av5, c5);
                        var av6 = Vector256.Create(Ap6[j]);
                        c6 = Fma.MultiplyAdd(bv, av6, c6);
                        var av7 = Vector256.Create(Ap7[j]);
                        c7 = Fma.MultiplyAdd(bv, av7, c7);
                    }
                    Cp0[0] = c0;
                    Cp1[0] = c1;
                    Cp2[0] = c2;
                    Cp3[0] = c3;
                    Cp4[0] = c4;
                    Cp5[0] = c5;
                    Cp6[0] = c6;
                    Cp7[0] = c7;
                }
            }
        }
        int rem = K - blocked;
        if (rem > 0)
        {
            float* T = P + tiles * N * panelWidth;
            int nq = rem / Vector256<float>.Count;
            for (int tt = 0; tt < nq; tt++)
            {
                for (int i = 0; i < M; i += 8)
                {
                    var Ap0 = A + i * N;
                    var Ap1 = Ap0 + N;
                    var Ap2 = Ap1 + N;
                    var Ap3 = Ap2 + N;
                    var Ap4 = Ap3 + N;
                    var Ap5 = Ap4 + N;
                    var Ap6 = Ap5 + N;
                    var Ap7 = Ap6 + N;
                    var rC0 = (Vector256<float>*)(C + i * K + blocked);
                    var rC1 = (Vector256<float>*)(C + (i + 1) * K + blocked);
                    var rC2 = (Vector256<float>*)(C + (i + 2) * K + blocked);
                    var rC3 = (Vector256<float>*)(C + (i + 3) * K + blocked);
                    var rC4 = (Vector256<float>*)(C + (i + 4) * K + blocked);
                    var rC5 = (Vector256<float>*)(C + (i + 5) * K + blocked);
                    var rC6 = (Vector256<float>*)(C + (i + 6) * K + blocked);
                    var rC7 = (Vector256<float>*)(C + (i + 7) * K + blocked);
                    Vector256<float> c0 = rC0[tt];
                    Vector256<float> c1 = rC1[tt];
                    Vector256<float> c2 = rC2[tt];
                    Vector256<float> c3 = rC3[tt];
                    Vector256<float> c4 = rC4[tt];
                    Vector256<float> c5 = rC5[tt];
                    Vector256<float> c6 = rC6[tt];
                    Vector256<float> c7 = rC7[tt];
                    for (int j = 0; j < N; ++j)
                    {
                        var Bpv = (Vector256<float>*)(T + j * rem + tt * Vector256<float>.Count);
                        var bv = Bpv[0];
                        c0 = Fma.MultiplyAdd(bv, Vector256.Create(Ap0[j]), c0);
                        c1 = Fma.MultiplyAdd(bv, Vector256.Create(Ap1[j]), c1);
                        c2 = Fma.MultiplyAdd(bv, Vector256.Create(Ap2[j]), c2);
                        c3 = Fma.MultiplyAdd(bv, Vector256.Create(Ap3[j]), c3);
                        c4 = Fma.MultiplyAdd(bv, Vector256.Create(Ap4[j]), c4);
                        c5 = Fma.MultiplyAdd(bv, Vector256.Create(Ap5[j]), c5);
                        c6 = Fma.MultiplyAdd(bv, Vector256.Create(Ap6[j]), c6);
                        c7 = Fma.MultiplyAdd(bv, Vector256.Create(Ap7[j]), c7);
                    }
                    rC0[tt] = c0;
                    rC1[tt] = c1;
                    rC2[tt] = c2;
                    rC3[tt] = c3;
                    rC4[tt] = c4;
                    rC5[tt] = c5;
                    rC6[tt] = c6;
                    rC7[tt] = c7;
                }
            }
            int vcols = nq * Vector256<float>.Count;
            // Narrow tail (fewer than 8 columns) accumulates in registers
            // instead of read-modify-writing C on every reduction step: the
            // per-element chain keeps the exact j-ascending mul-then-add order
            // of the loop it replaces, so results agree bit-wise while each C
            // row is loaded once and stored once.
            int tail = rem - vcols;
            if (tail > 0)
            for (int i = 0; i < M; i += 8)
            {
                var Ap0 = A + i * N;
                var Ap1 = Ap0 + N;
                var Ap2 = Ap1 + N;
                var Ap3 = Ap2 + N;
                var Ap4 = Ap3 + N;
                var Ap5 = Ap4 + N;
                var Ap6 = Ap5 + N;
                var Ap7 = Ap6 + N;
                var Cp0 = C + i * K + blocked + vcols;
                var Cp1 = Cp0 + K;
                var Cp2 = Cp1 + K;
                var Cp3 = Cp2 + K;
                var Cp4 = Cp3 + K;
                var Cp5 = Cp4 + K;
                var Cp6 = Cp5 + K;
                var Cp7 = Cp6 + K;
                switch (tail)
                {
                    case 1:
                    {
                        float c0 = Cp0[0], c1 = Cp1[0], c2 = Cp2[0], c3 = Cp3[0], c4 = Cp4[0], c5 = Cp5[0], c6 = Cp6[0], c7 = Cp7[0];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c0 += Ap0[j] * t[0]; c1 += Ap1[j] * t[0]; c2 += Ap2[j] * t[0]; c3 += Ap3[j] * t[0]; c4 += Ap4[j] * t[0]; c5 += Ap5[j] * t[0]; c6 += Ap6[j] * t[0]; c7 += Ap7[j] * t[0]; }
                        Cp0[0] = c0; Cp1[0] = c1; Cp2[0] = c2; Cp3[0] = c3; Cp4[0] = c4; Cp5[0] = c5; Cp6[0] = c6; Cp7[0] = c7;
                        break;
                    }
                    case 2:
                    {
                        float c00 = Cp0[0], c01 = Cp0[1], c10 = Cp1[0], c11 = Cp1[1], c20 = Cp2[0], c21 = Cp2[1], c30 = Cp3[0], c31 = Cp3[1];
                        float c40 = Cp4[0], c41 = Cp4[1], c50 = Cp5[0], c51 = Cp5[1], c60 = Cp6[0], c61 = Cp6[1], c70 = Cp7[0], c71 = Cp7[1];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c00 += Ap0[j] * t[0]; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c40 += Ap4[j] * t[0]; c50 += Ap5[j] * t[0]; c60 += Ap6[j] * t[0]; c70 += Ap7[j] * t[0]; c01 += Ap0[j] * t[1]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c41 += Ap4[j] * t[1]; c51 += Ap5[j] * t[1]; c61 += Ap6[j] * t[1]; c71 += Ap7[j] * t[1]; }
                        Cp0[0] = c00; Cp0[1] = c01; Cp1[0] = c10; Cp1[1] = c11; Cp2[0] = c20; Cp2[1] = c21; Cp3[0] = c30; Cp3[1] = c31;
                        Cp4[0] = c40; Cp4[1] = c41; Cp5[0] = c50; Cp5[1] = c51; Cp6[0] = c60; Cp6[1] = c61; Cp7[0] = c70; Cp7[1] = c71;
                        break;
                    }
                    case 3:
                    {
                        float c00 = Cp0[0], c01 = Cp0[1], c02 = Cp0[2], c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c30 = Cp3[0], c31 = Cp3[1], c32 = Cp3[2];
                        float c40 = Cp4[0], c41 = Cp4[1], c42 = Cp4[2], c50 = Cp5[0], c51 = Cp5[1], c52 = Cp5[2], c60 = Cp6[0], c61 = Cp6[1], c62 = Cp6[2], c70 = Cp7[0], c71 = Cp7[1], c72 = Cp7[2];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c00 += Ap0[j] * t[0]; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c40 += Ap4[j] * t[0]; c50 += Ap5[j] * t[0]; c60 += Ap6[j] * t[0]; c70 += Ap7[j] * t[0]; c01 += Ap0[j] * t[1]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c41 += Ap4[j] * t[1]; c51 += Ap5[j] * t[1]; c61 += Ap6[j] * t[1]; c71 += Ap7[j] * t[1]; c02 += Ap0[j] * t[2]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; c42 += Ap4[j] * t[2]; c52 += Ap5[j] * t[2]; c62 += Ap6[j] * t[2]; c72 += Ap7[j] * t[2]; }
                        Cp0[0] = c00; Cp0[1] = c01; Cp0[2] = c02; Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32;
                        Cp4[0] = c40; Cp4[1] = c41; Cp4[2] = c42; Cp5[0] = c50; Cp5[1] = c51; Cp5[2] = c52;
                        Cp6[0] = c60; Cp6[1] = c61; Cp6[2] = c62; Cp7[0] = c70; Cp7[1] = c71; Cp7[2] = c72;
                        break;
                    }
                    case 4:
                    {
                        float c00 = Cp0[0], c01 = Cp0[1], c02 = Cp0[2], c03 = Cp0[3], c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3], c30 = Cp3[0], c31 = Cp3[1], c32 = Cp3[2], c33 = Cp3[3];
                        float c40 = Cp4[0], c41 = Cp4[1], c42 = Cp4[2], c43 = Cp4[3], c50 = Cp5[0], c51 = Cp5[1], c52 = Cp5[2], c53 = Cp5[3];
                        float c60 = Cp6[0], c61 = Cp6[1], c62 = Cp6[2], c63 = Cp6[3], c70 = Cp7[0], c71 = Cp7[1], c72 = Cp7[2], c73 = Cp7[3];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c00 += Ap0[j] * t[0]; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c40 += Ap4[j] * t[0]; c50 += Ap5[j] * t[0]; c60 += Ap6[j] * t[0]; c70 += Ap7[j] * t[0]; c01 += Ap0[j] * t[1]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c41 += Ap4[j] * t[1]; c51 += Ap5[j] * t[1]; c61 += Ap6[j] * t[1]; c71 += Ap7[j] * t[1]; c02 += Ap0[j] * t[2]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; c42 += Ap4[j] * t[2]; c52 += Ap5[j] * t[2]; c62 += Ap6[j] * t[2]; c72 += Ap7[j] * t[2]; c03 += Ap0[j] * t[3]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c33 += Ap3[j] * t[3]; c43 += Ap4[j] * t[3]; c53 += Ap5[j] * t[3]; c63 += Ap6[j] * t[3]; c73 += Ap7[j] * t[3]; }
                        Cp0[0] = c00; Cp0[1] = c01; Cp0[2] = c02; Cp0[3] = c03; Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32; Cp3[3] = c33;
                        Cp4[0] = c40; Cp4[1] = c41; Cp4[2] = c42; Cp4[3] = c43; Cp5[0] = c50; Cp5[1] = c51; Cp5[2] = c52; Cp5[3] = c53;
                        Cp6[0] = c60; Cp6[1] = c61; Cp6[2] = c62; Cp6[3] = c63; Cp7[0] = c70; Cp7[1] = c71; Cp7[2] = c72; Cp7[3] = c73;
                        break;
                    }
                    case 5:
                    {
                        float c00 = Cp0[0], c01 = Cp0[1], c02 = Cp0[2], c03 = Cp0[3], c04 = Cp0[4], c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3], c14 = Cp1[4];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3], c24 = Cp2[4], c30 = Cp3[0], c31 = Cp3[1], c32 = Cp3[2], c33 = Cp3[3], c34 = Cp3[4];
                        float c40 = Cp4[0], c41 = Cp4[1], c42 = Cp4[2], c43 = Cp4[3], c44 = Cp4[4], c50 = Cp5[0], c51 = Cp5[1], c52 = Cp5[2], c53 = Cp5[3], c54 = Cp5[4];
                        float c60 = Cp6[0], c61 = Cp6[1], c62 = Cp6[2], c63 = Cp6[3], c64 = Cp6[4], c70 = Cp7[0], c71 = Cp7[1], c72 = Cp7[2], c73 = Cp7[3], c74 = Cp7[4];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c00 += Ap0[j] * t[0]; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c40 += Ap4[j] * t[0]; c50 += Ap5[j] * t[0]; c60 += Ap6[j] * t[0]; c70 += Ap7[j] * t[0]; c01 += Ap0[j] * t[1]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c41 += Ap4[j] * t[1]; c51 += Ap5[j] * t[1]; c61 += Ap6[j] * t[1]; c71 += Ap7[j] * t[1]; c02 += Ap0[j] * t[2]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; c42 += Ap4[j] * t[2]; c52 += Ap5[j] * t[2]; c62 += Ap6[j] * t[2]; c72 += Ap7[j] * t[2]; c03 += Ap0[j] * t[3]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c33 += Ap3[j] * t[3]; c43 += Ap4[j] * t[3]; c53 += Ap5[j] * t[3]; c63 += Ap6[j] * t[3]; c73 += Ap7[j] * t[3]; c04 += Ap0[j] * t[4]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; c34 += Ap3[j] * t[4]; c44 += Ap4[j] * t[4]; c54 += Ap5[j] * t[4]; c64 += Ap6[j] * t[4]; c74 += Ap7[j] * t[4]; }
                        Cp0[0] = c00; Cp0[1] = c01; Cp0[2] = c02; Cp0[3] = c03; Cp0[4] = c04;
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13; Cp1[4] = c14;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp2[4] = c24;
                        Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32; Cp3[3] = c33; Cp3[4] = c34;
                        Cp4[0] = c40; Cp4[1] = c41; Cp4[2] = c42; Cp4[3] = c43; Cp4[4] = c44;
                        Cp5[0] = c50; Cp5[1] = c51; Cp5[2] = c52; Cp5[3] = c53; Cp5[4] = c54;
                        Cp6[0] = c60; Cp6[1] = c61; Cp6[2] = c62; Cp6[3] = c63; Cp6[4] = c64;
                        Cp7[0] = c70; Cp7[1] = c71; Cp7[2] = c72; Cp7[3] = c73; Cp7[4] = c74;
                        break;
                    }
                    case 6:
                    {
                        float c00 = Cp0[0], c01 = Cp0[1], c02 = Cp0[2], c03 = Cp0[3], c04 = Cp0[4], c05 = Cp0[5];
                        float c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3], c14 = Cp1[4], c15 = Cp1[5];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3], c24 = Cp2[4], c25 = Cp2[5];
                        float c30 = Cp3[0], c31 = Cp3[1], c32 = Cp3[2], c33 = Cp3[3], c34 = Cp3[4], c35 = Cp3[5];
                        float c40 = Cp4[0], c41 = Cp4[1], c42 = Cp4[2], c43 = Cp4[3], c44 = Cp4[4], c45 = Cp4[5];
                        float c50 = Cp5[0], c51 = Cp5[1], c52 = Cp5[2], c53 = Cp5[3], c54 = Cp5[4], c55 = Cp5[5];
                        float c60 = Cp6[0], c61 = Cp6[1], c62 = Cp6[2], c63 = Cp6[3], c64 = Cp6[4], c65 = Cp6[5];
                        float c70 = Cp7[0], c71 = Cp7[1], c72 = Cp7[2], c73 = Cp7[3], c74 = Cp7[4], c75 = Cp7[5];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c00 += Ap0[j] * t[0]; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c40 += Ap4[j] * t[0]; c50 += Ap5[j] * t[0]; c60 += Ap6[j] * t[0]; c70 += Ap7[j] * t[0]; c01 += Ap0[j] * t[1]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c41 += Ap4[j] * t[1]; c51 += Ap5[j] * t[1]; c61 += Ap6[j] * t[1]; c71 += Ap7[j] * t[1]; c02 += Ap0[j] * t[2]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; c42 += Ap4[j] * t[2]; c52 += Ap5[j] * t[2]; c62 += Ap6[j] * t[2]; c72 += Ap7[j] * t[2]; c03 += Ap0[j] * t[3]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c33 += Ap3[j] * t[3]; c43 += Ap4[j] * t[3]; c53 += Ap5[j] * t[3]; c63 += Ap6[j] * t[3]; c73 += Ap7[j] * t[3]; c04 += Ap0[j] * t[4]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; c34 += Ap3[j] * t[4]; c44 += Ap4[j] * t[4]; c54 += Ap5[j] * t[4]; c64 += Ap6[j] * t[4]; c74 += Ap7[j] * t[4]; c05 += Ap0[j] * t[5]; c15 += Ap1[j] * t[5]; c25 += Ap2[j] * t[5]; c35 += Ap3[j] * t[5]; c45 += Ap4[j] * t[5]; c55 += Ap5[j] * t[5]; c65 += Ap6[j] * t[5]; c75 += Ap7[j] * t[5]; }
                        Cp0[0] = c00; Cp0[1] = c01; Cp0[2] = c02; Cp0[3] = c03; Cp0[4] = c04; Cp0[5] = c05;
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13; Cp1[4] = c14; Cp1[5] = c15;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp2[4] = c24; Cp2[5] = c25;
                        Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32; Cp3[3] = c33; Cp3[4] = c34; Cp3[5] = c35;
                        Cp4[0] = c40; Cp4[1] = c41; Cp4[2] = c42; Cp4[3] = c43; Cp4[4] = c44; Cp4[5] = c45;
                        Cp5[0] = c50; Cp5[1] = c51; Cp5[2] = c52; Cp5[3] = c53; Cp5[4] = c54; Cp5[5] = c55;
                        Cp6[0] = c60; Cp6[1] = c61; Cp6[2] = c62; Cp6[3] = c63; Cp6[4] = c64; Cp6[5] = c65;
                        Cp7[0] = c70; Cp7[1] = c71; Cp7[2] = c72; Cp7[3] = c73; Cp7[4] = c74; Cp7[5] = c75;
                        break;
                    }
                    default:
                    {
                        float c00 = Cp0[0], c01 = Cp0[1], c02 = Cp0[2], c03 = Cp0[3], c04 = Cp0[4], c05 = Cp0[5], c06 = Cp0[6];
                        float c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3], c14 = Cp1[4], c15 = Cp1[5], c16 = Cp1[6];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3], c24 = Cp2[4], c25 = Cp2[5], c26 = Cp2[6];
                        float c30 = Cp3[0], c31 = Cp3[1], c32 = Cp3[2], c33 = Cp3[3], c34 = Cp3[4], c35 = Cp3[5], c36 = Cp3[6];
                        float c40 = Cp4[0], c41 = Cp4[1], c42 = Cp4[2], c43 = Cp4[3], c44 = Cp4[4], c45 = Cp4[5], c46 = Cp4[6];
                        float c50 = Cp5[0], c51 = Cp5[1], c52 = Cp5[2], c53 = Cp5[3], c54 = Cp5[4], c55 = Cp5[5], c56 = Cp5[6];
                        float c60 = Cp6[0], c61 = Cp6[1], c62 = Cp6[2], c63 = Cp6[3], c64 = Cp6[4], c65 = Cp6[5], c66 = Cp6[6];
                        float c70 = Cp7[0], c71 = Cp7[1], c72 = Cp7[2], c73 = Cp7[3], c74 = Cp7[4], c75 = Cp7[5], c76 = Cp7[6];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c00 += Ap0[j] * t[0]; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c30 += Ap3[j] * t[0]; c40 += Ap4[j] * t[0]; c50 += Ap5[j] * t[0]; c60 += Ap6[j] * t[0]; c70 += Ap7[j] * t[0]; c01 += Ap0[j] * t[1]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c31 += Ap3[j] * t[1]; c41 += Ap4[j] * t[1]; c51 += Ap5[j] * t[1]; c61 += Ap6[j] * t[1]; c71 += Ap7[j] * t[1]; c02 += Ap0[j] * t[2]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c32 += Ap3[j] * t[2]; c42 += Ap4[j] * t[2]; c52 += Ap5[j] * t[2]; c62 += Ap6[j] * t[2]; c72 += Ap7[j] * t[2]; c03 += Ap0[j] * t[3]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c33 += Ap3[j] * t[3]; c43 += Ap4[j] * t[3]; c53 += Ap5[j] * t[3]; c63 += Ap6[j] * t[3]; c73 += Ap7[j] * t[3]; c04 += Ap0[j] * t[4]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; c34 += Ap3[j] * t[4]; c44 += Ap4[j] * t[4]; c54 += Ap5[j] * t[4]; c64 += Ap6[j] * t[4]; c74 += Ap7[j] * t[4]; c05 += Ap0[j] * t[5]; c15 += Ap1[j] * t[5]; c25 += Ap2[j] * t[5]; c35 += Ap3[j] * t[5]; c45 += Ap4[j] * t[5]; c55 += Ap5[j] * t[5]; c65 += Ap6[j] * t[5]; c75 += Ap7[j] * t[5]; c06 += Ap0[j] * t[6]; c16 += Ap1[j] * t[6]; c26 += Ap2[j] * t[6]; c36 += Ap3[j] * t[6]; c46 += Ap4[j] * t[6]; c56 += Ap5[j] * t[6]; c66 += Ap6[j] * t[6]; c76 += Ap7[j] * t[6]; }
                        Cp0[0] = c00; Cp0[1] = c01; Cp0[2] = c02; Cp0[3] = c03; Cp0[4] = c04; Cp0[5] = c05; Cp0[6] = c06;
                        Cp1[0] = c10; Cp1[1] = c11; Cp1[2] = c12; Cp1[3] = c13; Cp1[4] = c14; Cp1[5] = c15; Cp1[6] = c16;
                        Cp2[0] = c20; Cp2[1] = c21; Cp2[2] = c22; Cp2[3] = c23; Cp2[4] = c24; Cp2[5] = c25; Cp2[6] = c26;
                        Cp3[0] = c30; Cp3[1] = c31; Cp3[2] = c32; Cp3[3] = c33; Cp3[4] = c34; Cp3[5] = c35; Cp3[6] = c36;
                        Cp4[0] = c40; Cp4[1] = c41; Cp4[2] = c42; Cp4[3] = c43; Cp4[4] = c44; Cp4[5] = c45; Cp4[6] = c46;
                        Cp5[0] = c50; Cp5[1] = c51; Cp5[2] = c52; Cp5[3] = c53; Cp5[4] = c54; Cp5[5] = c55; Cp5[6] = c56;
                        Cp6[0] = c60; Cp6[1] = c61; Cp6[2] = c62; Cp6[3] = c63; Cp6[4] = c64; Cp6[5] = c65; Cp6[6] = c66;
                        Cp7[0] = c70; Cp7[1] = c71; Cp7[2] = c72; Cp7[3] = c73; Cp7[4] = c74; Cp7[5] = c75; Cp7[6] = c76;
                        break;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Register-tiled matrix multiplication reading panel-packed B with bias
    /// epilogue, two rows per group (E70).
    /// </summary>
    /// <param name="M">A rows (must be even).</param>
    /// <param name="N">A columns (reduction axis).</param>
    /// <param name="K">B columns.</param>
    /// <param name="A">Left matrix.</param>
    /// <param name="P">Panel-packed right matrix from PackPanelsB.</param>
    /// <param name="C">Result matrix.</param>
    /// <param name="Bias">Row bias of length K, added once per stored element.</param>
    /// <remarks>
    /// Same panels and per-element order as the 2-row packed kernel with one
    /// add per stored element in the unfused position, so results agree
    /// bit-wise with packed kernel followed by Add. Tails likewise.
    /// </remarks>
    public unsafe static void mm_unsafe_vectorized_intrinsics_2x4packed_bias(int M,
                          int N,
                          int K,
                          float* A,
                          float* P,
                          float* C,
                          float* Bias)
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
            var bb0 = ((Vector256<float>*)(Bias + kb))[0];
            var bb1 = ((Vector256<float>*)(Bias + kb + 8))[0];
            var bb2 = ((Vector256<float>*)(Bias + kb + 16))[0];
            var bb3 = ((Vector256<float>*)(Bias + kb + 24))[0];
            for (int i = 0; i < M; i += 2)
            {
                var Ap1 = A + i * N;
                var Ap2 = Ap1 + N;
                var Cpv1 = (Vector256<float>*)(C + i * K + kb);
                var Cpv2 = (Vector256<float>*)(C + (i + 1) * K + kb);
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
                Cpv1[0] = c00 + bb0;
                Cpv1[1] = c01 + bb1;
                Cpv1[2] = c02 + bb2;
                Cpv1[3] = c03 + bb3;
                Cpv2[0] = c10 + bb0;
                Cpv2[1] = c11 + bb1;
                Cpv2[2] = c12 + bb2;
                Cpv2[3] = c13 + bb3;
            }
        }
        int rem = K - blocked;
        if (rem > 0)
        {
            float* T = P + tiles * N * (4 * Vector256<float>.Count);
            int rv = rem / Vector256<float>.Count;
            for (int tt = 0; tt < rv; tt++)
            {
                var bbt = (Vector256<float>*)(Bias + blocked + tt * Vector256<float>.Count);
                for (int i = 0; i < M; i += 2)
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
                    rC1[tt] = c1 + bbt[0];
                    rC2[tt] = c2 + bbt[0];
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
                        float c1 = Cp1[0], c2 = Cp2[0];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c1 += Ap1[j] * t[0]; c2 += Ap2[j] * t[0]; }
                        Cp1[0] = c1 + Bias[blocked + vcols + 0]; Cp2[0] = c2 + Bias[blocked + vcols + 0];
                        break;
                    }
                    case 2:
                    {
                        float c10 = Cp1[0], c11 = Cp1[1], c20 = Cp2[0], c21 = Cp2[1];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; }
                        Cp1[0] = c10 + Bias[blocked + vcols + 0]; Cp1[1] = c11 + Bias[blocked + vcols + 1]; Cp2[0] = c20 + Bias[blocked + vcols + 0]; Cp2[1] = c21 + Bias[blocked + vcols + 1];
                        break;
                    }
                    case 3:
                    {
                        float c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; }
                        Cp1[0] = c10 + Bias[blocked + vcols + 0]; Cp1[1] = c11 + Bias[blocked + vcols + 1]; Cp1[2] = c12 + Bias[blocked + vcols + 2]; Cp2[0] = c20 + Bias[blocked + vcols + 0]; Cp2[1] = c21 + Bias[blocked + vcols + 1]; Cp2[2] = c22 + Bias[blocked + vcols + 2];
                        break;
                    }
                    case 4:
                    {
                        float c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; }
                        Cp1[0] = c10 + Bias[blocked + vcols + 0]; Cp1[1] = c11 + Bias[blocked + vcols + 1]; Cp1[2] = c12 + Bias[blocked + vcols + 2]; Cp1[3] = c13 + Bias[blocked + vcols + 3];
                        Cp2[0] = c20 + Bias[blocked + vcols + 0]; Cp2[1] = c21 + Bias[blocked + vcols + 1]; Cp2[2] = c22 + Bias[blocked + vcols + 2]; Cp2[3] = c23 + Bias[blocked + vcols + 3];
                        break;
                    }
                    case 5:
                    {
                        float c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3], c14 = Cp1[4];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3], c24 = Cp2[4];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; }
                        Cp1[0] = c10 + Bias[blocked + vcols + 0]; Cp1[1] = c11 + Bias[blocked + vcols + 1]; Cp1[2] = c12 + Bias[blocked + vcols + 2]; Cp1[3] = c13 + Bias[blocked + vcols + 3]; Cp1[4] = c14 + Bias[blocked + vcols + 4];
                        Cp2[0] = c20 + Bias[blocked + vcols + 0]; Cp2[1] = c21 + Bias[blocked + vcols + 1]; Cp2[2] = c22 + Bias[blocked + vcols + 2]; Cp2[3] = c23 + Bias[blocked + vcols + 3]; Cp2[4] = c24 + Bias[blocked + vcols + 4];
                        break;
                    }
                    case 6:
                    {
                        float c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3], c14 = Cp1[4], c15 = Cp1[5];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3], c24 = Cp2[4], c25 = Cp2[5];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; c15 += Ap1[j] * t[5]; c25 += Ap2[j] * t[5]; }
                        Cp1[0] = c10 + Bias[blocked + vcols + 0]; Cp1[1] = c11 + Bias[blocked + vcols + 1]; Cp1[2] = c12 + Bias[blocked + vcols + 2]; Cp1[3] = c13 + Bias[blocked + vcols + 3]; Cp1[4] = c14 + Bias[blocked + vcols + 4]; Cp1[5] = c15 + Bias[blocked + vcols + 5];
                        Cp2[0] = c20 + Bias[blocked + vcols + 0]; Cp2[1] = c21 + Bias[blocked + vcols + 1]; Cp2[2] = c22 + Bias[blocked + vcols + 2]; Cp2[3] = c23 + Bias[blocked + vcols + 3]; Cp2[4] = c24 + Bias[blocked + vcols + 4]; Cp2[5] = c25 + Bias[blocked + vcols + 5];
                        break;
                    }
                    default:
                    {
                        float c10 = Cp1[0], c11 = Cp1[1], c12 = Cp1[2], c13 = Cp1[3], c14 = Cp1[4], c15 = Cp1[5], c16 = Cp1[6];
                        float c20 = Cp2[0], c21 = Cp2[1], c22 = Cp2[2], c23 = Cp2[3], c24 = Cp2[4], c25 = Cp2[5], c26 = Cp2[6];
                        for (int j = 0; j < N; ++j) { var t = T + j * rem + vcols; c10 += Ap1[j] * t[0]; c20 += Ap2[j] * t[0]; c11 += Ap1[j] * t[1]; c21 += Ap2[j] * t[1]; c12 += Ap1[j] * t[2]; c22 += Ap2[j] * t[2]; c13 += Ap1[j] * t[3]; c23 += Ap2[j] * t[3]; c14 += Ap1[j] * t[4]; c24 += Ap2[j] * t[4]; c15 += Ap1[j] * t[5]; c25 += Ap2[j] * t[5]; c16 += Ap1[j] * t[6]; c26 += Ap2[j] * t[6]; }
                        Cp1[0] = c10 + Bias[blocked + vcols + 0]; Cp1[1] = c11 + Bias[blocked + vcols + 1]; Cp1[2] = c12 + Bias[blocked + vcols + 2]; Cp1[3] = c13 + Bias[blocked + vcols + 3]; Cp1[4] = c14 + Bias[blocked + vcols + 4]; Cp1[5] = c15 + Bias[blocked + vcols + 5]; Cp1[6] = c16 + Bias[blocked + vcols + 6];
                        Cp2[0] = c20 + Bias[blocked + vcols + 0]; Cp2[1] = c21 + Bias[blocked + vcols + 1]; Cp2[2] = c22 + Bias[blocked + vcols + 2]; Cp2[3] = c23 + Bias[blocked + vcols + 3]; Cp2[4] = c24 + Bias[blocked + vcols + 4]; Cp2[5] = c25 + Bias[blocked + vcols + 5]; Cp2[6] = c26 + Bias[blocked + vcols + 6];
                        break;
                    }
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
    /// <summary>Softmax-scoped vectorized base-e exponential (E66): same Cody-Waite
    /// reduction, clamps, reconstruction and guards as ExpVector, but the degree-7
    /// Taylor core in Estrin form (parallel sub-chains) instead of serial Horner.
    /// Same documented contract: order 1e-7 against MathF.Exp on finite inputs,
    /// NaN in NaN out, overflow to infinity, underflow to zero. Tanh keeps ExpVector.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> ExpVectorEstrin(Vector<float> v) =>
        AblationSwitches.EnableSoftmaxExpPrune
            ? (AblationSwitches.EnableSoftmaxExpInline ? ExpVectorEstrinPrunedInline(v) : ExpVectorEstrinPruned(v))
            : ExpVectorEstrinReference(v);

    internal static Vector<float> ExpVectorEstrinReference(Vector<float> v)
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
        var y = r * r;
        var tHi = Vector.FusedMultiplyAdd(new Vector<float>(1f / 5040f), r, new Vector<float>(1f / 720f));
        var tLo = Vector.FusedMultiplyAdd(new Vector<float>(1f / 6f), r, new Vector<float>(1f / 2f));
        var tMid = Vector.FusedMultiplyAdd(new Vector<float>(1f / 120f), r, new Vector<float>(1f / 24f));
        var tOne = Vector.FusedMultiplyAdd(new Vector<float>(1f), r, new Vector<float>(1f));
        var uHi = Vector.FusedMultiplyAdd(tHi, y, tMid);
        var uLo = Vector.FusedMultiplyAdd(tLo, y, tOne);
        var y2 = y * y;
        var p = Vector.FusedMultiplyAdd(uHi, y2, uLo);
        var scale = Vector.AsVectorSingle(Vector.ShiftLeft(clamped + new Vector<int>(127), 23));
        var yy = p * scale;
        yy = Vector.ConditionalSelect(Vector.GreaterThan(v, new Vector<float>(88.722839f)), new Vector<float>(float.PositiveInfinity), yy);
        yy = Vector.ConditionalSelect(Vector.LessThan(v, new Vector<float>(-88.722839f)), Vector<float>.Zero, yy);
        return Vector.ConditionalSelect(isFinite, yy, new Vector<float>(float.NaN));
    }
    // Lanes discarded by the existing cutoff use a benign polynomial input.
    // Finite lanes at/above the cutoff and the final IEEE guards are unchanged.
    internal static Vector<float> ExpVectorEstrinPruned(Vector<float> v)
    {
        var isFinite = Vector.Equals(v, v);
        var underflow = Vector.LessThan(v, new Vector<float>(-88.722839f));
        var x = Vector.Min(Vector.ConditionalSelect(underflow, Vector<float>.Zero, v), new Vector<float>(88.722839f));
        var scaled = x * new Vector<float>(1.44269504088896341f);
        var shifted = Vector.ConditionalSelect(Vector.GreaterThanOrEqual(scaled, Vector<float>.Zero), scaled + new Vector<float>(0.5f), scaled - new Vector<float>(0.5f));
        var n = Vector.ConvertToInt32(shifted);
        var clamped = Vector.Min(Vector.Max(n, new Vector<int>(-126)), new Vector<int>(127));
        var nf = Vector.ConvertToSingle(clamped);
        var r = Vector.FusedMultiplyAdd(nf, new Vector<float>(-0.693359375f), x);
        r = Vector.FusedMultiplyAdd(nf, new Vector<float>(2.12194440e-4f), r);
        var y = r * r;
        var tHi = Vector.FusedMultiplyAdd(new Vector<float>(1f / 5040f), r, new Vector<float>(1f / 720f));
        var tLo = Vector.FusedMultiplyAdd(new Vector<float>(1f / 6f), r, new Vector<float>(1f / 2f));
        var tMid = Vector.FusedMultiplyAdd(new Vector<float>(1f / 120f), r, new Vector<float>(1f / 24f));
        var tOne = Vector.FusedMultiplyAdd(new Vector<float>(1f), r, new Vector<float>(1f));
        var uHi = Vector.FusedMultiplyAdd(tHi, y, tMid);
        var uLo = Vector.FusedMultiplyAdd(tLo, y, tOne);
        var y2 = y * y;
        var p = Vector.FusedMultiplyAdd(uHi, y2, uLo);
        var scale = Vector.AsVectorSingle(Vector.ShiftLeft(clamped + new Vector<int>(127), 23));
        var yy = p * scale;
        yy = Vector.ConditionalSelect(Vector.GreaterThan(v, new Vector<float>(88.722839f)), new Vector<float>(float.PositiveInfinity), yy);
        yy = Vector.ConditionalSelect(underflow, Vector<float>.Zero, yy);
        return Vector.ConditionalSelect(isFinite, yy, new Vector<float>(float.NaN));
    }
    // Exact arithmetic twin. Keep bit agreement with the callable reference above;
    // the opt-in only lets callers retain live vectors across the polynomial.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<float> ExpVectorEstrinPrunedInline(Vector<float> v)
    {
        var isFinite = Vector.Equals(v, v);
        var underflow = Vector.LessThan(v, new Vector<float>(-88.722839f));
        var x = Vector.Min(Vector.ConditionalSelect(underflow, Vector<float>.Zero, v), new Vector<float>(88.722839f));
        var scaled = x * new Vector<float>(1.44269504088896341f);
        var shifted = Vector.ConditionalSelect(Vector.GreaterThanOrEqual(scaled, Vector<float>.Zero), scaled + new Vector<float>(0.5f), scaled - new Vector<float>(0.5f));
        var n = Vector.ConvertToInt32(shifted);
        var clamped = Vector.Min(Vector.Max(n, new Vector<int>(-126)), new Vector<int>(127));
        var nf = Vector.ConvertToSingle(clamped);
        var r = Vector.FusedMultiplyAdd(nf, new Vector<float>(-0.693359375f), x);
        r = Vector.FusedMultiplyAdd(nf, new Vector<float>(2.12194440e-4f), r);
        var y = r * r;
        var tHi = Vector.FusedMultiplyAdd(new Vector<float>(1f / 5040f), r, new Vector<float>(1f / 720f));
        var tLo = Vector.FusedMultiplyAdd(new Vector<float>(1f / 6f), r, new Vector<float>(1f / 2f));
        var tMid = Vector.FusedMultiplyAdd(new Vector<float>(1f / 120f), r, new Vector<float>(1f / 24f));
        var tOne = Vector.FusedMultiplyAdd(new Vector<float>(1f), r, new Vector<float>(1f));
        var uHi = Vector.FusedMultiplyAdd(tHi, y, tMid);
        var uLo = Vector.FusedMultiplyAdd(tLo, y, tOne);
        var y2 = y * y;
        var p = Vector.FusedMultiplyAdd(uHi, y2, uLo);
        var scale = Vector.AsVectorSingle(Vector.ShiftLeft(clamped + new Vector<int>(127), 23));
        var yy = p * scale;
        yy = Vector.ConditionalSelect(Vector.GreaterThan(v, new Vector<float>(88.722839f)), new Vector<float>(float.PositiveInfinity), yy);
        yy = Vector.ConditionalSelect(underflow, Vector<float>.Zero, yy);
        return Vector.ConditionalSelect(isFinite, yy, new Vector<float>(float.NaN));
    }
    // Only call after subtracting the maximum over the same row values. Finite
    // arguments are nonpositive; exceptional rows produce NaN or -Infinity.
    // The general exponential must retain its positive-input contract.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<float> ExpVectorSoftmax(Vector<float> v) =>
        AblationSwitches.EnableSoftmaxNonpositive
            ? ExpVectorNonpositive(v)
            : ExpVectorEstrin(v);

    /// <summary>Exact Estrin arithmetic for nonpositive inputs, including signed
    /// zero, NaN and negative infinity. Positive inputs are outside this contract.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<float> ExpVectorNonpositive(Vector<float> v)
    {
        var isNotNaN = Vector.Equals(v, v);
        var underflow = Vector.LessThan(v, new Vector<float>(-88.722839f));
        var x = Vector.ConditionalSelect(underflow, Vector<float>.Zero, v);
        var scaled = x * new Vector<float>(1.44269504088896341f);
        // Both +/-0.5 truncate to zero, including when the input is signed zero.
        var shifted = scaled - new Vector<float>(0.5f);
        var n = Vector.ConvertToInt32(shifted);
        var clamped = Vector.Max(n, new Vector<int>(-126));
        var nf = Vector.ConvertToSingle(clamped);
        var r = Vector.FusedMultiplyAdd(nf, new Vector<float>(-0.693359375f), x);
        r = Vector.FusedMultiplyAdd(nf, new Vector<float>(2.12194440e-4f), r);
        var y = r * r;
        var tHi = Vector.FusedMultiplyAdd(new Vector<float>(1f / 5040f), r, new Vector<float>(1f / 720f));
        var tLo = Vector.FusedMultiplyAdd(new Vector<float>(1f / 6f), r, new Vector<float>(1f / 2f));
        var tMid = Vector.FusedMultiplyAdd(new Vector<float>(1f / 120f), r, new Vector<float>(1f / 24f));
        var tOne = Vector.FusedMultiplyAdd(new Vector<float>(1f), r, new Vector<float>(1f));
        var uHi = Vector.FusedMultiplyAdd(tHi, y, tMid);
        var uLo = Vector.FusedMultiplyAdd(tLo, y, tOne);
        var y2 = y * y;
        var p = Vector.FusedMultiplyAdd(uHi, y2, uLo);
        var scale = Vector.AsVectorSingle(Vector.ShiftLeft(clamped + new Vector<int>(127), 23));
        var yy = p * scale;
        yy = Vector.ConditionalSelect(underflow, Vector<float>.Zero, yy);
        return Vector.ConditionalSelect(isNotNaN, yy, new Vector<float>(float.NaN));
    }
    /// <summary>Softmax-scoped 512-bit base-e exponential probe (E81): lane-for-lane port of
    /// ExpVectorEstrin to explicit Vector512 arithmetic with the same Cody-Waite reduction,
    /// clamps, reconstruction and guards. Same documented contract (order 1e-7 against
    /// MathF.Exp, NaN in NaN out, overflow to infinity, underflow to zero). Test-reachable
    /// only; requires 512-bit vectors plus FMA, checked by callers. No dispatch yet.
    /// </summary>
    public static Vector512<float> ExpVector512(Vector512<float> v)
    {
        var isFinite = Vector512.Equals(v, v);
        var x = Vector512.Min(Vector512.Max(v, Vector512.Create(-88.722839f)), Vector512.Create(88.722839f));
        var scaled = x * Vector512.Create(1.44269504088896341f);
        var shifted = Vector512.ConditionalSelect(Vector512.GreaterThanOrEqual(scaled, Vector512<float>.Zero), scaled + Vector512.Create(0.5f), scaled - Vector512.Create(0.5f));
        var n = Vector512.ConvertToInt32(shifted);
        var clamped = Vector512.Min(Vector512.Max(n, Vector512.Create(-126)), Vector512.Create(127));
        var nf = Vector512.ConvertToSingle(clamped);
        var r = Vector512.FusedMultiplyAdd(nf, Vector512.Create(-0.693359375f), x);
        r = Vector512.FusedMultiplyAdd(nf, Vector512.Create(2.12194440e-4f), r);
        var y = r * r;
        var tHi = Vector512.FusedMultiplyAdd(Vector512.Create(1f / 5040f), r, Vector512.Create(1f / 720f));
        var tLo = Vector512.FusedMultiplyAdd(Vector512.Create(1f / 6f), r, Vector512.Create(1f / 2f));
        var tMid = Vector512.FusedMultiplyAdd(Vector512.Create(1f / 120f), r, Vector512.Create(1f / 24f));
        var tOne = Vector512.FusedMultiplyAdd(Vector512.Create(1f), r, Vector512.Create(1f));
        var uHi = Vector512.FusedMultiplyAdd(tHi, y, tMid);
        var uLo = Vector512.FusedMultiplyAdd(tLo, y, tOne);
        var y2 = y * y;
        var p = Vector512.FusedMultiplyAdd(uHi, y2, uLo);
        var shiftedBits = Vector512.ShiftLeft(clamped + Vector512.Create(127), 23);
        var scale = Unsafe.As<Vector512<int>, Vector512<float>>(ref shiftedBits);
        var yy = p * scale;
        yy = Vector512.ConditionalSelect(Vector512.GreaterThan(v, Vector512.Create(88.722839f)), Vector512.Create(float.PositiveInfinity), yy);
        yy = Vector512.ConditionalSelect(Vector512.LessThan(v, Vector512.Create(-88.722839f)), Vector512<float>.Zero, yy);
        return Vector512.ConditionalSelect(isFinite, yy, Vector512.Create(float.NaN));
    }

    /// <summary>Span driver for the 512-bit exponential probe: 16-wide vector core with a
    /// scalar MathF.Exp tail (same tail rule as the span-softmax scalar tails).</summary>
    internal static unsafe void ExpSpan512(ReadOnlySpan<float> xs, Span<float> ys)
    {
        int w = Vector512<float>.Count;
        int nvec = xs.Length / w;
        fixed (float* px = xs, py = ys)
        {
            var xv = (Vector512<float>*)px;
            var yv = (Vector512<float>*)py;
            for (int i = 0; i < nvec; i++) yv[i] = ExpVector512(xv[i]);
            for (int i = nvec * w; i < xs.Length; i++) ys[i] = MathF.Exp(xs[i]);
        }
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

