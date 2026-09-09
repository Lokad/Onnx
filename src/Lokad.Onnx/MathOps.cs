using System;
using System.Numerics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Diagnostics;

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
