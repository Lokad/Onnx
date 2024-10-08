﻿using System;
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

    public struct Conv2DOutputInfo
    {
        public PadInfo PadInfo;
        public int[] Shape;
    }

    public static Conv2DOutputInfo GetConv2DOutputInfo(PadType pad, int inHeight, int inWidth, int strideHeight, int strideWidth, int filterHeight, int filterWidth, int? padValue = null)
    {
        var padInfo = new PadInfo();
        var outHeight = 0;
        var outWidth = 0;
        switch (pad)
        {
            case PadType.Valid:
                padInfo.bottom = 0;
                padInfo.left = 0;
                padInfo.right = 0;
                padInfo.top = 0;
                outHeight = (int)Math.Ceiling((inHeight - filterHeight + 1d) / strideHeight);
                outWidth = (int)Math.Ceiling((inWidth - filterWidth + 1d) / strideWidth);
                padInfo.h = 0;
                padInfo.w = 0;
                break;

            case PadType.SameUpper:
            case PadType.SameLower:
                outHeight = (int)Math.Ceiling(inHeight / (float)strideHeight);
                outWidth = (int)Math.Ceiling(inWidth / (float)strideWidth);

                var padAlongHeight = (outHeight - 1) * strideHeight + filterHeight - inHeight;
                var padAlongWidth = (outWidth - 1) * strideWidth + filterWidth - inWidth;
                var top = (int)Math.Floor(padAlongHeight / 2f);
                var bottom = (int)padAlongHeight - top;
                var left = (int)Math.Floor(padAlongWidth / 2f);
                var right = (int)padAlongWidth - left;

                padInfo.bottom = bottom;
                padInfo.left = left;
                padInfo.right = right;
                padInfo.top = top;

                padInfo.h = padAlongHeight;
                padInfo.w = padAlongWidth;
                break;

            case PadType.Value:
                if (padValue == null) throw new ArgumentNullException(nameof(padValue));
                padInfo.bottom = padValue.Value;
                padInfo.left = padValue.Value;
                padInfo.right = padValue.Value;
                padInfo.top = padValue.Value;
                padInfo.h = padInfo.top + padInfo.bottom;
                padInfo.w = padInfo.right + padInfo.left;
                var outShape = GetConv2DOutputShape(new int[] { inHeight, inWidth}, filterHeight, filterWidth, strideHeight, strideWidth, padInfo.h, padInfo.w);
                outHeight = outShape[0];
                outWidth = outShape[1];
                break;
        }
        return new Conv2DOutputInfo { PadInfo = padInfo, Shape = new int[] { outHeight, outWidth } };
    }

    public static int[] GetConv2DOutputShape(int[] inputShape, int kernelHeight, int kernelWidth, int strideY, int strideX, int padY, int padX)
    {
        var outputHeight = (int) Math.Floor((inputShape[0] - kernelHeight + padY * 1.0f) / strideY) + 1;
        var outputWidth = (int) Math.Floor((inputShape[1] - kernelWidth + padX * 1.0f) / strideX) + 1;
        return new int[] { outputHeight, outputWidth};
    }

    public static int GetConv2DDefaultPad(int[] inputShape, int fieldSize, int stride, int dilation = 1) => 
        (int) Math.Floor(((float)inputShape[0] * (stride - 1) - stride + GetConv2DEffectiveFilterSize(fieldSize, dilation)) / 2);
    
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
        for (int i = 0; i < M; i++)
        {
            var Ap = A + i * N;
            var Cp = C + i * K;
            for (int j = 0; j < N; ++j)
            {
                var a = Ap[j];
                var Bp = B + j * K;
                for (int k = 0; k < K; ++k)
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
    public unsafe static void mm(int M,
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
                for (int k = 0; k < K; ++k)
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
    public unsafe static void mm(int M,
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
                for (int k = 0; k < K; ++k)
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
    public static void mm_managed(int M,
                          int N,
                          int K,
                          Memory<float> mA,
                          Memory<float> mB,
                          Memory<float> mC)
    {
        var A = mA.Span;
        var B = mB.Span;
        var C = mC.Span;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; ++j)
            {
                var a = A[i * N + j];
                var bp = j * K;
                var cp = i * K;

                for (int k = 0; k < K; ++k)
                {
                    C[cp + k] += a * B[bp + k];
                }
            }
        }
    }

    public static void mm_managed(int M,
                  int N,
                  int K,
                  Memory<int> mA,
                  Memory<int> mB,
                  Memory<int> mC)
    {
        var A = mA.Span;
        var B = mB.Span;
        var C = mC.Span;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; ++j)
            {
                var a = A[i * N + j];
                var bp = j * K;
                var cp = i * K;
                
                for (int k = 0; k < K; ++k)
                {
                    C[cp + k] +=  a * B[bp + k];
                }
            }
        }
    }

    public static void mm_vectorized(int M,
                 int N,
                 int K,
                 Memory<int> mA,
                 Memory<int> mB,
                 Memory<int> mC)
    {
        var v = Vector<int>.Count;
        var A = mA.Span;
        var B = mB.Span;
        var C = mC.Span;

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; ++j)
            {
                var a = A[i * N + j];
                var bp = j * K;
                var cp = i * K;

                for (int k = 0; k <= K - v; k+=v)
                {
                    var Bk = B.Slice(bp + k, v);
                    var Ck = C.Slice(cp + k, v);
                    Vector<int> vec1 = new Vector<int>(a);
                    Vector<int> vec2 = new Vector<int>(Bk);
                    Vector<int> vec3 = Vector.Multiply(vec1, vec2);
                    Vector<int> vec4 = new Vector<int>(Ck);
                    Vector<int> vec5 = Vector.Add(vec3, vec4);
                    vec5.CopyTo(Ck);
                }
            }
        }
    }

    public static void mm_vectorized(int M,
                 int N,
                 int K,
                 Memory<float> mA,
                 Memory<float> mB,
                 Memory<float> mC)
    {
        var v = Vector<float>.Count;
        var A = mA.Span;
        var B = mB.Span;
        var C = mC.Span;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; ++j)
            {
                var a = A[i * N + j];
                var bp = j * K;
                var cp = i * K;

                for (int k = 0; k <= K - v; k += v)
                {
                    var Bk = B.Slice(bp + k, v);
                    var Ck = C.Slice(cp + k, v);
                    
                    Vector<float> vec1 = new Vector<float>(a);
                    Vector<float> vec2 = new Vector<float>(Bk);
                    Vector<float> vec3 = Vector.Multiply(vec1, vec2);
                    Vector<float> vec4 = new Vector<float>(Ck);
                    Vector<float> vec5 = Vector.Add(vec3, vec4);
                    vec5.CopyTo(Ck);
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

        if (K % (4 * Vector256<float>.Count) != 0)
            throw new ArgumentException(nameof(K));

#if DEBUG
        var Aend = A + M * N;
        var Bend = B + N * K;
        var Cend = C + M * K;
#endif

        for (int i = 0; i < M; i += 2)
        {
            var Ap1 = A + i * N;
            var Ap2 = A + N;

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
                var Bepv = (Vector256<float>*)(Bp + K);

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
    /// <param name="padY">Zero padding by left side.</param>
    /// <param name="padX">Zero padding by top side.</param>
    /// <param name="padH">Zero padding by right side.</param>
    /// <param name="padW">Zero padding by bottom side.</param>
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
            var scsrcH = sc * srcH;
            for (int ky = 0; ky < kernelY; ++ky)
            {
                int sy_ = ky * dilationY - padY;
                for (int kx = 0; kx < kernelX; ++kx)
                {
                    int sx_ = kx * dilationX - padX;
                    for (int dy = 0; dy < dstH; ++dy)
                    {
                        int sy = sy_ + dy * strideY;
                        if ((sy < 0) || (sy >= srcH))
                        {
                            for (int dx = 0; dx < dstW; ++dx)
                            {
                                *buf++ = 0;
                            }
                            continue;
                        }
                        var src1 = src + (scsrcH + sy) * srcW;
                        for (int dx = 0; dx < dstW; ++dx)
                        {
                            int sx = sx_ + dx * strideX;
                            if ((sx >= 0) && (sx < srcW))
                            {
                                *buf++ = src1[sx];
                            }
                            else
                            {
                                *buf++ = 0;
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
            var scsrcH = sc * srcH;
            for (int ky = 0; ky < kernelY; ++ky)
            {
                int sy_ = ky * dilationY - padY;
                for (int kx = 0; kx < kernelX; ++kx)
                {
                    int sx_ = kx * dilationX - padX;
                    for (int dy = 0; dy < dstH; ++dy)
                    {
                        int sy = sy_ + dy * strideY;
                        if ((sy < 0) || (sy >= srcH))
                        {
                            for (int dx = 0; dx < dstW; ++dx)
                            {
                                *buf++ = 0;
                            }
                            continue;
                        }
                        var src1 = src + (scsrcH + sy) * srcW;
                        for (int dx = 0; dx < dstW; ++dx)
                        {
                            int sx = sx_ + dx * strideX;
                            if ((sx >= 0) && (sx < srcW))
                            {
                                *buf++ = src1[sx];
                            }
                            else
                            {
                                *buf++ = 0;
                            }
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
    /// <param name="padY">Zero padding by left side.</param>
    /// <param name="padX">Zero padding by top side.</param>
    /// <param name="padH">Zero padding by right side.</param>
    /// <param name="padW">Zero padding by bottom side.</param>
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
                                    float* bias = null)
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
        Marshal.FreeCoTaskMem((IntPtr)buf);
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
    /// <param name="padY">Zero padding by left side.</param>
    /// <param name="padX">Zero padding by top side.</param>
    /// <param name="padH">Zero padding by right side.</param>
    /// <param name="padW">Zero padding by bottom side.</param>
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
        for (int b = 0; b < batch; ++b)
        {
            Im2col(src, srcC, srcH, srcW, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, buf);
            for (int g = 0; g < group; ++g)
            {
                _mm(M, N, K, weight + M * K * g, buf + N * K * g, dst + M * N * g);
                //cs_BLAS.DGEMM("nota", "notb", M, N, K, 0.0f, )
            }
            for (int i = 0; i < dstC; ++i)
            {
                var pdst = dst + i * N;
                for (int j = 0; j < N; ++j)
                {
                    pdst[j] += bias[i];
                }
            }
            src += srcC * srcH * srcW;
            dst += dstC * dstH * dstW;
        }
        Marshal.FreeCoTaskMem((IntPtr)buf);
    }

    // From: https://www.johndcook.com/blog/2009/01/19/stand-alone-error-function-erf/
    public static float Erf(float x)
    {
        // constants
        float a1 = 0.254829592f;
        float a2 = -0.284496736f;
        float a3 = 1.421413741f;
        float a4 = -1.453152027f;
        float a5 = 1.061405429f;
        float p = 0.3275911f;

        // Save the sign of x
        int sign = 1;
        if (x < 0)
            sign = -1;
        x = Math.Abs(x);

        // A&S formula 7.1.26
        float t = 1.0f / (1.0f + p * x);
        float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);

        return sign * y;
    }

    // From: https://www.johndcook.com/blog/2009/01/19/stand-alone-error-function-erf/
    public static double Erf(double x)
    {
        // constants
        double a1 = 0.254829592;
        double a2 = -0.284496736;
        double a3 = 1.421413741;
        double a4 = -1.453152027;
        double a5 = 1.061405429;
        double p = 0.3275911;

        // Save the sign of x
        int sign = 1;
        if (x < 0)
            sign = -1;
        x = Math.Abs(x);

        // A&S formula 7.1.26
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

        return sign * y;
    }

    /// <summary>
    /// Returns the value of the gaussian error function at <paramref name="x"/>.
    /// </summary>
    public static double Erf2(double x)
    {
        /*
        Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
        *
        * Developed at SunPro, a Sun Microsystems, Inc. business.
        * Permission to use, copy, modify, and distribute this
        * software is freely granted, provided that this notice
        * is preserved.
        */

    #region Constants

    const double tiny = 1e-300;
        const double erx = 8.45062911510467529297e-01;

        // Coefficients for approximation to erf on [0, 0.84375]
        const double efx = 1.28379167095512586316e-01; /* 0x3FC06EBA; 0x8214DB69 */
        const double efx8 = 1.02703333676410069053e+00; /* 0x3FF06EBA; 0x8214DB69 */
        const double pp0 = 1.28379167095512558561e-01; /* 0x3FC06EBA; 0x8214DB68 */
        const double pp1 = -3.25042107247001499370e-01; /* 0xBFD4CD7D; 0x691CB913 */
        const double pp2 = -2.84817495755985104766e-02; /* 0xBF9D2A51; 0xDBD7194F */
        const double pp3 = -5.77027029648944159157e-03; /* 0xBF77A291; 0x236668E4 */
        const double pp4 = -2.37630166566501626084e-05; /* 0xBEF8EAD6; 0x120016AC */
        const double qq1 = 3.97917223959155352819e-01; /* 0x3FD97779; 0xCDDADC09 */
        const double qq2 = 6.50222499887672944485e-02; /* 0x3FB0A54C; 0x5536CEBA */
        const double qq3 = 5.08130628187576562776e-03; /* 0x3F74D022; 0xC4D36B0F */
        const double qq4 = 1.32494738004321644526e-04; /* 0x3F215DC9; 0x221C1A10 */
        const double qq5 = -3.96022827877536812320e-06; /* 0xBED09C43; 0x42A26120 */

        // Coefficients for approximation to erf in [0.84375, 1.25]
        const double pa0 = -2.36211856075265944077e-03; /* 0xBF6359B8; 0xBEF77538 */
        const double pa1 = 4.14856118683748331666e-01; /* 0x3FDA8D00; 0xAD92B34D */
        const double pa2 = -3.72207876035701323847e-01; /* 0xBFD7D240; 0xFBB8C3F1 */
        const double pa3 = 3.18346619901161753674e-01; /* 0x3FD45FCA; 0x805120E4 */
        const double pa4 = -1.10894694282396677476e-01; /* 0xBFBC6398; 0x3D3E28EC */
        const double pa5 = 3.54783043256182359371e-02; /* 0x3FA22A36; 0x599795EB */
        const double pa6 = -2.16637559486879084300e-03; /* 0xBF61BF38; 0x0A96073F */
        const double qa1 = 1.06420880400844228286e-01; /* 0x3FBB3E66; 0x18EEE323 */
        const double qa2 = 5.40397917702171048937e-01; /* 0x3FE14AF0; 0x92EB6F33 */
        const double qa3 = 7.18286544141962662868e-02; /* 0x3FB2635C; 0xD99FE9A7 */
        const double qa4 = 1.26171219808761642112e-01; /* 0x3FC02660; 0xE763351F */
        const double qa5 = 1.36370839120290507362e-02; /* 0x3F8BEDC2; 0x6B51DD1C */
        const double qa6 = 1.19844998467991074170e-02; /* 0x3F888B54; 0x5735151D */

        // Coefficients for approximation to erfc in [1.25, 1/0.35]
        const double ra0 = -9.86494403484714822705e-03; /* 0xBF843412; 0x600D6435 */
        const double ra1 = -6.93858572707181764372e-01; /* 0xBFE63416; 0xE4BA7360 */
        const double ra2 = -1.05586262253232909814e+01; /* 0xC0251E04; 0x41B0E726 */
        const double ra3 = -6.23753324503260060396e+01; /* 0xC04F300A; 0xE4CBA38D */
        const double ra4 = -1.62396669462573470355e+02; /* 0xC0644CB1; 0x84282266 */
        const double ra5 = -1.84605092906711035994e+02; /* 0xC067135C; 0xEBCCABB2 */
        const double ra6 = -8.12874355063065934246e+01; /* 0xC0545265; 0x57E4D2F2 */
        const double ra7 = -9.81432934416914548592e+00; /* 0xC023A0EF; 0xC69AC25C */
        const double sa1 = 1.96512716674392571292e+01; /* 0x4033A6B9; 0xBD707687 */
        const double sa2 = 1.37657754143519042600e+02; /* 0x4061350C; 0x526AE721 */
        const double sa3 = 4.34565877475229228821e+02; /* 0x407B290D; 0xD58A1A71 */
        const double sa4 = 6.45387271733267880336e+02; /* 0x40842B19; 0x21EC2868 */
        const double sa5 = 4.29008140027567833386e+02; /* 0x407AD021; 0x57700314 */
        const double sa6 = 1.08635005541779435134e+02; /* 0x405B28A3; 0xEE48AE2C */
        const double sa7 = 6.57024977031928170135e+00; /* 0x401A47EF; 0x8E484A93 */
        const double sa8 = -6.04244152148580987438e-02; /* 0xBFAEEFF2; 0xEE749A62 */

        // Coefficients for approximation to erfc in [1/0.35, 28]
        const double rb0 = -9.86494292470009928597e-03; /* 0xBF843412; 0x39E86F4A */
        const double rb1 = -7.99283237680523006574e-01; /* 0xBFE993BA; 0x70C285DE */
        const double rb2 = -1.77579549177547519889e+01; /* 0xC031C209; 0x555F995A */
        const double rb3 = -1.60636384855821916062e+02; /* 0xC064145D; 0x43C5ED98 */
        const double rb4 = -6.37566443368389627722e+02; /* 0xC083EC88; 0x1375F228 */
        const double rb5 = -1.02509513161107724954e+03; /* 0xC0900461; 0x6A2E5992 */
        const double rb6 = -4.83519191608651397019e+02; /* 0xC07E384E; 0x9BDC383F */
        const double sb1 = 3.03380607434824582924e+01; /* 0x403E568B; 0x261D5190 */
        const double sb2 = 3.25792512996573918826e+02; /* 0x40745CAE; 0x221B9F0A */
        const double sb3 = 1.53672958608443695994e+03; /* 0x409802EB; 0x189D5118 */
        const double sb4 = 3.19985821950859553908e+03; /* 0x40A8FFB7; 0x688C246A */
        const double sb5 = 2.55305040643316442583e+03; /* 0x40A3F219; 0xCEDF3BE6 */
        const double sb6 = 4.74528541206955367215e+02; /* 0x407DA874; 0xE79FE763 */
        const double sb7 = -2.24409524465858183362e+01; /* 0xC03670E2; 0x42712D62 */

        #endregion

        if (double.IsNaN(x))
            return double.NaN;

        if (double.IsNegativeInfinity(x))
            return -1.0;

        if (double.IsPositiveInfinity(x))
            return 1.0;

        int n0, hx, ix;
        double R, S, P, Q, s, y, z, r;
        unsafe
        {
            double one = 1.0;
            n0 = ((*(int*)&one) >> 29) ^ 1;
            hx = *(n0 + (int*)&x);
        }
        ix = hx & 0x7FFFFFFF;

        if (ix < 0x3FEB0000) // |x| < 0.84375
        {
            if (ix < 0x3E300000) // |x| < 2**-28
            {
                if (ix < 0x00800000)
                    return 0.125 * (8.0 * x + efx8 * x); // avoid underflow
                return x + efx * x;
            }
            z = x * x;
            r = pp0 + z * (pp1 + z * (pp2 + z * (pp3 + z * pp4)));
            s = 1.0 + z * (qq1 + z * (qq2 + z * (qq3 + z * (qq4 + z * qq5))));
            y = r / s;
            return x + x * y;
        }
        if (ix < 0x3FF40000) // 0.84375 <= |x| < 1.25
        {
            s = Math.Abs(x) - 1.0;
            P = pa0 + s * (pa1 + s * (pa2 + s * (pa3 + s * (pa4 + s * (pa5 + s * pa6)))));
            Q = 1.0 + s * (qa1 + s * (qa2 + s * (qa3 + s * (qa4 + s * (qa5 + s * qa6)))));
            if (hx >= 0)
                return erx + P / Q;
            else
                return -erx - P / Q;
        }
        if (ix >= 0x40180000) // inf > |x| >= 6
        {
            if (hx >= 0)
                return 1.0 - tiny;
            else
                return tiny - 1.0;
        }
        x = Math.Abs(x);
        s = 1.0 / (x * x);
        if (ix < 0x4006DB6E) // |x| < 1/0.35
        {
            R = ra0 + s * (ra1 + s * (ra2 + s * (ra3 + s * (ra4 + s * (ra5 + s * (ra6 + s * ra7))))));
            S = 1.0 + s * (sa1 + s * (sa2 + s * (sa3 + s * (sa4 + s * (sa5 + s * (sa6 + s * (sa7 + s * sa8)))))));
        }
        else // |x| >= 1/0.35
        {
            R = rb0 + s * (rb1 + s * (rb2 + s * (rb3 + s * (rb4 + s * (rb5 + s * rb6)))));
            S = 1.0 + s * (sb1 + s * (sb2 + s * (sb3 + s * (sb4 + s * (sb5 + s * (sb6 + s * sb7))))));
        }
        z = x;
        unsafe { *(1 - n0 + (int*)&z) = 0; }
        r = Math.Exp(-z * z - 0.5625) * Math.Exp((z - x) * (z + x) + R / S);
        if (hx >= 0)
            return 1.0 - r / x;
        else
            return r / x - 1.0;
    }

    public static float Erf2(float x)
    {
        /*
        Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
        *
        * Developed at SunPro, a Sun Microsystems, Inc. business.
        * Permission to use, copy, modify, and distribute this
        * software is freely granted, provided that this notice
        * is preserved.
        */

        #region Constants

        const float tiny = 1e-300f;
        const float erx = 8.45062911510467529297e-01f;

        // Coefficients for approximation to erf on [0, 0.84375]
        const float efx = 1.28379167095512586316e-01f; /* 0x3FC06EBA; 0x8214DB69 */
        const float efx8 = 1.02703333676410069053e+00f; /* 0x3FF06EBA; 0x8214DB69 */
        const float pp0 = 1.28379167095512558561e-01f; /* 0x3FC06EBA; 0x8214DB68 */
        const float pp1 = -3.25042107247001499370e-01f; /* 0xBFD4CD7D; 0x691CB913 */
        const float pp2 = -2.84817495755985104766e-02f; /* 0xBF9D2A51; 0xDBD7194F */
        const float pp3 = -5.77027029648944159157e-03f; /* 0xBF77A291; 0x236668E4 */
        const float pp4 = -2.37630166566501626084e-05f; /* 0xBEF8EAD6; 0x120016AC */
        const float qq1 = 3.97917223959155352819e-01f; /* 0x3FD97779; 0xCDDADC09 */
        const float qq2 = 6.50222499887672944485e-02f; /* 0x3FB0A54C; 0x5536CEBA */
        const float qq3 = 5.08130628187576562776e-03f; /* 0x3F74D022; 0xC4D36B0F */
        const float qq4 = 1.32494738004321644526e-04f; /* 0x3F215DC9; 0x221C1A10 */
        const float qq5 = -3.96022827877536812320e-06f; /* 0xBED09C43; 0x42A26120 */

        // Coefficients for approximation to erf in [0.84375, 1.25]
        const float pa0 = -2.36211856075265944077e-03f; /* 0xBF6359B8; 0xBEF77538 */
        const float pa1 = 4.14856118683748331666e-01f; /* 0x3FDA8D00; 0xAD92B34D */
        const float pa2 = -3.72207876035701323847e-01f; /* 0xBFD7D240; 0xFBB8C3F1 */
        const float pa3 = 3.18346619901161753674e-01f; /* 0x3FD45FCA; 0x805120E4 */
        const float pa4 = -1.10894694282396677476e-01f; /* 0xBFBC6398; 0x3D3E28EC */
        const float pa5 = 3.54783043256182359371e-02f; /* 0x3FA22A36; 0x599795EB */
        const float pa6 = -2.16637559486879084300e-03f; /* 0xBF61BF38; 0x0A96073F */
        const float qa1 = 1.06420880400844228286e-01f; /* 0x3FBB3E66; 0x18EEE323 */
        const float qa2 = 5.40397917702171048937e-01f; /* 0x3FE14AF0; 0x92EB6F33 */
        const float qa3 = 7.18286544141962662868e-02f; /* 0x3FB2635C; 0xD99FE9A7 */
        const float qa4 = 1.26171219808761642112e-01f; /* 0x3FC02660; 0xE763351F */
        const float qa5 = 1.36370839120290507362e-02f; /* 0x3F8BEDC2; 0x6B51DD1C */
        const float qa6 = 1.19844998467991074170e-02f; /* 0x3F888B54; 0x5735151D */

        // Coefficients for approximation to erfc in [1.25, 1/0.35]
        const float ra0 = -9.86494403484714822705e-03f; /* 0xBF843412; 0x600D6435 */
        const float ra1 = -6.93858572707181764372e-01f; /* 0xBFE63416; 0xE4BA7360 */
        const float ra2 = -1.05586262253232909814e+01f; /* 0xC0251E04; 0x41B0E726 */
        const float ra3 = -6.23753324503260060396e+01f; /* 0xC04F300A; 0xE4CBA38D */
        const float ra4 = -1.62396669462573470355e+02f; /* 0xC0644CB1; 0x84282266 */
        const float ra5 = -1.84605092906711035994e+02f; /* 0xC067135C; 0xEBCCABB2 */
        const float ra6 = -8.12874355063065934246e+01f; /* 0xC0545265; 0x57E4D2F2 */
        const float ra7 = -9.81432934416914548592e+00f; /* 0xC023A0EF; 0xC69AC25C */
        const float sa1 = 1.96512716674392571292e+01f; /* 0x4033A6B9; 0xBD707687 */
        const float sa2 = 1.37657754143519042600e+02f; /* 0x4061350C; 0x526AE721 */
        const float sa3 = 4.34565877475229228821e+02f; /* 0x407B290D; 0xD58A1A71 */
        const float sa4 = 6.45387271733267880336e+02f; /* 0x40842B19; 0x21EC2868 */
        const float sa5 = 4.29008140027567833386e+02f; /* 0x407AD021; 0x57700314 */
        const float sa6 = 1.08635005541779435134e+02f; /* 0x405B28A3; 0xEE48AE2C */
        const float sa7 = 6.57024977031928170135e+00f; /* 0x401A47EF; 0x8E484A93 */
        const float sa8 = -6.04244152148580987438e-02f; /* 0xBFAEEFF2; 0xEE749A62 */

        // Coefficients for approximation to erfc in [1/0.35, 28]
        const float rb0 = -9.86494292470009928597e-03f; /* 0xBF843412; 0x39E86F4A */
        const float rb1 = -7.99283237680523006574e-01f; /* 0xBFE993BA; 0x70C285DE */
        const float rb2 = -1.77579549177547519889e+01f; /* 0xC031C209; 0x555F995A */
        const float rb3 = -1.60636384855821916062e+02f; /* 0xC064145D; 0x43C5ED98 */
        const float rb4 = -6.37566443368389627722e+02f; /* 0xC083EC88; 0x1375F228 */
        const float rb5 = -1.02509513161107724954e+03f; /* 0xC0900461; 0x6A2E5992 */
        const float rb6 = -4.83519191608651397019e+02f; /* 0xC07E384E; 0x9BDC383F */
        const float sb1 = 3.03380607434824582924e+01f; /* 0x403E568B; 0x261D5190 */
        const float sb2 = 3.25792512996573918826e+02f; /* 0x40745CAE; 0x221B9F0A */
        const float sb3 = 1.53672958608443695994e+03f; /* 0x409802EB; 0x189D5118 */
        const float sb4 = 3.19985821950859553908e+03f; /* 0x40A8FFB7; 0x688C246A */
        const float sb5 = 2.55305040643316442583e+03f; /* 0x40A3F219; 0xCEDF3BE6 */
        const float sb6 = 4.74528541206955367215e+02f; /* 0x407DA874; 0xE79FE763 */
        const float sb7 = -2.24409524465858183362e+01f; /* 0xC03670E2; 0x42712D62 */

        #endregion

        if (float.IsNaN(x))
            return float.NaN;

        if (float.IsNegativeInfinity(x))
            return -1.0f;

        if (float.IsPositiveInfinity(x))
            return 1.0f;

        int n0, hx, ix;
        float R, S, P, Q, s, y, z, r;
        unsafe
        {
            float one = 1.0f;
            n0 = ((*(int*)&one) >> 29) ^ 1;
            hx = *(n0 + (int*)&x);
        }
        ix = hx & 0x7FFFFFFF;

        if (ix < 0x3FEB0000) // |x| < 0.84375
        {
            if (ix < 0x3E300000) // |x| < 2**-28
            {
                if (ix < 0x00800000)
                    return 0.125f * (8.0f * x + efx8 * x); // avoid underflow
                return x + efx * x;
            }
            z = x * x;
            r = pp0 + z * (pp1 + z * (pp2 + z * (pp3 + z * pp4)));
            s = 1.0f + z * (qq1 + z * (qq2 + z * (qq3 + z * (qq4 + z * qq5))));
            y = r / s;
            return x + x * y;
        }
        if (ix < 0x3FF40000) // 0.84375 <= |x| < 1.25
        {
            s = MathF.Abs(x) - 1.0f;
            P = pa0 + s * (pa1 + s * (pa2 + s * (pa3 + s * (pa4 + s * (pa5 + s * pa6)))));
            Q = 1.0f + s * (qa1 + s * (qa2 + s * (qa3 + s * (qa4 + s * (qa5 + s * qa6)))));
            if (hx >= 0)
                return erx + P / Q;
            else
                return -erx - P / Q;
        }
        if (ix >= 0x40180000) // inf > |x| >= 6
        {
            if (hx >= 0)
                return 1.0f - tiny;
            else
                return tiny - 1.0f;
        }
        x = MathF.Abs(x);
        s = 1.0f / (x * x);
        if (ix < 0x4006DB6E) // |x| < 1/0.35
        {
            R = ra0 + s * (ra1 + s * (ra2 + s * (ra3 + s * (ra4 + s * (ra5 + s * (ra6 + s * ra7))))));
            S = 1.0f + s * (sa1 + s * (sa2 + s * (sa3 + s * (sa4 + s * (sa5 + s * (sa6 + s * (sa7 + s * sa8)))))));
        }
        else // |x| >= 1/0.35
        {
            R = rb0 + s * (rb1 + s * (rb2 + s * (rb3 + s * (rb4 + s * (rb5 + s * rb6)))));
            S = 1.0f + s * (sb1 + s * (sb2 + s * (sb3 + s * (sb4 + s * (sb5 + s * (sb6 + s * sb7))))));
        }
        z = x;
        unsafe { *(1 - n0 + (int*)&z) = 0; }
        r = MathF.Exp(-z * z - 0.5625f) * MathF.Exp((z - x) * (z + x) + R / S);
        if (hx >= 0)
            return 1.0f - r / x;
        else
            return r / x - 1.0f;
    }
}
