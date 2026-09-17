using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using static Lokad.Onnx.MathOps;
using static Lokad.Onnx.Profiler;

namespace Lokad.Onnx;

public abstract partial class Tensor<T> : TensorBase, IList, IList<T>, IReadOnlyList<T>, IStructuralComparable, IStructuralEquatable, ITensor, INumericTensor
where T : unmanaged
{
    /// <summary>
    /// Direct rank-four float single-channel convolution (B1 lane). A C=1,
    /// group=1 layer has no channel reuse, so the generic patch matrix is
    /// pure setup overhead: the whole im2col build plus the packed-GEMM
    /// call serve a K=kH*kW dot per output. The direct lane streams the one
    /// input channel straight into per-output dots with the same
    /// stride-one AVX256 FMA interior / scalar border structure as the
    /// depthwise lane. Admission is deliberately narrow (M at most 64): the
    /// pricing case is the embedding stem (32x1x3x3); wider single-channel
    /// layers keep the generic path until a mirror shows otherwise.
    /// </summary>
    public static bool TryConvSingleChannel2D(Tensor<float> input, Tensor<float> weight, Tensor<float>? bias, int group, int[]? pads, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options, bool fuseRelu, out Tensor<float>? output)
    {
        output = null;
        if (input.Rank != 4 || weight.Rank != 4) return false;
        int N = input.Dimensions[0], C = input.Dimensions[1], H = input.Dimensions[2], W = input.Dimensions[3];
        int M = weight.Dimensions[0];
        // Only the no-reuse geometry: one input channel, one group, a
        // modest output count. Everything else keeps the patch path.
        if (group != 1 || N < 1 || C != 1 || M < 1 || M > 64 || weight.Dimensions[1] != 1) return false;
        int KH = weight.Dimensions[2], KW = weight.Dimensions[3];
        if (KH < 1 || KW < 1) return false;
        if (kernelshape is not null && (kernelshape.Length != 2 || kernelshape[0] != KH || kernelshape[1] != KW)) return false;
        int sH = 1, sW = 1;
        if (strides is not null)
        {
            if (strides.Length != 2 || strides[0] < 1 || strides[1] < 1) return false;
            sH = strides[0];
            sW = strides[1];
        }
        int dH = 1, dW = 1;
        if (dilations is not null)
        {
            if (dilations.Length != 2 || dilations[0] < 1 || dilations[1] < 1) return false;
            dH = dilations[0];
            dW = dilations[1];
        }
        int padT = 0, padL = 0, padB = 0, padR = 0;
        if (pads is not null)
        {
            if (pads.Length != 4 || pads[0] < 0 || pads[1] < 0 || pads[2] < 0 || pads[3] < 0) return false;
            padT = pads[0];
            padL = pads[1];
            padB = pads[2];
            padR = pads[3];
        }
        if (bias is not null && bias.Length != M) return false;
        int effKH = GetConv2DEffectiveFilterSize(KH, dH);
        int effKW = GetConv2DEffectiveFilterSize(KW, dW);
        int[] outShape = GetConv2DOutputShape(new int[] { H, W }, effKH, effKW, sH, sW, padT + padB, padL + padR);
        int outH = outShape[0], outW = outShape[1];
        if (outH < 1 || outW < 1) return false;
        options.Validate();
        output = RunSingleChannel2DFloat(input.ToDenseTensor(), weight.ToDenseTensor(), bias?.ToDenseTensor(), N, H, W, M, KH, KW, sH, sW, dH, dW, padT, padL, outH, outW, options, fuseRelu);
        return true;
    }

    /// <summary>
    /// Direct rank-four float single-channel convolution. Output positions
    /// run over a contiguous width axis, so a stride-one AVX256 FMA vector
    /// loop covers the explicit no-check interior of interior rows at any
    /// dilation (each tap loads eight contiguous inputs), with scalar
    /// border handling; other width strides stay scalar. Batches and
    /// outputs are independent, so parallel degrees split over
    /// batch-outputs with identical per-element results.
    /// </summary>
    static Tensor<float> RunSingleChannel2DFloat(DenseTensor<float> x, DenseTensor<float> w, DenseTensor<float>? b, int N, int H, int W, int M, int KH, int KW, int sH, int sW, int dH, int dW, int padT, int padL, int outH, int outW, TensorExecutionOptions options, bool fuseRelu)
    {
        var output = new DenseTensor<float>((ReadOnlySpan<int>)new int[] { N, M, outH, outW });
        var xMem = x.Buffer;
        var wMem = w.Buffer;
        var oMem = output.Buffer;
        var bMem = b is null ? default : b.Buffer;
        bool hasBias = b is not null;
        // Same vector contract as the depthwise lane: each tap eight
        // outputs load eight contiguous inputs at sW * ox + kx * dW - padL,
        // which is only contiguous for sW == 1 (other strides would need
        // gathers). Dilation stays vectorized and row validity already
        // accounts for sH/dH, with scalar borders.
        bool vector = options.UseSimd && options.UseIntrinsics && sW == 1 && Avx.IsSupported && Fma.IsSupported;
        int jobs = N * M;
        int dop = options.MaxDegreeOfParallelism < 2 || jobs < 2 ? 1 : Math.Min(options.MaxDegreeOfParallelism, jobs);
        if (dop > 1)
        {
            Parallel.For(0, jobs, new ParallelOptions { MaxDegreeOfParallelism = dop }, job =>
            {
                RunSingleChannelOutputFloat(xMem, wMem, bMem, hasBias, oMem, job / M, job % M, H, W, M, KH, KW, sH, sW, dH, dW, padT, padL, outH, outW, vector, fuseRelu);
            });
        }
        else
        {
            for (int job = 0; job < jobs; job++)
                RunSingleChannelOutputFloat(xMem, wMem, bMem, hasBias, oMem, job / M, job % M, H, W, M, KH, KW, sH, sW, dH, dW, padT, padL, outH, outW, vector, fuseRelu);
        }
        return output;
    }
    static void RunSingleChannelOutputFloat(Memory<float> xMem, Memory<float> wMem, Memory<float> bMem, bool hasBias, Memory<float> oMem, int b, int m, int H, int W, int M, int KH, int KW, int sH, int sW, int dH, int dW, int padT, int padL, int outH, int outW, bool vector, bool fuseRelu)
    {
        var xs = xMem.Span;
        var ws = wMem.Span;
        var os = oMem.Span;
        int xBase = b * H * W;
        int wBase = m * KH * KW;
        int oBase = (b * M + m) * outH * outW;
        float bi = hasBias ? bMem.Span[m] : 0f;
        int effKH = (KH - 1) * dH + 1;
        int yLo = (padT + sH - 1) / sH;
        int yHiNum = H - 1 + padT - (effKH - 1);
        int yHi = (yHiNum >= 0 ? yHiNum / sH : -1) + 1;
        if (yLo < 0) yLo = 0;
        if (yHi > outH) yHi = outH;
        if (vector)
        {
            int effKW = (KW - 1) * dW + 1;
            int xLo = (padL + sW - 1) / sW;
            int xHiNum = W - 1 + padL - (effKW - 1);
            int xHi = (xHiNum >= 0 ? xHiNum / sW : -1) + 1;
            if (xLo < 0) xLo = 0;
            if (xHi > outW) xHi = outW;
            var zero = Vector256<float>.Zero;
            int vecEnd = xLo + ((xHi - xLo) & ~7);
            unsafe
            {
                fixed (float* xp = xs, wp = ws, op = os)
                {
                    for (int oy = 0; oy < outH; oy++)
                    {
                        float* oc = op + oBase + oy * outW;
                        if (oy < yLo || oy >= yHi)
                        {
                            for (int ox = 0; ox < outW; ox++)
                                oc[ox] = SingleChannelTapScalar(xs, ws, xBase, wBase, bi, oy, ox, KH, KW, sH, sW, dH, dW, padT, padL, H, W, fuseRelu);
                            continue;
                        }
                        float* xc = xp + xBase + (oy * sH - padT) * W;
                        for (int ox = 0; ox < xLo; ox++)
                            oc[ox] = SingleChannelTapScalar(xs, ws, xBase, wBase, bi, oy, ox, KH, KW, sH, sW, dH, dW, padT, padL, H, W, fuseRelu);
                        for (int ox = xLo; ox < vecEnd; ox += 8)
                        {
                            var acc = Vector256.Create(bi);
                            for (int ky = 0; ky < KH; ky++)
                            {
                                float* row = xc + (ky * dH) * W;
                                for (int kx = 0; kx < KW; kx++)
                                {
                                    var xv = *(Vector256<float>*)(row + sW * ox + kx * dW - padL);
                                    acc = Fma.MultiplyAdd(Vector256.Create(wp[wBase + ky * KW + kx]), xv, acc);
                                }
                            }
                            if (fuseRelu)
                            {
                                var keep = Vector256.GreaterThan(acc, zero) | Vector256.Equals(acc, zero) | ~Vector256.Equals(acc, acc);
                                acc = Vector256.ConditionalSelect(keep, acc, zero);
                            }
                            *(Vector256<float>*)(oc + ox) = acc;
                        }
                        for (int ox = vecEnd; ox < outW; ox++)
                        {
                            if (ox >= xHi) oc[ox] = SingleChannelTapScalar(xs, ws, xBase, wBase, bi, oy, ox, KH, KW, sH, sW, dH, dW, padT, padL, H, W, fuseRelu);
                            else
                            {
                                float acc2 = bi;
                                for (int ky = 0; ky < KH; ky++)
                                {
                                    float* row2 = xc + (ky * dH) * W;
                                    for (int kx = 0; kx < KW; kx++)
                                        acc2 += wp[wBase + ky * KW + kx] * row2[sW * ox + kx * dW - padL];
                                }
                                oc[ox] = fuseRelu && acc2 < 0f ? 0f : acc2;
                            }
                        }
                    }
                }
            }
            return;
        }
        for (int oy = 0; oy < outH; oy++)
            for (int ox = 0; ox < outW; ox++)
                os[oBase + oy * outW + ox] = SingleChannelTapScalar(xs, ws, xBase, wBase, bi, oy, ox, KH, KW, sH, sW, dH, dW, padT, padL, H, W, fuseRelu);
    }

    /// <summary>Scalar single-channel 2D tap accumulation with explicit zero padding.</summary>
    static float SingleChannelTapScalar(ReadOnlySpan<float> xs, ReadOnlySpan<float> ws, int xBase, int wBase, float bi, int oy, int ox, int KH, int KW, int sH, int sW, int dH, int dW, int padT, int padL, int H, int W, bool fuseRelu)
    {
        float acc = bi;
        for (int ky = 0; ky < KH; ky++)
        {
            int iy = oy * sH + ky * dH - padT;
            for (int kx = 0; kx < KW; kx++)
            {
                int ix = ox * sW + kx * dW - padL;
                float xv = (uint)iy < (uint)H && (uint)ix < (uint)W ? xs[xBase + iy * W + ix] : 0f;
                acc += ws[wBase + ky * KW + kx] * xv;
            }
        }
        return fuseRelu && acc < 0f ? 0f : acc;
    }
}
