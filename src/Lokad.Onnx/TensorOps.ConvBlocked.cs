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
    /// Direct rank-four float blocked convolution, W4 prototype v1 (structure proof, not tuned, not wired into dispatch): NCHWc-16 input, OIHW16o filter, NCHWc-16 accumulation with fused bias and optional ReLU. Vector-width loops use System.Numerics (no FMA fusing yet -- that is v2); border taps skip out-of-range reads in one kernel.
    /// </summary>
    public static bool TryConvBlocked2D(Tensor<float> input, Tensor<float> weight, Tensor<float>? bias, int group, int[]? pads, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options, bool fuseRelu, TensorBufferPool? pool, out Tensor<float>? output)
    {
        output = null;
        if (input.Rank != 4 || weight.Rank != 4) return false;
        int N = input.Dimensions[0], C = input.Dimensions[1], H = input.Dimensions[2], W = input.Dimensions[3];
        int M = weight.Dimensions[0];
        if (N != 1 || group != 1) return false;
        if (C < 16 || (C & 15) != 0 || M < 16 || (M & 15) != 0 || weight.Dimensions[1] != C) return false;
        int KH = weight.Dimensions[2], KW = weight.Dimensions[3];
        if (KH != 3 || KW != 3) return false;
        if (kernelshape is not null && (kernelshape.Length != 2 || kernelshape[0] != 3 || kernelshape[1] != 3)) return false;
        if (strides is not null && (strides.Length != 2 || strides[0] != 1 || strides[1] != 1)) return false;
        if (dilations is not null && (dilations.Length != 2 || dilations[0] != 1 || dilations[1] != 1)) return false;
        int padT = 0, padL = 0, padB = 0, padR = 0;
        if (pads is not null)
        {
            if (pads.Length != 4 || pads[0] < 0 || pads[1] < 0 || pads[2] < 0 || pads[3] < 0) return false;
            padT = pads[0]; padL = pads[1]; padB = pads[2]; padR = pads[3];
        }
        if (bias is not null && bias.Length != M) return false;
        int[] outShape = GetConv2DOutputShape(new int[] { H, W }, 3, 3, 1, 1, padT + padB, padL + padR);
        int outH = outShape[0], outW = outShape[1];
        if (outH < 1 || outW < 1) return false;
        options.Validate();
        var xd = input.ToDenseTensor();
        var wd = weight.ToDenseTensor();
        var bd = bias?.ToDenseTensor();
        StartOpStage(OpStage.Math);
        ReportKernelRoute("conv-blocked");
        int Cb = C / 16, Mb = M / 16;
        var xs = xd.Buffer.Span;
        var ws = wd.Buffer.Span;
        var packedX = new float[(long)Cb * H * W * 16];
        var packedW = new float[(long)Mb * C * 9 * 16];
        for (int cb = 0; cb < Cb; cb++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                {
                    int d = ((cb * H) + h) * W + w; d *= 16;
                    int s = (cb * 16) * H * W + h * W + w;
                    for (int cc = 0; cc < 16; cc++) packedX[d + cc] = xs[s + cc * H * W];
                }
        for (int mb = 0; mb < Mb; mb++)
            for (int c = 0; c < C; c++)
                for (int t = 0; t < 9; t++)
                {
                    int d = ((mb * C) + c) * 9 + t; d *= 16;
                    int s = ((mb * 16) * C + c) * 9 + t;
                    for (int mm = 0; mm < 16; mm++) packedW[d + mm] = ws[s + mm * C * 9];
                }
        var y = pool is null ? new DenseTensor<float>((ReadOnlySpan<int>)new int[] { 1, M, outH, outW }) : new DenseTensor<float>(new Memory<float>(pool.Rent<float>((int)((long)M * outH * outW))), new int[] { 1, M, outH, outW });
        var ys = y.Buffer.Span;
        var bs = bd is null ? default(ReadOnlySpan<float>) : bd.Buffer.Span;
        int vw = Vector<float>.Count;
        var zero = Vector<float>.Zero;
        Span<float> acc = stackalloc float[16];
        for (int mb = 0; mb < Mb; mb++)
            for (int oh = 0; oh < outH; oh++)
                for (int ow = 0; ow < outW; ow++)
                {
                    for (int mm = 0; mm < 16; mm++) acc[mm] = bd is null ? 0f : bs[mb * 16 + mm];
                    if (Avx512F.IsSupported && Fma.IsSupported)
                    {
                        var acc512 = Vector512.LoadUnsafe(ref acc[0]);
                        for (int cb = 0; cb < Cb; cb++)
                            for (int kh = 0; kh < 3; kh++)
                            {
                                int ih = oh + kh - padT;
                                if ((uint)ih >= (uint)H) continue;
                                for (int kw = 0; kw < 3; kw++)
                                {
                                    int iw = ow + kw - padL;
                                    if ((uint)iw >= (uint)W) continue;
                                    int io = ((cb * H) + ih) * W * 16 + iw * 16;
                                    int woBase = (((mb * C) + cb * 16) * 9 + kh * 3 + kw) * 16;
                                    for (int ccb = 0; ccb < 16; ccb++)
                                    {
                                        var sv = Vector512.Create(packedX[io + ccb]);
                                        acc512 = Avx512F.FusedMultiplyAdd(sv, Vector512.LoadUnsafe(ref packedW[woBase + ccb * 144]), acc512);
                                    }
                                }
                            }
                        acc512.StoreUnsafe(ref acc[0]);
                    }
                    else
                    {
                        for (int cb = 0; cb < Cb; cb++)
                            for (int kh = 0; kh < 3; kh++)
                            {
                                int ih = oh + kh - padT;
                                if ((uint)ih >= (uint)H) continue;
                                for (int kw = 0; kw < 3; kw++)
                                {
                                    int iw = ow + kw - padL;
                                    if ((uint)iw >= (uint)W) continue;
                                    int io = ((cb * H) + ih) * W * 16 + iw * 16;
                                    int woBase = (((mb * C) + cb * 16) * 9 + kh * 3 + kw) * 16;
                                    for (int ccb = 0; ccb < 16; ccb++)
                                    {
                                        var sv = new Vector<float>(packedX[io + ccb]);
                                        for (int v = 0; v < 16; v += vw)
                                        {
                                            var av = new Vector<float>(acc.Slice(v));
                                            av += sv * new Vector<float>(packedW.AsSpan(woBase + ccb * 144 + v, vw));
                                            av.CopyTo(acc.Slice(v));
                                        }
                                    }
                                }
                            }
                    }
                    for (int mm = 0; mm < 16; mm++)
                    {
                        float v = acc[mm];
                        if (fuseRelu && v < 0f) v = 0f;
                        ys[((mb * 16 + mm) * outH + oh) * outW + ow] = v;
                    }
                }
        output = y;
        return true;
    }
}
