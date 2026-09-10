namespace Lokad.Onnx;

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

using static Lokad.Onnx.MathOps;
using static Lokad.Onnx.Profiler;

public abstract partial class Tensor<T> : TensorBase, IList, IList<T>, IReadOnlyList<T>, IStructuralComparable, IStructuralEquatable, ITensor, INumericTensor
where T : unmanaged
{
    // Shared Conv2D preparation for PadType padding: validates ranks, fills
    // default strides and dilations, resolves dims and kernel extents, and
    // computes the padded output geometry. Exception parameter names below
    // are contractual. The typed cores stay per-dtype.
    static (int N, int C, int H, int W, int M, int kH, int kW, int dH, int dW, int sH, int sW, PadInfo pad, int outH, int outW) PlanConvPadType(
        ITensor input, ITensor weight, int group, PadType padtype, int? padvalue, int[]? kernelshape, int[]? strides, int[]? dilations, int biasLength)
    {
        if (input.Rank != 4)
        {
            throw new ArgumentException("input", "Input tensors must be of rank 4 with the layout NxCxHxW.");
        }
        if (weight.Rank != 4)
        {
            throw new ArgumentException("weight", "Weight tensors must be of rank 4 with the layout M x C/group x kH x kW.");
        }
        if (strides == null)
        {
            strides = new int[2] { 1, 1 };
        }
        if (dilations == null)
        {
            dilations = new int[2] { 1, 1 };
        }
        int N = input.Dims[0];
        int C = input.Dims[1];
        int H = input.Dims[2];
        int W = input.Dims[3];
        int M = weight.Dims[0];
        int kH = kernelshape == null ? weight.Dims[2] : kernelshape[0];
        int kW = kernelshape == null ? weight.Dims[3] : kernelshape[1];
        ValidateConv2D(N, C, H, W, M, weight.Dims[1], kH, kW, group, strides, dilations, kernelshape, weight.Dims.ToArray(), input.Length, weight.Length, biasLength);
        var info = GetConv2DOutputInfo(padtype, H, W, strides[0], strides[1], GetConv2DEffectiveFilterSize(kH, dilations[0]), GetConv2DEffectiveFilterSize(kW, dilations[1]), padvalue);
        if (info.Shape[0] <= 0 || info.Shape[1] <= 0) throw new ArgumentException("Conv output spatial dims must be positive.");
        return (N, C, H, W, M, kH, kW, dilations[0], dilations[1], strides[0], strides[1], info.PadInfo, info.Shape[0], info.Shape[1]);
    }

    // Shared Conv2D preparation for explicit pads: validates ranks and pads,
    // fills defaults, resolves dims and kernel extents, and computes the
    // explicit output geometry. The typed cores stay per-dtype.
    static (int N, int C, int H, int W, int M, int kH, int kW, int dH, int dW, int sH, int sW, PadInfo pad, int outH, int outW) PlanConvExplicit(
        ITensor input, ITensor weight, int group, int[] pads, int[]? kernelshape, int[]? strides, int[]? dilations, int biasLength)
    {
        if (input.Rank != 4)
        {
            throw new ArgumentException("input", "Input tensors must be of rank 4 with the layout NxCxHxW.");
        }
        if (weight.Rank != 4)
        {
            throw new ArgumentException("weight", "Weight tensors must be of rank 4 with the layout M x C/group x kH x kW.");
        }
        if (pads is null || pads.Length != 4)
        {
            throw new ArgumentException(nameof(pads), "Explicit pads must have four values [begin_h, begin_w, end_h, end_w].");
        }
        if (pads.Any(p => p < 0))
        {
            throw new ArgumentException(nameof(pads), "Explicit pads must be non-negative.");
        }
        if (strides == null)
        {
            strides = new int[2] { 1, 1 };
        }
        if (dilations == null)
        {
            dilations = new int[2] { 1, 1 };
        }
        int N = input.Dims[0];
        int C = input.Dims[1];
        int H = input.Dims[2];
        int W = input.Dims[3];
        int M = weight.Dims[0];
        int kH = kernelshape == null ? weight.Dims[2] : kernelshape[0];
        int kW = kernelshape == null ? weight.Dims[3] : kernelshape[1];
        ValidateConv2D(N, C, H, W, M, weight.Dims[1], kH, kW, group, strides, dilations, kernelshape, weight.Dims.ToArray(), input.Length, weight.Length, biasLength);
        int effKH = GetConv2DEffectiveFilterSize(kH, dilations[0]);
        int effKW = GetConv2DEffectiveFilterSize(kW, dilations[1]);
        var outShape = GetConv2DOutputShape(new int[] { H, W }, effKH, effKW, strides[0], strides[1], pads[0] + pads[2], pads[1] + pads[3]);
        if (outShape[0] <= 0 || outShape[1] <= 0) throw new ArgumentException("Conv output spatial dims must be positive.");
        var pad = new PadInfo { top = pads[0], left = pads[1], bottom = pads[2], right = pads[3], h = pads[0] + pads[2], w = pads[1] + pads[3] };
        return (N, C, H, W, M, kH, kW, dilations[0], dilations[1], strides[0], strides[1], pad, outShape[0], outShape[1]);
    }


    public static Tensor<float> Conv2D(Tensor<float> input, Tensor<float> weight, int group, PadType padtype, int? padvalue, Tensor<float>? bias, int[]? kernelshape, int[]? strides, int[]? dilations) =>
        Conv2D(input, weight, group, padtype, padvalue, bias, kernelshape, strides, dilations, TensorExecutionOptions.Auto);

    /// <summary>Two-dimensional convolution with explicit execution options.</summary>
    public static Tensor<float> Conv2D(Tensor<float> input, Tensor<float> weight, int group, PadType padtype, int? padvalue, Tensor<float>? bias, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options)
    {
        var (N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW) = PlanConvPadType(input, weight, group, padtype, padvalue, kernelshape, strides, dilations, bias is null ? -1 : (int)bias.Length);
        return Conv2DFloatCore(input, weight, group, N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, bias, options);

    }

    public static Tensor<float> Conv2D(Tensor<float> input, Tensor<float> weight, int group, int[] pads, Tensor<float>? bias, int[]? kernelshape, int[]? strides, int[]? dilations) =>
        Conv2D(input, weight, group, pads, bias, kernelshape, strides, dilations, TensorExecutionOptions.Auto);

    /// <summary>Two-dimensional convolution with explicit execution options.</summary>
    public static Tensor<float> Conv2D(Tensor<float> input, Tensor<float> weight, int group, int[] pads, Tensor<float>? bias, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options)
    {
        var (N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW) = PlanConvExplicit(input, weight, group, pads, kernelshape, strides, dilations, bias is null ? -1 : (int)bias.Length);
        return Conv2DFloatCore(input, weight, group, N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, bias, options);

    }

    static Tensor<float> Conv2DFloatCore(Tensor<float> input, Tensor<float> weight, int group, int N, int C, int H, int W, int M, int kH, int kW, int dH, int dW, int sH, int sW, PadInfo pad, int outH, int outW, Tensor<float>? bias, TensorExecutionOptions options)
    {
        options.Validate();
        var output = new DenseTensor<float>((ReadOnlySpan<int>)new int[] { N, M, outH, outW });
        var xd = input.ToDenseTensor();
        var wd = weight.ToDenseTensor();
        var bd = bias?.ToDenseTensor();
        int inBatch = C * H * W;
        int outBatch = M * outH * outW;
        int patchSize = C * kH * kW * outH * outW;
        int dop = options.MaxDegreeOfParallelism < 2 || N < 2 ? 1 : Math.Min(options.MaxDegreeOfParallelism, N);
        var xMem = xd.Buffer;
        var wMem = wd.Buffer;
        var oMem = output.Buffer;
        var bMem = bd is null ? default : bd.Buffer;
        bool hasBias = bd is not null;
        if (dop > 1)
        {
            Parallel.For(0, N, new ParallelOptions { MaxDegreeOfParallelism = dop },
                () => ArrayPool<float>.Shared.Rent(patchSize),
                (b, state, scratch) =>
                {
                    RunConvBatchFloat(xMem, wMem, bMem, hasBias, oMem, scratch, patchSize, b, group, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, inBatch, outBatch, options);
                    return scratch;
                },
                scratch => ArrayPool<float>.Shared.Return(scratch));
        }
        else
        {
            var scratch = ArrayPool<float>.Shared.Rent(patchSize);
            try
            {
                for (int b = 0; b < N; b++)
                    RunConvBatchFloat(xMem, wMem, bMem, hasBias, oMem, scratch, patchSize, b, group, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, inBatch, outBatch, options);
            }
            finally { ArrayPool<float>.Shared.Return(scratch); }
        }

        return output;

    }

    /// <summary>
    /// Runs one batch of float convolution: im2col into pooled scratch, then one
    /// shared-dispatcher product per group (which clears each destination tile,
    /// preserving the legacy clearing semantics), then bias.
    /// </summary>
    static void RunConvBatchFloat(Memory<float> xMem, Memory<float> wMem, Memory<float> bMem, bool hasBias, Memory<float> oMem, float[] scratch, int patchSize, int b, int group, int C, int H, int W, int M, int kH, int kW, int dH, int dW, int sH, int sW, PadInfo pad, int outH, int outW, int inBatch, int outBatch, TensorExecutionOptions options)
    {
        var patchMem = new Memory<float>(scratch, 0, patchSize);
        unsafe
        {
            fixed (float* src = xMem.Span.Slice(b * inBatch, inBatch))
            fixed (float* patch = patchMem.Span)
            {
                MathOps.Im2col(src, C, H, W, kH, kW, dH, dW, sH, sW, pad.top, pad.left, pad.bottom, pad.right, patch);
            }
        }
        int tileM = M / group;
        int tileN = outH * outW;
        int tileK = C * kH * kW / group;
        for (int g = 0; g < group; g++)
        {
            var wView = new DenseTensor<float>(wMem.Slice(g * tileM * tileK, tileM * tileK), new int[] { tileM, tileK });
            var pView = new DenseTensor<float>(patchMem.Slice(g * tileK * tileN, tileK * tileN), new int[] { tileK, tileN });
            var dView = new DenseTensor<float>(oMem.Slice(b * outBatch + g * tileM * tileN, tileM * tileN), new int[] { tileM, tileN });
            Tensor<float>.MatMul2D(wView, pView, dView, options);
        }
        if (hasBias)
        {
            var bs = bMem.Span;
            var os = oMem.Span;
            for (int i = 0; i < M; i++)
            {
                float bi = bs[i];
                int row = b * outBatch + i * tileN;
                for (int j = 0; j < tileN; j++) os[row + j] += bi;
            }
        }
    }

    public static Tensor<double> Conv2D(Tensor<double> input, Tensor<double> weight, int group, PadType padtype, int? padvalue, Tensor<double>? bias, int[]? kernelshape, int[]? strides, int[]? dilations) =>
        Conv2D(input, weight, group, padtype, padvalue, bias, kernelshape, strides, dilations, TensorExecutionOptions.Auto);

    /// <summary>Two-dimensional convolution with explicit execution options.</summary>
    public static Tensor<double> Conv2D(Tensor<double> input, Tensor<double> weight, int group, PadType padtype, int? padvalue, Tensor<double>? bias, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options)
    {
        var (N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW) = PlanConvPadType(input, weight, group, padtype, padvalue, kernelshape, strides, dilations, bias is null ? -1 : (int)bias.Length);
        return Conv2DDoubleCore(input, weight, group, N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, bias, options);

    }

    public static Tensor<double> Conv2D(Tensor<double> input, Tensor<double> weight, int group, int[] pads, Tensor<double>? bias, int[]? kernelshape, int[]? strides, int[]? dilations) =>
        Conv2D(input, weight, group, pads, bias, kernelshape, strides, dilations, TensorExecutionOptions.Auto);

    /// <summary>Two-dimensional convolution with explicit execution options.</summary>
    public static Tensor<double> Conv2D(Tensor<double> input, Tensor<double> weight, int group, int[] pads, Tensor<double>? bias, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options)
    {
        var (N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW) = PlanConvExplicit(input, weight, group, pads, kernelshape, strides, dilations, bias is null ? -1 : (int)bias.Length);
        return Conv2DDoubleCore(input, weight, group, N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, bias, options);

    }

    static Tensor<double> Conv2DDoubleCore(Tensor<double> input, Tensor<double> weight, int group, int N, int C, int H, int W, int M, int kH, int kW, int dH, int dW, int sH, int sW, PadInfo pad, int outH, int outW, Tensor<double>? bias, TensorExecutionOptions options)
    {
        options.Validate();
        var output = new DenseTensor<double>((ReadOnlySpan<int>)new int[] { N, M, outH, outW });
        var xd = input.ToDenseTensor();
        var wd = weight.ToDenseTensor();
        var bd = bias?.ToDenseTensor();
        int inBatch = C * H * W;
        int outBatch = M * outH * outW;
        int patchSize = C * kH * kW * outH * outW;
        int dop = options.MaxDegreeOfParallelism < 2 || N < 2 ? 1 : Math.Min(options.MaxDegreeOfParallelism, N);
        var xMem = xd.Buffer;
        var wMem = wd.Buffer;
        var oMem = output.Buffer;
        var bMem = bd is null ? default : bd.Buffer;
        bool hasBias = bd is not null;
        if (dop > 1)
        {
            Parallel.For(0, N, new ParallelOptions { MaxDegreeOfParallelism = dop },
                () => ArrayPool<double>.Shared.Rent(patchSize),
                (b, state, scratch) =>
                {
                    RunConvBatchDouble(xMem, wMem, bMem, hasBias, oMem, scratch, patchSize, b, group, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, inBatch, outBatch, options);
                    return scratch;
                },
                scratch => ArrayPool<double>.Shared.Return(scratch));
        }
        else
        {
            var scratch = ArrayPool<double>.Shared.Rent(patchSize);
            try
            {
                for (int b = 0; b < N; b++)
                    RunConvBatchDouble(xMem, wMem, bMem, hasBias, oMem, scratch, patchSize, b, group, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, inBatch, outBatch, options);
            }
            finally { ArrayPool<double>.Shared.Return(scratch); }
        }

        return output;

    }

    /// <summary>
    /// Runs one batch of double convolution: im2col into pooled scratch, then one
    /// shared-dispatcher product per group (which clears each destination tile,
    /// preserving the legacy clearing semantics), then bias.
    /// </summary>
    static void RunConvBatchDouble(Memory<double> xMem, Memory<double> wMem, Memory<double> bMem, bool hasBias, Memory<double> oMem, double[] scratch, int patchSize, int b, int group, int C, int H, int W, int M, int kH, int kW, int dH, int dW, int sH, int sW, PadInfo pad, int outH, int outW, int inBatch, int outBatch, TensorExecutionOptions options)
    {
        var patchMem = new Memory<double>(scratch, 0, patchSize);
        unsafe
        {
            fixed (double* src = xMem.Span.Slice(b * inBatch, inBatch))
            fixed (double* patch = patchMem.Span)
            {
                MathOps.Im2col(src, C, H, W, kH, kW, dH, dW, sH, sW, pad.top, pad.left, pad.bottom, pad.right, patch);
            }
        }
        int tileM = M / group;
        int tileN = outH * outW;
        int tileK = C * kH * kW / group;
        for (int g = 0; g < group; g++)
        {
            var wView = new DenseTensor<double>(wMem.Slice(g * tileM * tileK, tileM * tileK), new int[] { tileM, tileK });
            var pView = new DenseTensor<double>(patchMem.Slice(g * tileK * tileN, tileK * tileN), new int[] { tileK, tileN });
            var dView = new DenseTensor<double>(oMem.Slice(b * outBatch + g * tileM * tileN, tileM * tileN), new int[] { tileM, tileN });
            Tensor<double>.MatMul2D(wView, pView, dView, options);
        }
        if (hasBias)
        {
            var bs = bMem.Span;
            var os = oMem.Span;
            for (int i = 0; i < M; i++)
            {
                double bi = bs[i];
                int row = b * outBatch + i * tileN;
                for (int j = 0; j < tileN; j++) os[row + j] += bi;
            }
        }
    }

    // Shared MaxPool preparation for PadType padding: validates the kernel
    // and ranks, fills default strides and dilations, resolves dims, and
    // computes the padded output geometry. The typed cores stay per-dtype.
    static (int N, int C, int H, int W, int KH, int KW, int SH, int SW, int DH, int DW, PadInfo Pad, int OutH, int OutW) PlanPoolPadType(
        ITensor input, int[] kernelshape, PadType padtype, int? padvalue, int[]? strides, int[]? dilations)
    {
        if (kernelshape is null)
        {
            throw new ArgumentNullException("kernelshape");
        }
        if (input.Rank != 4)
        {
            throw new ArgumentException("Input tensors must be of rank 4 with the layout NxCxHxW.");
        }
        if (kernelshape.Rank != 1 || kernelshape.Length != 2)
        {
            throw new ArgumentException("The kernel must have shape m x n.");
        }
        if (strides == null)
        {
            strides = new int[] { 1, 1 };
        }
        if (dilations == null)
        {
            dilations = new int[] { 1, 1 };
        }
        int N = input.Dims[0];
        int C = input.Dims[1];
        int H = input.Dims[2];
        int W = input.Dims[3];
        int kH = kernelshape[0];
        int kW = kernelshape[1];
        var info = GetConv2DOutputInfo(padtype, H, W, strides[0], strides[1], GetConv2DEffectiveFilterSize(kH, dilations[0]), GetConv2DEffectiveFilterSize(kW, dilations[1]), padvalue);
        return (N, C, H, W, kH, kW, strides[0], strides[1], dilations[0], dilations[1], info.PadInfo, info.Shape[0], info.Shape[1]);
    }

    // Shared MaxPool preparation for explicit pads: validates the kernel,
    // ranks, and pads, fills defaults, resolves dims, and computes the
    // explicit output geometry with the caller ceil mode. The typed cores
    // stay per-dtype.
    static (int N, int C, int H, int W, int KH, int KW, int SH, int SW, int DH, int DW, PadInfo Pad, int OutH, int OutW) PlanPoolExplicit(
        ITensor input, int[] kernelshape, int[] pads, int[]? strides, int[]? dilations, bool ceilMode)
    {
        if (kernelshape is null)
        {
            throw new ArgumentNullException("kernelshape");
        }
        if (input.Rank != 4)
        {
            throw new ArgumentException("Input tensors must be of rank 4 with the layout NxCxHxW.");
        }
        if (kernelshape.Rank != 1 || kernelshape.Length != 2)
        {
            throw new ArgumentException("The kernel must have shape m x n.");
        }
        if (pads is null || pads.Length != 4)
        {
            throw new ArgumentException(nameof(pads), "Explicit pads must have four values [begin_h, begin_w, end_h, end_w].");
        }
        if (pads.Any(p => p < 0))
        {
            throw new ArgumentException(nameof(pads), "Explicit pads must be non-negative.");
        }
        if (strides == null)
        {
            strides = new int[] { 1, 1 };
        }
        if (dilations == null)
        {
            dilations = new int[] { 1, 1 };
        }
        int N = input.Dims[0];
        int C = input.Dims[1];
        int H = input.Dims[2];
        int W = input.Dims[3];
        int kH = kernelshape[0];
        int kW = kernelshape[1];
        int effKH = GetConv2DEffectiveFilterSize(kH, dilations[0]);
        int effKW = GetConv2DEffectiveFilterSize(kW, dilations[1]);
        var outShape = MaxPoolOutputShape(H, W, effKH, effKW, strides[0], strides[1], pads[0] + pads[2], pads[1] + pads[3], ceilMode);
        if (outShape[0] <= 0 || outShape[1] <= 0) throw new ArgumentException("MaxPool output spatial dims must be positive.");
        var pad = new PadInfo { top = pads[0], left = pads[1], bottom = pads[2], right = pads[3], h = pads[0] + pads[2], w = pads[1] + pads[3] };
        return (N, C, H, W, kH, kW, strides[0], strides[1], dilations[0], dilations[1], pad, outShape[0], outShape[1]);
    }

    public static Tensor<float> MaxPool2D(Tensor<float> input, int[] kernelshape, PadType padtype, int? padvalue, int[]? strides, int[]? dilations, bool ceilMode)
    {
        var (N, C, H, W, kH, kW, sH, sW, dH, dW, pad, outH, outW) = PlanPoolPadType(input, kernelshape, padtype, padvalue, strides, dilations);
        if (!ceilMode) return MaxPoolFloatCore(input, N, C, H, W, kH, kW, sH, sW, dH, dW, pad, outH, outW);
        int effKH = GetConv2DEffectiveFilterSize(kH, dH);
        int effKW = GetConv2DEffectiveFilterSize(kW, dW);
        var ceilShape = MaxPoolOutputShape(H, W, effKH, effKW, sH, sW, pad.top + pad.bottom, pad.left + pad.right, true);
        return MaxPoolFloatCore(input, N, C, H, W, kH, kW, sH, sW, dH, dW, pad, ceilShape[0], ceilShape[1]);
    }

    public static Tensor<float> MaxPool2D(Tensor<float> input, int[] kernelshape, int[] pads, int[]? strides, int[]? dilations, bool ceilMode)
    {
        var (N, C, H, W, kH, kW, sH, sW, dH, dW, pad, outH, outW) = PlanPoolExplicit(input, kernelshape, pads, strides, dilations, ceilMode);
        return MaxPoolFloatCore(input, N, C, H, W, kH, kW, sH, sW, dH, dW, pad, outH, outW);
    }

    static int[] MaxPoolOutputShape(int H, int W, int effKH, int effKW, int sH, int sW, int padH, int padW, bool ceilMode)
    {
        int outH = ceilMode
            ? (int)Math.Ceiling((H + padH - effKH) / (float)sH) + 1
            : (int)Math.Floor((H + padH - effKH) / (float)sH) + 1;
        int outW = ceilMode
            ? (int)Math.Ceiling((W + padW - effKW) / (float)sW) + 1
            : (int)Math.Floor((W + padW - effKW) / (float)sW) + 1;
        return new int[] { outH, outW };
    }

    static Tensor<float> MaxPoolFloatCore(Tensor<float> input, int N, int C, int H, int W, int kH, int kW, int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, PadInfo pad, int outH, int outW)
    {
        var Y = DenseTensor<float>.OfShape(N, C, outH, outW);

        for (var n = 0; n < N; ++n)
        {
            for (var d = 0; d < C; ++d)
            {
                for (var yR = 0; yR < outH; ++yR)
                {
                    var xRCorner = yR * strideHeight - pad.top;
                    for (var yC = 0; yC < outW; ++yC)
                    {
                        var xCCorner = yC * strideWidth - pad.left;

                        // Negative float max like the native reference: windows
                        // with no finite value stay there, and NaN never wins a
                        // comparison. (Multi-NaN full-vector windows differ on
                        // the native side by SIMD width, which no deterministic
                        // contract can match; ours is stable.)
                        var maxValue = -float.MaxValue;

                        for (var tR = 0; tR < kH; ++tR)
                        {
                            var xR = xRCorner + tR * dilationHeight;
                            if (xR < 0 || xR >= H) continue;
                            for (var tC = 0; tC < kW; ++tC)
                            {
                                var xC = xCCorner + tC * dilationWidth;
                                if (xC < 0 || xC >= W) continue;
                                var v = input[n, d, xR, xC];

                                if (v > maxValue)
                                {
                                    maxValue = v;
                                }
                            }
                        }
                        Y[n, d, yR, yC] = maxValue;
                    }
                }
            }
        }
        return Y;
    }

    public static Tensor<double> MaxPool2D(Tensor<double> input, int[] kernelshape, PadType padtype, int? padvalue, int[]? strides, int[]? dilations, bool ceilMode)
    {
        var (N, C, H, W, kH, kW, sH, sW, dH, dW, pad, outH, outW) = PlanPoolPadType(input, kernelshape, padtype, padvalue, strides, dilations);
        if (!ceilMode) return MaxPoolDoubleCore(input, N, C, H, W, kH, kW, sH, sW, dH, dW, pad, outH, outW);
        int effKH = GetConv2DEffectiveFilterSize(kH, dH);
        int effKW = GetConv2DEffectiveFilterSize(kW, dW);
        var ceilShape = MaxPoolOutputShape(H, W, effKH, effKW, sH, sW, pad.top + pad.bottom, pad.left + pad.right, true);
        return MaxPoolDoubleCore(input, N, C, H, W, kH, kW, sH, sW, dH, dW, pad, ceilShape[0], ceilShape[1]);
    }

    public static Tensor<double> MaxPool2D(Tensor<double> input, int[] kernelshape, int[] pads, int[]? strides, int[]? dilations, bool ceilMode)
    {
        var (N, C, H, W, kH, kW, sH, sW, dH, dW, pad, outH, outW) = PlanPoolExplicit(input, kernelshape, pads, strides, dilations, ceilMode);
        return MaxPoolDoubleCore(input, N, C, H, W, kH, kW, sH, sW, dH, dW, pad, outH, outW);
    }

    static Tensor<double> MaxPoolDoubleCore(Tensor<double> input, int N, int C, int H, int W, int kH, int kW, int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, PadInfo pad, int outH, int outW)
    {
        var Y = DenseTensor<double>.OfShape(N, C, outH, outW);

        for (var n = 0; n < N; ++n)
        {
            for (var d = 0; d < C; ++d)
            {
                for (var yR = 0; yR < outH; ++yR)
                {
                    var xRCorner = yR * strideHeight - pad.top;
                    for (var yC = 0; yC < outW; ++yC)
                    {
                        var xCCorner = yC * strideWidth - pad.left;

                        // Unlike the float core (which skips NaN like ORT
                        // float), ORT double propagates NaN, so a NaN value
                        // always wins here.
                        var maxValue = double.NegativeInfinity;

                        for (var tR = 0; tR < kH; ++tR)
                        {
                            var xR = xRCorner + tR * dilationHeight;
                            if (xR < 0 || xR >= H) continue;
                            for (var tC = 0; tC < kW; ++tC)
                            {
                                var xC = xCCorner + tC * dilationWidth;
                                if (xC < 0 || xC >= W) continue;
                                var v = input[n, d, xR, xC];

                                if (v > maxValue || double.IsNaN(v))
                                {
                                    maxValue = v;
                                }
                            }
                            if (maxValue == double.NegativeInfinity)
                            {
                                break;
                            }
                        }
                        Y[n, d, yR, yC] = maxValue;
                    }
                }
            }
        }
        return Y;
    }

    public static Tensor<int> MaxPool2D(Tensor<int> input, int[] kernelshape, PadType padtype, int? padvalue, int[]? strides, int[]? dilations)
    {
        var (N, C, H, W, kH, kW, sH, sW, dH, dW, pad, outH, outW) = PlanPoolPadType(input, kernelshape, padtype, padvalue, strides, dilations);
        return MaxPoolIntCore(input, N, C, H, W, kH, kW, sH, sW, dH, dW, pad, outH, outW);
    }

    static Tensor<int> MaxPoolIntCore(Tensor<int> input, int N, int C, int H, int W, int kH, int kW, int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, PadInfo pad, int outH, int outW)
    {
        var Y = DenseTensor<int>.OfShape(N, C, outH, outW);

        for (var n = 0; n < N; ++n)
        {
            for (var d = 0; d < C; ++d)
            {
                for (var yR = 0; yR < outH; ++yR)
                {
                    var xRCorner = yR * strideHeight - pad.top;
                    for (var yC = 0; yC < outW; ++yC)
                    {
                        var xCCorner = yC * strideWidth - pad.left;

                        var maxValue = 0;

                        for (var tR = 0; tR < kH; ++tR)
                        {
                            var xR = xRCorner + tR * dilationHeight;
                            if (xR < 0 || xR >= H) continue;
                            for (var tC = 0; tC < kW; ++tC)
                            {
                                var xC = xCCorner + tC * dilationWidth;
                                if (xC < 0 || xC >= W) continue;
                                var v = input[n, d, xR, xC];

                                if (v > maxValue)
                                {
                                    maxValue = v;
                                }
                            }
                            if (maxValue == 0)
                            {
                                break;
                            }
                        }
                        Y[n, d, yR, yC] = maxValue;
                    }
                }
            }
        }
        return Y;
    }

    static void ValidateConv2D(
        int N, int C, int H, int W, int M, int Cpg, int kH, int kW,
        int group, int[] strides, int[] dilations, int[]? kernelshape,
        int[] weightDims, long inputLength, long weightLength, int biasLength)
    {
        if (N < 0) throw new ArgumentException("Conv input batch must be non-negative.", nameof(N));
        if (C < 0) throw new ArgumentException("Conv input channels must be non-negative.", nameof(C));
        if (H <= 0 || W <= 0) throw new ArgumentException("Conv spatial dims must be positive.");
        if (M < 0) throw new ArgumentException("Conv output channels must be non-negative.", nameof(M));
        if (group <= 0) throw new ArgumentException("Conv group must be positive.", nameof(group));
        if (C % group != 0) throw new ArgumentException("Conv input channels must be divisible by group.");
        if (M % group != 0) throw new ArgumentException("Conv output channels must be divisible by group.");
        if (strides.Length != 2 || strides[0] <= 0 || strides[1] <= 0) throw new ArgumentException("Conv strides must be two positive values.");
        if (dilations.Length != 2 || dilations[0] <= 0 || dilations[1] <= 0) throw new ArgumentException("Conv dilations must be two positive values.");
        if (kH <= 0 || kW <= 0) throw new ArgumentException("Conv kernel dims must be positive.");
        if (kernelshape is not null)
        {
            if (kernelshape.Length != 2) throw new ArgumentException("Conv kernel_shape must have two values.");
            if (kernelshape[0] != weightDims[2] || kernelshape[1] != weightDims[3]) throw new ArgumentException("Conv kernel_shape must match weight spatial dims.");
        }
        if (weightDims.Length != 4) throw new ArgumentException("Conv weight must be rank 4.");
        if (weightDims[0] != M || weightDims[1] != Cpg || weightDims[2] != kH || weightDims[3] != kW) throw new ArgumentException("Conv weight shape must be [M, C/group, kH, kW].");
        if (Cpg != C / group) throw new ArgumentException("Conv weight channels must equal C/group.");
        if (biasLength >= 0 && biasLength != M) throw new ArgumentException("Conv bias length must equal M.");
        checked
        {
            long expectInput = (long)N * C * H * W;
            long expectWeight = (long)M * Cpg * kH * kW;
            if (inputLength != expectInput) throw new ArgumentException("Conv input backing length does not match shape.");
            if (weightLength != expectWeight) throw new ArgumentException("Conv weight backing length does not match shape.");
        }
    }
}
