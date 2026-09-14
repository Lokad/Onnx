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
using System.Runtime.Intrinsics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

using static Lokad.Onnx.MathOps;
using static Lokad.Onnx.Profiler;

public abstract partial class Tensor<T> : TensorBase, IList, IList<T>, IReadOnlyList<T>, IStructuralComparable, IStructuralEquatable, ITensor, INumericTensor
where T : unmanaged
{
    /// <summary>Rents ArrayPool scratch and reports the requested bytes to the run accountant, if any.</summary>
    static U[] RentScratch<U>(int length, TensorExecutionOptions options) where U : unmanaged
    {
        var rented = ArrayPool<U>.Shared.Rent(length);
        options.ScratchReporter?.AddScratchBytes((long)length * Unsafe.SizeOf<U>());
        return rented;
    }

    /// <summary>Bounds one tiled-convolution column block (patch plus GEMM output tile) to L2-resident scratch.</summary>
    const int ConvTileBudgetBytes = 256 * 1024;

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
        return Conv2DFloatCore(input, weight, group, N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, bias, options, false);

    }

    public static Tensor<float> Conv2D(Tensor<float> input, Tensor<float> weight, int group, int[] pads, Tensor<float>? bias, int[]? kernelshape, int[]? strides, int[]? dilations) =>
        Conv2D(input, weight, group, pads, bias, kernelshape, strides, dilations, TensorExecutionOptions.Auto);

    /// <summary>Two-dimensional convolution with explicit execution options.</summary>
    public static Tensor<float> Conv2D(Tensor<float> input, Tensor<float> weight, int group, int[] pads, Tensor<float>? bias, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options)
    {
        var (N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW) = PlanConvExplicit(input, weight, group, pads, kernelshape, strides, dilations, bias is null ? -1 : (int)bias.Length);
        return Conv2DFloatCore(input, weight, group, N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, bias, options, false);

    }

    /// <summary>Two-dimensional convolution with explicit execution options and a fused ReLU epilogue.</summary>
    public static Tensor<float> Conv2D(Tensor<float> input, Tensor<float> weight, int group, PadType padtype, int? padvalue, Tensor<float>? bias, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options, bool fuseRelu)
    {
        var (fN, fC, fH, fW, fM, fkH, fkW, fdH, fdW, fsH, fsW, fpad, foutH, foutW) = PlanConvPadType(input, weight, group, padtype, padvalue, kernelshape, strides, dilations, bias is null ? -1 : (int)bias.Length);
        return Conv2DFloatCore(input, weight, group, fN, fC, fH, fW, fM, fkH, fkW, fdH, fdW, fsH, fsW, fpad, foutH, foutW, bias, options, fuseRelu);
    }

    /// <summary>Two-dimensional convolution with explicit execution options and a fused ReLU epilogue.</summary>
    public static Tensor<float> Conv2D(Tensor<float> input, Tensor<float> weight, int group, int[] pads, Tensor<float>? bias, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options, bool fuseRelu)
    {
        var (eN, eC, eH, eW, eM, ekH, ekW, edH, edW, esH, esW, epad, eoutH, eoutW) = PlanConvExplicit(input, weight, group, pads, kernelshape, strides, dilations, bias is null ? -1 : (int)bias.Length);
        return Conv2DFloatCore(input, weight, group, eN, eC, eH, eW, eM, ekH, ekW, edH, edW, esH, esW, epad, eoutH, eoutW, bias, options, fuseRelu);
    }

    static Tensor<float> Conv2DFloatCore(Tensor<float> input, Tensor<float> weight, int group, int N, int C, int H, int W, int M, int kH, int kW, int dH, int dW, int sH, int sW, PadInfo pad, int outH, int outW, Tensor<float>? bias, TensorExecutionOptions options, bool fuseRelu)
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
        if (kH == 1 && kW == 1 && sH == 1 && sW == 1 && dH == 1 && dW == 1
            && pad.top == 0 && pad.left == 0 && pad.bottom == 0 && pad.right == 0
            && outH == H && outW == W)
        {
            RunPointwiseBatchesFloat(xMem, wMem, bMem, hasBias, oMem, N, group, C, H, W, M, outH, outW, inBatch, outBatch, options, fuseRelu);
            return output;
        }
        int tileN = outH * outW;
        int tileKFull = C * kH * kW;
        int tileM = M / group;
        int blockN = tileN;
        if ((long)tileKFull * tileN * sizeof(float) > ConvTileBudgetBytes)
        {
            long perColumn = ((long)tileKFull + M) * sizeof(float);
            long fit = ConvTileBudgetBytes / perColumn;
            // Align full tiles down to the matrix kernel micro-panel so only
            // the final spatial remainder (handled per block below) pays a
            // remainder; narrower fits still take one panel, which the scratch
            // accountant reports exactly like any other rental.
            int panel = 4 * Vector256<float>.Count;
            long aligned = (fit / panel) * panel;
            if (aligned < panel) aligned = panel;
            if (aligned < tileN) blockN = (int)aligned;
        }
        if (blockN < tileN)
        {
            RunTiledConvFloat(xMem, wMem, bMem, hasBias, oMem, N, group, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, inBatch, outBatch, tileN, blockN, dop, options, fuseRelu);
            return output;
        }
        if (dop > 1)
        {
            Parallel.For(0, N, new ParallelOptions { MaxDegreeOfParallelism = dop },
                () => RentScratch<float>(patchSize, options),
                (b, state, scratch) =>
                {
                    RunConvBatchFloat(xMem, wMem, bMem, hasBias, oMem, scratch, patchSize, b, group, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, inBatch, outBatch, options, fuseRelu);
                    return scratch;
                },
                scratch => ArrayPool<float>.Shared.Return(scratch));
        }
        else
        {
            var scratch = RentScratch<float>(patchSize, options);
            try
            {
                for (int b = 0; b < N; b++)
                    RunConvBatchFloat(xMem, wMem, bMem, hasBias, oMem, scratch, patchSize, b, group, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, inBatch, outBatch, options, fuseRelu);
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
    static void RunConvBatchFloat(Memory<float> xMem, Memory<float> wMem, Memory<float> bMem, bool hasBias, Memory<float> oMem, float[] scratch, int patchSize, int b, int group, int C, int H, int W, int M, int kH, int kW, int dH, int dW, int sH, int sW, PadInfo pad, int outH, int outW, int inBatch, int outBatch, TensorExecutionOptions options, bool fuseRelu)
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
        // Fused ReLU epilogue: with fuseRelu the max applies in the same pass
        // as the bias add, so a fused Conv+Relu pair costs no extra tensor
        // pass. Unfused behavior is bit-identical (same add order, same max,
        // including signed zero and NaN handling).
        if (hasBias || fuseRelu)
        {
            var bs = bMem.Span;
            var os = oMem.Span;
            for (int i = 0; i < M; i++)
            {
                float bi = hasBias ? bs[i] : 0f;
                int row = b * outBatch + i * tileN;
                for (int j = 0; j < tileN; j++)
                {
                    float v = hasBias ? os[row + j] + bi : os[row + j];
                    os[row + j] = fuseRelu && v < 0f ? 0f : v;
                }
            }
        }
    }
    /// <summary>
    /// Runs convolution in bounded column tiles when the full patch would
    /// exceed L2-resident scratch: each tile converts one output-column
    /// block, multiplies it through the shared dispatcher into a block
    /// output buffer, and streams it through the bias/ReLU epilogue.
    /// Blocking covers independent outputs only, so each dot product keeps
    /// the single-pass order; the shared dispatcher may still pick different
    /// vectorized kernels per block shape, so agreement is within float
    /// rounding (validated at the 1e-4 gate), not bit for bit.
    /// </summary>
    static void RunTiledConvFloat(Memory<float> xMem, Memory<float> wMem, Memory<float> bMem, bool hasBias, Memory<float> oMem, int N, int group, int C, int H, int W, int M, int kH, int kW, int dH, int dW, int sH, int sW, PadInfo pad, int outH, int outW, int inBatch, int outBatch, int tileN, int blockN, int dop, TensorExecutionOptions options, bool fuseRelu)
    {
        int blockPatch = C * kH * kW * blockN;
        int blockOut = M * blockN;
        if (dop > 1)
        {
            Parallel.For(0, N, new ParallelOptions { MaxDegreeOfParallelism = dop },
                () => RentScratch<float>(blockPatch + blockOut, options),
                (b, state, scratch) =>
                {
                    RunTiledBatchFloat(xMem, wMem, bMem, hasBias, oMem, scratch, b, group, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, inBatch, outBatch, tileN, blockN, options, fuseRelu);
                    return scratch;
                },
                scratch => ArrayPool<float>.Shared.Return(scratch));
        }
        else
        {
            var scratch = RentScratch<float>(blockPatch + blockOut, options);
            try
            {
                for (int b = 0; b < N; b++)
                    RunTiledBatchFloat(xMem, wMem, bMem, hasBias, oMem, scratch, b, group, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, inBatch, outBatch, tileN, blockN, options, fuseRelu);
            }
            finally { ArrayPool<float>.Shared.Return(scratch); }
        }
    }

    /// <summary>
    /// Runs one batch of tiled float convolution: for each output-column
    /// block, converts the block patch, runs one shared-dispatcher product
    /// per group into the block output buffer (cleared by the dispatcher),
    /// then streams the block through the bias/ReLU epilogue. The epilogue
    /// keeps the single-pass add order and max, including NaN handling.
    /// </summary>
    static void RunTiledBatchFloat(Memory<float> xMem, Memory<float> wMem, Memory<float> bMem, bool hasBias, Memory<float> oMem, float[] scratch, int b, int group, int C, int H, int W, int M, int kH, int kW, int dH, int dW, int sH, int sW, PadInfo pad, int outH, int outW, int inBatch, int outBatch, int tileN, int blockN, TensorExecutionOptions options, bool fuseRelu)
    {
        int tileKFull = C * kH * kW;
        int tileM = M / group;
        int tileKg = tileKFull / group;
        var patchMem = new Memory<float>(scratch, 0, tileKFull * blockN);
        var outMem = new Memory<float>(scratch, tileKFull * blockN, M * blockN);
        int numBlocks = (tileN + blockN - 1) / blockN;
        var bs = bMem.Span;
        var os = oMem.Span;
        var ds = outMem.Span;
        for (int s = 0; s < numBlocks; s++)
        {
            int colStart = s * blockN;
            int colCount = Math.Min(blockN, tileN - colStart);
            unsafe
            {
                fixed (float* src = xMem.Span.Slice(b * inBatch, inBatch))
                fixed (float* patch = patchMem.Span)
                {
                    MathOps.Im2colRange(src, C, H, W, kH, kW, dH, dW, sH, sW, pad.top, pad.left, pad.bottom, pad.right, outW, colStart, colCount, patch);
                }
            }
            for (int g = 0; g < group; g++)
            {
                var wView = new DenseTensor<float>(wMem.Slice(g * tileM * tileKg, tileM * tileKg), new int[] { tileM, tileKg });
                var pView = new DenseTensor<float>(patchMem.Slice(g * tileKg * colCount, tileKg * colCount), new int[] { tileKg, colCount });
                var dView = new DenseTensor<float>(outMem.Slice(g * tileM * colCount, tileM * colCount), new int[] { tileM, colCount });
                Tensor<float>.MatMul2D(wView, pView, dView, options);
                int outBase = b * outBatch + g * tileM * tileN;
                int blkBase = g * tileM * colCount;
                for (int i = 0; i < tileM; i++)
                {
                    float bi = hasBias ? bs[g * tileM + i] : 0f;
                    int outRow = outBase + i * tileN + colStart;
                    int blkRow = blkBase + i * colCount;
                    for (int j = 0; j < colCount; j++)
                    {
                        float v = hasBias ? ds[blkRow + j] + bi : ds[blkRow + j];
                        os[outRow + j] = fuseRelu && v < 0f ? 0f : v;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Tries a direct float rank-three depthwise convolution: every channel
    /// owns one length-K filter (group, input and output channels equal,
    /// single filter channel), so each output is one row dot with no patch
    /// matrix and no per-group dispatch. Anything else, including automatic
    /// padding modes, declines for the rank-three adapter, which validates
    /// and reports unsupported contracts authoritatively.
    /// </summary>
    public static bool TryConvDepthwise1D(Tensor<float> input, Tensor<float> weight, Tensor<float>? bias, int group, int[]? pads, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options, bool fuseRelu, out Tensor<float>? output)
    {
        output = null;
        if (input.Rank != 3 || weight.Rank != 3) return false;
        int N = input.Dimensions[0], C = input.Dimensions[1], L = input.Dimensions[2];
        int M = weight.Dimensions[0];
        if (N < 1 || C < 1 || M != C || group != C || weight.Dimensions[1] != 1) return false;
        int K = weight.Dimensions[2];
        if (K < 1) return false;
        if (kernelshape is not null && (kernelshape.Length != 1 || kernelshape[0] != K)) return false;
        int s = 1;
        if (strides is not null)
        {
            if (strides.Length != 1 || strides[0] < 1) return false;
            s = strides[0];
        }
        int d = 1;
        if (dilations is not null)
        {
            if (dilations.Length != 1 || dilations[0] < 1) return false;
            d = dilations[0];
        }
        int padL = 0, padR = 0;
        if (pads is not null)
        {
            if (pads.Length != 2 || pads[0] < 0 || pads[1] < 0) return false;
            padL = pads[0];
            padR = pads[1];
        }
        if (bias is not null && bias.Length != M) return false;
        int effK = (K - 1) * d + 1;
        int outL = (L + padL + padR - effK) / s + 1;
        if (outL < 1) return false;
        options.Validate();
        output = RunDepthwise1DFloat(input.ToDenseTensor(), weight.ToDenseTensor(), bias?.ToDenseTensor(), N, C, L, M, K, s, d, padL, padR, outL, options, fuseRelu);
        return true;
    }

    /// <summary>
    /// Direct rank-three float depthwise convolution. Output positions run
    /// over a contiguous time axis, so stride/dilation one takes an AVX256
    /// FMA vector loop over an explicit no-check interior, with scalar
    /// border/tail handling; anything else stays scalar. Batches and channels
    /// are independent, so parallel degrees split over batch-channels with
    /// identical per-element results.
    /// </summary>
    static Tensor<float> RunDepthwise1DFloat(DenseTensor<float> x, DenseTensor<float> w, DenseTensor<float>? b, int N, int C, int L, int M, int K, int s, int d, int padL, int padR, int outL, TensorExecutionOptions options, bool fuseRelu)
    {
        var output = new DenseTensor<float>((ReadOnlySpan<int>)new int[] { N, M, outL });
        var xMem = x.Buffer;
        var wMem = w.Buffer;
        var oMem = output.Buffer;
        var bMem = b is null ? default : b.Buffer;
        bool hasBias = b is not null;
        bool vector = options.UseSimd && options.UseIntrinsics && s == 1 && d == 1 && Avx.IsSupported && Fma.IsSupported;
        int jobs = N * C;
        int dop = options.MaxDegreeOfParallelism < 2 || jobs < 2 ? 1 : Math.Min(options.MaxDegreeOfParallelism, jobs);
        if (dop > 1)
        {
            Parallel.For(0, jobs, new ParallelOptions { MaxDegreeOfParallelism = dop }, job =>
            {
                RunDepthwiseChannelFloat(xMem, wMem, bMem, hasBias, oMem, job / C, job % C, C, L, M, K, s, d, padL, outL, vector, fuseRelu);
            });
        }
        else
        {
            for (int job = 0; job < jobs; job++)
                RunDepthwiseChannelFloat(xMem, wMem, bMem, hasBias, oMem, job / C, job % C, C, L, M, K, s, d, padL, outL, vector, fuseRelu);
        }
        return output;
    }

    /// <summary>
    /// One batch-channel of direct depthwise convolution: bias-seeded
    /// accumulation over the K filter taps per output position, vectorized
    /// across the explicit interior for stride/dilation one, scalar with
    /// zero padding elsewhere.
    /// </summary>
    static void RunDepthwiseChannelFloat(Memory<float> xMem, Memory<float> wMem, Memory<float> bMem, bool hasBias, Memory<float> oMem, int b, int c, int C, int L, int M, int K, int s, int d, int padL, int outL, bool vector, bool fuseRelu)
    {
        var xs = xMem.Span;
        var ws = wMem.Span;
        var os = oMem.Span;
        int xBase = (b * C + c) * L;
        int wBase = c * K;
        int oBase = (b * M + c) * outL;
        float bi = hasBias ? bMem.Span[c] : 0f;
        int effK = (K - 1) * d + 1;
        int tLo = (padL + s - 1) / s;
        int hiNum = L - 1 + padL - (effK - 1);
        int tHi = (hiNum >= 0 ? hiNum / s : -1) + 1;
        if (tLo < 0) tLo = 0;
        if (tHi > outL) tHi = outL;
        if (vector)
        {
            var zero = Vector256<float>.Zero;
            int vecEnd = tLo + ((tHi - tLo) & ~7);
            unsafe
            {
                fixed (float* xp = xs, wp = ws, op = os)
                {
                    float* xc = xp + xBase;
                    float* wc = wp + wBase;
                    float* oc = op + oBase;
                    for (int t = tLo; t < vecEnd; t += 8)
                    {
                        var acc = Vector256.Create(bi);
                        for (int k = 0; k < K; k++)
                        {
                            var xv = *(Vector256<float>*)(xc + t + k - padL);
                            acc = Fma.MultiplyAdd(Vector256.Create(wc[k]), xv, acc);
                        }
                        if (fuseRelu)
                        {
                            var keep = Vector256.GreaterThan(acc, zero) | Vector256.Equals(acc, zero) | ~Vector256.Equals(acc, acc);
                            acc = Vector256.ConditionalSelect(keep, acc, zero);
                        }
                        *(Vector256<float>*)(oc + t) = acc;
                    }
                }
            }
            for (int t = 0; t < tLo; t++) os[oBase + t] = DepthwiseTapScalar(xs, ws, xBase, wBase, bi, t, K, s, d, padL, L, fuseRelu);
            for (int t = vecEnd; t < outL; t++)
            {
                if (t >= tHi) os[oBase + t] = DepthwiseTapScalar(xs, ws, xBase, wBase, bi, t, K, s, d, padL, L, fuseRelu);
                else
                {
                    float acc2 = bi;
                    for (int k = 0; k < K; k++) acc2 += ws[wBase + k] * xs[xBase + t + k - padL];
                    os[oBase + t] = fuseRelu && acc2 < 0f ? 0f : acc2;
                }
            }
            return;
        }
        for (int t = 0; t < outL; t++) os[oBase + t] = DepthwiseTapScalar(xs, ws, xBase, wBase, bi, t, K, s, d, padL, L, fuseRelu);
    }

    /// <summary>Scalar depthwise tap accumulation with explicit zero padding.</summary>
    static float DepthwiseTapScalar(ReadOnlySpan<float> xs, ReadOnlySpan<float> ws, int xBase, int wBase, float bi, int t, int K, int s, int d, int padL, int L, bool fuseRelu)
    {
        float acc = bi;
        for (int k = 0; k < K; k++)
        {
            int ix = t * s + k * d - padL;
            float xv = (uint)ix < (uint)L ? xs[xBase + ix] : 0f;
            acc += ws[wBase + k] * xv;
        }
        return fuseRelu && acc < 0f ? 0f : acc;
    }

    /// <summary>
    /// Runs 1x1 stride-1 no-pad batches with no patch matrix: the input slice
    /// already lays out as the GEMM right-hand side, so each group multiplies
    /// directly through the shared dispatcher with the same bias epilogue.
    /// </summary>
    static void RunPointwiseBatchesFloat(Memory<float> xMem, Memory<float> wMem, Memory<float> bMem, bool hasBias, Memory<float> oMem, int N, int group, int C, int H, int W, int M, int outH, int outW, int inBatch, int outBatch, TensorExecutionOptions options, bool fuseRelu)
    {
        int tileN = outH * outW;
        int tileM = M / group;
        int tileK = C / group;
        for (int b = 0; b < N; b++)
        {
            for (int g = 0; g < group; g++)
            {
                var wView = new DenseTensor<float>(wMem.Slice(g * tileM * tileK, tileM * tileK), new int[] { tileM, tileK });
                var pView = new DenseTensor<float>(xMem.Slice(b * inBatch + g * tileK * tileN, tileK * tileN), new int[] { tileK, tileN });
                var dView = new DenseTensor<float>(oMem.Slice(b * outBatch + g * tileM * tileN, tileM * tileN), new int[] { tileM, tileN });
                Tensor<float>.MatMul2D(wView, pView, dView, options);
            }
        }
        if (hasBias || fuseRelu)
        {
            var bs = bMem.Span;
            var os = oMem.Span;
            for (int i = 0; i < M; i++)
            {
                float bi = hasBias ? bs[i] : 0f;
                for (int n = 0; n < N; n++)
                {
                    int row = n * outBatch + i * tileN;
                    for (int j = 0; j < tileN; j++)
                    {
                        float v = hasBias ? os[row + j] + bi : os[row + j];
                        os[row + j] = fuseRelu && v < 0f ? 0f : v;
                    }
                }
            }
        }
    }

    public static Tensor<double> Conv2D(Tensor<double> input, Tensor<double> weight, int group, PadType padtype, int? padvalue, Tensor<double>? bias, int[]? kernelshape, int[]? strides, int[]? dilations) =>
        Conv2D(input, weight, group, padtype, padvalue, bias, kernelshape, strides, dilations, TensorExecutionOptions.Auto);

    /// <summary>Two-dimensional convolution with explicit execution options.</summary>
    public static Tensor<double> Conv2D(Tensor<double> input, Tensor<double> weight, int group, PadType padtype, int? padvalue, Tensor<double>? bias, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options)
    {
        var (N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW) = PlanConvPadType(input, weight, group, padtype, padvalue, kernelshape, strides, dilations, bias is null ? -1 : (int)bias.Length);
        return Conv2DDoubleCore(input, weight, group, N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, bias, options, false);

    }

    public static Tensor<double> Conv2D(Tensor<double> input, Tensor<double> weight, int group, int[] pads, Tensor<double>? bias, int[]? kernelshape, int[]? strides, int[]? dilations) =>
        Conv2D(input, weight, group, pads, bias, kernelshape, strides, dilations, TensorExecutionOptions.Auto);

    /// <summary>Two-dimensional convolution with explicit execution options.</summary>
    public static Tensor<double> Conv2D(Tensor<double> input, Tensor<double> weight, int group, int[] pads, Tensor<double>? bias, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options)
    {
        var (N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW) = PlanConvExplicit(input, weight, group, pads, kernelshape, strides, dilations, bias is null ? -1 : (int)bias.Length);
        return Conv2DDoubleCore(input, weight, group, N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, bias, options, false);

    }

    /// <summary>Two-dimensional convolution with explicit execution options and a fused ReLU epilogue.</summary>
    public static Tensor<double> Conv2D(Tensor<double> input, Tensor<double> weight, int group, PadType padtype, int? padvalue, Tensor<double>? bias, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options, bool fuseRelu)
    {
        var (dN, dC, dH, dW2, dM, dkH, dkW, ddH, ddW, dsH, dsW, dpad, doutH, doutW) = PlanConvPadType(input, weight, group, padtype, padvalue, kernelshape, strides, dilations, bias is null ? -1 : (int)bias.Length);
        return Conv2DDoubleCore(input, weight, group, dN, dC, dH, dW2, dM, dkH, dkW, ddH, ddW, dsH, dsW, dpad, doutH, doutW, bias, options, fuseRelu);
    }

    /// <summary>Two-dimensional convolution with explicit execution options and a fused ReLU epilogue.</summary>
    public static Tensor<double> Conv2D(Tensor<double> input, Tensor<double> weight, int group, int[] pads, Tensor<double>? bias, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options, bool fuseRelu)
    {
        var (qN, qC, qH, qW, qM, qkH, qkW, qdH, qdW, qsH, qsW, qpad, qoutH, qoutW) = PlanConvExplicit(input, weight, group, pads, kernelshape, strides, dilations, bias is null ? -1 : (int)bias.Length);
        return Conv2DDoubleCore(input, weight, group, qN, qC, qH, qW, qM, qkH, qkW, qdH, qdW, qsH, qsW, qpad, qoutH, qoutW, bias, options, fuseRelu);
    }

    static Tensor<double> Conv2DDoubleCore(Tensor<double> input, Tensor<double> weight, int group, int N, int C, int H, int W, int M, int kH, int kW, int dH, int dW, int sH, int sW, PadInfo pad, int outH, int outW, Tensor<double>? bias, TensorExecutionOptions options, bool fuseRelu)
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
                () => RentScratch<double>(patchSize, options),
                (b, state, scratch) =>
                {
                    RunConvBatchDouble(xMem, wMem, bMem, hasBias, oMem, scratch, patchSize, b, group, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, inBatch, outBatch, options, fuseRelu);
                    return scratch;
                },
                scratch => ArrayPool<double>.Shared.Return(scratch));
        }
        else
        {
            var scratch = RentScratch<double>(patchSize, options);
            try
            {
                for (int b = 0; b < N; b++)
                    RunConvBatchDouble(xMem, wMem, bMem, hasBias, oMem, scratch, patchSize, b, group, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, inBatch, outBatch, options, fuseRelu);
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
    static void RunConvBatchDouble(Memory<double> xMem, Memory<double> wMem, Memory<double> bMem, bool hasBias, Memory<double> oMem, double[] scratch, int patchSize, int b, int group, int C, int H, int W, int M, int kH, int kW, int dH, int dW, int sH, int sW, PadInfo pad, int outH, int outW, int inBatch, int outBatch, TensorExecutionOptions options, bool fuseRelu)
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
        // Fused ReLU epilogue: same pass, same max, bit-identical when unfused.
        if (hasBias || fuseRelu)
        {
            var bs = bMem.Span;
            var os = oMem.Span;
            for (int i = 0; i < M; i++)
            {
                double bi = hasBias ? bs[i] : 0.0;
                int row = b * outBatch + i * tileN;
                for (int j = 0; j < tileN; j++)
                {
                    double v = hasBias ? os[row + j] + bi : os[row + j];
                    os[row + j] = fuseRelu && v < 0.0 ? 0.0 : v;
                }
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
        if (kernelshape[0] <= 0 || kernelshape[1] <= 0)
        {
            throw new ArgumentException("MaxPool kernel dims must be positive.", nameof(kernelshape));
        }
        if (strides == null)
        {
            strides = new int[] { 1, 1 };
        }
        if (dilations == null)
        {
            dilations = new int[] { 1, 1 };
        }
        if (dilations[0] <= 0 || dilations[1] <= 0)
        {
            throw new ArgumentException("MaxPool dilations must be positive.", nameof(dilations));
        }
        if (strides[0] <= 0 || strides[1] <= 0)
        {
            throw new ArgumentException("MaxPool strides must be positive.", nameof(strides));
        }
        int N = input.Dims[0];
        int C = input.Dims[1];
        int H = input.Dims[2];
        int W = input.Dims[3];
        // Only the batch dimension may be zero (verified against ORT 1.29);
        // zero computed extents below still flow to empty outputs.
        if (C == 0 || H == 0 || W == 0)
        {
            throw new ArgumentException("MaxPool input channels and spatial dims must be non-zero; only the batch dimension may be zero.", nameof(input));
        }
        int kH = kernelshape[0];
        int kW = kernelshape[1];
        var info = GetConv2DOutputInfo(padtype, H, W, strides[0], strides[1], GetConv2DEffectiveFilterSize(kH, dilations[0]), GetConv2DEffectiveFilterSize(kW, dilations[1]), padvalue);
        // VALID sizes use the same truncating division as the explicit path
        // (verified against ORT 1.29; identical to the ceiling form except
        // for degenerate geometries, where zero flows to empty outputs).
        int validH = info.Shape[0], validW = info.Shape[1];
        if (padtype == PadType.Valid)
        {
            validH = (H - GetConv2DEffectiveFilterSize(kH, dilations[0])) / strides[0] + 1;
            validW = (W - GetConv2DEffectiveFilterSize(kW, dilations[1])) / strides[1] + 1;
        }
        return (N, C, H, W, kH, kW, strides[0], strides[1], dilations[0], dilations[1], info.PadInfo, validH, validW);
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
        if (kernelshape[0] <= 0 || kernelshape[1] <= 0)
        {
            throw new ArgumentException("MaxPool kernel dims must be positive.", nameof(kernelshape));
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
        if (dilations[0] <= 0 || dilations[1] <= 0)
        {
            throw new ArgumentException("MaxPool dilations must be positive.", nameof(dilations));
        }
        if (strides[0] <= 0 || strides[1] <= 0)
        {
            throw new ArgumentException("MaxPool strides must be positive.", nameof(strides));
        }
        int N = input.Dims[0];
        int C = input.Dims[1];
        int H = input.Dims[2];
        int W = input.Dims[3];
        // Only the batch dimension may be zero (verified against ORT 1.29);
        // zero computed extents below still flow to empty outputs.
        if (C == 0 || H == 0 || W == 0)
        {
            throw new ArgumentException("MaxPool input channels and spatial dims must be non-zero; only the batch dimension may be zero.", nameof(input));
        }
        int kH = kernelshape[0];
        int kW = kernelshape[1];
        int effKH = GetConv2DEffectiveFilterSize(kH, dilations[0]);
        int effKW = GetConv2DEffectiveFilterSize(kW, dilations[1]);
        var outShape = MaxPoolOutputShape(H, W, effKH, effKW, strides[0], strides[1], pads[0], pads[1], pads[0] + pads[2], pads[1] + pads[3], ceilMode);
        if (outShape[0] < 0 || outShape[1] < 0) throw new ArgumentException("MaxPool output spatial dims must be non-negative.");
        var pad = new PadInfo { top = pads[0], left = pads[1], bottom = pads[2], right = pads[3], h = pads[0] + pads[2], w = pads[1] + pads[3] };
        return (N, C, H, W, kH, kW, strides[0], strides[1], dilations[0], dilations[1], pad, outShape[0], outShape[1]);
    }

    public static Tensor<float> MaxPool2D(Tensor<float> input, int[] kernelshape, PadType padtype, int? padvalue, int[]? strides, int[]? dilations, bool ceilMode)
    {
        var (N, C, H, W, kH, kW, sH, sW, dH, dW, pad, outH, outW) = PlanPoolPadType(input, kernelshape, padtype, padvalue, strides, dilations);
        if (!ceilMode) return MaxPoolFloatCore(input, N, C, H, W, kH, kW, sH, sW, dH, dW, pad, outH, outW);
        int effKH = GetConv2DEffectiveFilterSize(kH, dH);
        int effKW = GetConv2DEffectiveFilterSize(kW, dW);
        var ceilShape = MaxPoolOutputShape(H, W, effKH, effKW, sH, sW, pad.top, pad.left, pad.top + pad.bottom, pad.left + pad.right, true);
        return MaxPoolFloatCore(input, N, C, H, W, kH, kW, sH, sW, dH, dW, pad, ceilShape[0], ceilShape[1]);
    }

    public static Tensor<float> MaxPool2D(Tensor<float> input, int[] kernelshape, int[] pads, int[]? strides, int[]? dilations, bool ceilMode)
    {
        var (N, C, H, W, kH, kW, sH, sW, dH, dW, pad, outH, outW) = PlanPoolExplicit(input, kernelshape, pads, strides, dilations, ceilMode);
        return MaxPoolFloatCore(input, N, C, H, W, kH, kW, sH, sW, dH, dW, pad, outH, outW);
    }

    static int[] MaxPoolOutputShape(int H, int W, int effKH, int effKW, int sH, int sW, int padBeginH, int padBeginW, int padH, int padW, bool ceilMode)
    {
        // Truncating integer division, not floor: verified against ORT 1.29
        // (degenerate (2-3)/2+1 is 1, not 0); identical to floor whenever the
        // numerator is non-negative. Ceil mode keeps the ceiling spelling,
        // then drops trailing windows that start in post-padding: the last
        // window must start strictly inside input plus begin-pad
        // ((out-1)*stride < dim+padBegin), verified across 11 ORT 1.29
        // geometries including asymmetric pads (begin-pad only), VALID trim,
        // dilations, and negative numerators (flow to empty, never below 0:
        // the loop condition is already false at out 0).
        int outH = ceilMode
            ? (int)Math.Ceiling((H + padH - effKH) / (float)sH) + 1
            : (H + padH - effKH) / sH + 1;
        int outW = ceilMode
            ? (int)Math.Ceiling((W + padW - effKW) / (float)sW) + 1
            : (W + padW - effKW) / sW + 1;
        if (ceilMode)
        {
            while (outH > 0 && ((long)outH - 1) * sH >= (long)H + padBeginH) outH--;
            while (outW > 0 && ((long)outW - 1) * sW >= (long)W + padBeginW) outW--;
        }
        return new int[] { outH, outW };
    }

    /// <summary>
    /// MaxPool over dense standard-stride spans with precomputed row offsets.
    /// Matches the indexer loop bit for bit (negative-maximum start, NaN never
    /// wins, padded cells skipped) with no per-element allocation.
    /// </summary>
    static void MaxPoolSpanFloat(System.Span<float> xs, System.Span<float> ys, int N, int C, int H, int W, int kH, int kW, int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, PadInfo pad, int outH, int outW)
    {
        for (int n = 0; n < N; n++)
            for (int d = 0; d < C; d++)
                for (int yR = 0; yR < outH; yR++)
                {
                    int xCorner = yR * strideHeight - pad.top;
                    for (int yC = 0; yC < outW; yC++)
                    {
                        int xCCorner = yC * strideWidth - pad.left;
                        float best = -float.MaxValue;
                        for (int tR = 0; tR < kH; tR++)
                        {
                            int xR = xCorner + tR * dilationHeight;
                            if ((uint)xR >= (uint)H) continue;
                            int rowBase = ((n * C + d) * H + xR) * W;
                            for (int tC = 0; tC < kW; tC++)
                            {
                                int xC = xCCorner + tC * dilationWidth;
                                if ((uint)xC >= (uint)W) continue;
                                float v = xs[rowBase + xC];
                                if (v > best) best = v;
                            }
                        }
                        ys[((n * C + d) * outH + yR) * outW + yC] = best;
                    }
                }
    }
    static Tensor<float> MaxPoolFloatCore(Tensor<float> input, int N, int C, int H, int W, int kH, int kW, int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, PadInfo pad, int outH, int outW)
    {
        var Y = DenseTensor<float>.OfShape(N, C, outH, outW);
        if (input is DenseTensor<float> dense && !dense.IsReversedStride && HasStandardStrides(dense) && dense.Buffer.Length == (int)dense.Length
            && Y.Buffer.Length == (int)Y.Length)
        {
            MaxPoolSpanFloat(dense.Buffer.Span, Y.Buffer.Span, N, C, H, W, kH, kW, strideHeight, strideWidth, dilationHeight, dilationWidth, pad, outH, outW);
            return Y;
        }

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
        var ceilShape = MaxPoolOutputShape(H, W, effKH, effKW, sH, sW, pad.top, pad.left, pad.top + pad.bottom, pad.left + pad.right, true);
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
