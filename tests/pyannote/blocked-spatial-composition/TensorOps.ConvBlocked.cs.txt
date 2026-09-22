namespace Lokad.Onnx;

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;

using static Lokad.Onnx.MathOps;

public abstract partial class Tensor<T>
where T : unmanaged
{
    internal const long ConvBlockedScratchBytes = 64L * 1024 * 1024;

    internal static bool PlanConvBlockedScratch(int c, int m, int h, int w, int oh, int ow, out int input, out int output)
    {
        input = output = 0;
        if (c < 1 || m < 1 || h < 1 || w < 1 || oh < 1 || ow < 1) return false;
        try
        {
            long a = checked((long)c * ((long)h + 2) * ((long)w + 2));
            long b = checked((long)m * oh * ow);
            if (a > ConvBlockedScratchBytes / sizeof(float) - b) return false;
            input = (int)a; output = (int)b; return true;
        }
        catch (OverflowException) { return false; }
    }

    static bool TryConvBlockedSpatial(DenseTensor<float> input, Tensor<float> sourceWeight,
        DenseTensor<float> denseWeight, DenseTensor<float>? bias, DenseTensor<float> output,
        int group, int n, int c, int h, int w, int m, int kh, int kw, int dh, int dw,
        int sh, int sw, PadInfo pad, int oh, int ow, TensorExecutionOptions options)
    {
        if (!options.UseIntrinsics || !options.UseSimd || options.UseSegmentedConvolution
            || AblationSwitches.EnableSegmentedConvolution || n != 1 || group != 1
            || kh != 3 || kw != 3 || dh != 1 || dw != 1 || sh != sw || sh is not (1 or 2)
            || pad.top != 1 || pad.bottom != 1 || pad.left != 1 || pad.right != 1
            || h < 1 || w < 1 || c < 16 || c % 16 != 0 || m < 32 || m % 16 != 0
            || !GraphConvPacking.Standard(input) || !GraphConvPacking.Standard(denseWeight)
            || !GraphConvPacking.Standard(output) || bias is not null && !GraphConvPacking.Standard(bias)) return false;
        int lanes = GraphConvPacking.Lanes;
        if (lanes == 0 || GraphConvPacking.Resolve(options.PackedConvWeights, sourceWeight, lanes) is not float[] prepared) return false;
        if (!PlanConvBlockedScratch(c, m, h, w, oh, ow, out int inputCount, out int outputCount)) return false;
        float[]? packedInput = null, packedOutput = null;
        try
        {
            packedInput = ArrayPool<float>.Shared.Rent(inputCount);
            options.ScratchReporter?.AddScratchBytes((long)inputCount * sizeof(float));
            packedOutput = ArrayPool<float>.Shared.Rent(outputCount);
            options.ScratchReporter?.AddScratchBytes((long)outputCount * sizeof(float));
            return ConvBlockedSpatial.Execute(input.Buffer.Span, prepared,
                bias is null ? default : bias.Buffer.Span, default, output.Buffer.Span,
                packedInput, packedOutput, c, m, h, w, sh, lanes, false);
        }
        finally
        {
            if (packedOutput is not null) ArrayPool<float>.Shared.Return(packedOutput);
            if (packedInput is not null) ArrayPool<float>.Shared.Return(packedInput);
        }
    }
}
