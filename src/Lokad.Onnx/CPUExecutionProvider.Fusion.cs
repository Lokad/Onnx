namespace Lokad.Onnx;

using System;
using System.Linq;

using static OpResult;
public partial class CPUExecutionProvider
{
    /// <summary>
    /// Convolution with fused Relu epilogue: runs the existing Conv path
    /// unchanged, then applies the existing float/double span Relu core in
    /// place on the fresh output. Bitwise identity with the two-node form
    /// holds by construction (same computation, same per-element function);
    /// exotic layouts fall back to the allocating public Relu path.
    /// </summary>
    public static OpResult ConvRelu(ITensor? X, ITensor? W, ITensor? B, string? auto_pad, int[]? dilations, int? group, int[]? kernel_shape, int[]? pads, int[]? strides, ExecutionOptions? options)
    {
        var op = OpType.ConvRelu;
        var inner = Conv(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides, options);
        if (inner.Status != OpStatus.Success || inner.Outputs is null || inner.Outputs.Length != 1 || inner.Outputs[0] is null)
            return inner;
        return FinishRelu(op, inner.Outputs[0], options);
    }

    /// <summary>
    /// Add with fused Relu epilogue: same delegation and in-place contract as ConvRelu.
    /// </summary>
    public static OpResult AddRelu(ITensor? A, ITensor? B, ExecutionOptions? options, TensorBufferPool? pool)
    {
        var op = OpType.AddRelu;
        var inner = Add(A, B, options, pool);
        if (inner.Status != OpStatus.Success || inner.Outputs is null || inner.Outputs.Length != 1 || inner.Outputs[0] is null)
            return inner;
        return FinishRelu(op, inner.Outputs[0], options);
    }

    static OpResult FinishRelu(OpType op, ITensor output, ExecutionOptions? options)
    {
        if (output is DenseTensor<float> df && !df.IsReversedStride
            && df.strides.SequenceEqual(ArrayUtilities.GetStrides(df.dimensions))
            && df.Buffer.Length == (int)df.Length)
        {
            Tensor<float>.ReluSpanFloat(df.Buffer.Span, df.Buffer.Span);
            return Success(op, output);
        }
        if (output is DenseTensor<double> dd && !dd.IsReversedStride
            && dd.strides.SequenceEqual(ArrayUtilities.GetStrides(dd.dimensions))
            && dd.Buffer.Length == (int)dd.Length)
        {
            Tensor<double>.ReluSpanDouble(dd.Buffer.Span, dd.Buffer.Span);
            return Success(op, output);
        }
        var relu = Relu(output, options);
        if (relu.Status != OpStatus.Success || relu.Outputs is null)
            return relu;
        return Success(op, relu.Outputs);
    }
}
