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

    /// <summary>
    /// Bias-add fused with exact GELU: runs the single-pass span kernel when
    /// the data is dense float32 of rank one or more in standard row-major layout
    /// with a rank-one float bias matching the last dimension, else the legacy
    /// two-step path with identical values (scalar and reversed-layout inputs
    /// must not reach the pointer kernel). Bitwise
    /// identity on the hot path holds by construction (identical add, then the
    /// identical erf call and combine order per element).
    /// </summary>
    public static OpResult BiasGelu(ITensor? X, ITensor? Bias, ExecutionOptions? options, TensorBufferPool? pool)
    {
        var op = OpType.BiasGelu;
        if (X is null) return MissingInput(op, nameof(X));
        if (Bias is null) return MissingInput(op, nameof(Bias));
        var opts = (options ?? ExecutionOptions.Default).Validated();
        if (X.ElementType == TensorElementType.Float && Bias.ElementType == TensorElementType.Float
            && X is DenseTensor<float> xd && xd.Rank > 0 && !xd.IsReversedStride
            && xd.strides.SequenceEqual(ArrayUtilities.GetStrides(xd.dimensions))
            && xd.Buffer.Length == (int)xd.Length
            && Bias is DenseTensor<float> bd && bd.Rank == 1 && bd.Buffer.Length == (int)bd.Length)
        {
            int lastDim = xd.Dimensions[^1];
            if (lastDim > 0 && bd.Length == lastDim)
            {
                DenseTensor<float> output = pool is null
                    ? DenseTensor<float>.OfShape(xd.Dimensions.ToArray())
                    : new DenseTensor<float>(new Memory<float>(pool.Rent<float>((int)xd.Length)), xd.Dimensions.ToArray());
                if (UseGeluTanhTrial())
                    Tensor<float>.BiasGeluTanhSpanFloat(xd.Buffer.Span, bd.Buffer.Span, output.Buffer.Span);
                else if (AblationSwitches.EnableBiasGeluInline)
                    Tensor<float>.BiasGeluSpanFloatInline(xd.Buffer.Span, bd.Buffer.Span, output.Buffer.Span);
                else
                    Tensor<float>.BiasGeluSpanFloatPtr4x(xd.Buffer.Span, bd.Buffer.Span, output.Buffer.Span);
                return Success(op, output);
            }
        }
        var add = Add(X, Bias, options, pool);
        if (add.Status != OpStatus.Success || add.Outputs is null || add.Outputs.Length != 1 || add.Outputs[0] is null)
            return add;
        var gelu = Gelu(add.Outputs[0], null, options, pool);
        if (gelu.Status != OpStatus.Success || gelu.Outputs is null)
            return gelu;
        return Success(op, gelu.Outputs);
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
