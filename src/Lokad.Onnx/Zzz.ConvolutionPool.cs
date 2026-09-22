namespace Lokad.Onnx;

using static Lokad.Onnx.MathOps;

// Compile new overloads after existing partial-class methods to keep the
// isolated candidate's unrelated compiler-generated method identities stable.
public partial class CPUExecutionProvider
{
    public static OpResult Conv(ITensor? X, ITensor? W, ITensor? B, string? auto_pad, int[]? dilations, int? group, int[]? kernel_shape, int[]? pads, int[]? strides, ExecutionOptions? options) =>
        Conv(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides, options, null);

    public static OpResult ConvRelu(ITensor? X, ITensor? W, ITensor? B, string? auto_pad, int[]? dilations, int? group, int[]? kernel_shape, int[]? pads, int[]? strides, ExecutionOptions? options) =>
        ConvRelu(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides, options, null);
}

public abstract partial class Tensor<T> where T : unmanaged
{
    internal static Tensor<float> Conv2D(Tensor<float> input, Tensor<float> weight, int group, PadType padtype, int? padvalue, Tensor<float>? bias, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options, TensorBufferPool? pool)
    {
        var (N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW) = PlanConvPadType(input, weight, group, padtype, padvalue, kernelshape, strides, dilations, bias is null ? -1 : (int)bias.Length);
        return Conv2DFloatCore(input, weight, group, N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, bias, options, pool);

    }

    internal static Tensor<float> Conv2D(Tensor<float> input, Tensor<float> weight, int group, int[] pads, Tensor<float>? bias, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options, TensorBufferPool? pool)
    {
        var (N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW) = PlanConvExplicit(input, weight, group, pads, kernelshape, strides, dilations, bias is null ? -1 : (int)bias.Length);
        return Conv2DFloatCore(input, weight, group, N, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, bias, options, pool);

    }
}
