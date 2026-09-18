namespace Lokad.Onnx;

public sealed record ExecutionOptions(OptimizationMode Optimization, TensorExecutionOptions Tensor)
{
    public static ExecutionOptions Default => new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Auto);

    /// <summary>Uses automatic tensor dispatch and releases eligible dead intermediate views.</summary>
    /// <remarks>IntermediateOutputs entries may be cleared after their final use so their
    /// owned storage can be reused. Inputs, initializers, graph outputs and previously returned
    /// tensors remain valid. This policy does not impose a process memory ceiling.</remarks>
    public static ExecutionOptions Memory => new ExecutionOptions(OptimizationMode.Memory, TensorExecutionOptions.Auto);

    public static ExecutionOptions Scalar => new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Scalar);

    public static ExecutionOptions Simd => new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Simd);

    public static ExecutionOptions Intrinsics => new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Intrinsics);

    /// <summary>Returns this instance after validating its tensor options.</summary>
    public ExecutionOptions Validated()
    {
        Tensor.Validate();
        return this;
    }
}
