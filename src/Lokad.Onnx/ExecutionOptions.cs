namespace Lokad.Onnx;

public sealed record ExecutionOptions(OptimizationMode Optimization, TensorExecutionOptions Tensor)
{
    public static ExecutionOptions Default => new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Auto);

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
