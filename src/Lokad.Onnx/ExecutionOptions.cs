namespace Lokad.Onnx;

public sealed record ExecutionOptions(OptimizationMode Optimization, TensorExecutionOptions Tensor)
{
    static readonly ExecutionOptions defaultOptions = new(OptimizationMode.Speed, TensorExecutionOptions.Auto);
    static readonly ExecutionOptions memoryOptions = new(OptimizationMode.Memory, TensorExecutionOptions.Auto);
    static readonly ExecutionOptions scalarOptions = new(OptimizationMode.Speed, TensorExecutionOptions.Scalar);
    static readonly ExecutionOptions simdOptions = new(OptimizationMode.Speed, TensorExecutionOptions.Simd);
    static readonly ExecutionOptions intrinsicsOptions = new(OptimizationMode.Speed, TensorExecutionOptions.Intrinsics);

    public static ExecutionOptions Default => defaultOptions;

    /// <summary>Uses automatic tensor dispatch and releases eligible dead intermediate views.</summary>
    /// <remarks>IntermediateOutputs entries may be cleared after their final use so their
    /// owned storage can be reused. Inputs, initializers, graph outputs and previously returned
    /// tensors remain valid. This policy does not impose a process memory ceiling.</remarks>
    public static ExecutionOptions Memory => memoryOptions;

    public static ExecutionOptions Scalar => scalarOptions;

    public static ExecutionOptions Simd => simdOptions;

    public static ExecutionOptions Intrinsics => intrinsicsOptions;

    /// <summary>Returns this instance after validating its tensor options.</summary>
    public ExecutionOptions Validated()
    {
        Tensor.Validate();
        return this;
    }
}
