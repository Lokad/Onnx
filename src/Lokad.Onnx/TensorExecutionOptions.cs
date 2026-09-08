using System;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

/// <summary>MaxDegreeOfParallelism caps worker threads for batch-parallel and row-split kernels (1 keeps the historical sequential path). Values above the available splits are clamped by the kernel; only explicit values above 1 opt in.</summary>
public readonly record struct TensorExecutionOptions(bool UseSimd, bool UseIntrinsics, int MaxDegreeOfParallelism)
{
    public static TensorExecutionOptions Scalar => new TensorExecutionOptions(false, false, 1);

    public static TensorExecutionOptions Simd => new TensorExecutionOptions(true, false, 1);

    public static TensorExecutionOptions Intrinsics => new TensorExecutionOptions(true, true, 1);

    public static TensorExecutionOptions Parallel(int maxDegreeOfParallelism) =>
        maxDegreeOfParallelism < 1
            ? throw new ArgumentOutOfRangeException(nameof(maxDegreeOfParallelism), "Parallelism must request at least 1 worker.")
            : new TensorExecutionOptions(true, true, maxDegreeOfParallelism);

    public static TensorExecutionOptions Auto => new TensorExecutionOptions(true, Fma.IsSupported, 1);

    public void Validate()
    {
        if (MaxDegreeOfParallelism < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(MaxDegreeOfParallelism), "Parallelism must request at least 1 worker.");
        }
        if (UseIntrinsics && !UseSimd)
        {
            throw new ArgumentException("Tensor intrinsics require SIMD mode.", nameof(UseIntrinsics));
        }
        if (UseIntrinsics && !Fma.IsSupported)
        {
            throw new InvalidOperationException("Tensor intrinsics were explicitly requested but x86 FMA is not supported on this machine.");
        }
    }
}
