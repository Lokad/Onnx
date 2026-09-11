using System;
using System.Runtime.Intrinsics.X86;
using System.Threading;

namespace Lokad.Onnx;

/// <summary>Selects the kernel paths execution may use. SIMD enables portable-vectorized elementwise, reduction, and softmax paths with scalar fallback; intrinsics additionally enables x86-intrinsic paths and requires SIMD with x86 FMA (see Validate). MaxDegreeOfParallelism caps worker threads for batch-parallel and row-split kernels (1 keeps the sequential path). Values above the available splits are clamped by the kernel; only explicit values above 1 opt in.</summary>
/// <summary>Collects transient kernel scratch bytes (im2col patches, GEMM panel packing) for one execution.</summary>
/// <remarks>Implementations must be thread-safe: parallel kernel workers report concurrently.
/// A null reporter (the default) disables accounting with a single branch per rent.</remarks>
public interface IScratchAccountant
{
    void AddScratchBytes(long bytes);
    long TotalScratchBytes { get; }
}

/// <summary>Thread-safe scratch-byte accumulator for one execution.</summary>
public sealed class ScratchAccountant : IScratchAccountant
{
    long total;
    public long TotalScratchBytes => Interlocked.Read(ref total);
    public void AddScratchBytes(long bytes) => Interlocked.Add(ref total, bytes);
}

/// <summary>Collects tensor-copy bytes (view materialization) for one execution.</summary>
/// <remarks>Implementations must be thread-safe: parallel kernel workers report concurrently.
/// A null reporter (the default) disables accounting with a single branch per copy.</remarks>
public interface ICopyAccountant
{
    void AddCopyBytes(long bytes);
    long TotalCopyBytes { get; }
}

/// <summary>Thread-safe copy-byte accumulator for one execution.</summary>
public sealed class CopyAccountant : ICopyAccountant
{
    long total;
    public long TotalCopyBytes => Interlocked.Read(ref total);
    public void AddCopyBytes(long bytes) => Interlocked.Add(ref total, bytes);
}

public readonly record struct TensorExecutionOptions(bool UseSimd, bool UseIntrinsics, int MaxDegreeOfParallelism)
{
    /// <summary>Optional per-run scratch-byte sink; null disables accounting.</summary>
    public IScratchAccountant? ScratchReporter { get; init; }

    /// <summary>Optional per-run copy-byte sink; null disables accounting.</summary>
    public ICopyAccountant? CopyReporter { get; init; }
    /// <summary>Optional prepared packed-MatMul map; null disables packed routing.</summary>
    internal IReadOnlyDictionary<float[], PackedMatMulWeight>? PackedMatMulWeights { get; init; }

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
