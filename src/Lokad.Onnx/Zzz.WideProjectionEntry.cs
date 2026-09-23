using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using static Lokad.Onnx.Profiler;

namespace Lokad.Onnx;

public abstract partial class Tensor<T> where T : unmanaged
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Tensor<float> DispatchWideProjectionMatMul2DCore(Tensor<float> x, Tensor<float> y, DenseTensor<float> destination, TensorExecutionOptions options, bool clearDestination)
    {
        // Preserve the original validation path for malformed or smaller inputs.
        if (x is not null && y is not null && x.Rank == 2 && y.Rank == 2
            && x.Dimensions[0] >= 48 && x.Dimensions[1] >= 1024 && y.Dimensions[1] >= 1024
            && (long)x.Dimensions[1] * y.Dimensions[1] <= 67108864
            && options.UseSimd && options.UseIntrinsics && Fma.IsSupported)
            return RunWideProjectionMatMul2DCore(x, y, destination, options, clearDestination);
        return MatMul2DCore(x, y, destination, options, clearDestination);
    }

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    static Tensor<float> RunWideProjectionMatMul2DCore(Tensor<float> x, Tensor<float> y, DenseTensor<float> destination, TensorExecutionOptions options, bool clearDestination)
    {
        options.Validate();
        StartOpStage(OpStage.ValidateArguments);
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException($"The number of columns in the first matrix ({x.Dimensions[1]}) is not equal to the number of rows in the second matrix ({y.Dimensions[0]}).");
        if (destination.Dimensions.Length != 2 || destination.Dimensions[0] != x.Dimensions[0] || destination.Dimensions[1] != y.Dimensions[1]) throw new ArgumentException(nameof(destination), "Destination shape must match the matrix product shape.");
        if (!HasStandardStrides(destination)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        if (ReferenceEquals(destination, x) || ReferenceEquals(destination, y) || TensorAlias.SharesBackingMemory(destination, x) || TensorAlias.SharesBackingMemory(destination, y)) throw new ArgumentException(nameof(destination), "Destination must not alias the input matrices.");
        if (clearDestination) destination.Buffer.Span.Clear();
        var m = x.Dimensions[0];
        var n = x.Dimensions[1];
        var k = y.Dimensions[1];

        if (ResolvePackedKernel(options, y, m) is { } packedB)
        {
            var dx = RequireContiguous(x, nameof(x), options.CopyReporter);
            StartOpStage(OpStage.Math);
            using var xh = dx.Buffer.Pin();
            using var ph = packedB.Buffer.Pin();
            using var oh = destination.Buffer.Pin();
            unsafe
            {
                RunPreparedPackedRows(m, n, k, (float*)xh.Pointer, (float*)ph.Pointer, (float*)oh.Pointer);
            }
            return destination;
        }

        var (_x, _y) = DensifyFloatOperands(x, y, options.CopyReporter);

        StartOpStage(OpStage.Math);
        int rowDop = options.MaxDegreeOfParallelism < 2 || m < 64
            ? 1
            : Math.Min(options.MaxDegreeOfParallelism, m);
        if (rowDop > 1)
        {
            int chunk = (m + rowDop - 1) / rowDop;
            Parallel.For(0, rowDop, new ParallelOptions { MaxDegreeOfParallelism = rowDop }, w =>
            {
                int start = w * chunk;
                int rows = Math.Min(chunk, m - start);
                if (rows <= 0) return;
                using var xh = _x.Buffer.Pin();
                using var yh = _y.Buffer.Pin();
                using var oh = destination.Buffer.Pin();
                unsafe
                {
                    RunIsolatedShortWideKernel(rows, n, k,
                        (float*)xh.Pointer + start * n,
                        (float*)yh.Pointer,
                        (float*)oh.Pointer + start * k, options);
                }
            });
        }
        else
        {
            using var xh = _x.Buffer.Pin();
            using var yh = _y.Buffer.Pin();
            using var oh = destination.Buffer.Pin();
            unsafe
            {
                RunIsolatedShortWideKernel(m, n, k, (float*)xh.Pointer, (float*)yh.Pointer, (float*)oh.Pointer, options);
            }
        }
        return destination;
    }
}
