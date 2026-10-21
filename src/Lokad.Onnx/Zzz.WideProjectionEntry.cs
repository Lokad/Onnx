using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;

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
    static Tensor<float> RunWideProjectionMatMul2DCore(Tensor<float> x, Tensor<float> y, DenseTensor<float> destination, TensorExecutionOptions options, bool clearDestination) =>
        MatMul2DCore(x, y, destination, options, clearDestination, allowOwnedPacking: true);
}
