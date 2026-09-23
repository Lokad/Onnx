using System;
using System.Runtime.CompilerServices;

namespace Lokad.Onnx;

public abstract partial class Tensor<T> where T : unmanaged
{
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static bool TryUniformScalarWhere(Tensor<bool> condition, Tensor<T> x,
        Tensor<T> y, out Tensor<T> output)
    {
        output = null!;
        // Exact classes exclude subclasses with different GetValue semantics.
        // Dense memory windows are supported; their spans respect the offset.
        if (condition is null || x is null || y is null
            || condition.GetType() != typeof(DenseTensor<bool>)
            || x.GetType() != typeof(DenseTensor<T>) || y.GetType() != typeof(DenseTensor<T>)
            || condition.IsReversedStride || x.IsReversedStride || y.IsReversedStride
            || x.Length != 1 || y.Rank < 1 || y.Rank > 8
            || x.Rank > y.Rank || condition.Rank > y.Rank)
            return false;

        // A one-element x has only singleton axes. Both broadcasts must leave
        // y's shape unchanged, even when the condition selects only one input.
        var shape = y.Dimensions;
        var maskShape = condition.Dimensions;
        int offset = shape.Length - maskShape.Length;
        for (int axis = 0; axis < maskShape.Length; axis++)
            if (maskShape[axis] != 1 && maskShape[axis] != shape[offset + axis])
                return false;

        if (y.Length == 0)
        {
            output = new DenseTensor<T>(shape);
            return true;
        }

        var mask = ((DenseTensor<bool>)condition).Buffer.Span;
        bool chooseTrue = mask[0];
        if (mask.IndexOf(!chooseTrue) >= 0)
            return false;

        // Always return independent storage, including an all-false mask.
        // Copy/Fill select bits without floating-point arithmetic or merging.
        var result = new DenseTensor<T>(shape);
        if (chooseTrue)
            result.Buffer.Span.Fill(((DenseTensor<T>)x).Buffer.Span[0]);
        else
            ((DenseTensor<T>)y).Buffer.Span.CopyTo(result.Buffer.Span);
        output = result;
        return true;
    }
}
