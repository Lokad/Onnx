using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Lokad.Onnx;

internal static class DenseScalarWhere
{
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    internal static bool Try<T>(Tensor<bool> condition, Tensor<T> x,
        Tensor<T> y, out Tensor<T> output) where T : unmanaged
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

        // Truth is zero/nonzero, including noncanonical Boolean storage.
        var mask = MemoryMarshal.AsBytes(((DenseTensor<bool>)condition).Buffer.Span);
        bool chooseTrue = mask[0] != 0;
        bool uniform = chooseTrue ? mask.IndexOf((byte)0) < 0 : mask.IndexOfAnyExcept((byte)0) < 0;
        var result = new DenseTensor<T>(shape);
        T scalar = ((DenseTensor<T>)x).Buffer.Span[0];
        var values = ((DenseTensor<T>)y).Buffer.Span;
        if (uniform)
        {
            if (chooseTrue) result.Buffer.Span.Fill(scalar);
            else values.CopyTo(result.Buffer.Span);
        }
        else
        {
            SelectMixed(mask, scalar, values, result.Buffer.Span, shape, maskShape, condition.Strides);
        }
        output = result;
        return true;
    }

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    private static void SelectMixed<T>(ReadOnlySpan<byte> mask, T scalar,
        ReadOnlySpan<T> values, Span<T> destination, ReadOnlySpan<int> shape,
        ReadOnlySpan<int> maskShape, ReadOnlySpan<int> maskStrides) where T : unmanaged
    {
        int rank = shape.Length, offset = rank - maskShape.Length;
        Span<int> steps = stackalloc int[rank];
        Span<int> positions = stackalloc int[rank];
        positions.Clear();
        for (int axis = 0; axis < rank; axis++)
        {
            int sourceAxis = axis - offset;
            steps[axis] = sourceAxis < 0 || maskShape[sourceAxis] == 1 ? 0 : maskStrides[sourceAxis];
        }

        // Coalesce a trailing run whose condition is either scalar throughout
        // or contiguous. Outer coordinates advance once per run, not per value.
        int run = 1, outer = rank - 1;
        bool scalarRun = true, modeSet = false;
        for (; outer >= 0; outer--)
        {
            if (shape[outer] == 1) continue;
            bool axisScalar = steps[outer] == 0;
            if (modeSet && axisScalar != scalarRun) break;
            if (!axisScalar && steps[outer] != run) break;
            scalarRun = axisScalar;
            modeSet = true;
            run *= shape[outer];
        }

        int maskOffset = 0;
        for (int start = 0; start < destination.Length; start += run)
        {
            var target = destination.Slice(start, run);
            var source = values.Slice(start, run);
            if (scalarRun)
            {
                if (mask[maskOffset] != 0) target.Fill(scalar);
                else source.CopyTo(target);
            }
            else
            {
                var conditions = mask.Slice(maskOffset, run);
                for (int i = 0; i < run; i++)
                    target[i] = conditions[i] != 0 ? scalar : source[i];
            }

            for (int axis = outer; axis >= 0; axis--)
            {
                positions[axis]++;
                maskOffset += steps[axis];
                if (positions[axis] < shape[axis]) break;
                positions[axis] = 0;
                maskOffset -= steps[axis] * shape[axis];
            }
        }
    }
}
