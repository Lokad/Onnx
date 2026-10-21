namespace Lokad.Onnx;

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

using static Lokad.Onnx.MathOps;
using static Lokad.Onnx.Profiler;

public abstract partial class Tensor<T> : TensorBase, IList, IList<T>, IReadOnlyList<T>, IStructuralComparable, IStructuralEquatable, ITensor, INumericTensor
where T : unmanaged
{
    public static Tensor<T>[] Broadcast(Tensor<T> inA, Tensor<T> inB)
    {
        return Broadcast(inA, inB, out var left, out var right)
            ? new[] { left, right } : Array.Empty<Tensor<T>>();
    }

    public static bool Broadcast(Tensor<T> x, Tensor<T> y, [NotNullWhen(true)] out Tensor<T>? outx, [NotNullWhen(true)] out Tensor<T>? outy)
    {
        if (ITensor.TryBroadcast(x, x.Dimensions, y, y.Dimensions, out var left, out var right))
        {
            outx = (Tensor<T>)left;
            outy = (Tensor<T>)right;
            return true;
        }
        outx = outy = null;
        return false;
    }

    /// <summary>
    /// Broadcasts a tensor against a target shape span, building only view
    /// metadata (no element storage). Shares compatibility with <see cref="BroadcastShape"/>.
    /// </summary>
    public static bool Broadcast(Tensor<T> x, ReadOnlySpan<int> y, [NotNullWhen(true)] out Tensor<T>? bx)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (!BroadcastShape(x.Dimensions, y, out var shape))
        {
            bx = null;
            return false;
        }
        bx = BroadcastTo(x, shape);
        return true;
    }

    /// <summary>
    /// Computes right-aligned broadcast compatibility over dimension spans only.
    /// Allocates in proportion to rank, never element count.
    /// </summary>
    public static bool BroadcastShape(ReadOnlySpan<int> x, ReadOnlySpan<int> y, [NotNullWhen(true)] out int[]? b) =>
        ArrayUtilities.TryBroadcastShape(x, y, out b);

    public static bool BroadcastShape(Tensor<T> x, Tensor<T> y, [NotNullWhen(true)] out int[]? b) => BroadcastShape(x.Dimensions, y.Dimensions, out b);

    /// <summary>
    /// Elementwise binary operation with NumPy-style broadcasting, writing a new
    /// tensor. Scalar operands are read directly, dense operands stream through
    /// contiguous runs, and any other layout uses a stride-based general path;
    /// no expanded operand is ever materialized. The options are validated. The
    /// destination is overwritten; hazardous source overlap is snapshotted
    /// before writes, while identical dense element mappings run in place.
    /// </summary>
    public Tensor<T> BroadcastApply<TOp>(Tensor<T> other, TensorExecutionOptions options) where TOp : struct, IBroadcastOperator<T>
    {
        options.Validate();
        if (other is null) throw new ArgumentNullException(nameof(other));
        if (!BroadcastShape(Dimensions, other.Dimensions, out var shape)) throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
        var output = DenseTensor<T>.OfShape(shape);
        BroadcastApplyInto<TOp>(other, output, options, shape);
        return output;
    }

    public Tensor<T> BroadcastApply<TOp>(Tensor<T> other) where TOp : struct, IBroadcastOperator<T> =>
        BroadcastApply<TOp>(other, TensorExecutionOptions.Auto);

    /// <summary>
    /// Elementwise binary operation with NumPy-style broadcasting into an existing
    /// dense destination, which must match the broadcast shape. Same contract as
    /// the allocating overload.
    /// </summary>
    public Tensor<T> BroadcastApply<TOp>(Tensor<T> other, Tensor<T> destination, TensorExecutionOptions options) where TOp : struct, IBroadcastOperator<T>
    {
        options.Validate();
        if (other is null) throw new ArgumentNullException(nameof(other));
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        if (!BroadcastShape(Dimensions, other.Dimensions, out var shape)) throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
        if (!destination.Dimensions.SequenceEqual(shape)) throw new ArgumentException("Destination shape must match the broadcast shape.", nameof(destination));
        if (destination is not DenseTensor<T> dense || !HasStandardStrides(dense)) throw new ArgumentException("Destination must have standard row-major strides.", nameof(destination));
        BroadcastApplyInto<TOp>(other, dense, options, shape);
        return dense;
    }

    /// <summary>
    /// Scalar-function fallback for operators without a vector form (such as
    /// power). Same broadcast contract as the specialized overloads.
    /// </summary>
    public Tensor<T> BroadcastApply(Tensor<T> other, Func<T, T, T> op, TensorExecutionOptions options)
    {
        options.Validate();
        if (other is null) throw new ArgumentNullException(nameof(other));
        if (op is null) throw new ArgumentNullException(nameof(op));
        if (!BroadcastShape(Dimensions, other.Dimensions, out var shape)) throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
        var output = DenseTensor<T>.OfShape(shape);
        BroadcastApplyFuncInto(other, op, output, shape);
        return output;
    }

    public Tensor<T> BroadcastApply(Tensor<T> other, Func<T, T, T> op) => BroadcastApply(other, op, TensorExecutionOptions.Auto);

    public Tensor<T> BroadcastApply(Tensor<T> other, Func<T, T, T> op, Tensor<T> destination, TensorExecutionOptions options)
    {
        options.Validate();
        if (other is null) throw new ArgumentNullException(nameof(other));
        if (op is null) throw new ArgumentNullException(nameof(op));
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        if (!BroadcastShape(Dimensions, other.Dimensions, out var shape)) throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
        if (!destination.Dimensions.SequenceEqual(shape)) throw new ArgumentException("Destination shape must match the broadcast shape.", nameof(destination));
        if (destination is not DenseTensor<T> dense || !HasStandardStrides(dense)) throw new ArgumentException("Destination must have standard row-major strides.", nameof(destination));
        BroadcastApplyFuncInto(other, op, dense, shape);
        return dense;
    }

    /// <summary>Specialized scalar, contiguous-run and stride-based broadcast tiers.</summary>
    /// <remarks>Internal callers must validate options and broadcast compatibility,
    /// and supply a standard row-major destination matching the prepared shape.</remarks>
    internal void BroadcastApplyInto<TOp>(Tensor<T> other, DenseTensor<T> destination, TensorExecutionOptions options, int[] shape) where TOp : struct, IBroadcastOperator<T>
    {
        var left = PreserveElementwiseSource(this, destination);
        var right = ReferenceEquals(this, other) ? left : PreserveElementwiseSource(other, destination);
        left.BroadcastApplyIntoCore<TOp>(right, destination, options, shape);
    }

    void BroadcastApplyIntoCore<TOp>(Tensor<T> other, DenseTensor<T> destination, TensorExecutionOptions options, int[] shape) where TOp : struct, IBroadcastOperator<T>
    {
        if (destination.Length == 0) return;
        if (other.Length == 1)
        {
            BroadcastScalarInto<TOp>(this, other.GetValue(0), destination, options, false);
            return;
        }
        if (Length == 1)
        {
            BroadcastScalarInto<TOp>(other, GetValue(0), destination, options, true);
            return;
        }
        BroadcastApplyRuns<TOp>(this, other, destination, options, shape, null);
    }

    static void BroadcastScalarInto<TOp>(Tensor<T> input, T scalar, DenseTensor<T> destination,
        TensorExecutionOptions options, bool scalarOnLeft) where TOp : struct, IBroadcastOperator<T>
    {
        if (options.UseSimd && !input.IsReversedStride && IsDirectSpanCompatible(input))
        {
            ApplyBroadcastScalarRow<TOp>(((DenseTensor<T>)input).Buffer.Span, scalar, destination.Buffer.Span, scalarOnLeft);
            return;
        }
        for (int i = 0; i < destination.Length; i++)
        {
            T value = input.GetLogicalValue(i);
            destination.SetValue(i, scalarOnLeft ? TOp.Scalar(scalar, value) : TOp.Scalar(value, scalar));
        }
    }

    // Only built-in view chains promise an affine mapping into the root buffer.
    static bool TryElementwiseLayout(Tensor<T> input, out Memory<T> storage, out int origin, out int[] steps)
    {
        storage = default;
        origin = 0;
        steps = Array.Empty<int>();
        if (input.GetType() == typeof(DenseTensor<T>))
        {
            storage = ((DenseTensor<T>)input).Buffer;
            steps = input.strides;
            return true;
        }
        var owner = input;
        while (owner.GetType() != typeof(DenseTensor<T>))
        {
            if (owner.GetType() == typeof(TensorSlice<T>)) owner = ((TensorSlice<T>)owner).parent;
            else if (owner.GetType() == typeof(BroadcastedTensor<T>)) owner = ((BroadcastedTensor<T>)owner).source;
            else return false;
        }
        storage = ((DenseTensor<T>)owner).Buffer;
        var coordinates = new int[input.Rank];
        steps = new int[input.Rank];
        origin = input.GetStorageIndex(coordinates);
        for (int axis = 0; axis < input.Rank; axis++)
        {
            if (input.dimensions[axis] <= 1) continue;
            coordinates[axis] = 1;
            steps[axis] = input.GetStorageIndex(coordinates) - origin;
            coordinates[axis] = 0;
        }
        return true;
    }

    static void BroadcastApplyIndexed<TOp>(Tensor<T> left, Tensor<T> right, DenseTensor<T> destination,
        int[] shape, Func<T, T, T>? operation) where TOp : struct, IBroadcastOperator<T>
    {
        var leftCoordinates = new int[left.Rank];
        var rightCoordinates = new int[right.Rank];
        int position = 0;
        foreach (var coordinate in new TensorDimensionsIterator(shape))
        {
            for (int axis = 0; axis < left.Rank; axis++)
                leftCoordinates[axis] = left.dimensions[axis] == 1 ? 0 : coordinate[shape.Length - left.Rank + axis];
            for (int axis = 0; axis < right.Rank; axis++)
                rightCoordinates[axis] = right.dimensions[axis] == 1 ? 0 : coordinate[shape.Length - right.Rank + axis];
            T x = left[leftCoordinates], y = right[rightCoordinates];
            destination.SetValue(position++, operation is null ? TOp.Scalar(x, y) : operation(x, y));
        }
    }

    /// <summary>Shared run-tier engine used by the specialized and fallback paths.</summary>
    void BroadcastApplyRuns<TOp>(Tensor<T> left, Tensor<T> right, DenseTensor<T> destination, TensorExecutionOptions options, int[] shape, Func<T, T, T>? sop) where TOp : struct, IBroadcastOperator<T>
    {
        if (destination.Length == 0) return;
        if (!TryElementwiseLayout(left, out var xStorage, out int xOrigin, out var xStrides)
            || !TryElementwiseLayout(right, out var yStorage, out int yOrigin, out var yStrides))
        {
            BroadcastApplyIndexed<TOp>(left, right, destination, shape, sop);
            return;
        }
        int rank = shape.Length;
        var xs = xStorage.Span;
        var ys = yStorage.Span;
        var ds = destination.Buffer.Span;
        var xSteps = new int[rank];
        var ySteps = new int[rank];
        for (int d = rank - 1; d >= 0; d--)
        {
            int xAxis = d - (rank - left.Rank), yAxis = d - (rank - right.Rank);
            xSteps[d] = xAxis < 0 || left.dimensions[xAxis] == 1 ? 0 : xStrides[xAxis];
            ySteps[d] = yAxis < 0 || right.dimensions[yAxis] == 1 ? 0 : yStrides[yAxis];
        }
        // Row normalization broadcasts one value across the contiguous final
        // axis. Compute outer coordinates once per row and broadcast that value
        // in SIMD registers instead of doing coordinate divisions per element.
        // Keep overlapping buffers on the existing path.
        bool rowScalar = sop is null && options.UseSimd && rank > 0 && shape[rank - 1] >= Vector<T>.Count &&
            ((xSteps[rank - 1] == 0 && ySteps[rank - 1] == 1) || (xSteps[rank - 1] == 1 && ySteps[rank - 1] == 0)) &&
            !xs.Overlaps(ds) && !ys.Overlaps(ds);
        // Longest trailing span where both sides advance one element per step,
        // so the inner loop pairs positions directly (and vectorizes).
        int inner = rowScalar ? shape[rank - 1] : 1;
        int od = rowScalar ? rank - 2 : rank - 1;
        for (; !rowScalar && od >= 0; od--)
        {
            if (shape[od] <= 1) continue;
            if (xSteps[od] != inner || ySteps[od] != inner) break;
            inner *= shape[od];
        }
        bool vectorize = sop is null && options.UseSimd && inner >= Vector<T>.Count;
        int total = (int)destination.Length;
        int blocks = total == 0 ? 0 : total / inner;
        for (int b = 0; b < blocks; b++)
        {
            int rem = b, ox = xOrigin, oy = yOrigin;
            for (int d = od; d >= 0; d--)
            {
                int q = shape[d];
                int c = q == 0 ? 0 : rem % q;
                rem = q == 0 ? rem : rem / q;
                ox += c * xSteps[d];
                oy += c * ySteps[d];
            }
            int dstBase = b * inner;
            if (rowScalar)
            {
                if (xSteps[rank - 1] == 0)
                    ApplyBroadcastScalarRow<TOp>(ys.Slice(oy, inner), xs[ox], ds.Slice(dstBase, inner), true);
                else
                    ApplyBroadcastScalarRow<TOp>(xs.Slice(ox, inner), ys[oy], ds.Slice(dstBase, inner), false);
            }
            else if (vectorize)
            {
                var xv = MemoryMarshal.Cast<T, Vector<T>>(xs.Slice(ox, inner));
                var yv = MemoryMarshal.Cast<T, Vector<T>>(ys.Slice(oy, inner));
                var dv = MemoryMarshal.Cast<T, Vector<T>>(ds.Slice(dstBase, inner));
                int k = 0;
                int kend = Math.Min(xv.Length, Math.Min(yv.Length, dv.Length));
                for (; k < kend; k++) dv[k] = TOp.Vector(xv[k], yv[k]);
                int done = k * Vector<T>.Count;
                for (int i = done; i < inner; i++) ds[dstBase + i] = TOp.Scalar(xs[ox + i], ys[oy + i]);
            }
            else if (sop is null)
            {
                for (int i = 0; i < inner; i++) ds[dstBase + i] = TOp.Scalar(xs[ox + i], ys[oy + i]);
            }
            else
            {
                for (int i = 0; i < inner; i++) ds[dstBase + i] = sop(xs[ox + i], ys[oy + i]);
            }
        }
    }

    static void ApplyBroadcastScalarRow<TOp>(ReadOnlySpan<T> row, T scalar, Span<T> destination, bool scalarOnLeft)
        where TOp : struct, IBroadcastOperator<T>
    {
        var rv = MemoryMarshal.Cast<T, Vector<T>>(row);
        var dv = MemoryMarshal.Cast<T, Vector<T>>(destination);
        var sv = new Vector<T>(scalar);
        int done = rv.Length * Vector<T>.Count;
        if (scalarOnLeft)
        {
            for (int i = 0; i < rv.Length; i++) dv[i] = TOp.Vector(sv, rv[i]);
            for (int i = done; i < row.Length; i++) destination[i] = TOp.Scalar(scalar, row[i]);
        }
        else
        {
            for (int i = 0; i < rv.Length; i++) dv[i] = TOp.Vector(rv[i], sv);
            for (int i = done; i < row.Length; i++) destination[i] = TOp.Scalar(row[i], scalar);
        }
    }

    /// <summary>Delegate-based run engine for operators without a vector form.</summary>
    void BroadcastApplyFuncInto(Tensor<T> other, Func<T, T, T> op, DenseTensor<T> destination, int[] shape)
    {
        var left = PreserveElementwiseSource(this, destination);
        var right = ReferenceEquals(this, other) ? left : PreserveElementwiseSource(other, destination);
        if (right.Rank == 0)
        {
            var scalar = right.GetValue(0);
            for (int i = 0; i < left.Length; i++) destination.SetValue(i, op(left.GetLogicalValue(i), scalar));
            return;
        }
        if (left.Rank == 0)
        {
            var scalar = left.GetValue(0);
            for (int i = 0; i < right.Length; i++) destination.SetValue(i, op(scalar, right.GetLogicalValue(i)));
            return;
        }
        BroadcastApplyRuns<NoBroadcastOperator<T>>(left, right, destination, TensorExecutionOptions.Scalar, shape, op);
    }

    /// <summary>Placeholder operator selecting the delegate path of the run engine.</summary>
    readonly struct NoBroadcastOperator<TElement> : IBroadcastOperator<TElement> where TElement : unmanaged
    {
        public static TElement Scalar(TElement left, TElement right) => throw new NotSupportedException();
        public static Vector<TElement> Vector(Vector<TElement> left, Vector<TElement> right) => throw new NotSupportedException();
    }


    public static Tensor<T> BroadcastTo(Tensor<T> input, int[] targetShape)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (targetShape is null) throw new ArgumentNullException(nameof(targetShape));

        var targetRank = targetShape.Length;
        if (input.Rank > targetRank)
        {
            throw new ArgumentException("Target shape has fewer dimensions than the input tensor.", nameof(targetShape));
        }

        var paddedInputDims = new int[targetRank];
        var offset = targetRank - input.Rank;
        for (int i = 0; i < targetRank; i++)
        {
            paddedInputDims[i] = i < offset ? 1 : input.Dimensions[i - offset];
        }

        var normalizedShape = new int[targetRank];
        for (int i = 0; i < targetRank; i++)
        {
            var dim = targetShape[i];
            if (dim == -1)
            {
                dim = paddedInputDims[i];
            }
            // Zero extents broadcast from 0/1 extents to empty views (NumPy/ONNX:
            // e.g. MatMul batch [] over [0] yields empty, not an error). Other
            // mismatches still throw below in the per-dimension loop.
            if (dim < 0 || (dim == 0 && paddedInputDims[i] != 0 && paddedInputDims[i] != 1))
            {
                throw new ArgumentException("Target shape dimensions must be positive or -1.", nameof(targetShape));
            }
            normalizedShape[i] = dim;
        }

        StartOpStage(OpStage.CalculateIndices);
        for (int i = 0; i < targetRank; i++)
            if (paddedInputDims[i] != normalizedShape[i] && paddedInputDims[i] != 1)
                throw new ArgumentException($"Cannot broadcast dimension {paddedInputDims[i]} to {normalizedShape[i]}.", nameof(targetShape));
        return (Tensor<T>)ITensor.BroadcastView(input, input.Dimensions, normalizedShape);
    }

    /// <summary>
    /// Broadcasts data to the target shape following ONNX broadcast rules.
    /// A target dimension of 1 preserves the input dimension; added dimensions are stride-zero views.
    /// </summary>
    public static Tensor<T> Expand(Tensor<T> data, int[] targetShape)
    {
        // Broadcast is view-based: added dimensions are stride-zero views, so Expand itself allocates no element storage.
        StartOpStage(OpStage.ValidateArguments);
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (targetShape is null) throw new ArgumentNullException(nameof(targetShape));
        // Right-aligned broadcasting: a shorter shape aligns to the trailing
        // dimensions (verified against ORT 1.29: [2,3] data with [3] shape
        // yields [2,3]; an empty shape is the identity). Only the local binding
        // is replaced; the caller array is never mutated.
        if (targetShape.Length < data.Rank)
        {
            var padded = new int[data.Rank];
            for (int i = 0; i < padded.Length - targetShape.Length; i++) padded[i] = 1;
            targetShape.CopyTo(padded, padded.Length - targetShape.Length);
            targetShape = padded;
        }
        var targetRank = targetShape.Length;

        StartOpStage(OpStage.CalculateIndices);
        var result = data;
        var offset = targetRank - data.Rank;
        for (int i = 0; i < offset; i++)
        {
            result = result.InsertDim(0);
        }

        for (int i = 0; i < targetRank; i++)
        {
            var inputDim = result.Dimensions[i];
            var targetDim = targetShape[i];
            // Negative dims fail like the native engine (ORT 1.29); -1 is not
            // a keep marker here, unlike Reshape.
            if (targetDim < 0) throw new ArgumentException("Expand shape dimensions must be non-negative.", nameof(targetShape));
            if (targetDim == 1 && inputDim > 1)
            {
                // Keep-dim on target 1: variable-size models (verified on the DINOv2 position-embedding interpolation path at 224 and 518 pixels) emit shape masks that collapse to 1 for dimensions that must be preserved, so 1 means keep here rather than broadcast.
                targetDim = inputDim;
            }

            if (inputDim == targetDim)
            {
                continue;
            }
            if (inputDim == 1)
            {
                result = result.BroadcastDim(i, targetDim);
                continue;
            }
            throw new ArgumentException($"Cannot broadcast dimension {inputDim} to {targetDim}.", nameof(targetShape));
        }

        return result;
    }
}
