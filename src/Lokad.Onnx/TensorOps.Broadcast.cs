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
        if (inA.dimensions.SequenceEqual(inB.dimensions))
        {
            return [inA, inB];
        }
        else if (inA.Rank == 0 && inB.Rank != 0)
        {
            var _A = inB.CloneEmpty();
            for (int i = 0; i < _A.Length; i++)
            {
                _A.SetValue(i, inA.GetValue(0));
            }
            return [_A, inB ];
        }
        else if (inB.Rank == 0 && inA.Rank != 0)
        {
            var _B = inA.CloneEmpty();
            for (int i = 0; i < _B.Length; i++)
            {
                _B.SetValue(i, inB.GetValue(0));
            }
            return [inA, _B ];
        }

        var broadcastRank = Math.Max(inA.Rank, inB.Rank);
        var outA = inA;
        var outB = inB;
        for (var i = 0; i < broadcastRank; i++)
        {
            var idxA = i - broadcastRank + inA.Rank;
            var idxB = i - broadcastRank + inB.Rank;
            if (i < broadcastRank - inA.Rank)
            {
                outA = outA.InsertDim(i);
                outA = outA.BroadcastDim(i, inB.Dimensions[idxB]);
            }
            else if (i < broadcastRank - inB.Rank)
            {
                outB = outB.InsertDim(i);
                outB = outB.BroadcastDim(i, inA.Dimensions[idxA]);
            }
            else if (inA.Dimensions[idxA] == inB.Dimensions[idxB])
            {
                continue;
            }
            else if (inA.Dimensions[idxA] == 1)
            {
                outA = outA.BroadcastDim(i, inB.Dimensions[idxB]);
            }
            else if (inB.Dimensions[idxB] == 1)
            {
                outB = outB.BroadcastDim(i, inA.Dimensions[idxA]);
            }
            else
            {
                return Array.Empty<Tensor<T>>();
            }
        }
        return [outA, outB ];
    }

    public static bool Broadcast(Tensor<T> x, Tensor<T> y, [NotNullWhen(true)] out Tensor<T>? outx, [NotNullWhen(true)] out Tensor<T>? outy)
    {
        var b = Broadcast(x, y);
        if (b.Length == 0)
        {
            outx = null;
            outy = null;
            return false;
        }
        else
        {
            outx = b[0];
            outy = b[1];
            return true;
        }
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
    public static bool BroadcastShape(ReadOnlySpan<int> x, ReadOnlySpan<int> y, [NotNullWhen(true)] out int[]? b)
    {
        int rank = Math.Max(x.Length, y.Length);
        var dims = new int[rank];
        int ox = rank - x.Length;
        int oy = rank - y.Length;
        for (int i = 0; i < rank; i++)
        {
            int dx = i < ox ? 1 : x[i - ox];
            int dy = i < oy ? 1 : y[i - oy];
            if (dx == dy) dims[i] = dx;
            else if (dx == 1) dims[i] = dy;
            else if (dy == 1) dims[i] = dx;
            else
            {
                b = null;
                return false;
            }
        }
        b = dims;
        return true;
    }

    public static bool BroadcastShape(Tensor<T> x, Tensor<T> y, [NotNullWhen(true)] out int[]? b) => BroadcastShape(x.Dimensions, y.Dimensions, out b);

    /// <summary>
    /// Elementwise binary operation with NumPy-style broadcasting, writing a new
    /// tensor. Scalar operands are read directly, dense operands stream through
    /// contiguous runs, and any other layout uses a stride-based general path;
    /// no expanded operand is ever materialized. The options are validated. The
    /// destination is overwritten; aliasing it with either source is allowed
    /// because every element is read before it is written.
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
        if (!destination.Dimensions.SequenceEqual(shape)) throw new ArgumentException(nameof(destination), "Destination shape must match the broadcast shape.");
        if (destination is not DenseTensor<T> dense || !HasStandardStrides(dense)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
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
        if (!destination.Dimensions.SequenceEqual(shape)) throw new ArgumentException(nameof(destination), "Destination shape must match the broadcast shape.");
        if (destination is not DenseTensor<T> dense || !HasStandardStrides(dense)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        BroadcastApplyFuncInto(other, op, dense, shape);
        return dense;
    }

    /// <summary>Specialized scalar, contiguous-run and stride-based broadcast tiers.</summary>
    void BroadcastApplyInto<TOp>(Tensor<T> other, DenseTensor<T> destination, TensorExecutionOptions options, int[] shape) where TOp : struct, IBroadcastOperator<T>
    {
        if (other.Rank == 0)
        {
            var scalar = other.GetValue(0);
            if (options.UseSimd && this is DenseTensor<T> xd0 && IsDirectSpanCompatible(xd0) && IsDirectSpanCompatible(destination))
            {
                var sv = new Vector<T>(scalar);
                var xv = MemoryMarshal.Cast<T, Vector<T>>(xd0.Buffer.Span);
                var dv = MemoryMarshal.Cast<T, Vector<T>>(destination.Buffer.Span);
                int k = 0;
                int kend = Math.Min(xv.Length, dv.Length);
                for (; k < kend; k++) dv[k] = TOp.Vector(xv[k], sv);
                int done = k * Vector<T>.Count;
                var xs = xd0.Buffer.Span;
                var ds = destination.Buffer.Span;
                for (int i = done; i < (int)Length; i++) ds[i] = TOp.Scalar(xs[i], scalar);
                return;
            }
            for (int i = 0; i < Length; i++) destination.SetValue(i, TOp.Scalar(GetValue(i), scalar));
            return;
        }
        if (Rank == 0)
        {
            var scalar = GetValue(0);
            if (options.UseSimd && other is DenseTensor<T> yd0 && IsDirectSpanCompatible(yd0) && IsDirectSpanCompatible(destination))
            {
                var sv = new Vector<T>(scalar);
                var yv = MemoryMarshal.Cast<T, Vector<T>>(yd0.Buffer.Span);
                var dv = MemoryMarshal.Cast<T, Vector<T>>(destination.Buffer.Span);
                int k = 0;
                int kend = Math.Min(yv.Length, dv.Length);
                for (; k < kend; k++) dv[k] = TOp.Vector(sv, yv[k]);
                int done = k * Vector<T>.Count;
                var ys = yd0.Buffer.Span;
                var ds = destination.Buffer.Span;
                for (int i = done; i < (int)other.Length; i++) ds[i] = TOp.Scalar(scalar, ys[i]);
                return;
            }
            for (int i = 0; i < other.Length; i++) destination.SetValue(i, TOp.Scalar(scalar, other.GetValue(i)));
            return;
        }
        BroadcastApplyRuns<TOp>(this, other, destination, options, shape, null);
    }

    /// <summary>Shared run-tier engine used by the specialized and fallback paths.</summary>
    void BroadcastApplyRuns<TOp>(Tensor<T> left, Tensor<T> right, DenseTensor<T> destination, TensorExecutionOptions options, int[] shape, Func<T, T, T>? sop) where TOp : struct, IBroadcastOperator<T>
    {
        var xd = left as DenseTensor<T> is { IsReversedStride: false } ownX ? ownX : left.ToDenseTensor();
        var yd = right as DenseTensor<T> is { IsReversedStride: false } ownY ? ownY : right.ToDenseTensor();
        int rank = shape.Length;
        var xs = xd.Buffer.Span;
        var ys = yd.Buffer.Span;
        var ds = destination.Buffer.Span;
        // Right-aligned span steps; 0 marks a reused (broadcast or missing) dim.
        var xSteps = new int[rank];
        var ySteps = new int[rank];
        int sx = 1, sy = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            int xDim = d < rank - xd.Rank ? 1 : xd.Dimensions[d - (rank - xd.Rank)];
            int yDim = d < rank - yd.Rank ? 1 : yd.Dimensions[d - (rank - yd.Rank)];
            xSteps[d] = xDim == 1 ? 0 : sx;
            ySteps[d] = yDim == 1 ? 0 : sy;
            if (xDim != 1) sx *= xDim;
            if (yDim != 1) sy *= yDim;
        }
        // Longest trailing span where both sides advance one element per step,
        // so the inner loop pairs positions directly (and vectorizes).
        int inner = 1;
        int od = rank - 1;
        for (; od >= 0; od--)
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
            int rem = b, ox = 0, oy = 0;
            for (int d = od; d >= 0; d--)
            {
                int q = shape[d];
                int c = q == 0 ? 0 : rem % q;
                rem = q == 0 ? rem : rem / q;
                ox += c * xSteps[d];
                oy += c * ySteps[d];
            }
            int dstBase = b * inner;
            if (vectorize)
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

    /// <summary>Delegate-based run engine for operators without a vector form.</summary>
    void BroadcastApplyFuncInto(Tensor<T> other, Func<T, T, T> op, DenseTensor<T> destination, int[] shape)
    {
        if (other.Rank == 0)
        {
            Apply(v => op(v, other.GetValue(0)), destination);
            return;
        }
        if (Rank == 0)
        {
            var scalar = GetValue(0);
            other.Apply(v => op(scalar, v), destination);
            return;
        }
        BroadcastApplyRuns<NoBroadcastOperator<T>>(this, other, destination, TensorExecutionOptions.Scalar, shape, op);
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
            throw new ArgumentException(nameof(targetShape), "Target shape has fewer dimensions than the input tensor.");
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
            if (dim < 0 || (dim == 0 && paddedInputDims[i] != 0))
            {
                throw new ArgumentException(nameof(targetShape), "Target shape dimensions must be positive or -1.");
            }
            normalizedShape[i] = dim;
        }

        StartOpStage(OpStage.CalculateIndices);
        var result = input;
        for (int i = 0; i < offset; i++)
        {
            result = result.InsertDim(0);
        }
        for (int i = 0; i < targetRank; i++)
        {
            var dim = result.Dimensions[i];
            var targetDim = normalizedShape[i];
            if (dim == targetDim)
            {
                continue;
            }
            if (dim == 1)
            {
                result = result.BroadcastDim(i, targetDim);
                continue;
            }
            throw new ArgumentException(nameof(targetShape), $"Cannot broadcast dimension {dim} to {targetDim}.");
        }
        return result;
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
            if (targetDim == -1)
            {
                targetDim = inputDim;
            }
            else if (targetDim == 1 && inputDim > 1)
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
            throw new ArgumentException(nameof(targetShape), $"Cannot broadcast dimension {inputDim} to {targetDim}.");
        }

        return result;
    }
}
