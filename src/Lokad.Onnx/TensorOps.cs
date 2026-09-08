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

public abstract partial class Tensor<T> : TensorBase, IList, IList<T>, IReadOnlyList<T>, IStructuralComparable, IStructuralEquatable, ITensor
where T : unmanaged
{
    /// <summary>
    /// Checks an elementwise destination: it must exist and hold exactly one
    /// slot per source element. The destination is fully overwritten; it may
    /// alias the sources because every write pairs flat index to flat index.
    /// </summary>
    void CheckDestination(Tensor<T> destination)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        if (destination.Length != Length) throw new ArgumentException("Destination length must match the source length.", nameof(destination));
    }

    void CheckBinaryDestination(Tensor<T> tensor2, Tensor<T> destination)
    {
        if (tensor2 is null) throw new ArgumentNullException(nameof(tensor2));
        if (tensor2.Length != Length) throw new ArgumentException("Second operand length must match the source length.", nameof(tensor2));
        CheckDestination(destination);
    }

    /// <summary>
    /// True when the whole logical contents are addressable as one flat span,
    /// which is what the vector fast paths require. Dense backing always matches
    /// the logical length by construction; anything else uses the index path.
    /// </summary>
    static bool IsDirectSpanCompatible(Tensor<T> tensor) =>
        tensor is DenseTensor<T> dense && dense.Buffer.Length == dense.Length;

    /// <summary>
    /// Writes op(element) for every element, overwriting the destination.
    /// The destination may alias this tensor; pairing is by flat index.
    /// </summary>
    public virtual void Apply(Func<T, T> op, Tensor<T> destination)
    {
        CheckDestination(destination);
  
        for (int index = 0; index < Length; index++)
        {
            destination.SetValue(index, op(GetValue(index)));
        }        
    }

    public Tensor<T> Apply(Func<T, T> op)
    {
        var output = CloneEmpty();
        Apply(op, output);
        return output;
    }

    public virtual void VectorizedApply(Func<Vector<T>, Vector<T>> op, Func<T, T> sop, Tensor<T> destination)
        => VectorizedApply(op, sop, destination, TensorExecutionOptions.Auto);

    /// <summary>
    /// Writes op(element) for every element, overwriting the destination.
    /// The destination may alias this tensor; pairing is by flat index.
    /// The vector path runs only on directly spannable storage.
    /// </summary>
    public virtual void VectorizedApply(Func<Vector<T>, Vector<T>> op, Func<T, T> sop, Tensor<T> destination, TensorExecutionOptions options)
    {
        CheckDestination(destination);

        if (options.UseSimd && this is DenseTensor<T> d1 && destination is DenseTensor<T> d2
            && IsDirectSpanCompatible(this) && IsDirectSpanCompatible(destination))
        {
            var vspan1 = MemoryMarshal.Cast<T, Vector<T>>(d1.Buffer.Span);
            var vspan2 = MemoryMarshal.Cast<T, Vector<T>>(d2.Buffer.Span);
            var ceiling = (Convert.ToInt32(this.length) / Vector<T>.Count) * Vector<T>.Count;
            for (int i = 0; i < vspan1.Length; i++)
            {
                vspan2[i] = op(vspan1[i]);
            }
            for (int i = ceiling; i < this.length; i++)
            {
                destination.SetValue(i, sop(GetValue(i)));
            }
        }
        else
        {
            Apply(sop, destination);
        }
    }

    public Tensor<T> VectorizedApply(Func<Vector<T>, Vector<T>> op, Func<T, T> sop)
    {
        var output = CloneEmpty();
        VectorizedApply(op, sop, output);
        return output;
    }

    public Tensor<T> VectorizedApply(Func<Vector<T>, Vector<T>> op, Func<T, T> sop, TensorExecutionOptions options)
    {
        var output = CloneEmpty();
        VectorizedApply(op, sop, output, options);
        return output;
    }

    /// <summary>
    /// Writes op(left, right) for every element pair, overwriting the destination.
    /// The destination may alias either source; pairing is by flat index.
    /// </summary>
    public virtual void Apply(Func<T, T, T> op, Tensor<T> tensor2, Tensor<T> destination)
    {
        CheckBinaryDestination(tensor2, destination);

        for (int index = 0; index < this.Length; index++)
        {
            destination.SetValue(index, op(GetValue(index), tensor2.GetValue(index)));
        }        
    }

    public Tensor<T> Apply(Func<T, T, T> op, Tensor<T> tensor2)
    {
        var output = CloneEmpty();
        Apply(op, tensor2, output);
        return output;
    }

    public virtual void VectorizedApply(Func<Vector<T>, Vector<T>, Vector<T>> op, Func<T, T, T> sop, Tensor<T> tensor2, Tensor<T> destination)
        => VectorizedApply(op, sop, tensor2, destination, TensorExecutionOptions.Auto);

    /// <summary>
    /// Writes op(left, right) for every element pair, overwriting the destination.
    /// The destination may alias either source; pairing is by flat index.
    /// The vector path runs only on directly spannable storage.
    /// </summary>
    public virtual void VectorizedApply(Func<Vector<T>, Vector<T>, Vector<T>> op, Func<T, T, T> sop, Tensor<T> tensor2, Tensor<T> destination, TensorExecutionOptions options)
    {
        CheckBinaryDestination(tensor2, destination);

        if (options.UseSimd && this is DenseTensor<T> d1 && tensor2 is DenseTensor<T> d2 && destination is DenseTensor<T> d3
            && IsDirectSpanCompatible(this) && IsDirectSpanCompatible(tensor2) && IsDirectSpanCompatible(destination))
        {
            var vspan1 = MemoryMarshal.Cast<T, Vector<T>>(d1.Buffer.Span);
            var vspan2 = MemoryMarshal.Cast<T, Vector<T>>(d2.Buffer.Span);
            var vspan3 = MemoryMarshal.Cast<T, Vector<T>>(d3.Buffer.Span);
            var ceiling = (Convert.ToInt32(this.length) / Vector<T>.Count) * Vector<T>.Count;
            for (int i = 0; i < vspan1.Length; i++)
            {
                vspan3[i] = op(vspan1[i], vspan2[i]);
            }
            for (int i = ceiling; i < this.length; i++)
            {
                destination.SetValue(i, sop(GetValue(i), tensor2.GetValue(i)));
            }
        }
        else
        {
            Apply(sop, tensor2, destination);
        }
    }

    public virtual Tensor<T> VectorizedApply(Func<Vector<T>, Vector<T>, Vector<T>> op, Func<T, T, T> sop, Tensor<T> tensor2)
    {
        var output = CloneEmpty();
        VectorizedApply(op, sop, tensor2, output);
        return output;
    }

    public virtual Tensor<T> VectorizedApply(Func<Vector<T>, Vector<T>, Vector<T>> op, Func<T, T, T> sop, Tensor<T> tensor2, TensorExecutionOptions options)
    {
        var output = CloneEmpty();
        VectorizedApply(op, sop, tensor2, output, options);
        return output;
    }

    public virtual T Accumulate(Func<T, T, T> op, T state)
    {
        var result = state;
        for (int index = 0; index < Length; index++)
        {
            result = op(result, GetValue(index));
        }
        return result;
    }

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
            if (dim < 1)
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

    public static Tensor<bool> Equal(Tensor<T> x, Tensor<T> y)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (!Broadcast(x, y, out var bx, out var by))
        {
            throw new ArgumentException("Inputs are not broadcastable.");
        }
        var output = DenseTensor<bool>.OfShape(bx.Dimensions.ToArray());
        // ONNX numeric semantics: NaN equals nothing, including itself. The generic
        // EqualityComparer treats NaN as equal to NaN, so floats use == (verified
        // against ORT 1.29: NaN==NaN is false, +0==-0 is true).
        if (typeof(T) == typeof(float))
        {
            var fx = (Tensor<float>)(object)bx;
            var fy = (Tensor<float>)(object)by;
            for (int i = 0; i < output.Length; i++)
            {
                output.SetValue(i, fx.GetValue(i) == fy.GetValue(i));
            }
            return output;
        }
        if (typeof(T) == typeof(double))
        {
            var dx = (Tensor<double>)(object)bx;
            var dy = (Tensor<double>)(object)by;
            for (int i = 0; i < output.Length; i++)
            {
                output.SetValue(i, dx.GetValue(i) == dy.GetValue(i));
            }
            return output;
        }
        for (int i = 0; i < output.Length; i++)
        {
            output.SetValue(i, EqualityComparer<T>.Default.Equals(bx.GetValue(i), by.GetValue(i)));
        }
        return output;
    }

    public static Tensor<bool> Less(Tensor<T> x, Tensor<T> y)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (!Broadcast(x, y, out var bx, out var by))
        {
            throw new ArgumentException("Inputs are not broadcastable.");
        }
        var output = DenseTensor<bool>.OfShape(bx.Dimensions.ToArray());
        // ONNX numeric semantics: ordered comparisons with NaN are always false.
        // Comparer<T> orders NaN below every value, so floats use < (verified
        // against ORT 1.29: NaN<0 and 0<NaN are both false).
        if (typeof(T) == typeof(float))
        {
            var fx = (Tensor<float>)(object)bx;
            var fy = (Tensor<float>)(object)by;
            for (int i = 0; i < output.Length; i++)
            {
                output.SetValue(i, fx.GetValue(i) < fy.GetValue(i));
            }
            return output;
        }
        if (typeof(T) == typeof(double))
        {
            var dx = (Tensor<double>)(object)bx;
            var dy = (Tensor<double>)(object)by;
            for (int i = 0; i < output.Length; i++)
            {
                output.SetValue(i, dx.GetValue(i) < dy.GetValue(i));
            }
            return output;
        }
        for (int i = 0; i < output.Length; i++)
        {
            output.SetValue(i, Comparer<T>.Default.Compare(bx.GetValue(i), by.GetValue(i)) < 0);
        }
        return output;
    }

    public static Tensor<T> Where(Tensor<bool> condition, Tensor<T> x, Tensor<T> y)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (!Broadcast(x, y, out var bx, out var by))
        {
            throw new ArgumentException("Inputs are not broadcastable.");
        }
        var bcond = Tensor<bool>.BroadcastTo(condition, bx.Dimensions.ToArray());
        var output = bx.CloneEmpty();
        for (int i = 0; i < output.Length; i++)
        {
            output.SetValue(i, bcond.GetValue(i) ? bx.GetValue(i) : by.GetValue(i));
        }
        return output;
    }

    public static Tensor<byte> Add(Tensor<byte> x, Tensor<byte> y) => Add(x, y, TensorExecutionOptions.Auto);
    public static Tensor<byte> Add(Tensor<byte> x, Tensor<byte> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => (l + r), (l, r) => (byte) (l + r), y, options);

    public static Tensor<byte> Add(Tensor<byte> x, byte y) => x.Apply(l => (byte)(l + y));

    public static Tensor<int> Add(Tensor<int> x, Tensor<int> y) => Add(x, y, TensorExecutionOptions.Auto);
    public static Tensor<int> Add(Tensor<int> x, Tensor<int> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l + r, (l, r) => l + r, y, options);

    public static Tensor<int> Add(Tensor<int> x, int y) => Add(x, y, TensorExecutionOptions.Auto);
    public static Tensor<int> Add(Tensor<int> x, int y, TensorExecutionOptions options) => x.VectorizedApply(l => l + new Vector<int>(y), l => l + y, options);

    public static Tensor<float> Add(Tensor<float> x, Tensor<float> y) => Add(x, y, TensorExecutionOptions.Auto);
    public static Tensor<float> Add(Tensor<float> x, Tensor<float> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l + r, (l, r) => l + r, y, options);

    /// <summary>Writes the float sum into an existing destination tensor.</summary>
    public static Tensor<float> Add(Tensor<float> x, Tensor<float> y, Tensor<float> destination, TensorExecutionOptions options)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        return x.BroadcastApply<AddBroadcast<float>>(y, destination, options);
    }

    public static Tensor<float> Add(Tensor<float> x, float y) => Add(x, y, TensorExecutionOptions.Auto);
    public static Tensor<float> Add(Tensor<float> x, float y, TensorExecutionOptions options) => x.VectorizedApply(l => l + new Vector<float>(y), l => l + y, options);

    public static Tensor<double> Add(Tensor<double> x, Tensor<double> y) => Add(x, y, TensorExecutionOptions.Auto);
    public static Tensor<double> Add(Tensor<double> x, Tensor<double> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l + r, (l, r) => l + r, y, options);

    public static Tensor<double> Add(Tensor<double> x, double y) => Add(x, y, TensorExecutionOptions.Auto);
    public static Tensor<double> Add(Tensor<double> x, double y, TensorExecutionOptions options) => x.VectorizedApply(l => l + new Vector<double>(y), l => l + y, options);

    public static Tensor<byte> Subtract(Tensor<byte> x, Tensor<byte> y) => Subtract(x, y, TensorExecutionOptions.Auto);
    public static Tensor<byte> Subtract(Tensor<byte> x, Tensor<byte> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => (l - r), (l, r) => (byte)(l - r), y, options);

    public static Tensor<byte> Subtract(Tensor<byte> x, byte y) => x.Apply(l => (byte)(l - y));

    public static Tensor<int> Subtract(Tensor<int> x, Tensor<int> y) => Subtract(x, y, TensorExecutionOptions.Auto);
    public static Tensor<int> Subtract(Tensor<int> x, Tensor<int> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l - r, (l, r) => l - r, y, options);

    public static Tensor<int> Subtract(Tensor<int> x, int y) => Subtract(x, y, TensorExecutionOptions.Auto);
    public static Tensor<int> Subtract(Tensor<int> x, int y, TensorExecutionOptions options) => x.VectorizedApply(l => l - new Vector<int>(y), l => l - y, options);

    public static Tensor<float> Subtract(Tensor<float> x, Tensor<float> y) => Subtract(x, y, TensorExecutionOptions.Auto);
    public static Tensor<float> Subtract(Tensor<float> x, Tensor<float> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l - r, (l, r) => l - r, y, options);

    public static Tensor<float> Subtract(Tensor<float> x, float y) => Subtract(x, y, TensorExecutionOptions.Auto);
    public static Tensor<float> Subtract(Tensor<float> x, float y, TensorExecutionOptions options) => x.VectorizedApply(l => l - new Vector<float>(y), l => l - y, options);

    public static Tensor<double> Subtract(Tensor<double> x, Tensor<double> y) => Subtract(x, y, TensorExecutionOptions.Auto);
    public static Tensor<double> Subtract(Tensor<double> x, Tensor<double> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l - r, (l, r) => l - r, y, options);

    public static Tensor<double> Subtract(Tensor<double> x, double y) => Subtract(x, y, TensorExecutionOptions.Auto);
    public static Tensor<double> Subtract(Tensor<double> x, double y, TensorExecutionOptions options) => x.VectorizedApply(l => l - new Vector<double>(y), l => l - y, options);

    public static Tensor<byte> Multiply(Tensor<byte> x, Tensor<byte> y) => Multiply(x, y, TensorExecutionOptions.Auto);
    public static Tensor<byte> Multiply(Tensor<byte> x, Tensor<byte> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => (l * r), (l, r) => (byte)(l * r), y, options);

    public static Tensor<byte> Multiply(Tensor<byte> x, byte y) => x.Apply(l => (byte)(l * y));

    public static Tensor<int> Multiply(Tensor<int> x, Tensor<int> y) => Multiply(x, y, TensorExecutionOptions.Auto);
    public static Tensor<int> Multiply(Tensor<int> x, Tensor<int> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l * r, (l, r) => l * r, y, options);

    public static Tensor<int> Multiply(Tensor<int> x, int y) => Multiply(x, y, TensorExecutionOptions.Auto);
    public static Tensor<int> Multiply(Tensor<int> x, int y, TensorExecutionOptions options) => x.VectorizedApply(l => l * new Vector<int>(y), l => l * y, options);

    public static Tensor<float> Multiply(Tensor<float> x, Tensor<float> y) => Multiply(x, y, TensorExecutionOptions.Auto);
    public static Tensor<float> Multiply(Tensor<float> x, Tensor<float> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l * r, (l, r) => l * r, y, options);

    /// <summary>Writes the float product into an existing destination tensor.</summary>
    public static Tensor<float> Multiply(Tensor<float> x, Tensor<float> y, Tensor<float> destination, TensorExecutionOptions options)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        return x.BroadcastApply<MultiplyBroadcast<float>>(y, destination, options);
    }

    public static Tensor<float> Multiply(Tensor<float> x, float y) => Multiply(x, y, TensorExecutionOptions.Auto);
    public static Tensor<float> Multiply(Tensor<float> x, float y, TensorExecutionOptions options) => x.VectorizedApply(l => l * new Vector<float>(y), l => l * y, options);

    public static Tensor<double> Multiply(Tensor<double> x, Tensor<double> y) => Multiply(x, y, TensorExecutionOptions.Auto);
    public static Tensor<double> Multiply(Tensor<double> x, Tensor<double> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l * r, (l, r) => l * r, y, options);

    public static Tensor<double> Multiply(Tensor<double> x, double y) => Multiply(x, y, TensorExecutionOptions.Auto);
    public static Tensor<double> Multiply(Tensor<double> x, double y, TensorExecutionOptions options) => x.VectorizedApply(l => l * new Vector<double>(y), l => l * y, options);

    public static Tensor<byte> Divide(Tensor<byte> x, Tensor<byte> y) => Divide(x, y, TensorExecutionOptions.Auto);
    public static Tensor<byte> Divide(Tensor<byte> x, Tensor<byte> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => (l / r), (l, r) => (byte)(l / r), y, options);

    public static Tensor<byte> Divide(Tensor<byte> x, byte y) => x.Apply(l => (byte)(l / y));

    public static Tensor<int> Divide(Tensor<int> x, Tensor<int> y) => Divide(x, y, TensorExecutionOptions.Auto);
    public static Tensor<int> Divide(Tensor<int> x, Tensor<int> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l / r, (l, r) => l / r, y, options);

    public static Tensor<int> Divide(Tensor<int> x, int y) => Divide(x, y, TensorExecutionOptions.Auto);
    public static Tensor<int> Divide(Tensor<int> x, int y, TensorExecutionOptions options) => x.VectorizedApply(l => l / new Vector<int>(y), l => l / y, options);

    public static Tensor<long> Add(Tensor<long> x, Tensor<long> y) => Add(x, y, TensorExecutionOptions.Auto);
    public static Tensor<long> Add(Tensor<long> x, Tensor<long> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l + r, (l, r) => l + r, y, options);

    public static Tensor<long> Subtract(Tensor<long> x, Tensor<long> y) => Subtract(x, y, TensorExecutionOptions.Auto);
    public static Tensor<long> Subtract(Tensor<long> x, Tensor<long> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l - r, (l, r) => l - r, y, options);

    public static Tensor<long> Multiply(Tensor<long> x, Tensor<long> y) => Multiply(x, y, TensorExecutionOptions.Auto);
    public static Tensor<long> Multiply(Tensor<long> x, Tensor<long> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l * r, (l, r) => l * r, y, options);

    public static Tensor<long> Divide(Tensor<long> x, Tensor<long> y) => x.Apply((l, r) => l / r, y);

    public static Tensor<long> Divide(Tensor<long> x, long y) => x.Apply(l => l / y);

    public static Tensor<float> Divide(Tensor<float> x, Tensor<float> y) => Divide(x, y, TensorExecutionOptions.Auto);
    public static Tensor<float> Divide(Tensor<float> x, Tensor<float> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l / r, (l, r) => l / r, y, options);

    /// <summary>Writes the float quotient into an existing destination tensor.</summary>
    public static Tensor<float> Divide(Tensor<float> x, Tensor<float> y, Tensor<float> destination, TensorExecutionOptions options)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        return x.BroadcastApply<DivideBroadcast<float>>(y, destination, options);
    }

    public static Tensor<float> Divide(Tensor<float> x, float y) => Divide(x, y, TensorExecutionOptions.Auto);
    public static Tensor<float> Divide(Tensor<float> x, float y, TensorExecutionOptions options) => x.VectorizedApply(l => l / new Vector<float>(y), l => l / y, options);

    public static Tensor<double> Divide(Tensor<double> x, Tensor<double> y) => Divide(x, y, TensorExecutionOptions.Auto);
    public static Tensor<double> Divide(Tensor<double> x, Tensor<double> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l / r, (l, r) => l / r, y, options);

    public static Tensor<double> Divide(Tensor<double> x, double y) => Divide(x, y, TensorExecutionOptions.Auto);
    public static Tensor<double> Divide(Tensor<double> x, double y, TensorExecutionOptions options) => x.VectorizedApply(l => l / new Vector<double>(y), l => l / y, options);

    public static Tensor<float> Negate(Tensor<float> x) => Negate(x, TensorExecutionOptions.Auto);
    public static Tensor<float> Negate(Tensor<float> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.Negate, l => -l, options);

    public static Tensor<double> Negate(Tensor<double> x) => Negate(x, TensorExecutionOptions.Auto);
    public static Tensor<double> Negate(Tensor<double> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.Negate, l => -l, options);

    public static Tensor<float> Pow(Tensor<float> x, Tensor<float> y) => x.Apply(MathF.Pow, y);

    public static Tensor<double> Pow(Tensor<double> x, Tensor<double> y) => x.Apply(Math.Pow, y);

    public static Tensor<float> Square(Tensor<float> x) => x.Apply(l => l * l);

    public static Tensor<double> Square(Tensor<double> x) => x.Apply(l => l * l);

    // MathF.Abs clears the NaN sign bit, matching ORT (Abs(NaN) is +NaN).
    public static Tensor<float> Abs(Tensor<float> x) => x.Apply(MathF.Abs);

    public static Tensor<double> Abs(Tensor<double> x) => x.Apply(Math.Abs);
    public static Tensor<float> Cos(Tensor<float> x) => Cos(x, TensorExecutionOptions.Auto);
    public static Tensor<float> Cos(Tensor<float> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.Cos, MathF.Cos, options);

    public static Tensor<double> Cos(Tensor<double> x) => Cos(x, TensorExecutionOptions.Auto);
    public static Tensor<double> Cos(Tensor<double> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.Cos, Math.Cos, options);

    public static Tensor<float> Sin(Tensor<float> x) => Sin(x, TensorExecutionOptions.Auto);
    public static Tensor<float> Sin(Tensor<float> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.Sin, MathF.Sin, options);

    public static Tensor<double> Sin(Tensor<double> x) => Sin(x, TensorExecutionOptions.Auto);
    public static Tensor<double> Sin(Tensor<double> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.Sin, Math.Sin, options);

    public static Tensor<int> Negate(Tensor<int> x) => Negate(x, TensorExecutionOptions.Auto);
    public static Tensor<int> Negate(Tensor<int> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.Negate, l => -l, options);

    public static Tensor<long> Negate(Tensor<long> x) => Negate(x, TensorExecutionOptions.Auto);
    public static Tensor<long> Negate(Tensor<long> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.Negate, l => -l, options);

    public static Tensor<int> Abs(Tensor<int> x) => x.Apply(l => l >= 0 ? l : -l);

    public static Tensor<long> Abs(Tensor<long> x) => x.Apply(l => l >= 0L ? l : -l);

    /// <summary>
    /// Exact Gaussian error linear unit: 0.5 * x * (1 + erf(x / sqrt(2))).
    /// </summary>
    public static Tensor<float> Gelu(Tensor<float> x) => Gelu(x, TensorExecutionOptions.Auto);

    public static Tensor<float> Gelu(Tensor<float> x, TensorExecutionOptions options) =>
        x.VectorizedApply((Vector<float> v) => new Vector<float>(0.5f) * v * (Vector<float>.One + ErfVector(new Vector<float>(0.7071067811865476f) * v)), (float v) => 0.5f * v * (1f + MathOps.Erf(v * 0.7071067811865476f)), options);

    public static Tensor<float> Gelu(Tensor<float> x, Tensor<float> destination) => Gelu(x, destination, TensorExecutionOptions.Auto);

    public static Tensor<float> Gelu(Tensor<float> x, Tensor<float> destination, TensorExecutionOptions options)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        x.VectorizedApply((Vector<float> v) => new Vector<float>(0.5f) * v * (Vector<float>.One + ErfVector(new Vector<float>(0.7071067811865476f) * v)), (float v) => 0.5f * v * (1f + MathOps.Erf(v * 0.7071067811865476f)), destination, options);
        return destination;
    }

    /// <summary>
    /// Exact Gaussian error linear unit: 0.5 * x * (1 + erf(x / sqrt(2))).
    /// </summary>
    public static Tensor<double> Gelu(Tensor<double> x) => Gelu(x, TensorExecutionOptions.Auto);

    /// <summary>Double-precision GELU runs a fixed scalar path; the options are validated but select no kernel variant.</summary>
    public static Tensor<double> Gelu(Tensor<double> x, TensorExecutionOptions options)
    {
        options.Validate();
        return x.Apply((double v) => 0.5 * v * (1.0 + MathOps.Erf(v * 0.7071067811865476)));
    }

    /// <summary>
    /// Validates LayerNormalization arguments and derives the normalization block geometry.
    /// The axis indexes the input tensor; negative values count from the back.
    /// </summary>
    static (int Block, int Outer) LayerNormalizationPlan(int rank, int axis, int[] dimensions, long inputLength, long scaleLength, long? biasLength)
    {
        int normalizedAxis = axis < 0 ? axis + rank : axis;
        if (normalizedAxis < 0 || normalizedAxis >= rank) throw new ArgumentException(nameof(axis));
        int block = 1;
        for (int dimension = normalizedAxis; dimension < rank; dimension++) block *= dimensions[dimension];
        if (scaleLength != block) throw new ArgumentException("Scale length must match the normalized dimensions.", "scale");
        if (biasLength.HasValue && biasLength.Value != block) throw new ArgumentException("Bias length must match the normalized dimensions.", "bias");
        return (block, (int)(inputLength / block));
    }

    /// <summary>
    /// Normalizes over the input dimensions from axis to the last one using scale, optional bias, and epsilon.
    /// Statistics accumulate in double precision; the axis indexes the input tensor.
    /// </summary>
    public static Tensor<float> LayerNormalization(Tensor<float> x, Tensor<float> scale, Tensor<float>? bias, int axis, float epsilon)
    {
        var xd = x.ToDenseTensor();
        var sd = scale.ToDenseTensor();
        var bd = bias?.ToDenseTensor();
        var plan = LayerNormalizationPlan(x.Rank, axis, xd.Dimensions.ToArray(), xd.Length, sd.Length, bd?.Length);
        var output = new DenseTensor<float>(xd.Dimensions);
        LayerNormFloatInto(xd, sd, bd, output, plan.Block, plan.Outer, epsilon);
        return output;
    }

    /// <summary>Shared float layer-normalization kernel used by both entries.</summary>
    static void LayerNormFloatInto(DenseTensor<float> xd, DenseTensor<float> sd, DenseTensor<float>? bd, DenseTensor<float> destination, int block, int outer, float epsilon)
    {
        var xs = xd.Buffer.Span;
        var ss = sd.Buffer.Span;
        var bs = bd is null ? new Span<float>() : bd.Buffer.Span;
        var os = destination.Buffer.Span;
        for (int o = 0; o < outer; o++)
        {
            double mean = 0.0;
            for (int i = 0; i < block; i++) mean += xs[o * block + i];
            mean /= block;
            double variance = 0.0;
            for (int i = 0; i < block; i++) { double d = xs[o * block + i] - mean; variance += d * d; }
            variance /= block;
            double inv = 1.0 / Math.Sqrt(variance + epsilon);
            for (int i = 0; i < block; i++) os[o * block + i] = (float)((xs[o * block + i] - mean) * inv * ss[i] + (bd is null ? 0f : bs[i]));
        }
    }

    /// <summary>Writes float layer normalization into an existing dense destination.</summary>
    public static Tensor<float> LayerNormalization(Tensor<float> x, Tensor<float> scale, Tensor<float>? bias, DenseTensor<float> destination, int axis, float epsilon)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        var xd = x.ToDenseTensor();
        var sd = scale.ToDenseTensor();
        var bd = bias?.ToDenseTensor();
        var plan = LayerNormalizationPlan(x.Rank, axis, xd.Dimensions.ToArray(), xd.Length, sd.Length, bd?.Length);
        if (!destination.Dimensions.SequenceEqual(xd.Dimensions.ToArray())) throw new ArgumentException(nameof(destination), "Destination shape must match the input shape.");
        if (!HasStandardStrides(destination)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        LayerNormFloatInto(xd, sd, bd, destination, plan.Block, plan.Outer, epsilon);
        return destination;
    }

    /// <summary>
    /// Normalizes over the input dimensions from axis to the last one using scale, optional bias, and epsilon.
    /// Statistics accumulate in double precision; the axis indexes the input tensor.
    /// </summary>
    public static Tensor<double> LayerNormalization(Tensor<double> x, Tensor<double> scale, Tensor<double>? bias, int axis, double epsilon)
    {
        var xd = x.ToDenseTensor();
        var sd = scale.ToDenseTensor();
        var bd = bias?.ToDenseTensor();
        var plan = LayerNormalizationPlan(x.Rank, axis, xd.Dimensions.ToArray(), xd.Length, sd.Length, bd?.Length);
        var output = new DenseTensor<double>(xd.Dimensions);
        LayerNormDoubleInto(xd, sd, bd, output, plan.Block, plan.Outer, epsilon);
        return output;
    }

    /// <summary>Writes double layer normalization into an existing dense destination.</summary>
    public static Tensor<double> LayerNormalization(Tensor<double> x, Tensor<double> scale, Tensor<double>? bias, DenseTensor<double> destination, int axis, double epsilon)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        var xd = x.ToDenseTensor();
        var sd = scale.ToDenseTensor();
        var bd = bias?.ToDenseTensor();
        var plan = LayerNormalizationPlan(x.Rank, axis, xd.Dimensions.ToArray(), xd.Length, sd.Length, bd?.Length);
        if (!destination.Dimensions.SequenceEqual(xd.Dimensions.ToArray())) throw new ArgumentException(nameof(destination), "Destination shape must match the input shape.");
        if (!HasStandardStrides(destination)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        LayerNormDoubleInto(xd, sd, bd, destination, plan.Block, plan.Outer, epsilon);
        return destination;
    }

    /// <summary>Shared double layer-normalization kernel used by both entries.</summary>
    static void LayerNormDoubleInto(DenseTensor<double> xd, DenseTensor<double> sd, DenseTensor<double>? bd, DenseTensor<double> destination, int block, int outer, double epsilon)
    {
        var xs = xd.Buffer.Span;
        var ss = sd.Buffer.Span;
        var bs = bd is null ? new Span<double>() : bd.Buffer.Span;
        var os = destination.Buffer.Span;
        for (int o = 0; o < outer; o++)
        {
            double mean = 0.0;
            for (int i = 0; i < block; i++) mean += xs[o * block + i];
            mean /= block;
            double variance = 0.0;
            for (int i = 0; i < block; i++) { double d = xs[o * block + i] - mean; variance += d * d; }
            variance /= block;
            double inv = 1.0 / Math.Sqrt(variance + epsilon);
            for (int i = 0; i < block; i++) os[o * block + i] = (xs[o * block + i] - mean) * inv * ss[i] + (bd is null ? 0.0 : bs[i]);
        }
    }

    /// <summary>
    /// Applies rotary position embedding: out = x * cos + rotate_half(x) * sin with
    /// rotate_half(x)[i] = i &lt; span ? -x[i + half] : x[i - span] along the axis,
    /// where span = dim - half (the classic rotation at dim == 2 * half). Cos/sin
    /// follow standard right-aligned broadcast against x. Fused form of the
    /// Slice/Slice/Neg/Concat/Mul/Mul/Add pattern; computes every output element
    /// with the same operations in the same order, so results match it bitwise.
    /// </summary>
    public static Tensor<float> RotaryEmbedding(Tensor<float> x, Tensor<float> cos, Tensor<float> sin, int half, int axis, int concatAxis)
    {
        var output = new DenseTensor<float>(x.ToDenseTensor().Dimensions);
        return RotaryEmbedding(x, cos, sin, output, half, axis, -1);
    }

    /// <summary>Writes the rotary position embedding into an existing dense destination.</summary>
    public static Tensor<float> RotaryEmbedding(Tensor<float> x, Tensor<float> cos, Tensor<float> sin, DenseTensor<float> destination, int half, int axis, int concatAxis)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        StartOpStage(OpStage.ValidateArguments);
        var xd = x.ToDenseTensor();
        var cd = cos.ToDenseTensor();
        var sd = sin.ToDenseTensor();
        if (!destination.Dimensions.SequenceEqual(xd.Dimensions.ToArray())) throw new ArgumentException(nameof(destination), "Destination shape must match the input shape.");
        if (!HasStandardStrides(destination)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        if (half <= 0) throw new ArgumentException(nameof(half), "Half size must be positive.");
        int rank = xd.Rank;
        int a = axis < 0 ? axis + rank : axis;
        if (a < 0 || a >= rank) throw new ArgumentException(nameof(axis), "Axis is out of range for the input rank.");
        int ca = concatAxis < 0 ? concatAxis + rank : concatAxis;
        if (ca != a) throw new ArgumentException(nameof(concatAxis), "Slice and concat axes disagree after rank normalization.");
        if (!cd.Dimensions.SequenceEqual(sd.Dimensions.ToArray())) throw new ArgumentException(nameof(sin), "Cos and sin must have identical shapes.");
        if (cd.Rank > rank) throw new ArgumentException(nameof(cos), "Cos rank must not exceed input rank.");
        int inner = xd.Dimensions[a];
        if (inner < half) throw new ArgumentException(nameof(half), "Half size must not exceed the axis dimension.");
        int span = inner - half;
        var dims = xd.Dimensions.ToArray();
        var xstrides = new int[rank];
        var cstrides = new int[rank];
        var cshape = new int[rank];
        int stride = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            xstrides[d] = stride;
            int cd2 = d < rank - cd.Rank ? 1 : cd.Dimensions[d - (rank - cd.Rank)];
            if (cd2 != 1 && cd2 != dims[d]) throw new ArgumentException(nameof(cos), "Cos shape must broadcast against the input shape.");
            cshape[d] = cd2;
            cstrides[d] = cd2 == 1 ? 0 : stride;
            stride *= dims[d];
        }
        var xs = xd.Buffer.Span;
        var cs = cd.Buffer.Span;
        var ss = sd.Buffer.Span;
        var os = destination.Buffer.Span;
        StartOpStage(OpStage.Math);
        var index = new int[rank];
        int cx = 0;
        int cc = 0;
        int total = (int)xd.Length;
        for (int n = 0; n < total; n++)
        {
            int pos = index[a];
            int rot = pos < span ? cx + half * xstrides[a] : cx - span * xstrides[a];
            float rotated = pos < span ? -xs[rot] : xs[rot];
            os[cx] = xs[cx] * cs[cc] + rotated * ss[cc];
            for (int d = rank - 1; d >= 0; d--)
            {
                index[d]++;
                cx += xstrides[d];
                cc += cstrides[d];
                if (index[d] < dims[d]) break;
                index[d] = 0;
                cx -= xstrides[d] * dims[d];
                cc -= cstrides[d] * cshape[d];
            }
        }
        return destination;
    }
    /// <summary>
    /// Generates start, start plus delta, and so on, stopping before limit. Zero delta throws.
    /// </summary>
    public static Tensor<float> Range(float start, float limit, float delta)
    {
        if (delta == 0f) throw new ArgumentException(nameof(delta));
        int count = Math.Max((int)Math.Ceiling((limit - start) / delta), 0);
        var output = new DenseTensor<float>(count);
        var span = output.Buffer.Span;
        for (int i = 0; i < count; i++) span[i] = start + i * delta;
        return output;
    }

    /// <summary>
    /// Generates start, start plus delta, and so on, stopping before limit. Zero delta throws.
    /// </summary>
    public static Tensor<double> Range(double start, double limit, double delta)
    {
        if (delta == 0.0) throw new ArgumentException(nameof(delta));
        int count = Math.Max((int)Math.Ceiling((limit - start) / delta), 0);
        var output = new DenseTensor<double>(count);
        var span = output.Buffer.Span;
        for (int i = 0; i < count; i++) span[i] = start + i * delta;
        return output;
    }

    /// <summary>
    /// Generates start, start plus delta, and so on, stopping before limit. Zero delta throws.
    /// </summary>
    public static Tensor<long> Range(long start, long limit, long delta)
    {
        if (delta == 0L) throw new ArgumentException(nameof(delta));
        int count = Math.Max((int)Math.Ceiling(((double)limit - start) / delta), 0);
        var output = new DenseTensor<long>(count);
        var span = output.Buffer.Span;
        for (int i = 0; i < count; i++) span[i] = start + i * delta;
        return output;
    }

    /// <summary>
    /// Generates start, start plus delta, and so on, stopping before limit. Zero delta throws.
    /// </summary>
    public static Tensor<int> Range(int start, int limit, int delta)
    {
        if (delta == 0) throw new ArgumentException(nameof(delta));
        int count = Math.Max((int)Math.Ceiling(((double)limit - start) / delta), 0);
        var output = new DenseTensor<int>(count);
        var span = output.Buffer.Span;
        for (int i = 0; i < count; i++) span[i] = start + i * delta;
        return output;
    }

    static DenseTensor<U> TileCore<U>(Tensor<U> x, int[] repeats) where U : unmanaged
    {
        if (repeats.Length != x.Rank) throw new ArgumentException("Repeats rank must match input rank.", nameof(repeats));
        var xd = x.ToDenseTensor();
        var outDims = new int[x.Rank];
        for (int i = 0; i < x.Rank; i++)
        {
            if (repeats[i] < 0) throw new ArgumentException("Repeats must be non-negative.", nameof(repeats));
            outDims[i] = xd.Dimensions[i] * repeats[i];
        }
        var output = new DenseTensor<U>((ReadOnlySpan<int>)outDims);
        var src = new int[x.Rank];
        foreach (var index in output.GetDimensionsIterator())
        {
            for (int i = 0; i < x.Rank; i++) src[i] = xd.Dimensions[i] == 0 ? 0 : index[i] % xd.Dimensions[i];
            output[index] = xd[src];
        }
        return output;
    }

    /// <summary>
    /// Repeats the input repeats[i] times along each dimension.
    /// The repeats rank must match the input rank and every repeat must be non-negative.
    /// </summary>
    public static Tensor<float> Tile(Tensor<float> x, int[] repeats) => TileCore(x, repeats);

    /// <summary>
    /// Repeats the input repeats[i] times along each dimension.
    /// The repeats rank must match the input rank and every repeat must be non-negative.
    /// </summary>
    public static Tensor<double> Tile(Tensor<double> x, int[] repeats) => TileCore(x, repeats);

    /// <summary>
    /// Repeats the input repeats[i] times along each dimension.
    /// The repeats rank must match the input rank and every repeat must be non-negative.
    /// </summary>
    public static Tensor<int> Tile(Tensor<int> x, int[] repeats) => TileCore(x, repeats);

    /// <summary>
    /// Repeats the input repeats[i] times along each dimension.
    /// The repeats rank must match the input rank and every repeat must be non-negative.
    /// </summary>
    public static Tensor<long> Tile(Tensor<long> x, int[] repeats) => TileCore(x, repeats);

    /// <summary>
    /// Copies the length elements starting at start along the axis.
    /// Negative axes resolve against the input rank; out-of-range chunks throw.
    /// </summary>
    public static DenseTensor<U> ChunkCopy<U>(Tensor<U> x, int axis, int start, int length) where U : unmanaged
    {
        int rank = x.Rank;
        int ax = axis < 0 ? axis + rank : axis;
        if (ax < 0 || ax >= rank) throw new ArgumentException(nameof(axis));
        if (start < 0 || length < 0 || start + length > x.Dimensions[ax]) throw new ArgumentException("Chunk is out of range.");
        var xd = x.ToDenseTensor();
        var outDims = xd.Dimensions.ToArray();
        outDims[ax] = length;
        var output = new DenseTensor<U>((ReadOnlySpan<int>)outDims);
        int inner = 1;
        for (int trailing = ax + 1; trailing < rank; trailing++) inner *= outDims[trailing];
        int outer = 1;
        for (int leading = 0; leading < ax; leading++) outer *= outDims[leading];
        if (HasStandardStrides(xd))
        {
            ArrayUtilities.CopyAxisChunks(xd.Buffer.Span, xd.Dimensions[ax], output.Buffer.Span, length, outer, inner, start, 0, length);
            return output;
        }
        var src = new int[rank];
        foreach (var index in output.GetDimensionsIterator())
        {
            for (int i = 0; i < rank; i++) src[i] = index[i];
            src[ax] += start;
            output[index] = xd[src];
        }
        return output;
    }

    public static Tensor<float> Sqrt(Tensor<float> x) => Sqrt(x, TensorExecutionOptions.Auto);
    public static Tensor<float> Sqrt(Tensor<float> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.SquareRoot, MathF.Sqrt, options);

    public static Tensor<double> Sqrt(Tensor<double> x) => Sqrt(x, TensorExecutionOptions.Auto);
    public static Tensor<double> Sqrt(Tensor<double> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.SquareRoot, Math.Sqrt, options);

    public static Tensor<float> Resize(Tensor<float> input, int[] sizes, MathOps.ResizeMode mode, MathOps.ResizeCoordinateTransformation coordinateTransformationMode, MathOps.ResizeNearestMode nearestMode, float cubicCoeffA, double[]? scales)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (input.Rank != 4) throw new ArgumentException(nameof(input), "Resize currently supports only 4D tensors (NCHW).");
        if (sizes is null || sizes.Length != 4) throw new ArgumentException(nameof(sizes), "Resize sizes must be a 1D array of length 4.");

        var nOut = sizes[0];
        var cOut = sizes[1];
        var hOut = sizes[2];
        var wOut = sizes[3];
        var nIn = input.Dimensions[0];
        var cIn = input.Dimensions[1];
        var hIn = input.Dimensions[2];
        var wIn = input.Dimensions[3];

        if (nOut != nIn || cOut != cIn)
        {
            throw new ArgumentException(nameof(sizes), "Resize currently requires N and C dimensions to remain unchanged.");
        }

        var output = DenseTensor<float>.OfShape(sizes);
        var xd = input.ToDenseTensor();
        var xs = xd.Buffer.Span;
        var os = output.Buffer.Span;
        // Coordinates use the true scales when the caller derived sizes from
        // them: floored sizes would otherwise give back a different scale
        // (verified against ORT 1.29: 5 * 1.5 floors to 7 but samples with 1.5).
        var scaleH = scales is null ? (float)hOut / hIn : (float)scales[2];
        var scaleW = scales is null ? (float)wOut / wIn : (float)scales[3];

        float TransformCoordinate(int outIndex, int inSize, int outSize, float scale)
        {
            return coordinateTransformationMode switch
            {
                MathOps.ResizeCoordinateTransformation.HalfPixel => (outIndex + 0.5f) / scale - 0.5f,
                MathOps.ResizeCoordinateTransformation.AlignCorners => outSize == 1 ? 0f : outIndex * (inSize - 1f) / (outSize - 1f),
                MathOps.ResizeCoordinateTransformation.Asymmetric => outIndex / scale,
                _ => throw new NotSupportedException($"coordinate_transformation_mode {coordinateTransformationMode} is not supported."),
            };
        }

        // round_prefer_floor under half_pixel rounds exact halves down, i.e.
        // ceil(coord - 0.5), while asymmetric/align_corners use floor(coord + 0.5)
        // (verified against ORT 1.29: the coordinate must be the TRUE scale, and
        // mixing scales with floored sizes creates phantom contradictions).
        int NearestIndex(float coord, int inSize)
        {
            var value = nearestMode switch
            {
                MathOps.ResizeNearestMode.Floor => (int)MathF.Floor(coord),
                MathOps.ResizeNearestMode.Ceil => (int)MathF.Ceiling(coord),
                MathOps.ResizeNearestMode.RoundPreferFloor => coordinateTransformationMode == MathOps.ResizeCoordinateTransformation.HalfPixel
                    ? (int)MathF.Ceiling(coord - 0.5f)
                    : (int)MathF.Floor(coord + 0.5f),
                MathOps.ResizeNearestMode.RoundPreferCeil => (int)MathF.Floor(coord + 0.5f),
                _ => throw new NotSupportedException($"nearest_mode {nearestMode} is not supported."),
            };
            return Math.Clamp(value, 0, inSize - 1);
        }

        static float CubicWeight(float x, float a)
        {
            var t = MathF.Abs(x);
            if (t <= 1f)
            {
                return ((a + 2f) * t * t * t) - ((a + 3f) * t * t) + 1f;
            }
            if (t < 2f)
            {
                return (a * t * t * t) - (5f * a * t * t) + (8f * a * t) - (4f * a);
            }
            return 0f;
        }

        for (int n = 0; n < nOut; n++)
        {
            for (int c = 0; c < cOut; c++)
            {
                for (int oy = 0; oy < hOut; oy++)
                {
                    var inY = TransformCoordinate(oy, hIn, hOut, scaleH);
                    if (mode == MathOps.ResizeMode.Nearest)
                    {
                        var ny = NearestIndex(inY, hIn);
                        for (int ox = 0; ox < wOut; ox++)
                        {
                            var inX = TransformCoordinate(ox, wIn, wOut, scaleW);
                            var nx = NearestIndex(inX, wIn);
                            os[(((n * cIn) + c) * hOut + oy) * wOut + ox] = xs[(((n * cIn) + c) * hIn + ny) * wIn + nx];
                        }
                    }
                    else if (mode == MathOps.ResizeMode.Linear)
                    {
                        var y0 = MathF.Floor(inY);
                        var y1 = y0 + 1f;
                        var y0i = Math.Clamp((int)y0, 0, hIn - 1);
                        var y1i = Math.Clamp((int)y1, 0, hIn - 1);
                        var ly = inY - y0;
                        var wy0 = 1f - ly;
                        var wy1 = ly;
                        for (int ox = 0; ox < wOut; ox++)
                        {
                            var inX = TransformCoordinate(ox, wIn, wOut, scaleW);
                            var x0 = MathF.Floor(inX);
                            var x1 = x0 + 1f;
                            var x0i = Math.Clamp((int)x0, 0, wIn - 1);
                            var x1i = Math.Clamp((int)x1, 0, wIn - 1);
                            var lx = inX - x0;
                            var wx0 = 1f - lx;
                            var wx1 = lx;
                            int y0Row = (((n * cIn) + c) * hIn + y0i) * wIn;
                            int y1Row = (((n * cIn) + c) * hIn + y1i) * wIn;
                            int dstIdx = (((n * cIn) + c) * hOut + oy) * wOut + ox;
                            var v00 = xs[y0Row + x0i];
                            var v01 = xs[y0Row + x1i];
                            var v10 = xs[y1Row + x0i];
                            var v11 = xs[y1Row + x1i];
                            os[dstIdx] = (v00 * wy0 * wx0) + (v01 * wy0 * wx1) + (v10 * wy1 * wx0) + (v11 * wy1 * wx1);
                        }
                    }
                    else if (mode == MathOps.ResizeMode.Cubic)
                    {
                        var yBase = (int)MathF.Floor(inY);
                        var wy = new float[4];
                        var yIdx = new int[4];
                        for (int i = 0; i < 4; i++)
                        {
                            var yi = yBase - 1 + i;
                            yIdx[i] = Math.Clamp(yi, 0, hIn - 1);
                            wy[i] = CubicWeight(inY - yi, cubicCoeffA);
                        }
                        var wx = new float[4];
                        var xIdx = new int[4];
                        for (int ox = 0; ox < wOut; ox++)
                        {
                            var inX = TransformCoordinate(ox, wIn, wOut, scaleW);
                            var xBase = (int)MathF.Floor(inX);
                            for (int i = 0; i < 4; i++)
                            {
                                var xi = xBase - 1 + i;
                                xIdx[i] = Math.Clamp(xi, 0, wIn - 1);
                                wx[i] = CubicWeight(inX - xi, cubicCoeffA);
                            }
                            var sum = 0f;
                            for (int iy = 0; iy < 4; iy++)
                            {
                                var wyv = wy[iy];
                                for (int ix = 0; ix < 4; ix++)
                                {
                                    sum += wyv * wx[ix] * xs[(((n * cIn) + c) * hIn + yIdx[iy]) * wIn + xIdx[ix]];
                                }
                            }
                            os[(((n * cIn) + c) * hOut + oy) * wOut + ox] = sum;
                        }
                    }
                    else
                    {
                        throw new NotSupportedException($"Resize mode {mode} is not supported.");
                    }
                }
            }
        }

        return output;
    }

    public static Tensor<double> Resize(Tensor<double> input, int[] sizes, MathOps.ResizeMode mode, MathOps.ResizeCoordinateTransformation coordinateTransformationMode, MathOps.ResizeNearestMode nearestMode, double cubicCoeffA, double[]? scales)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (input.Rank != 4) throw new ArgumentException(nameof(input), "Resize currently supports only 4D tensors (NCHW).");
        if (sizes is null || sizes.Length != 4) throw new ArgumentException(nameof(sizes), "Resize sizes must be a 1D array of length 4.");

        var nOut = sizes[0];
        var cOut = sizes[1];
        var hOut = sizes[2];
        var wOut = sizes[3];
        var nIn = input.Dimensions[0];
        var cIn = input.Dimensions[1];
        var hIn = input.Dimensions[2];
        var wIn = input.Dimensions[3];

        if (nOut != nIn || cOut != cIn)
        {
            throw new ArgumentException(nameof(sizes), "Resize currently requires N and C dimensions to remain unchanged.");
        }

        var output = DenseTensor<double>.OfShape(sizes);
        var xd = input.ToDenseTensor();
        var xs = xd.Buffer.Span;
        var os = output.Buffer.Span;
        var scaleH = scales is null ? (double)hOut / hIn : scales[2];
        var scaleW = scales is null ? (double)wOut / wIn : scales[3];

        double TransformCoordinate(int outIndex, int inSize, int outSize, double scale)
        {
            return coordinateTransformationMode switch
            {
                MathOps.ResizeCoordinateTransformation.HalfPixel => (outIndex + 0.5) / scale - 0.5,
                MathOps.ResizeCoordinateTransformation.AlignCorners => outSize == 1 ? 0d : outIndex * (inSize - 1d) / (outSize - 1d),
                MathOps.ResizeCoordinateTransformation.Asymmetric => outIndex / scale,
                _ => throw new NotSupportedException($"coordinate_transformation_mode {coordinateTransformationMode} is not supported."),
            };
        }

        // See the float copy: half_pixel round_prefer_floor rounds halves down.
        int NearestIndex(double coord, int inSize)
        {
            var value = nearestMode switch
            {
                MathOps.ResizeNearestMode.Floor => (int)Math.Floor(coord),
                MathOps.ResizeNearestMode.Ceil => (int)Math.Ceiling(coord),
                MathOps.ResizeNearestMode.RoundPreferFloor => coordinateTransformationMode == MathOps.ResizeCoordinateTransformation.HalfPixel
                    ? (int)Math.Ceiling(coord - 0.5)
                    : (int)Math.Floor(coord + 0.5),
                MathOps.ResizeNearestMode.RoundPreferCeil => (int)Math.Floor(coord + 0.5),
                _ => throw new NotSupportedException($"nearest_mode {nearestMode} is not supported."),
            };
            return Math.Clamp(value, 0, inSize - 1);
        }

        static double CubicWeight(double x, double a)
        {
            var t = Math.Abs(x);
            if (t <= 1d)
            {
                return ((a + 2d) * t * t * t) - ((a + 3d) * t * t) + 1d;
            }
            if (t < 2d)
            {
                return (a * t * t * t) - (5d * a * t * t) + (8d * a * t) - (4d * a);
            }
            return 0d;
        }

        for (int n = 0; n < nOut; n++)
        {
            for (int c = 0; c < cOut; c++)
            {
                for (int oy = 0; oy < hOut; oy++)
                {
                    var inY = TransformCoordinate(oy, hIn, hOut, scaleH);
                    if (mode == MathOps.ResizeMode.Nearest)
                    {
                        var ny = NearestIndex(inY, hIn);
                        for (int ox = 0; ox < wOut; ox++)
                        {
                            var inX = TransformCoordinate(ox, wIn, wOut, scaleW);
                            var nx = NearestIndex(inX, wIn);
                            os[(((n * cIn) + c) * hOut + oy) * wOut + ox] = xs[(((n * cIn) + c) * hIn + ny) * wIn + nx];
                        }
                    }
                    else if (mode == MathOps.ResizeMode.Linear)
                    {
                        var y0 = Math.Floor(inY);
                        var y1 = y0 + 1d;
                        var y0i = Math.Clamp((int)y0, 0, hIn - 1);
                        var y1i = Math.Clamp((int)y1, 0, hIn - 1);
                        var ly = inY - y0;
                        var wy0 = 1d - ly;
                        var wy1 = ly;
                        for (int ox = 0; ox < wOut; ox++)
                        {
                            var inX = TransformCoordinate(ox, wIn, wOut, scaleW);
                            var x0 = Math.Floor(inX);
                            var x1 = x0 + 1d;
                            var x0i = Math.Clamp((int)x0, 0, wIn - 1);
                            var x1i = Math.Clamp((int)x1, 0, wIn - 1);
                            var lx = inX - x0;
                            var wx0 = 1d - lx;
                            var wx1 = lx;
                            int y0Row = (((n * cIn) + c) * hIn + y0i) * wIn;
                            int y1Row = (((n * cIn) + c) * hIn + y1i) * wIn;
                            int dstIdx = (((n * cIn) + c) * hOut + oy) * wOut + ox;
                            var v00 = xs[y0Row + x0i];
                            var v01 = xs[y0Row + x1i];
                            var v10 = xs[y1Row + x0i];
                            var v11 = xs[y1Row + x1i];
                            os[dstIdx] = (v00 * wy0 * wx0) + (v01 * wy0 * wx1) + (v10 * wy1 * wx0) + (v11 * wy1 * wx1);
                        }
                    }
                    else if (mode == MathOps.ResizeMode.Cubic)
                    {
                        var yBase = (int)Math.Floor(inY);
                        var wy = new double[4];
                        var yIdx = new int[4];
                        for (int i = 0; i < 4; i++)
                        {
                            var yi = yBase - 1 + i;
                            yIdx[i] = Math.Clamp(yi, 0, hIn - 1);
                            wy[i] = CubicWeight(inY - yi, cubicCoeffA);
                        }
                        var wx = new double[4];
                        var xIdx = new int[4];
                        for (int ox = 0; ox < wOut; ox++)
                        {
                            var inX = TransformCoordinate(ox, wIn, wOut, scaleW);
                            var xBase = (int)Math.Floor(inX);
                            for (int i = 0; i < 4; i++)
                            {
                                var xi = xBase - 1 + i;
                                xIdx[i] = Math.Clamp(xi, 0, wIn - 1);
                                wx[i] = CubicWeight(inX - xi, cubicCoeffA);
                            }
                            var sum = 0d;
                            for (int iy = 0; iy < 4; iy++)
                            {
                                var wyv = wy[iy];
                                for (int ix = 0; ix < 4; ix++)
                                {
                                    sum += wyv * wx[ix] * xs[(((n * cIn) + c) * hIn + yIdx[iy]) * wIn + xIdx[ix]];
                                }
                            }
                            os[(((n * cIn) + c) * hOut + oy) * wOut + ox] = sum;
                        }
                    }
                    else
                    {
                        throw new NotSupportedException($"Resize mode {mode} is not supported.");
                    }
                }
            }
        }

        return output;
    }

    public static Tensor<int> MatMul2D(Tensor<int> x, Tensor<int> y) => MatMul2D(x, y, TensorExecutionOptions.Auto);

    public static Tensor<int> MatMul2D(Tensor<int> x, Tensor<int> y, TensorExecutionOptions options)
    {
        options.Validate();
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
        var m = x.Dimensions[0];
        var n = x.Dimensions[1];
        var k = y.Dimensions[1];

        var dx = x as DenseTensor<int>;
        var dy = y as DenseTensor<int>;
        var _x = dx is not null && !dx.IsReversedStride ? dx : x.ToDenseTensor();
        var _y = dy is not null && !dy.IsReversedStride ? dy : y.ToDenseTensor();
        var output = DenseTensor<int>.OfShape(new int[] { x.Dimensions[0], y.Dimensions[1] });

        var xh = _x.Buffer.Pin();
        var yh = _y.Buffer.Pin();
        var oh = output.Buffer.Pin();
        if (options.UseSimd)
        {
            unsafe
            {
                mm_unsafe_vectorized(m, n, k, (int*)xh.Pointer, (int*)yh.Pointer, (int*)oh.Pointer);
            }
        }
        else
        {
            unsafe
            {
                mm(m, n, k, (int*)xh.Pointer, (int*)yh.Pointer, (int*)oh.Pointer);
            }
        }

        xh.Dispose();
        yh.Dispose();
        oh.Dispose();
        return output;
    }

    static (DenseTensor<float> x, DenseTensor<float> y) DensifyFloatOperands(Tensor<float> x, Tensor<float> y)
    {
        var dx = x as DenseTensor<float>;
        var dy = y as DenseTensor<float>;
        if (dx is not { IsReversedStride: false } || dy is not { IsReversedStride: false })
        {
            StartOpStage(OpStage.Copy);
        }
        DenseTensor<float> ddx = dx is { IsReversedStride: false } ownX ? ownX : x.ToDenseTensor();
        DenseTensor<float> ddy = dy is { IsReversedStride: false } ownY ? ownY : y.ToDenseTensor();
        return (ddx, ddy);
    }

    static unsafe void RunFloatMatMulKernel(int m, int n, int k, float* x, float* y, float* output, TensorExecutionOptions options)
    {
        if (options.UseSimd && options.UseIntrinsics && Fma.IsSupported && k % 32 == 0 && m >= 2)
        {
            int blocked = m - (m % 2);
            mm_unsafe_vectorized_intrinsics_2x4(blocked, n, k, x, y, output);
            if (blocked != m)
            {
                mm_unsafe_vectorized_intrinsics(1, n, k, x + blocked * n, y, output + blocked * k);
            }
        }
        else if (options.UseSimd && options.UseIntrinsics && Fma.IsSupported)
        {
            mm_unsafe_vectorized_intrinsics(m, n, k, x, y, output);
        }
        else if (options.UseSimd)
        {
            mm_unsafe_vectorized(m, n, k, x, y, output);
        }
        else
        {
            mm(m, n, k, x, y, output);
        }
    }

    public static Tensor<float> MatMul2D(Tensor<float> x, Tensor<float> y) => MatMul2D(x, y, TensorExecutionOptions.Auto);

    public static Tensor<float> MatMul2D(Tensor<float> x, Tensor<float> y, TensorExecutionOptions options)
    {
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        return MatMul2DCore(x, y, DenseTensor<float>.OfShape(new int[] { x.Dimensions[0], y.Dimensions[1] }), options, clearDestination: false);
    }

    /// <summary>
    /// Writes the 2D float matrix product into an existing dense destination,
    /// overwriting it. The destination must not alias either input. The raw
    /// kernels accumulate, so this entry point clears the destination first;
    /// callers that already hold a zeroed buffer use the renting overload.
    /// </summary>
    public static Tensor<float> MatMul2D(Tensor<float> x, Tensor<float> y, DenseTensor<float> destination, TensorExecutionOptions options)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        return MatMul2DCore(x, y, destination, options, clearDestination: true);
    }

    static Tensor<float> MatMul2DCore(Tensor<float> x, Tensor<float> y, DenseTensor<float> destination, TensorExecutionOptions options, bool clearDestination)
    {
        options.Validate();
        StartOpStage(OpStage.ValidateArguments);
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException($"The number of columns in the first matrix ({x.Dimensions[1]}) is not equal to the number of rows in the second matrix ({y.Dimensions[0]}).");
        if (destination.Dimensions.Length != 2 || destination.Dimensions[0] != x.Dimensions[0] || destination.Dimensions[1] != y.Dimensions[1]) throw new ArgumentException(nameof(destination), "Destination shape must match the matrix product shape.");
        if (!HasStandardStrides(destination)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        if (ReferenceEquals(destination, x) || ReferenceEquals(destination, y)) throw new ArgumentException(nameof(destination), "Destination must not alias the input matrices.");
        if (clearDestination) destination.Buffer.Span.Clear();
        var m = x.Dimensions[0];
        var n = x.Dimensions[1];
        var k = y.Dimensions[1];

        var (_x, _y) = DensifyFloatOperands(x, y);

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
                    RunFloatMatMulKernel(rows, n, k,
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
                RunFloatMatMulKernel(m, n, k, (float*)xh.Pointer, (float*)yh.Pointer, (float*)oh.Pointer, options);
            }
        }
        return destination;
    }

    /// <summary>Computes the 2D float matrix product, renting the output from the pool when provided.</summary>
    public static Tensor<float> MatMul2D(Tensor<float> x, Tensor<float> y, TensorExecutionOptions options, TensorBufferPool? pool)
    {
        if (pool is null) return MatMul2D(x, y, options);
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        var dims = new int[] { x.Dimensions[0], y.Dimensions[1] };
        int flat;
        checked { flat = dims[0] * dims[1]; }
        var destination = new DenseTensor<float>(new Memory<float>(pool.RentCleared<float>(flat)), dims);
        return MatMul2DCore(x, y, destination, options, clearDestination: false);
    }

    public static Tensor<double> MatMul2D(Tensor<double> x, Tensor<double> y) => MatMul2D(x, y, TensorExecutionOptions.Auto);

    public static Tensor<double> MatMul2D(Tensor<double> x, Tensor<double> y, TensorExecutionOptions options)
    {
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        return MatMul2DCoreDouble(x, y, DenseTensor<double>.OfShape(new int[] { x.Dimensions[0], y.Dimensions[1] }), options, clearDestination: false);
    }

    /// <summary>
    /// Writes the 2D double matrix product into an existing dense destination,
    /// overwriting it. The destination must not alias either input. The raw
    /// kernels accumulate, so this entry point clears the destination first.
    /// </summary>
    public static Tensor<double> MatMul2D(Tensor<double> x, Tensor<double> y, DenseTensor<double> destination, TensorExecutionOptions options)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        return MatMul2DCoreDouble(x, y, destination, options, clearDestination: true);
    }

    static Tensor<double> MatMul2DCoreDouble(Tensor<double> x, Tensor<double> y, DenseTensor<double> destination, TensorExecutionOptions options, bool clearDestination)
    {
        options.Validate();
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
        if (destination.Dimensions.Length != 2 || destination.Dimensions[0] != x.Dimensions[0] || destination.Dimensions[1] != y.Dimensions[1]) throw new ArgumentException(nameof(destination), "Destination shape must match the matrix product shape.");
        if (!HasStandardStrides(destination)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        if (ReferenceEquals(destination, x) || ReferenceEquals(destination, y)) throw new ArgumentException(nameof(destination), "Destination must not alias the input matrices.");
        if (clearDestination) destination.Buffer.Span.Clear();
        var m = x.Dimensions[0];
        var n = x.Dimensions[1];
        var k = y.Dimensions[1];


        var dx = x as DenseTensor<double>;
        var dy = y as DenseTensor<double>;
        var _x = dx is not null && !dx.IsReversedStride ? dx : x.ToDenseTensor();
        var _y = dy is not null && !dy.IsReversedStride ? dy : y.ToDenseTensor();
        var output = destination;

        var xh = _x.Buffer.Pin();
        var yh = _y.Buffer.Pin();
        var oh = output.Buffer.Pin();
        if (options.UseSimd && options.UseIntrinsics && Fma.IsSupported)
        {
            unsafe
            {
                mm_unsafe_vectorized_intrinsics(m, n, k, (double*)xh.Pointer, (double*)yh.Pointer, (double*)oh.Pointer);
            }
        }
        else if (options.UseSimd)
        {
            unsafe
            {
                mm_unsafe_vectorized(m, n, k, (double*)xh.Pointer, (double*)yh.Pointer, (double*)oh.Pointer);
            }
        }
        else
            unsafe
            {
                mm(m, n, k, (double*)xh.Pointer, (double*)yh.Pointer, (double*)oh.Pointer);
            }
        xh.Dispose();
        yh.Dispose();
        oh.Dispose();
        return output;
    }

    public static Tensor<int> MatMul2D_managed(Tensor<int> x, Tensor<int> y)
    {
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
        int rA = x.Dimensions[0];
        int cA = x.Dimensions[1];
        int cB = y.Dimensions[1];
        var output = DenseTensor<int>.OfShape(new int[] { rA, cB });
        int temp;
       
        for (int i = 0; i < rA; i++)
        {
            for (int j = 0; j < cB; j++)
            {
                temp = 0;
                for (int k = 0; k < cA; k++)
                {
                    temp += x[i, k] * y[k, j];
                }
                output.SetValue((i * rA + j), temp);
            }
        }
        return output;
    }

    public static Tensor<float> MatMul2D_managed(Tensor<float> x, Tensor<float> y)
    {
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
        int rA = x.Dimensions[0];
        int cA = x.Dimensions[1];
        int cB = y.Dimensions[1];
        var output = DenseTensor<float>.OfShape(new int[] { rA, cB });
        float temp;

        for (int i = 0; i < rA; i++)
        {
            for (int j = 0; j < cB; j++)
            {
                temp = 0;
                for (int k = 0; k < cA; k++)
                {
                    temp += x[i, k] * y[k, j];
                }
                output.SetValue((i * rA + j), temp);
            }
        }
        return output;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static Tensor<int> MatMul(Tensor<int> x, Tensor<int> y) => MatMul(x, y, TensorExecutionOptions.Auto);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static Tensor<int> MatMul(Tensor<int> x, Tensor<int> y, TensorExecutionOptions options)
    
    {
        if (x.Rank == 0 || y.Rank == 0) throw new ArgumentException("The rank of each tensor in matrix multiplication must be greater than 1.");
        var plan = MatMulShapes.Create(x.Dimensions, y.Dimensions);
        var px = plan.PromoteX ? x.InsertDim(0) : x;
        var py = plan.PromoteY ? y.InsertDim(y.Rank) : y;
        Tensor<int> core;
        if (px.Rank == 2 && py.Rank == 2)
        {
            core = Tensor<int>.MatMul2D(px, py, options);
        }
        else
        {
            var xdl = px.Dimensions[^2..];
            var ydl = py.Dimensions[^2..];
            if (xdl[1] != ydl[0])
            {
                throw new ArgumentException($"The number of columns in the first matrix ({xdl[1]}) is not equal to the number of rows in the second matrix ({ydl[0]}).");
            }
            StartOpStage(OpStage.Broadcast);
            if (!BroadcastShape(px.Dimensions[0..^2], py.Dimensions[0..^2], out var bd))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }
            var bdx = bd.Append(xdl[0]).Append(xdl[1]).ToArray();
            var bdy = bd.Append(ydl[0]).Append(ydl[1]).ToArray();
            if (!Tensor<int>.Broadcast(py, bdy, out var by))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }
            if (!Tensor<int>.Broadcast(px, bdx, out var bx))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }
            var z = DenseTensor<int>.OfShape(bd.Append(xdl[0]).Append(ydl[1]).ToArray());
            var cbx = RequireContiguousInt(bx, nameof(bx));
            var cby = RequireContiguousInt(by, nameof(by));
            bx = cbx;
            by = cby;
            var batchDims = bx.Dimensions[0..^2];
            var xSteps = BatchSteps(batchDims, bx.Dimensions, bx.strides);
            var ySteps = BatchSteps(batchDims, by.Dimensions, by.strides);
            var zSteps = BatchSteps(batchDims, z.Dimensions, z.strides);
            int batchCount = BatchCount(batchDims);
            using var xh = bx.Storage.Pin();
            using var yh = by.Storage.Pin();
            using var zh = z.Storage.Pin();
            var m = bx.Dimensions[^2];
            var n = bx.Dimensions[^1];
            var k = by.Dimensions[^1];
            StartOpStage(OpStage.Math);
            unsafe
            {
                var xp = (int*)xh.Pointer;
                var yp = (int*)yh.Pointer;
                var zp = (int*)zh.Pointer;
                int r = batchDims.Length;
                var coords = new int[r];
                int ox = 0, oy = 0, oz = 0;
                for (int b = 0; b < batchCount; b++)
                {
                    if (options.UseSimd)
                    {
                        mm_unsafe_vectorized(m, n, k, xp + ox, yp + oy, zp + oz);
                    }
                    else
                    {
                        mm(m, n, k, xp + ox, yp + oy, zp + oz);
                    }
                    for (int d = r - 1; d >= 0; d--)
                    {
                        coords[d]++;
                        ox += xSteps[d]; oy += ySteps[d]; oz += zSteps[d];
                        if (coords[d] < batchDims[d]) break;
                        coords[d] = 0;
                        ox -= xSteps[d] * batchDims[d]; oy -= ySteps[d] * batchDims[d]; oz -= zSteps[d] * batchDims[d];
                    }
                }
            }
            core = z;
        }
        return MatMulShapes.Squeeze(core, plan);
    }


    static int[] MatMulOutputShape(ReadOnlySpan<int> xd, ReadOnlySpan<int> yd) => MatMulShapes.Create(xd, yd).OutputShape;

    /// <summary>
    /// Flat batch strides for one standard-dense operand: zero where the operand
    /// reuses a batch entry, flat stride otherwise. Matches GetStorageIndex on
    /// the same coordinates for densified operands.
    /// </summary>
    static int[] BatchSteps(ReadOnlySpan<int> batchDims, ReadOnlySpan<int> operandDims, int[] operandStrides)
    {
        var steps = new int[batchDims.Length];
        for (int d = 0; d < batchDims.Length; d++)
            steps[d] = (d < operandDims.Length - 2 && operandDims[d] != 1) ? operandStrides[d] : 0;
        return steps;
    }

    static int BatchCount(ReadOnlySpan<int> batchDims)
    {
        int n = 1;
        foreach (var d in batchDims) n *= d;
        return n;
    }

    /// <summary>Fills per-batch storage offsets for three operands sharing batchDims.</summary>
    static void FillBatchOffsets(ReadOnlySpan<int> batchDims, int[] xSteps, int[] ySteps, int[] zSteps, int[] xOff, int[] yOff, int[] zOff)
    {
        int r = batchDims.Length;
        var coords = new int[r];
        int ox = 0, oy = 0, oz = 0;
        for (int b = 0; b < xOff.Length; b++)
        {
            xOff[b] = ox; yOff[b] = oy; zOff[b] = oz;
            for (int d = r - 1; d >= 0; d--)
            {
                coords[d]++;
                ox += xSteps[d]; oy += ySteps[d]; oz += zSteps[d];
                if (coords[d] < batchDims[d]) break;
                coords[d] = 0;
                ox -= xSteps[d] * batchDims[d]; oy -= ySteps[d] * batchDims[d]; oz -= zSteps[d] * batchDims[d];
            }
        }
    }

    static void RunBatchedFloatMatMul(Tensor<float> bx, Tensor<float> by, Tensor<float> z, TensorExecutionOptions options)
    {
        bx = RequireContiguousFloat(bx, nameof(bx));
        by = RequireContiguousFloat(by, nameof(by));
        z = RequireContiguousFloat(z, nameof(z));
        var batchDims = bx.Dimensions[0..^2];
        var m = bx.Dimensions[^2];
        var n = bx.Dimensions[^1];
        var k = by.Dimensions[^1];
        var xSteps = BatchSteps(batchDims, bx.Dimensions, bx.strides);
        var ySteps = BatchSteps(batchDims, by.Dimensions, by.strides);
        var zSteps = BatchSteps(batchDims, z.Dimensions, z.strides);
        int batchCount = BatchCount(batchDims);
        int dop = options.MaxDegreeOfParallelism < 2 || batchCount < 2
            ? 1
            : Math.Min(options.MaxDegreeOfParallelism, batchCount);
        using var xh = bx.Storage.Pin();
        using var yh = by.Storage.Pin();
        using var zh = z.Storage.Pin();
        IntPtr xp0, yp0, zp0;
        unsafe { xp0 = (IntPtr)xh.Pointer; yp0 = (IntPtr)yh.Pointer; zp0 = (IntPtr)zh.Pointer; }
        if (dop > 1)
        {
            var xOff = new int[batchCount];
            var yOff = new int[batchCount];
            var zOff = new int[batchCount];
            FillBatchOffsets(batchDims, xSteps, ySteps, zSteps, xOff, yOff, zOff);
            Parallel.For(0, batchCount, new ParallelOptions { MaxDegreeOfParallelism = dop }, bi =>
            {
                unsafe
                {
                    RunFloatMatMulKernel(m, n, k,
                        (float*)xp0 + xOff[bi],
                        (float*)yp0 + yOff[bi],
                        (float*)zp0 + zOff[bi], options);
                }
            });
        }
        else
        {
            unsafe
            {
                var xp = (float*)xp0;
                var yp = (float*)yp0;
                var zp = (float*)zp0;
                int r = batchDims.Length;
                var coords = new int[r];
                int ox = 0, oy = 0, oz = 0;
                for (int b = 0; b < batchCount; b++)
                {
                    RunFloatMatMulKernel(m, n, k, xp + ox, yp + oy, zp + oz, options);
                    for (int d = r - 1; d >= 0; d--)
                    {
                        coords[d]++;
                        ox += xSteps[d]; oy += ySteps[d]; oz += zSteps[d];
                        if (coords[d] < batchDims[d]) break;
                        coords[d] = 0;
                        ox -= xSteps[d] * batchDims[d]; oy -= ySteps[d] * batchDims[d]; oz -= zSteps[d] * batchDims[d];
                    }
                }
            }
        }
    }

    /// <summary>
    /// Writes the float matrix product into an existing dense destination,
    /// overwriting it. The destination must not alias either input.
    /// </summary>
    public static Tensor<float> MatMul(Tensor<float> x, Tensor<float> y, DenseTensor<float> destination, TensorExecutionOptions options)

    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        if (ReferenceEquals(destination, x) || ReferenceEquals(destination, y)) throw new ArgumentException(nameof(destination), "Destination must not alias the input tensors.");
        var plan = MatMulShapes.Create(x.Dimensions, y.Dimensions);
        if (!destination.Dimensions.SequenceEqual(plan.OutputShape)) throw new ArgumentException(nameof(destination), "Destination shape must match the matrix product shape.");
        if (!HasStandardStrides(destination)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        var px = plan.PromoteX ? x.InsertDim(0) : x;
        var py = plan.PromoteY ? y.InsertDim(y.Rank) : y;
        Tensor<float> core;
        if (px.Rank == 2 && py.Rank == 2)
        {
            core = Tensor<float>.MatMul2D(px, py, options);
        }
        else
        {
            var xdl = px.Dimensions[^2..];
            var ydl = py.Dimensions[^2..];
            if (xdl[1] != ydl[0])
            {
                throw new ArgumentException($"The number of columns in the first matrix ({xdl[1]}) is not equal to the number of rows in the second matrix ({ydl[0]}).");
            }
            StartOpStage(OpStage.Broadcast);
            if (!BroadcastShape(px.Dimensions[0..^2], py.Dimensions[0..^2], out var bd))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }

            var bdx = bd.Append(xdl[0]).Append(xdl[1]).ToArray();
            if (!Tensor<float>.Broadcast(px, bdx, out var bx))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }
            var bdy = bd.Append(ydl[0]).Append(ydl[1]).ToArray();
            if (!Tensor<float>.Broadcast(py, bdy, out var by))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }

            StartOpStage(OpStage.Math);
            var z = DenseTensor<float>.OfShape(bd.Append(xdl[0]).Append(ydl[1]).ToArray());
            RunBatchedFloatMatMul(bx, by, z, options);
            core = z;
        }
        var squeezed = MatMulShapes.Squeeze(core, plan);
        for (int i = 0; i < (int)squeezed.Length; i++) destination.SetValue(i, squeezed.GetValue(i));
        return destination;
    }


    /// <summary>Computes the float matrix product, renting the output from the pool when provided.</summary>
    public static Tensor<float> MatMul(Tensor<float> x, Tensor<float> y, TensorExecutionOptions options, TensorBufferPool? pool)
    {
        if (pool is null) return MatMul(x, y, options);
        var dims = MatMulOutputShape(x.Dimensions, y.Dimensions);
        long length = 1;
        checked
        {
            foreach (var d in dims) length *= d;
        }
        if (length > int.MaxValue) throw new ArgumentException("MatMul output element count exceeds maximum backing-store length.");
        var destination = new DenseTensor<float>(new Memory<float>(pool.RentCleared<float>((int)length)), dims);
        return MatMul(x, y, destination, options);
    }
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]  
    public static Tensor<float> MatMul(Tensor<float> x, Tensor<float> y) => MatMul(x, y, TensorExecutionOptions.Auto);

    public static Tensor<float> MatMul(Tensor<float> x, Tensor<float> y, TensorExecutionOptions options)
    
    {
        if (x.Rank == 0 || y.Rank == 0) throw new ArgumentException("The rank of each tensor in matrix multiplication must be greater than 1.");
        var plan = MatMulShapes.Create(x.Dimensions, y.Dimensions);
        var px = plan.PromoteX ? x.InsertDim(0) : x;
        var py = plan.PromoteY ? y.InsertDim(y.Rank) : y;
        Tensor<float> core;
        if (px.Rank == 2 && py.Rank == 2)
        {
            core = Tensor<float>.MatMul2D(px, py, options);
        }
        else
        {
            var xdl = px.Dimensions[^2..];
            var ydl = py.Dimensions[^2..];
            if (xdl[1] != ydl[0])
            {
                throw new ArgumentException($"The number of columns in the first matrix ({xdl[1]}) is not equal to the number of rows in the second matrix ({ydl[0]}).");
            }
            StartOpStage(OpStage.Broadcast);
            if (!BroadcastShape(px.Dimensions[0..^2], py.Dimensions[0..^2], out var bd))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }

            var bdx = bd.Append(xdl[0]).Append(xdl[1]).ToArray();
            if (!Tensor<float>.Broadcast(px, bdx, out var bx))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }
            var bdy = bd.Append(ydl[0]).Append(ydl[1]).ToArray();
            if (!Tensor<float>.Broadcast(py, bdy, out var by))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }

            StartOpStage(OpStage.Math);

            var z = DenseTensor<float>.OfShape(bd.Append(xdl[0]).Append(ydl[1]).ToArray());
            RunBatchedFloatMatMul(bx, by, z, options);
            core = z;
        }
        return MatMulShapes.Squeeze(core, plan);
    }


    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static Tensor<double> MatMul(Tensor<double> x, Tensor<double> y) => MatMul(x, y, TensorExecutionOptions.Auto);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static Tensor<double> MatMul(Tensor<double> x, Tensor<double> y, TensorExecutionOptions options)
    
    {
        if (x.Rank == 0 || y.Rank == 0) throw new ArgumentException("The rank of each tensor in matrix multiplication must be greater than 1.");
        var plan = MatMulShapes.Create(x.Dimensions, y.Dimensions);
        var px = plan.PromoteX ? x.InsertDim(0) : x;
        var py = plan.PromoteY ? y.InsertDim(y.Rank) : y;
        Tensor<double> core;
        if (px.Rank == 2 && py.Rank == 2)
        {
            core = Tensor<double>.MatMul2D(px, py, options);
        }
        else
        {
            var xdl = px.Dimensions[^2..];
            var ydl = py.Dimensions[^2..];
            if (xdl[1] != ydl[0])
            {
                throw new ArgumentException($"The number of columns in the first matrix ({xdl[1]}) is not equal to the number of rows in the second matrix ({ydl[0]}).");
            }
            StartOpStage(OpStage.Broadcast);
            if (!BroadcastShape(px.Dimensions[0..^2], py.Dimensions[0..^2], out var bd))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }

            var bdx = bd.Append(xdl[0]).Append(xdl[1]).ToArray();
            if (!Tensor<double>.Broadcast(px, bdx, out var bx))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }
            var bdy = bd.Append(ydl[0]).Append(ydl[1]).ToArray();
            if (!Tensor<double>.Broadcast(py, bdy, out var by))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }

            StartOpStage(OpStage.Math);
            var z = DenseTensor<double>.OfShape(bd.Append(xdl[0]).Append(ydl[1]).ToArray());
            var cbx = RequireContiguousDouble(bx, nameof(bx));
            var cby = RequireContiguousDouble(by, nameof(by));
            bx = cbx;
            by = cby;
            var batchDims = bx.Dimensions[0..^2];
            var xSteps = BatchSteps(batchDims, bx.Dimensions, bx.strides);
            var ySteps = BatchSteps(batchDims, by.Dimensions, by.strides);
            var zSteps = BatchSteps(batchDims, z.Dimensions, z.strides);
            int batchCount = BatchCount(batchDims);
            using var xh = bx.Storage.Pin();
            using var yh = by.Storage.Pin();
            using var zh = z.Storage.Pin();
            var m = bx.Dimensions[^2];
            var n = bx.Dimensions[^1];
            var k = by.Dimensions[^1];
            unsafe
            {
                var xp = (double*)xh.Pointer;
                var yp = (double*)yh.Pointer;
                var zp = (double*)zh.Pointer;
                int r = batchDims.Length;
                var coords = new int[r];
                int ox = 0, oy = 0, oz = 0;
                for (int b = 0; b < batchCount; b++)
                {
                    if (options.UseSimd && options.UseIntrinsics && Fma.IsSupported)
                    {
                        mm_unsafe_vectorized_intrinsics(m, n, k, xp + ox, yp + oy, zp + oz);
                    }
                    else if (options.UseSimd)
                    {
                        mm_unsafe_vectorized(m, n, k, xp + ox, yp + oy, zp + oz);
                    }
                    else
                    {
                        mm(m, n, k, xp + ox, yp + oy, zp + oz);
                    }
                    for (int d = r - 1; d >= 0; d--)
                    {
                        coords[d]++;
                        ox += xSteps[d]; oy += ySteps[d]; oz += zSteps[d];
                        if (coords[d] < batchDims[d]) break;
                        coords[d] = 0;
                        ox -= xSteps[d] * batchDims[d]; oy -= ySteps[d] * batchDims[d]; oz -= zSteps[d] * batchDims[d];
                    }
                }
            }
            core = z;
        }
        return MatMulShapes.Squeeze(core, plan);
    }


    public static Tensor<float> Conv2D(Tensor<float> input, Tensor<float> weight, int group, PadType padtype, int? padvalue, Tensor<float>? bias, int[]? kernelshape, int[]? strides, int[]? dilations) =>
        Conv2D(input, weight, group, padtype, padvalue, bias, kernelshape, strides, dilations, TensorExecutionOptions.Auto);

    /// <summary>Two-dimensional convolution with explicit execution options.</summary>
    public static Tensor<float> Conv2D(Tensor<float> input, Tensor<float> weight, int group, PadType padtype, int? padvalue, Tensor<float>? bias, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options)
    {
        if (input.Rank != 4)
        {
            throw new ArgumentException(nameof(input), "Input tensors must be of rank 4 with the layout NxCxHxW.");
        }
        if (weight.Rank != 4)
        {
            throw new ArgumentException(nameof(weight), "Weight tensors must be of rank 4 with the layout M x C/group x kH x kW.");
        }
        if (strides == null)
        {
            strides = new int[2] { 1, 1 };
        }
        if (dilations == null)
        {
            dilations = new int[2] { 1, 1 };
        }
        var N = input.Dimensions[0];
        var C = input.Dimensions[1];
        var H = input.Dimensions[2];
        var W = input.Dimensions[3];
        var M = weight.Dimensions[0];
        var kH = kernelshape == null ? weight.Dimensions[2] : kernelshape[0];
        var kW = kernelshape == null ? weight.Dimensions[3] : kernelshape[1];
        ValidateConv2D(N, C, H, W, M, weight.Dimensions[1], kH, kW, group, strides, dilations, kernelshape, weight.Dimensions.ToArray(), input.Length, weight.Length, bias is null ? -1 : (int)bias.Length);
        var info = GetConv2DOutputInfo(padtype, H, W, strides[0], strides[1], GetConv2DEffectiveFilterSize(kH, dilations[0]), GetConv2DEffectiveFilterSize(kW, dilations[1]), padvalue);
        if (info.Shape[0] <= 0 || info.Shape[1] <= 0) throw new ArgumentException("Conv output spatial dims must be positive.");
        return Conv2DFloatCore(input, weight, group, N, C, H, W, M, kH, kW, dilations[0], dilations[1], strides[0], strides[1], info.PadInfo, info.Shape[0], info.Shape[1], bias, options);

    }

    public static Tensor<float> Conv2D(Tensor<float> input, Tensor<float> weight, int group, int[] pads, Tensor<float>? bias, int[]? kernelshape, int[]? strides, int[]? dilations) =>
        Conv2D(input, weight, group, pads, bias, kernelshape, strides, dilations, TensorExecutionOptions.Auto);

    /// <summary>Two-dimensional convolution with explicit execution options.</summary>
    public static Tensor<float> Conv2D(Tensor<float> input, Tensor<float> weight, int group, int[] pads, Tensor<float>? bias, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options)
    {
        if (input.Rank != 4)
        {
            throw new ArgumentException(nameof(input), "Input tensors must be of rank 4 with the layout NxCxHxW.");
        }
        if (weight.Rank != 4)
        {
            throw new ArgumentException(nameof(weight), "Weight tensors must be of rank 4 with the layout M x C/group x kH x kW.");
        }
        if (pads is null || pads.Length != 4)
        {
            throw new ArgumentException(nameof(pads), "Explicit pads must have four values [begin_h, begin_w, end_h, end_w].");
        }
        if (strides == null)
        {
            strides = new int[2] { 1, 1 };
        }
        if (dilations == null)
        {
            dilations = new int[2] { 1, 1 };
        }
        var N = input.Dimensions[0];
        var C = input.Dimensions[1];
        var H = input.Dimensions[2];
        var W = input.Dimensions[3];
        var M = weight.Dimensions[0];
        var kH = kernelshape == null ? weight.Dimensions[2] : kernelshape[0];
        var kW = kernelshape == null ? weight.Dimensions[3] : kernelshape[1];
        ValidateConv2D(N, C, H, W, M, weight.Dimensions[1], kH, kW, group, strides, dilations, kernelshape, weight.Dimensions.ToArray(), input.Length, weight.Length, bias is null ? -1 : (int)bias.Length);
        int effKH = GetConv2DEffectiveFilterSize(kH, dilations[0]);
        int effKW = GetConv2DEffectiveFilterSize(kW, dilations[1]);
        var outShape = GetConv2DOutputShape(new int[] { H, W }, effKH, effKW, strides[0], strides[1], pads[0] + pads[2], pads[1] + pads[3]);
        if (outShape[0] <= 0 || outShape[1] <= 0) throw new ArgumentException("Conv output spatial dims must be positive.");
        var pad = new PadInfo { top = pads[0], left = pads[1], bottom = pads[2], right = pads[3], h = pads[0] + pads[2], w = pads[1] + pads[3] };
        return Conv2DFloatCore(input, weight, group, N, C, H, W, M, kH, kW, dilations[0], dilations[1], strides[0], strides[1], pad, outShape[0], outShape[1], bias, options);

    }

    static Tensor<float> Conv2DFloatCore(Tensor<float> input, Tensor<float> weight, int group, int N, int C, int H, int W, int M, int kH, int kW, int dH, int dW, int sH, int sW, PadInfo pad, int outH, int outW, Tensor<float>? bias, TensorExecutionOptions options)
    {
        options.Validate();
        var output = new DenseTensor<float>((ReadOnlySpan<int>)new int[] { N, M, outH, outW });
        var xd = input.ToDenseTensor();
        var wd = weight.ToDenseTensor();
        var bd = bias?.ToDenseTensor();
        int inBatch = C * H * W;
        int outBatch = M * outH * outW;
        int patchSize = C * kH * kW * outH * outW;
        int dop = options.MaxDegreeOfParallelism < 2 || N < 2 ? 1 : Math.Min(options.MaxDegreeOfParallelism, N);
        var xMem = xd.Buffer;
        var wMem = wd.Buffer;
        var oMem = output.Buffer;
        var bMem = bd is null ? default : bd.Buffer;
        bool hasBias = bd is not null;
        if (dop > 1)
        {
            Parallel.For(0, N, new ParallelOptions { MaxDegreeOfParallelism = dop },
                () => ArrayPool<float>.Shared.Rent(patchSize),
                (b, state, scratch) =>
                {
                    RunConvBatchFloat(xMem, wMem, bMem, hasBias, oMem, scratch, patchSize, b, group, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, inBatch, outBatch, options);
                    return scratch;
                },
                scratch => ArrayPool<float>.Shared.Return(scratch));
        }
        else
        {
            var scratch = ArrayPool<float>.Shared.Rent(patchSize);
            try
            {
                for (int b = 0; b < N; b++)
                    RunConvBatchFloat(xMem, wMem, bMem, hasBias, oMem, scratch, patchSize, b, group, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, inBatch, outBatch, options);
            }
            finally { ArrayPool<float>.Shared.Return(scratch); }
        }

        return output;

    }

    /// <summary>
    /// Runs one batch of float convolution: im2col into pooled scratch, then one
    /// shared-dispatcher product per group (which clears each destination tile,
    /// preserving the legacy clearing semantics), then bias.
    /// </summary>
    static void RunConvBatchFloat(Memory<float> xMem, Memory<float> wMem, Memory<float> bMem, bool hasBias, Memory<float> oMem, float[] scratch, int patchSize, int b, int group, int C, int H, int W, int M, int kH, int kW, int dH, int dW, int sH, int sW, PadInfo pad, int outH, int outW, int inBatch, int outBatch, TensorExecutionOptions options)
    {
        var patchMem = new Memory<float>(scratch, 0, patchSize);
        unsafe
        {
            fixed (float* src = xMem.Span.Slice(b * inBatch, inBatch))
            fixed (float* patch = patchMem.Span)
            {
                MathOps.Im2col(src, C, H, W, kH, kW, dH, dW, sH, sW, pad.top, pad.left, pad.bottom, pad.right, patch);
            }
        }
        int tileM = M / group;
        int tileN = outH * outW;
        int tileK = C * kH * kW / group;
        for (int g = 0; g < group; g++)
        {
            var wView = new DenseTensor<float>(wMem.Slice(g * tileM * tileK, tileM * tileK), new int[] { tileM, tileK });
            var pView = new DenseTensor<float>(patchMem.Slice(g * tileK * tileN, tileK * tileN), new int[] { tileK, tileN });
            var dView = new DenseTensor<float>(oMem.Slice(b * outBatch + g * tileM * tileN, tileM * tileN), new int[] { tileM, tileN });
            Tensor<float>.MatMul2D(wView, pView, dView, options);
        }
        if (hasBias)
        {
            var bs = bMem.Span;
            var os = oMem.Span;
            for (int i = 0; i < M; i++)
            {
                float bi = bs[i];
                int row = b * outBatch + i * tileN;
                for (int j = 0; j < tileN; j++) os[row + j] += bi;
            }
        }
    }

    public static Tensor<double> Conv2D(Tensor<double> input, Tensor<double> weight, int group, PadType padtype, int? padvalue, Tensor<double>? bias, int[]? kernelshape, int[]? strides, int[]? dilations) =>
        Conv2D(input, weight, group, padtype, padvalue, bias, kernelshape, strides, dilations, TensorExecutionOptions.Auto);

    /// <summary>Two-dimensional convolution with explicit execution options.</summary>
    public static Tensor<double> Conv2D(Tensor<double> input, Tensor<double> weight, int group, PadType padtype, int? padvalue, Tensor<double>? bias, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options)
    {
        if (input.Rank != 4)
        {
            throw new ArgumentException(nameof(input), "Input tensors must be of rank 4 with the layout NxCxHxW.");
        }
        if (weight.Rank != 4)
        {
            throw new ArgumentException(nameof(weight), "Weight tensors must be of rank 4 with the layout M x C/group x kH x kW.");
        }
        if (strides == null)
        {
            strides = new int[2] { 1, 1 };
        }
        if (dilations == null)
        {
            dilations = new int[2] { 1, 1 };
        }
        var N = input.Dimensions[0];
        var C = input.Dimensions[1];
        var H = input.Dimensions[2];
        var W = input.Dimensions[3];
        var M = weight.Dimensions[0];
        var kH = kernelshape == null ? weight.Dimensions[2] : kernelshape[0];
        var kW = kernelshape == null ? weight.Dimensions[3] : kernelshape[1];
        ValidateConv2D(N, C, H, W, M, weight.Dimensions[1], kH, kW, group, strides, dilations, kernelshape, weight.Dimensions.ToArray(), input.Length, weight.Length, bias is null ? -1 : (int)bias.Length);
        var info = GetConv2DOutputInfo(padtype, H, W, strides[0], strides[1], GetConv2DEffectiveFilterSize(kH, dilations[0]), GetConv2DEffectiveFilterSize(kW, dilations[1]), padvalue);
        if (info.Shape[0] <= 0 || info.Shape[1] <= 0) throw new ArgumentException("Conv output spatial dims must be positive.");
        return Conv2DDoubleCore(input, weight, group, N, C, H, W, M, kH, kW, dilations[0], dilations[1], strides[0], strides[1], info.PadInfo, info.Shape[0], info.Shape[1], bias, options);

    }

    public static Tensor<double> Conv2D(Tensor<double> input, Tensor<double> weight, int group, int[] pads, Tensor<double>? bias, int[]? kernelshape, int[]? strides, int[]? dilations) =>
        Conv2D(input, weight, group, pads, bias, kernelshape, strides, dilations, TensorExecutionOptions.Auto);

    /// <summary>Two-dimensional convolution with explicit execution options.</summary>
    public static Tensor<double> Conv2D(Tensor<double> input, Tensor<double> weight, int group, int[] pads, Tensor<double>? bias, int[]? kernelshape, int[]? strides, int[]? dilations, TensorExecutionOptions options)
    {
        if (input.Rank != 4)
        {
            throw new ArgumentException(nameof(input), "Input tensors must be of rank 4 with the layout NxCxHxW.");
        }
        if (weight.Rank != 4)
        {
            throw new ArgumentException(nameof(weight), "Weight tensors must be of rank 4 with the layout M x C/group x kH x kW.");
        }
        if (pads is null || pads.Length != 4)
        {
            throw new ArgumentException(nameof(pads), "Explicit pads must have four values [begin_h, begin_w, end_h, end_w].");
        }
        if (strides == null)
        {
            strides = new int[2] { 1, 1 };
        }
        if (dilations == null)
        {
            dilations = new int[2] { 1, 1 };
        }
        var N = input.Dimensions[0];
        var C = input.Dimensions[1];
        var H = input.Dimensions[2];
        var W = input.Dimensions[3];
        var M = weight.Dimensions[0];
        var kH = kernelshape == null ? weight.Dimensions[2] : kernelshape[0];
        var kW = kernelshape == null ? weight.Dimensions[3] : kernelshape[1];
        ValidateConv2D(N, C, H, W, M, weight.Dimensions[1], kH, kW, group, strides, dilations, kernelshape, weight.Dimensions.ToArray(), input.Length, weight.Length, bias is null ? -1 : (int)bias.Length);
        int effKH = GetConv2DEffectiveFilterSize(kH, dilations[0]);
        int effKW = GetConv2DEffectiveFilterSize(kW, dilations[1]);
        var outShape = GetConv2DOutputShape(new int[] { H, W }, effKH, effKW, strides[0], strides[1], pads[0] + pads[2], pads[1] + pads[3]);
        if (outShape[0] <= 0 || outShape[1] <= 0) throw new ArgumentException("Conv output spatial dims must be positive.");
        var pad = new PadInfo { top = pads[0], left = pads[1], bottom = pads[2], right = pads[3], h = pads[0] + pads[2], w = pads[1] + pads[3] };
        return Conv2DDoubleCore(input, weight, group, N, C, H, W, M, kH, kW, dilations[0], dilations[1], strides[0], strides[1], pad, outShape[0], outShape[1], bias, options);

    }

    static Tensor<double> Conv2DDoubleCore(Tensor<double> input, Tensor<double> weight, int group, int N, int C, int H, int W, int M, int kH, int kW, int dH, int dW, int sH, int sW, PadInfo pad, int outH, int outW, Tensor<double>? bias, TensorExecutionOptions options)
    {
        options.Validate();
        var output = new DenseTensor<double>((ReadOnlySpan<int>)new int[] { N, M, outH, outW });
        var xd = input.ToDenseTensor();
        var wd = weight.ToDenseTensor();
        var bd = bias?.ToDenseTensor();
        int inBatch = C * H * W;
        int outBatch = M * outH * outW;
        int patchSize = C * kH * kW * outH * outW;
        int dop = options.MaxDegreeOfParallelism < 2 || N < 2 ? 1 : Math.Min(options.MaxDegreeOfParallelism, N);
        var xMem = xd.Buffer;
        var wMem = wd.Buffer;
        var oMem = output.Buffer;
        var bMem = bd is null ? default : bd.Buffer;
        bool hasBias = bd is not null;
        if (dop > 1)
        {
            Parallel.For(0, N, new ParallelOptions { MaxDegreeOfParallelism = dop },
                () => ArrayPool<double>.Shared.Rent(patchSize),
                (b, state, scratch) =>
                {
                    RunConvBatchDouble(xMem, wMem, bMem, hasBias, oMem, scratch, patchSize, b, group, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, inBatch, outBatch, options);
                    return scratch;
                },
                scratch => ArrayPool<double>.Shared.Return(scratch));
        }
        else
        {
            var scratch = ArrayPool<double>.Shared.Rent(patchSize);
            try
            {
                for (int b = 0; b < N; b++)
                    RunConvBatchDouble(xMem, wMem, bMem, hasBias, oMem, scratch, patchSize, b, group, C, H, W, M, kH, kW, dH, dW, sH, sW, pad, outH, outW, inBatch, outBatch, options);
            }
            finally { ArrayPool<double>.Shared.Return(scratch); }
        }

        return output;

    }

    /// <summary>
    /// Runs one batch of double convolution: im2col into pooled scratch, then one
    /// shared-dispatcher product per group (which clears each destination tile,
    /// preserving the legacy clearing semantics), then bias.
    /// </summary>
    static void RunConvBatchDouble(Memory<double> xMem, Memory<double> wMem, Memory<double> bMem, bool hasBias, Memory<double> oMem, double[] scratch, int patchSize, int b, int group, int C, int H, int W, int M, int kH, int kW, int dH, int dW, int sH, int sW, PadInfo pad, int outH, int outW, int inBatch, int outBatch, TensorExecutionOptions options)
    {
        var patchMem = new Memory<double>(scratch, 0, patchSize);
        unsafe
        {
            fixed (double* src = xMem.Span.Slice(b * inBatch, inBatch))
            fixed (double* patch = patchMem.Span)
            {
                MathOps.Im2col(src, C, H, W, kH, kW, dH, dW, sH, sW, pad.top, pad.left, pad.bottom, pad.right, patch);
            }
        }
        int tileM = M / group;
        int tileN = outH * outW;
        int tileK = C * kH * kW / group;
        for (int g = 0; g < group; g++)
        {
            var wView = new DenseTensor<double>(wMem.Slice(g * tileM * tileK, tileM * tileK), new int[] { tileM, tileK });
            var pView = new DenseTensor<double>(patchMem.Slice(g * tileK * tileN, tileK * tileN), new int[] { tileK, tileN });
            var dView = new DenseTensor<double>(oMem.Slice(b * outBatch + g * tileM * tileN, tileM * tileN), new int[] { tileM, tileN });
            Tensor<double>.MatMul2D(wView, pView, dView, options);
        }
        if (hasBias)
        {
            var bs = bMem.Span;
            var os = oMem.Span;
            for (int i = 0; i < M; i++)
            {
                double bi = bs[i];
                int row = b * outBatch + i * tileN;
                for (int j = 0; j < tileN; j++) os[row + j] += bi;
            }
        }
    }

    public static Tensor<float> MaxPool2D(Tensor<float> input, int[] kernelshape, PadType padtype, int? padvalue, int[]? strides, int[]? dilations, bool ceilMode)
    {
        if (kernelshape is null)
        {
            throw new ArgumentNullException("kernelshape");
        }
        if (input.Rank != 4)
        {
            throw new ArgumentException("Input tensors must be of rank 4 with the layout NxCxHxW.");
        }
        if (kernelshape.Rank != 1 || kernelshape.Length != 2)
        {
            throw new ArgumentException("The kernel must have shape m x n.");
        }

        if (strides == null)
        {
            strides = kernelshape;
        }
        if (dilations == null)
        {
            dilations = new int[] { 1, 1 };
        }

        var N = input.Dimensions[0];
        var C = input.Dimensions[1];
        var H = input.Dimensions[2];
        var W = input.Dimensions[3];
        var kH = kernelshape[0];
        var kW = kernelshape[1];
        var strideHeight = strides[0];
        var strideWidth = strides[1];
        var info = GetConv2DOutputInfo(padtype, H, W, strides[0], strides[1], GetConv2DEffectiveFilterSize(kH, dilations[0]), GetConv2DEffectiveFilterSize(kW, dilations[1]), padvalue);
        if (!ceilMode) return MaxPoolFloatCore(input, N, C, H, W, kH, kW, strideHeight, strideWidth, dilations[0], dilations[1], info.PadInfo, info.Shape[0], info.Shape[1]);
        int effKH = GetConv2DEffectiveFilterSize(kH, dilations[0]);
        int effKW = GetConv2DEffectiveFilterSize(kW, dilations[1]);
        var ceilShape = MaxPoolOutputShape(H, W, effKH, effKW, strideHeight, strideWidth, info.PadInfo.top + info.PadInfo.bottom, info.PadInfo.left + info.PadInfo.right, true);
        return MaxPoolFloatCore(input, N, C, H, W, kH, kW, strideHeight, strideWidth, dilations[0], dilations[1], info.PadInfo, ceilShape[0], ceilShape[1]);
    }

    public static Tensor<float> MaxPool2D(Tensor<float> input, int[] kernelshape, int[] pads, int[]? strides, int[]? dilations, bool ceilMode)
    {
        if (kernelshape is null)
        {
            throw new ArgumentNullException("kernelshape");
        }
        if (input.Rank != 4)
        {
            throw new ArgumentException("Input tensors must be of rank 4 with the layout NxCxHxW.");
        }
        if (kernelshape.Rank != 1 || kernelshape.Length != 2)
        {
            throw new ArgumentException("The kernel must have shape m x n.");
        }
        if (pads is null || pads.Length != 4)
        {
            throw new ArgumentException(nameof(pads), "Explicit pads must have four values [begin_h, begin_w, end_h, end_w].");
        }

        if (strides == null)
        {
            strides = kernelshape;
        }
        if (dilations == null)
        {
            dilations = new int[] { 1, 1 };
        }

        var N = input.Dimensions[0];
        var C = input.Dimensions[1];
        var H = input.Dimensions[2];
        var W = input.Dimensions[3];
        var kH = kernelshape[0];
        var kW = kernelshape[1];
        int effKH = GetConv2DEffectiveFilterSize(kH, dilations[0]);
        int effKW = GetConv2DEffectiveFilterSize(kW, dilations[1]);
        var outShape = MaxPoolOutputShape(H, W, effKH, effKW, strides[0], strides[1], pads[0] + pads[2], pads[1] + pads[3], ceilMode);
        if (outShape[0] <= 0 || outShape[1] <= 0) throw new ArgumentException("MaxPool output spatial dims must be positive.");
        var pad = new PadInfo { top = pads[0], left = pads[1], bottom = pads[2], right = pads[3], h = pads[0] + pads[2], w = pads[1] + pads[3] };
        return MaxPoolFloatCore(input, N, C, H, W, kH, kW, strides[0], strides[1], dilations[0], dilations[1], pad, outShape[0], outShape[1]);
    }

    static int[] MaxPoolOutputShape(int H, int W, int effKH, int effKW, int sH, int sW, int padH, int padW, bool ceilMode)
    {
        int outH = ceilMode
            ? (int)Math.Ceiling((H + padH - effKH) / (float)sH) + 1
            : (int)Math.Floor((H + padH - effKH) / (float)sH) + 1;
        int outW = ceilMode
            ? (int)Math.Ceiling((W + padW - effKW) / (float)sW) + 1
            : (int)Math.Floor((W + padW - effKW) / (float)sW) + 1;
        return new int[] { outH, outW };
    }

    static Tensor<float> MaxPoolFloatCore(Tensor<float> input, int N, int C, int H, int W, int kH, int kW, int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, PadInfo pad, int outH, int outW)
    {
        var Y = DenseTensor<float>.OfShape(N, C, outH, outW);

        for (var n = 0; n < N; ++n)
        {
            for (var d = 0; d < C; ++d)
            {
                for (var yR = 0; yR < outH; ++yR)
                {
                    var xRCorner = yR * strideHeight - pad.top;
                    for (var yC = 0; yC < outW; ++yC)
                    {
                        var xCCorner = yC * strideWidth - pad.left;

                        var maxValue = float.NegativeInfinity;

                        for (var tR = 0; tR < kH; ++tR)
                        {
                            var xR = xRCorner + tR * dilationHeight;
                            if (xR < 0 || xR >= H) continue;
                            for (var tC = 0; tC < kW; ++tC)
                            {
                                var xC = xCCorner + tC * dilationWidth;
                                if (xC < 0 || xC >= W) continue;
                                var v = input[n, d, xR, xC];

                                if (v > maxValue)
                                {
                                    maxValue = v;
                                }
                            }
                        }
                        Y[n, d, yR, yC] = maxValue;
                    }
                }
            }
        }
        return Y;
    }

    public static Tensor<double> MaxPool2D(Tensor<double> input, int[] kernelshape, PadType padtype, int? padvalue, int[]? strides, int[]? dilations, bool ceilMode)
    {
        if (kernelshape is null)
        {
            throw new ArgumentNullException("kernelshape");
        }
        if (input.Rank != 4)
        {
            throw new ArgumentException("Input tensors must be of rank 4 with the layout NxCxHxW.");
        }
        if (kernelshape.Rank != 1 || kernelshape.Length != 2)
        {
            throw new ArgumentException("The kernel must have shape m x n.");
        }

        if (strides is null)
        {
            strides = kernelshape;
        }
        if (dilations is null)
        {
            dilations = new int[] { 1, 1 };
        }

        var N = input.Dimensions[0];
        var C = input.Dimensions[1];
        var H = input.Dimensions[2];
        var W = input.Dimensions[3];
        var kH = kernelshape[0];
        var kW = kernelshape[1];
        var strideHeight = strides[0];
        var strideWidth = strides[1];
        var info = GetConv2DOutputInfo(padtype, H, W, strides[0], strides[1], GetConv2DEffectiveFilterSize(kH, dilations[0]), GetConv2DEffectiveFilterSize(kW, dilations[1]), padvalue);
        if (!ceilMode) return MaxPoolDoubleCore(input, N, C, H, W, kH, kW, strideHeight, strideWidth, dilations[0], dilations[1], info.PadInfo, info.Shape[0], info.Shape[1]);
        int effKH = GetConv2DEffectiveFilterSize(kH, dilations[0]);
        int effKW = GetConv2DEffectiveFilterSize(kW, dilations[1]);
        var ceilShape = MaxPoolOutputShape(H, W, effKH, effKW, strideHeight, strideWidth, info.PadInfo.top + info.PadInfo.bottom, info.PadInfo.left + info.PadInfo.right, true);
        return MaxPoolDoubleCore(input, N, C, H, W, kH, kW, strideHeight, strideWidth, dilations[0], dilations[1], info.PadInfo, ceilShape[0], ceilShape[1]);
    }

    public static Tensor<double> MaxPool2D(Tensor<double> input, int[] kernelshape, int[] pads, int[]? strides, int[]? dilations, bool ceilMode)
    {
        if (kernelshape is null)
        {
            throw new ArgumentNullException("kernelshape");
        }
        if (input.Rank != 4)
        {
            throw new ArgumentException("Input tensors must be of rank 4 with the layout NxCxHxW.");
        }
        if (kernelshape.Rank != 1 || kernelshape.Length != 2)
        {
            throw new ArgumentException("The kernel must have shape m x n.");
        }
        if (pads is null || pads.Length != 4)
        {
            throw new ArgumentException(nameof(pads), "Explicit pads must have four values [begin_h, begin_w, end_h, end_w].");
        }

        if (strides == null)
        {
            strides = kernelshape;
        }
        if (dilations == null)
        {
            dilations = new int[] { 1, 1 };
        }

        var N = input.Dimensions[0];
        var C = input.Dimensions[1];
        var H = input.Dimensions[2];
        var W = input.Dimensions[3];
        var kH = kernelshape[0];
        var kW = kernelshape[1];
        int effKH = GetConv2DEffectiveFilterSize(kH, dilations[0]);
        int effKW = GetConv2DEffectiveFilterSize(kW, dilations[1]);
        var outShape = MaxPoolOutputShape(H, W, effKH, effKW, strides[0], strides[1], pads[0] + pads[2], pads[1] + pads[3], ceilMode);
        if (outShape[0] <= 0 || outShape[1] <= 0) throw new ArgumentException("MaxPool output spatial dims must be positive.");
        var pad = new PadInfo { top = pads[0], left = pads[1], bottom = pads[2], right = pads[3], h = pads[0] + pads[2], w = pads[1] + pads[3] };
        return MaxPoolDoubleCore(input, N, C, H, W, kH, kW, strides[0], strides[1], dilations[0], dilations[1], pad, outShape[0], outShape[1]);
    }

    static Tensor<double> MaxPoolDoubleCore(Tensor<double> input, int N, int C, int H, int W, int kH, int kW, int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, PadInfo pad, int outH, int outW)
    {
        var Y = DenseTensor<double>.OfShape(N, C, outH, outW);

        for (var n = 0; n < N; ++n)
        {
            for (var d = 0; d < C; ++d)
            {
                for (var yR = 0; yR < outH; ++yR)
                {
                    var xRCorner = yR * strideHeight - pad.top;
                    for (var yC = 0; yC < outW; ++yC)
                    {
                        var xCCorner = yC * strideWidth - pad.left;

                        var maxValue = double.NegativeInfinity;

                        for (var tR = 0; tR < kH; ++tR)
                        {
                            var xR = xRCorner + tR * dilationHeight;
                            if (xR < 0 || xR >= H) continue;
                            for (var tC = 0; tC < kW; ++tC)
                            {
                                var xC = xCCorner + tC * dilationWidth;
                                if (xC < 0 || xC >= W) continue;
                                var v = input[n, d, xR, xC];

                                if (v > maxValue)
                                {
                                    maxValue = v;
                                }
                            }
                            if (maxValue == double.NegativeInfinity)
                            {
                                break;
                            }
                        }
                        Y[n, d, yR, yC] = maxValue;
                    }
                }
            }
        }
        return Y;
    }

    public static Tensor<int> MaxPool2D(Tensor<int> input, int[] kernelshape, PadType padtype, int? padvalue, int[]? strides, int[]? dilations)
    {
        if (kernelshape == null)
        {
            throw new ArgumentNullException("kernelshape");
        }
        if (input.Rank != 4)
        {
            throw new ArgumentException("Input tensors must be of rank 4 with the layout NxCxHxW.");
        }
        if (kernelshape.Rank != 1 || kernelshape.Length != 2)
        {
            throw new ArgumentException("The kernel must have shape m x n.");
        }

        if (strides == null)
        {
            strides = kernelshape;
        }
        if (dilations == null)
        {
            dilations = new int[] { 1, 1 };
        }

        var N = input.Dimensions[0];
        var C = input.Dimensions[1];
        var H = input.Dimensions[2];
        var W = input.Dimensions[3];
        var kH = kernelshape[0];
        var kW = kernelshape[1];
        var strideHeight = strides[0];
        var strideWidth = strides[1];
        var info = GetConv2DOutputInfo(padtype, H, W, strides[0], strides[1], GetConv2DEffectiveFilterSize(kH, dilations[0]), GetConv2DEffectiveFilterSize(kW, dilations[1]), padvalue);
        return MaxPoolIntCore(input, N, C, H, W, kH, kW, strideHeight, strideWidth, dilations[0], dilations[1], info.PadInfo, info.Shape[0], info.Shape[1]);
    }

    static Tensor<int> MaxPoolIntCore(Tensor<int> input, int N, int C, int H, int W, int kH, int kW, int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, PadInfo pad, int outH, int outW)
    {
        var Y = DenseTensor<int>.OfShape(N, C, outH, outW);

        for (var n = 0; n < N; ++n)
        {
            for (var d = 0; d < C; ++d)
            {
                for (var yR = 0; yR < outH; ++yR)
                {
                    var xRCorner = yR * strideHeight - pad.top;
                    for (var yC = 0; yC < outW; ++yC)
                    {
                        var xCCorner = yC * strideWidth - pad.left;

                        var maxValue = 0;

                        for (var tR = 0; tR < kH; ++tR)
                        {
                            var xR = xRCorner + tR * dilationHeight;
                            if (xR < 0 || xR >= H) continue;
                            for (var tC = 0; tC < kW; ++tC)
                            {
                                var xC = xCCorner + tC * dilationWidth;
                                if (xC < 0 || xC >= W) continue;
                                var v = input[n, d, xR, xC];

                                if (v > maxValue)
                                {
                                    maxValue = v;
                                }
                            }
                            if (maxValue == 0)
                            {
                                break;
                            }
                        }
                        Y[n, d, yR, yC] = maxValue;
                    }
                }
            }
        }
        return Y;
    }
    // NaN propagates and signed zero is preserved (ORT: Relu(NaN)=NaN, Relu(-0)=-0); negatives map to +0.
    public static Tensor<float> Relu(Tensor<float> x) => x.Apply(l => l <= 0.0f ? (l == 0.0f ? l : 0.0f) : l);

    public static Tensor<double> Relu(Tensor<double> x) => x.Apply(l => l <= 0.0 ? (l == 0.0 ? l : 0.0) : l);

    public static Tensor<T> Reshape(Tensor<T> input, Tensor<long> shape, bool allowZero)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (shape.Rank != 1)
        {
            throw new ArgumentException(nameof(shape), "Shape tensors must be of rank 1.");
        }
        if (shape.Any(v => v < -1))
        {
            throw new ArgumentException(nameof(shape), $"A shape dimension cannot be < -1, got {shape.First(v => v < -1)}.");
        }
        if (shape.Count(v => v == -1) > 1)
        {
            throw new ArgumentException(nameof(shape), $"At most 1 shape dimension can be -1.");
        }

        StartOpStage(OpStage.CalculateIndices);
        int unknownDim = -1;
        List<int> newShapeDims = new List<int>();
        int newSize = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] == -1)
            {
                unknownDim = i;
                newShapeDims.Add(-1);
            }
            else if (shape[i] == 0 && !allowZero)
            {
                newShapeDims.Add(input.Dimensions[i]);
                newSize *= input.Dimensions[i];
            }
            else if (shape[i] == 0 && allowZero)
            {
                newShapeDims.Add(0);
            }
            else
            {
                newShapeDims.Add(Convert.ToInt32(shape[i]));
                newSize *= Convert.ToInt32(shape[i]);
            }
        }
        if (unknownDim != -1)
        {
            newShapeDims[unknownDim] = Convert.ToInt32(input.Length / newSize);
            newSize *= newShapeDims[unknownDim];
        }

        if (newSize != input.Length)
        {
            throw new ArgumentException(nameof(shape), $"The input tensor cannot be reshaped to the requested shape. Input shape:{input.PrintShape()}, requested shape:{newShapeDims.Print()}");
        }

        return input.Reshape(newShapeDims.ToArray());
    }

    public static Tensor<float> Softmax(Tensor<float> x, int axis, TensorExecutionOptions? options, int opsetVersion)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (x.Rank < 1) throw new ArgumentException(nameof(x), "Softmax requires a tensor of rank 1 or more.");
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(x.Rank, axis);
        if (axis < 0 || axis >= x.Rank) throw new ArgumentException(nameof(axis), "The specified axis must be a dimension of the tensor.");
        var denseInput = x.ToDenseTensor();
        var output = new DenseTensor<float>(denseInput.Dimensions);
        SoftmaxFloatInto(denseInput, output, axis, options ?? TensorExecutionOptions.Auto, opsetVersion);
        return output;
    }

    static void SoftmaxFloatInto(DenseTensor<float> input, DenseTensor<float> destination, int axis, TensorExecutionOptions options, int opsetVersion)
    {
        var dims = input.Dimensions.ToArray();
        var inputSpan = input.Buffer.Span;
        var outputSpan = destination.Buffer.Span;
        if (opsetVersion < 13)
        {
            int block = 1;
            for (int dimension = axis; dimension < dims.Length; dimension++) block *= dims[dimension];
            int outer = block == 0 ? 0 : (int)(input.Length / block);
            SoftmaxContiguousFloat(inputSpan, outputSpan, outer, block, options.UseSimd);
            return;
        }
        int outerCount = 1;
        for (int d = 0; d < axis; d++) outerCount *= dims[d];
        int dimLen = dims[axis];
        int inner = 1;
        for (int d = axis + 1; d < dims.Length; d++) inner *= dims[d];
        if (inner == 1)
        {
            int outer = dimLen == 0 ? 0 : (int)(input.Length / dimLen);
            SoftmaxContiguousFloat(inputSpan, outputSpan, outer, dimLen, options.UseSimd);
            return;
        }
        for (int o = 0; o < outerCount; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                float max = float.NegativeInfinity;
                for (int a = 0; a < dimLen; a++)
                {
                    float candidate = inputSpan[(o * dimLen + a) * inner + i];
                    if (float.IsNaN(candidate)) { max = float.NaN; break; }
                    if (candidate > max) max = candidate;
                }
                float sum = 0f;
                for (int a = 0; a < dimLen; a++)
                {
                    float activated = MathF.Exp(inputSpan[(o * dimLen + a) * inner + i] - max);
                    outputSpan[(o * dimLen + a) * inner + i] = activated;
                    sum += activated;
                }
                for (int a = 0; a < dimLen; a++) outputSpan[(o * dimLen + a) * inner + i] /= sum;
            }
        }
    }

    static void SoftmaxContiguousFloat(System.Span<float> inputSpan, System.Span<float> outputSpan, int outer, int block, bool useSimd)
    {
        for (int outerIndex = 0; outerIndex < outer; outerIndex++)
        {
            float max = float.NegativeInfinity;
            for (int blockIndex = 0; blockIndex < block; blockIndex++)
            {
                float candidate = inputSpan[outerIndex * block + blockIndex];
                if (float.IsNaN(candidate)) { max = float.NaN; break; }
                if (candidate > max) max = candidate;
            }
            float sum = 0f;
            int expIndex = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vmax = new Vector<float>(max);
                for (; expIndex <= block - width; expIndex += width)
                {
                    int baseIndex = outerIndex * block + expIndex;
                    var activated = MathOps.ExpVector(new Vector<float>(inputSpan.Slice(baseIndex, width)) - vmax);
                    activated.CopyTo(outputSpan.Slice(baseIndex, width));
                    for (int j = 0; j < width; j++) sum += outputSpan[baseIndex + j];
                }
            }
            for (; expIndex < block; expIndex++)
            {
                float activated = MathF.Exp(inputSpan[outerIndex * block + expIndex] - max);
                outputSpan[outerIndex * block + expIndex] = activated;
                sum += activated;
            }
            for (int blockIndex = 0; blockIndex < block; blockIndex++) outputSpan[outerIndex * block + blockIndex] /= sum;
        }
    }

    /// <summary>Writes the float softmax into an existing dense destination.</summary>
    public static Tensor<float> Softmax(Tensor<float> x, DenseTensor<float> destination, int axis, TensorExecutionOptions? options, int opsetVersion)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        StartOpStage(OpStage.ValidateArguments);
        if (x.Rank < 1) throw new ArgumentException(nameof(x), "Softmax requires a tensor of rank 1 or more.");
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(x.Rank, axis);
        if (axis < 0 || axis >= x.Rank) throw new ArgumentException(nameof(axis), "The specified axis must be a dimension of the tensor.");
        var denseInput = x.ToDenseTensor();
        if (!destination.Dimensions.SequenceEqual(denseInput.Dimensions.ToArray())) throw new ArgumentException(nameof(destination), "Destination shape must match the input shape.");
        if (!HasStandardStrides(destination)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        SoftmaxFloatInto(denseInput, destination, axis, options ?? TensorExecutionOptions.Auto, opsetVersion);
        return destination;
    }

    public static Tensor<double> Softmax(Tensor<double> x, int axis, TensorExecutionOptions? options, int opsetVersion)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (x.Rank < 1) throw new ArgumentException(nameof(x), "Softmax requires a tensor of rank 1 or more.");
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(x.Rank, axis);
        if (axis < 0 || axis >= x.Rank) throw new ArgumentException(nameof(axis), "The specified axis must be a dimension of the tensor.");
        var denseInput = x.ToDenseTensor();
        var output = new DenseTensor<double>(denseInput.Dimensions);
        SoftmaxDoubleInto(denseInput, output, axis, opsetVersion);
        return output;
    }

    static void SoftmaxDoubleInto(DenseTensor<double> input, DenseTensor<double> destination, int axis, int opsetVersion)
    {
        var dims = input.Dimensions.ToArray();
        var inputSpan = input.Buffer.Span;
        var outputSpan = destination.Buffer.Span;
        if (opsetVersion < 13)
        {
            int block = 1;
            for (int dimension = axis; dimension < dims.Length; dimension++) block *= dims[dimension];
            int outer = block == 0 ? 0 : (int)(input.Length / block);
            SoftmaxContiguousDouble(inputSpan, outputSpan, outer, block);
            return;
        }
        int outerCount = 1;
        for (int d = 0; d < axis; d++) outerCount *= dims[d];
        int dimLen = dims[axis];
        int inner = 1;
        for (int d = axis + 1; d < dims.Length; d++) inner *= dims[d];
        if (inner == 1)
        {
            int outer = dimLen == 0 ? 0 : (int)(input.Length / dimLen);
            SoftmaxContiguousDouble(inputSpan, outputSpan, outer, dimLen);
            return;
        }
        for (int o = 0; o < outerCount; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                double max = double.NegativeInfinity;
                for (int a = 0; a < dimLen; a++)
                {
                    double candidate = inputSpan[(o * dimLen + a) * inner + i];
                    if (double.IsNaN(candidate)) { max = double.NaN; break; }
                    if (candidate > max) max = candidate;
                }
                double sum = 0d;
                for (int a = 0; a < dimLen; a++)
                {
                    double activated = Math.Exp(inputSpan[(o * dimLen + a) * inner + i] - max);
                    outputSpan[(o * dimLen + a) * inner + i] = activated;
                    sum += activated;
                }
                for (int a = 0; a < dimLen; a++) outputSpan[(o * dimLen + a) * inner + i] /= sum;
            }
        }
    }

    static void SoftmaxContiguousDouble(System.Span<double> inputSpan, System.Span<double> outputSpan, int outer, int block)
    {
        for (int outerIndex = 0; outerIndex < outer; outerIndex++)
        {
            double max = double.NegativeInfinity;
            for (int blockIndex = 0; blockIndex < block; blockIndex++)
            {
                double candidate = inputSpan[outerIndex * block + blockIndex];
                if (double.IsNaN(candidate)) { max = double.NaN; break; }
                if (candidate > max) max = candidate;
            }
            double sum = 0d;
            for (int blockIndex = 0; blockIndex < block; blockIndex++)
            {
                double activated = Math.Exp(inputSpan[outerIndex * block + blockIndex] - max);
                outputSpan[outerIndex * block + blockIndex] = activated;
                sum += activated;
            }
            for (int blockIndex = 0; blockIndex < block; blockIndex++) outputSpan[outerIndex * block + blockIndex] /= sum;
        }
    }

    public static Tensor<float> Erf(Tensor<float> x) => Erf(x, TensorExecutionOptions.Auto);

    public static Tensor<float> Erf(Tensor<float> x, TensorExecutionOptions options) => x.VectorizedApply(MathOps.ErfVector, MathOps.Erf, options);

    /// <summary>Writes the float error function into an existing destination tensor.</summary>
    public static Tensor<float> Erf(Tensor<float> x, Tensor<float> destination) => Erf(x, destination, TensorExecutionOptions.Auto);

    public static Tensor<float> Erf(Tensor<float> x, Tensor<float> destination, TensorExecutionOptions options)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        x.VectorizedApply(MathOps.ErfVector, MathOps.Erf, destination, options);
        return destination;
    }

    public static Tensor<double> Erf(Tensor<double> x) => Erf(x, TensorExecutionOptions.Auto);

    /// <summary>Double-precision erf runs a fixed scalar path; the options are validated but select no kernel variant.</summary>
    public static Tensor<double> Erf(Tensor<double> x, TensorExecutionOptions options)
    {
        options.Validate();
        return x.Apply(MathOps.Erf);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    /// <summary>
    /// Validates and normalizes a transpose permutation without mutating the
    /// caller array (a private normalized copy is returned).
    /// </summary>
    static int[] NormalizeTransposePerm(int rank, int[]? perm)
    {
        int[] p;
        if (perm is not null)
        {
            if (perm.Length != rank)
            {
                throw new ArgumentException(nameof(perm), $"The size of the permutation array must be the rank of the tensor: {rank}.");
            }
            if (!perm.All(q => q < rank))
            {
                throw new ArgumentException(nameof(perm), $"The permuted dimension {perm.First(q => q >= rank)} exceeds the number of dimensions in the tensor.");
            }
            if (!ArrayUtilities.CheckNoRepeatedDims(perm))
            {
                throw new ArgumentException(nameof(perm), "The permutation array has a repeated dimension.");
            }
            p = (int[])perm.Clone();
            for (int i = 0; i < p.Length; i++)
            {
                p[i] = ArrayUtilities.HandleNegativeAxisOrIndex(rank, p[i]);
            }
        }
        else
        {
            p = Enumerable.Range(0, rank).Reverse().ToArray();
        }
        return p;
    }

    /// <summary>
    /// Shared transpose copy: odometer over the destination with source stride
    /// math, so no iterator objects or per-element virtual dispatch remain.
    /// The source is densified once up front; the destination must be standard.
    /// </summary>
    static void TransposeInto(Tensor<T> data, DenseTensor<T> destination, int[] perm)
    {
        int rank = data.Rank;
        if (rank <= 1)
        {
            for (int i = 0; i < (int)data.Length; i++) destination.SetValue(i, data.GetValue(i));
            return;
        }
        var xd = data as DenseTensor<T> is { IsReversedStride: false } own && HasStandardStrides(own) && own.Buffer.Length == (int)own.Length
            ? own
            : data.ToDenseTensor();
        var map = new int[rank];
        for (int d = 0; d < rank; d++) map[d] = xd.strides[perm[d]];
        var xs = xd.Buffer.Span;
        var ds = destination.Buffer.Span;
        var destDims = destination.Dimensions;
        var coords = new int[rank];
        int total = (int)destination.Length;
        for (int i = 0; i < total; i++)
        {
            int srcOff = 0;
            for (int d = 0; d < rank; d++) srcOff += coords[d] * map[d];
            ds[i] = xs[srcOff];
            for (int d = rank - 1; d >= 0; d--)
            {
                coords[d]++;
                if (coords[d] < destDims[d]) break;
                coords[d] = 0;
            }
        }
    }

    public static Tensor<T> Transpose(Tensor<T> data, int[]? perm)
    {
        StartOpStage(OpStage.ValidateArguments);
        var p = NormalizeTransposePerm(data.Rank, perm);
        if (data.Rank <= 1)
        {
            return data;
        }

        StartOpStage(OpStage.Copy);
        var r = DenseTensor<T>.OfShape(TransposedShape(data.Dimensions, p));
        TransposeInto(data, r, p);
        return r;
    }

    /// <summary>Computes the transposed shape without moving elements.</summary>
    public static int[] TransposedShape(ReadOnlySpan<int> dimensions, int[]? perm)
    {
        int rank = dimensions.Length;
        int[] p;
        if (perm is not null)
        {
            if (perm.Length != rank) throw new ArgumentException(nameof(perm), $"The size of the permutation array must be the rank of the tensor: {rank}.");
            if (!perm.All(q => q < rank)) throw new ArgumentException(nameof(perm), $"The permuted dimension {perm.First(q => q >= rank)} exceeds the number of dimensions in the tensor.");
            if (!ArrayUtilities.CheckNoRepeatedDims(perm)) throw new ArgumentException(nameof(perm), "The permutation array has a repeated dimension.");
            p = (int[])perm.Clone();
            for (int i = 0; i < p.Length; i++) p[i] = ArrayUtilities.HandleNegativeAxisOrIndex(rank, p[i]);
        }
        else p = Enumerable.Range(0, rank).Reverse().ToArray();
        var shape = new int[rank];
        for (int i = 0; i < p.Length; i++) shape[i] = dimensions[p[i]];
        return shape;
    }

    /// <summary>Writes the transposed tensor into an existing dense destination.</summary>
    public static Tensor<T> Transpose(Tensor<T> data, DenseTensor<T> destination, int[]? perm)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        StartOpStage(OpStage.ValidateArguments);
        var p = NormalizeTransposePerm(data.Rank, perm);
        if (!destination.Dimensions.SequenceEqual(TransposedShape(data.Dimensions, p))) throw new ArgumentException(nameof(destination), "Destination shape must match the transposed shape.");
        if (!HasStandardStrides(destination)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        if (ReferenceEquals(destination, data)) throw new ArgumentException(nameof(destination), "Destination must not alias the input tensor: permutation is not an in-place operation.");
        StartOpStage(OpStage.Copy);
        TransposeInto(data, destination, p);
        return destination;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]  
    public static Tensor<T> Gather(Tensor<T> data, Tensor<int> indices, int? _axis)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (data.Rank == 0) throw new ArgumentException(nameof (data), "Cannot gather from a tensor of rank 0.");

        StartOpStage(OpStage.CalculateIndices);
        var axis = _axis.HasValue ? ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, _axis.Value) : 0;    
        if (axis > data.Rank - 1)
        {
            throw new ArgumentException(nameof(axis), $"The specified axis {_axis} exceeds the number of dimensions in the tensor.");
        }
        var outputShape = new int[data.Rank - 1 + indices.Rank];
        int n = 0;
        for (int i = 0; i < axis; i++)
        {
            outputShape[n] = data.dimensions[i];
            n++;
        }
        for (int i = 0; i < indices.Rank; i++)
        {
            outputShape[n] = indices.dimensions[i];
            n++;
        }
        for (int i = axis + 1; i < data.Rank; i++)
        {
            outputShape[n] = data.dimensions[i];
            n++;
        }
        var output = DenseTensor<T>.OfShape(outputShape);
        StartOpStage(OpStage.Copy);
        if (data is DenseTensor<T> denseData && HasStandardStrides(denseData) && indices is DenseTensor<int> denseIndices)
        {
            int outer = 1;
            for (int d = 0; d < axis; d++) outer *= data.dimensions[d];
            int inner = 1;
            for (int d = axis + 1; d < data.Rank; d++) inner *= data.dimensions[d];
            int axisDim = data.dimensions[axis];
            var indexSpan = denseIndices.Buffer.Span;
            var srcSpan = denseData.Buffer.Span;
            var dstSpan = output.Buffer.Span;
            var rows = new int[indexSpan.Length];
            for (int j = 0; j < rows.Length; j++) rows[j] = ArrayUtilities.HandleNegativeAxisOrIndex(axisDim, indexSpan[j]);
            for (int o = 0; o < outer; o++)
            {
                for (int j = 0; j < rows.Length; j++)
                {
                    srcSpan.Slice((o * axisDim + rows[j]) * inner, inner).CopyTo(dstSpan.Slice((o * rows.Length + j) * inner, inner));
                }
            }
            return output;
        }
        foreach (var di in output.GetDimensionsIterator())
        {
            var a = di[0..axis];
            var k = ArrayUtilities.HandleNegativeAxisOrIndex(data.dimensions[axis], indices[di[axis..(axis + indices.Rank)]]);
            var b = di[(axis + (indices.Rank == 0 ? 1 : indices.Rank))..];
            var oloc = a.Append(k).Concat(b).ToArray();
            output[di] = data[oloc];
        }
        return output;  
    }

    static bool HasStandardStrides<TElement>(DenseTensor<TElement> tensor) where TElement : unmanaged
    {
        if (tensor.IsReversedStride) return false;
        return tensor.strides.SequenceEqual(ArrayUtilities.GetStrides(tensor.dimensions));
    }

    static DenseTensor<float> RequireContiguousFloat(Tensor<float> t, string name)
    {
        if (t is DenseTensor<float> d && !d.IsReversedStride && HasStandardStrides(d))
        {
            if (d.Buffer.Length != (int)d.Length) throw new ArgumentException(name + " backing length does not match shape.");
            return d;
        }
        return t.ToDenseTensor();
    }

    static DenseTensor<int> RequireContiguousInt(Tensor<int> t, string name)
    {
        if (t is DenseTensor<int> d && !d.IsReversedStride && HasStandardStrides(d))
        {
            if (d.Buffer.Length != (int)d.Length) throw new ArgumentException(name + " backing length does not match shape.");
            return d;
        }
        return t.ToDenseTensor();
    }

    static DenseTensor<double> RequireContiguousDouble(Tensor<double> t, string name)
    {
        if (t is DenseTensor<double> d && !d.IsReversedStride && HasStandardStrides(d))
        {
            if (d.Buffer.Length != (int)d.Length) throw new ArgumentException(name + " backing length does not match shape.");
            return d;
        }
        return t.ToDenseTensor();
    }

    static void ValidateConv2D(
        int N, int C, int H, int W, int M, int Cpg, int kH, int kW,
        int group, int[] strides, int[] dilations, int[]? kernelshape,
        int[] weightDims, long inputLength, long weightLength, int biasLength)
    {
        if (N <= 0) throw new ArgumentException("Conv input batch must be positive.", nameof(N));
        if (C <= 0) throw new ArgumentException("Conv input channels must be positive.", nameof(C));
        if (H <= 0 || W <= 0) throw new ArgumentException("Conv spatial dims must be positive.");
        if (M <= 0) throw new ArgumentException("Conv output channels must be positive.", nameof(M));
        if (group <= 0) throw new ArgumentException("Conv group must be positive.", nameof(group));
        if (C % group != 0) throw new ArgumentException("Conv input channels must be divisible by group.");
        if (M % group != 0) throw new ArgumentException("Conv output channels must be divisible by group.");
        if (strides.Length != 2 || strides[0] <= 0 || strides[1] <= 0) throw new ArgumentException("Conv strides must be two positive values.");
        if (dilations.Length != 2 || dilations[0] <= 0 || dilations[1] <= 0) throw new ArgumentException("Conv dilations must be two positive values.");
        if (kH <= 0 || kW <= 0) throw new ArgumentException("Conv kernel dims must be positive.");
        if (kernelshape is not null)
        {
            if (kernelshape.Length != 2) throw new ArgumentException("Conv kernel_shape must have two values.");
            if (kernelshape[0] != weightDims[2] || kernelshape[1] != weightDims[3]) throw new ArgumentException("Conv kernel_shape must match weight spatial dims.");
        }
        if (weightDims.Length != 4) throw new ArgumentException("Conv weight must be rank 4.");
        if (weightDims[0] != M || weightDims[1] != Cpg || weightDims[2] != kH || weightDims[3] != kW) throw new ArgumentException("Conv weight shape must be [M, C/group, kH, kW].");
        if (Cpg != C / group) throw new ArgumentException("Conv weight channels must equal C/group.");
        if (biasLength >= 0 && biasLength != M) throw new ArgumentException("Conv bias length must equal M.");
        checked
        {
            long expectInput = (long)N * C * H * W;
            long expectWeight = (long)M * Cpg * kH * kW;
            if (inputLength != expectInput) throw new ArgumentException("Conv input backing length does not match shape.");
            if (weightLength != expectWeight) throw new ArgumentException("Conv weight backing length does not match shape.");
        }
    }

    public static Tensor<T> Concat(Tensor<T> x, Tensor<T> y, int axis)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (x.Rank != y.Rank) throw new ArgumentException(nameof(y), "The rank of each tensor in a concat operation must be the same.");
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(x.Rank, axis);
        for (int i = 0; i < x.Rank; i++)
        {
            if (i == axis) continue;
            if (x.dimensions[i] != y.dimensions[i])
            {
                throw new ArgumentException(nameof(y), "The dimensions of each tensor in a concat operation must be the same, with the exception of the axis dimension.");
            }
        }
        var shape = x.dimensions.Copy();
        shape[axis] += y.dimensions[axis];

        StartOpStage(OpStage.Copy);
        var output = DenseTensor<T>.OfShape(shape);
        int inner = 1;
        for (int trailing = axis + 1; trailing < output.Rank; trailing++) inner *= shape[trailing];
        int outer = 1;
        for (int leading = 0; leading < axis; leading++) outer *= shape[leading];
        if (x is DenseTensor<T> denseX && y is DenseTensor<T> denseY && HasStandardStrides(denseX) && HasStandardStrides(denseY))
        {
            ArrayUtilities.CopyAxisChunks(denseX.Buffer.Span, x.dimensions[axis], output.Buffer.Span, x.dimensions[axis] + y.dimensions[axis], outer, inner, 0, 0, x.dimensions[axis]);
            ArrayUtilities.CopyAxisChunks(denseY.Buffer.Span, y.dimensions[axis], output.Buffer.Span, x.dimensions[axis] + y.dimensions[axis], outer, inner, 0, x.dimensions[axis], y.dimensions[axis]);
            return output;
        }
        var di = output.GetDimensionsIterator();    
        foreach (var index in di)
        {
            if (index[axis] < x.dimensions[axis])
            {
                output[index] = x[index];
            }
            else
            {
                var loc = index.Copy();
                loc[axis] -= x.dimensions[axis];
                output[index] = y[loc];
            }
        }
        return output;
    }
    public static Tensor<T> Concat(Tensor<T>[] inputs, int axis)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (inputs.Length < 2) throw new ArgumentException(nameof(inputs), "At least two tensors must be specified for the concat operation.");
        if (!inputs.All(i => i.Rank == inputs[0].Rank)) throw new ArgumentException(nameof(inputs), $"Each input tensor in a concat operation must be of the same rank.");
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(inputs[0].Rank, axis);
        if (!inputs.All(i => i.dimensions.Select((d, n) => n == axis ? 0 : d - inputs[0].dimensions[n]).All(s => s == 0)))
            throw new ArgumentException(nameof(inputs), "The dimensions of each tensor in a concat operation must be the same, with the exception of the axis dimension.");
        var shape = inputs[0].dimensions.Copy();
        shape[axis] = 0;
        foreach (var input in inputs) shape[axis] += input.dimensions[axis];

        StartOpStage(OpStage.Copy);
        var output = DenseTensor<T>.OfShape(shape);
        int inner = 1;
        for (int trailing = axis + 1; trailing < output.Rank; trailing++) inner *= shape[trailing];
        int outer = 1;
        for (int leading = 0; leading < axis; leading++) outer *= shape[leading];
        if (inputs.All(i => i is DenseTensor<T> dense && HasStandardStrides(dense)))
        {
            var outputSpan = output.Buffer.Span;
            int destinationAxisOffset = 0;
            foreach (var input in inputs)
            {
                var sourceSpan = ((DenseTensor<T>)input).Buffer.Span;
                int axisLength = input.dimensions[axis];
                for (int outerIndex = 0; outerIndex < outer; outerIndex++)
                {
                    sourceSpan.Slice(outerIndex * axisLength * inner, axisLength * inner).CopyTo(outputSpan.Slice((outerIndex * shape[axis] + destinationAxisOffset) * inner, axisLength * inner));
                }
                destinationAxisOffset += axisLength;
            }
            return output;
        }
        var axisOffsets = new int[inputs.Length];
        int runningAxisOffset = 0;
        for (int inputIndex = 0; inputIndex < inputs.Length; inputIndex++)
        {
            axisOffsets[inputIndex] = runningAxisOffset;
            runningAxisOffset += inputs[inputIndex].dimensions[axis];
        }
        var iterator = output.GetDimensionsIterator();
        foreach (var index in iterator)
        {
            int source = 0;
            while (source + 1 < inputs.Length && index[axis] >= axisOffsets[source + 1]) source++;
            var location = index.Copy();
            location[axis] -= axisOffsets[source];
            output[index] = inputs[source][location];
        }
        return output;
    }

    public static Tensor<T> Slice(Tensor<T> data, Tensor<int> start, Tensor<int> ends, Tensor<int>? axes, Tensor<int>? steps)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (data.Rank == 0) throw new ArgumentException(nameof(data), "Cannot slice a tensor of rank 0.");
        if (start.Rank != 1) throw new ArgumentException(nameof(start), "The rank of the start tensor must be 1.");
        if (start.Length > data.Rank) throw new ArgumentException(nameof(start), "The length of the start tensor must be less-than or equal to the rank of the data tensor.");
        if (ends.Rank != 1) throw new ArgumentException(nameof(start), "The rank of the end tensor must be 1");
        if (start.Length != ends.Length) throw new ArgumentException(nameof(ends), "The end tensor must be the same length as the start tensor.");
        if (axes is not null && (axes.Rank != 1 || axes.Length != start.Length)) throw new ArgumentException(nameof(axes), "The axes tensor must be a rank 1 tensor with the same length as the start tensor.");
        if (steps is not null && (steps.Rank != 1 || steps.Length != start.Length)) throw new ArgumentException(nameof(steps), "The steps tensor must be a rank 1 tensor with the same length as the start tensor.");
        
        StartOpStage(OpStage.CalculateIndices);
        int length = Convert.ToInt32(start.Length);
        if (axes is null)
        {
            axes = Enumerable.Range(0, length).ToArray().ToTensor<int>();
        }
        else
        {
            axes = axes.Select(a => ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, a)).ToArray().ToTensor<int>();
        }
        if (steps is null)
        {
            steps = Tensor<int>.Ones(length);
        }
       
        start = start.Select((s, i) => ArrayUtilities.Clamp(ArrayUtilities.HandleNegativeAxisOrIndex(data.Dimensions[axes[i]], s), 0, data.Dimensions[axes[i]])).ToArray().ToTensor<int>();
        ends = ends.Select((s, i) => ArrayUtilities.Clamp(ArrayUtilities.HandleNegativeAxisOrIndex(data.Dimensions[axes[i]], s), 0, data.Dimensions[axes[i]])).ToArray().ToTensor<int>();

        SliceIndex[] indices = new SliceIndex[data.Rank];
        for (int i = 0; i < data.Rank; i++) 
        {
            indices[i] = axes.Contains(i) ? new SliceIndex(start[axes.IndexOf(i)], ends[axes.IndexOf(i)], steps[axes.IndexOf(i)]) : new SliceIndex(0, data.dimensions[i]);
        }
        return data.Slice(indices); 
    }

    public static Tensor<T> Unsqueeze(Tensor<T> data, int[] axes)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (!ArrayUtilities.CheckNoRepeatedDims(axes)) throw new ArgumentException(nameof(axes), "axes contains a repeated dimension.");
        
        StartOpStage(OpStage.CalculateIndices);
        for (int i = 0; i < axes.Length; i++)
        {
            axes[i] = ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, axes[i]);
        }
        if (!Array.TrueForAll(axes, (a => a <= (data.Rank + axes.Length) - 1))) throw new ArgumentException(nameof(axes), $"Each specified axis must be less than the rank of the output tensor. Got {axes.First(a => a > data.Rank - 1)}");
        var newshape = new int[axes.Length + data.Rank];
        for (int i = 0; i < axes.Length; i++)
        {
            newshape[axes[i]] = 1;
        }
        var e = data.Dimensions.GetEnumerator();
        for (int i = 0; i < newshape.Length; i++)
        {
            if (newshape[i] == 0)
            {
                if (!e.MoveNext()) throw new InvalidOperationException("Out of dimensions.");
                newshape[i] = e.Current;
            }
        }
        return data.Reshape(newshape);
    }
    /// <summary>
    /// Shared reduction orchestration: optional transpose to innermost axes,
    /// densified standard input, output shape and inner extent. Typed kernels
    /// accumulate over the returned spans; empty and no-op cases are resolved
    /// by the caller through the shared plan.
    /// </summary>
    static (DenseTensor<TElement> input, int[] outputShape, int inner) PrepareReduction<TElement>(Tensor<TElement> data, int[] axes) where TElement : unmanaged
    {
        var permutation = ArrayUtilities.GetAxesPermutationForReduction(axes, data.Rank);
        Tensor<TElement> pdata;
        int[] paxes;
        if (permutation is not null)
        {
            pdata = Tensor<TElement>.Transpose(data, permutation);
            paxes = ArrayUtilities.GetInnerMostAxes(axes.Length, data.Rank);
        }
        else
        {
            pdata = data;
            paxes = axes;
        }
        var (oshape, rshape) = ArrayUtilities.ComputeShapesForReduction(pdata.dimensions, paxes);
        int r = ArrayUtilities.ComputeOffsetForReduction(rshape, 0);
        DenseTensor<TElement> dense;
        if (pdata is DenseTensor<TElement> d && !d.IsReversedStride && HasStandardStrides(d) && d.Buffer.Length == (int)d.Length)
            dense = d;
        else
            dense = pdata.ToDenseTensor();
        return (dense, oshape, r);
    }

    public static Tensor<int> ReduceSum(Tensor<int> data, Tensor<int>? axes) => ReduceSum(data, axes, null, null);

        public static Tensor<int> ReduceSum(Tensor<int> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
    {
        StartOpStage(OpStage.ValidateArguments);
        var keepDims = _keepDims.HasValue ? _keepDims.Value : false;
        // One validated plan owns absent/empty axes, normalization, dedupe and
        // keepdims (ORT 1.29: absent/empty + noop is a no-op; dupes reduce once).
        var plan = ReductionPlan.Create(data.Rank, axes, keepDims, _noOpWithEmptyAxes.HasValue && _noOpWithEmptyAxes.Value);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<int>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            int sum = 0;
            for (int j = 0; j < r; ++j) sum += xs[offset + j];
            os[i] = sum;
        }
        if (keepDims)
        {
            return Tensor<int>.Unsqueeze(output, plan.Axes);
        }
        else
        {
            return output;
        }
    }

    public static Tensor<float> ReduceSum(Tensor<float> data, Tensor<int>? axes) => ReduceSum(data, axes, null, null);

        public static Tensor<float> ReduceSum(Tensor<float> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
    {
        StartOpStage(OpStage.ValidateArguments);
        var keepDims = _keepDims.HasValue ? _keepDims.Value : false;
        // One validated plan owns absent/empty axes, normalization, dedupe and
        // keepdims (ORT 1.29: absent/empty + noop is a no-op; dupes reduce once).
        var plan = ReductionPlan.Create(data.Rank, axes, keepDims, _noOpWithEmptyAxes.HasValue && _noOpWithEmptyAxes.Value);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<float>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            float sum = 0f;
            for (int j = 0; j < r; ++j) sum += xs[offset + j];
            os[i] = sum;
        }
        if (keepDims)
        {
            return Tensor<float>.Unsqueeze(output, plan.Axes);
        }
        else
        {
            return output;
        }
    }

    public static Tensor<double> ReduceSum(Tensor<double> data, Tensor<int>? axes) => ReduceSum(data, axes, null, null);

        public static Tensor<double> ReduceSum(Tensor<double> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
    {
        StartOpStage(OpStage.ValidateArguments);
        var keepDims = _keepDims.HasValue ? _keepDims.Value : false;
        // One validated plan owns absent/empty axes, normalization, dedupe and
        // keepdims (ORT 1.29: absent/empty + noop is a no-op; dupes reduce once).
        var plan = ReductionPlan.Create(data.Rank, axes, keepDims, _noOpWithEmptyAxes.HasValue && _noOpWithEmptyAxes.Value);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<double>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            double sum = 0.0;
            for (int j = 0; j < r; ++j) sum += xs[offset + j];
            os[i] = sum;
        }
        if (keepDims)
        {
            return Tensor<double>.Unsqueeze(output, plan.Axes);
        }
        else
        {
            return output;
        }
    }

    public static Tensor<int> ReduceMean(Tensor<int> data, Tensor<int>? axes) => ReduceMean(data, axes, null, null);

        public static Tensor<int> ReduceMean(Tensor<int> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
        => ReduceMean(data, axes, _keepDims, _noOpWithEmptyAxes, TensorExecutionOptions.Auto);

    public static Tensor<int> ReduceMean(Tensor<int> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes, TensorExecutionOptions options)
    {
        StartOpStage(OpStage.ValidateArguments);
        var keepDims = _keepDims.HasValue ? _keepDims.Value : false;
        // One validated plan owns absent/empty axes, normalization, dedupe and
        // keepdims (ORT 1.29: absent/empty + noop is a no-op; dupes reduce once).
        var plan = ReductionPlan.Create(data.Rank, axes, keepDims, _noOpWithEmptyAxes.HasValue && _noOpWithEmptyAxes.Value);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<int>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            int sum = 0;
            for (int j = 0; j < r; ++j) sum += xs[offset + j];
            os[i] = sum / r;
        }
        if (keepDims)
        {
            return Tensor<int>.Unsqueeze(output, plan.Axes);
        }
        else
        {
            return output;
        }
    }

    public static Tensor<float> ReduceMean(Tensor<float> data, Tensor<int>? axes) => ReduceMean(data, axes, null, null);

        public static Tensor<float> ReduceMean(Tensor<float> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
        => ReduceMean(data, axes, _keepDims, _noOpWithEmptyAxes, TensorExecutionOptions.Auto);

    public static Tensor<float> ReduceMean(Tensor<float> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes, TensorExecutionOptions options)
    {
        options.Validate();
        StartOpStage(OpStage.ValidateArguments);
        var keepDims = _keepDims.HasValue ? _keepDims.Value : false;
        // One validated plan owns absent/empty axes, normalization, dedupe and
        // keepdims (ORT 1.29: absent/empty + noop is a no-op; dupes reduce once).
        var plan = ReductionPlan.Create(data.Rank, axes, keepDims, _noOpWithEmptyAxes.HasValue && _noOpWithEmptyAxes.Value);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<float>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            float sum = 0f;
            for (int j = 0; j < r; ++j) sum += xs[offset + j];
            os[i] = sum / r;
        }
        if (keepDims)
        {
            return Tensor<float>.Unsqueeze(output, plan.Axes);
        }
        else
        {
            return output;
        }
    }

    public static Tensor<double> ReduceMean(Tensor<double> data, Tensor<int>? axes) => ReduceMean(data, axes, null, null);

        public static Tensor<double> ReduceMean(Tensor<double> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
        => ReduceMean(data, axes, _keepDims, _noOpWithEmptyAxes, TensorExecutionOptions.Auto);

    public static Tensor<double> ReduceMean(Tensor<double> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes, TensorExecutionOptions options)
    {
        options.Validate();
        StartOpStage(OpStage.ValidateArguments);
        var keepDims = _keepDims.HasValue ? _keepDims.Value : false;
        // One validated plan owns absent/empty axes, normalization, dedupe and
        // keepdims (ORT 1.29: absent/empty + noop is a no-op; dupes reduce once).
        var plan = ReductionPlan.Create(data.Rank, axes, keepDims, _noOpWithEmptyAxes.HasValue && _noOpWithEmptyAxes.Value);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<double>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            double sum = 0.0;
            for (int j = 0; j < r; ++j) sum += xs[offset + j];
            os[i] = sum / r;
        }
        if (keepDims)
        {
            return Tensor<double>.Unsqueeze(output, plan.Axes);
        }
        else
        {
            return output;
        }
    }

    public static Tensor<float> ReduceMax(Tensor<float> data, Tensor<int>? axes) => ReduceMax(data, axes, null, null);

        public static Tensor<float> ReduceMax(Tensor<float> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
    {
        StartOpStage(OpStage.ValidateArguments);
        var keepDims = _keepDims.HasValue ? _keepDims.Value : true;
        // One validated plan owns absent/empty axes, normalization, dedupe and
        // keepdims (ORT 1.29: absent/empty + noop is a no-op; dupes reduce once).
        var plan = ReductionPlan.Create(data.Rank, axes, keepDims, _noOpWithEmptyAxes.HasValue && _noOpWithEmptyAxes.Value);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<float>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            // Empty reductions yield -Infinity; init from the first element so
            // all-negative spans keep their true max and a leading NaN persists (ORT).
            if (r == 0)
            {
                os[i] = float.NegativeInfinity;
                continue;
            }
            float max = xs[offset];
            for (int j = 1; j < r; ++j)
            {
                float v = xs[offset + j];
                if (v > max) max = v;
            }
            os[i] = max;
        }
        if (keepDims)
        {
            return Tensor<float>.Unsqueeze(output, plan.Axes);
        }
        else
        {
            return output;
        }
    }

    public static Tensor<double> ReduceMax(Tensor<double> data, Tensor<int>? axes) => ReduceMax(data, axes, null, null);

        public static Tensor<double> ReduceMax(Tensor<double> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
    {
        StartOpStage(OpStage.ValidateArguments);
        var keepDims = _keepDims.HasValue ? _keepDims.Value : true;
        // One validated plan owns absent/empty axes, normalization, dedupe and
        // keepdims (ORT 1.29: absent/empty + noop is a no-op; dupes reduce once).
        var plan = ReductionPlan.Create(data.Rank, axes, keepDims, _noOpWithEmptyAxes.HasValue && _noOpWithEmptyAxes.Value);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<double>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            if (r == 0)
            {
                os[i] = double.NegativeInfinity;
                continue;
            }
            double max = xs[offset];
            for (int j = 1; j < r; ++j)
            {
                double v = xs[offset + j];
                if (v > max) max = v;
            }
            os[i] = max;
        }
        if (keepDims)
        {
            return Tensor<double>.Unsqueeze(output, plan.Axes);
        }
        else
        {
            return output;
        }
    }
}

/// <summary>
/// Validated reduction plan shared by every ReduceSum/Mean/Max copy.
/// </summary>
/// <remarks>
/// Absent or explicitly empty axes are a no-op when flagged, else reduce all
/// axes. Otherwise axes normalize (negative += rank), range-check against
/// [-rank, rank-1], then deduplicate: the native engine reduces duplicated
/// axes once instead of rejecting them (verified against ORT 1.29).
/// </remarks>
internal readonly struct ReductionPlan
{
    public readonly bool IsNoOp;
    public readonly int[] Axes;
    public readonly bool KeepDims;

    private ReductionPlan(bool isNoOp, int[] axes, bool keepDims)
    {
        IsNoOp = isNoOp;
        Axes = axes;
        KeepDims = keepDims;
    }

    public static ReductionPlan Create(int rank, Tensor<int>? axes, bool keepDims, bool noOpWithEmptyAxes)
    {
        var raw = axes is null ? System.Array.Empty<int>() : axes.ToArray();
        if (raw.Length == 0)
        {
            if (noOpWithEmptyAxes) return new ReductionPlan(true, System.Array.Empty<int>(), keepDims);
            var all = new int[rank];
            for (int i = 0; i < rank; i++) all[i] = i;
            return new ReductionPlan(false, all, keepDims);
        }
        var normalized = new int[raw.Length];
        for (int i = 0; i < raw.Length; i++)
        {
            int a = raw[i] < 0 ? raw[i] + rank : raw[i];
            if (a < 0 || a >= rank)
                throw new System.ArgumentException(nameof(axes), $"Axis {raw[i]} is out of range for tensor rank {rank}.");
            normalized[i] = a;
        }
        return new ReductionPlan(false, normalized.Distinct().ToArray(), keepDims);
    }
}

internal static class MatMulShapes
{
    public readonly struct Plan
    {
        public readonly bool PromoteX;
        public readonly bool PromoteY;
        public readonly int[] OutputShape;
        public Plan(bool promoteX, bool promoteY, int[] outputShape)
        {
            PromoteX = promoteX;
            PromoteY = promoteY;
            OutputShape = outputShape;
        }
    }

    static int[] CoreOutputShape(System.ReadOnlySpan<int> xd, System.ReadOnlySpan<int> yd)
    {
        if (xd.Length == 2 && yd.Length == 2)
        {
            if (xd[1] != yd[0]) throw new System.ArgumentException($"The number of columns in the first matrix ({xd[1]}) is not equal to the number of rows in the second matrix ({yd[0]}).");
            return new int[] { xd[0], yd[1] };
        }
        var xdl = xd[^2..];
        var ydl = yd[^2..];
        if (xdl[1] != ydl[0]) throw new System.ArgumentException($"The number of columns in the first matrix ({xdl[1]}) is not equal to the number of rows in the second matrix ({ydl[0]}).");
        if (!Tensor<int>.BroadcastShape(xd[0..^2], yd[0..^2], out var bd)) throw new System.ArgumentException("The tensor shapes are not compatible for broadcasting.");
        return bd.Append(xdl[0]).Append(ydl[1]).ToArray();
    }

    public static Plan Create(System.ReadOnlySpan<int> xd, System.ReadOnlySpan<int> yd)
    {
        if (xd.Length == 0 || yd.Length == 0) throw new System.ArgumentException("The rank of each tensor in matrix multiplication must be greater than 1.");
        bool promoteX = xd.Length == 1;
        bool promoteY = yd.Length == 1;
        int[] px = promoteX ? new int[] { 1, xd[0] } : xd.ToArray();
        int[] py = promoteY ? new int[] { yd[0], 1 } : yd.ToArray();
        int[] core = CoreOutputShape(px, py);
        int[] output;
        if (promoteX && promoteY) output = core[0..^2];
        else if (promoteX) output = core[0..^2].Append(core[^1]).ToArray();
        else if (promoteY) output = core[0..^1];
        else output = core;
        return new Plan(promoteX, promoteY, output);
    }

    public static Tensor<U> Squeeze<U>(Tensor<U> core, Plan plan) where U : unmanaged
    {
        if (plan.PromoteX && plan.PromoteY)
        {
            var once = core.RemoveDim(core.Rank - 2);
            return once.RemoveDim(once.Rank - 1);
        }
        if (plan.PromoteX) return core.RemoveDim(core.Rank - 2);
        if (plan.PromoteY) return core.RemoveDim(core.Rank - 1);
        return core;
    }
}
