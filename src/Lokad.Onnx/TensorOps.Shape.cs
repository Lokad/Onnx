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
    /// <summary>
    /// Generates start, start plus delta, and so on, stopping before limit. Zero delta throws,
    /// as do ranges with no finite count (NaN anywhere, infinite start/limit, or a count
    /// beyond the maximum tensor length); an infinite delta yields an empty range.
    /// </summary>
    public static Tensor<float> Range(float start, float limit, float delta)
    {
        if (delta == 0f) throw new ArgumentException(nameof(delta));
        double quotient = ((double)limit - start) / delta;
        if (double.IsNaN(quotient)) throw new ArgumentException("Range start, limit and delta must be finite or yield a finite count.", nameof(limit));
        if (!(quotient > 0.0)) return new DenseTensor<float>(0);
        if (quotient > int.MaxValue) throw new ArgumentException("Range count exceeds the maximum tensor length.", nameof(limit));
        long count = 0;
        if (delta > 0f) { while (count <= int.MaxValue && (double)start + count * (double)delta < limit) count++; }
        else { while (count <= int.MaxValue && (double)start + count * (double)delta > limit) count++; }
        if (count > int.MaxValue) throw new ArgumentException("Range count exceeds the maximum tensor length.", nameof(limit));
        var output = new DenseTensor<float>((int)count);
        var span = output.Buffer.Span;
        float value = start;
        for (int i = 0; i < (int)count; i++) { span[i] = value; value += delta; }
        return output;
    }

    /// <summary>
    /// Generates start, start plus delta, and so on, stopping before limit. Zero delta throws,
    /// as do ranges with no finite count (NaN anywhere, infinite start/limit, or a count
    /// beyond the maximum tensor length); an infinite delta yields an empty range.
    /// </summary>
    public static Tensor<double> Range(double start, double limit, double delta)
    {
        if (delta == 0.0) throw new ArgumentException(nameof(delta));
        double quotient = (limit - start) / delta;
        if (double.IsNaN(quotient)) throw new ArgumentException("Range start, limit and delta must be finite or yield a finite count.", nameof(limit));
        if (!(quotient > 0.0)) return new DenseTensor<double>(0);
        if (quotient > int.MaxValue) throw new ArgumentException("Range count exceeds the maximum tensor length.", nameof(limit));
        int count = Math.Max((int)Math.Ceiling(quotient), 0);
        var output = new DenseTensor<double>(count);
        var span = output.Buffer.Span;
        double value = start;
        for (int i = 0; i < count; i++) { span[i] = value; value += delta; }
        return output;
    }

    /// <summary>
    /// Generates start, start plus delta, and so on, stopping before limit. Zero delta throws.
    /// </summary>
    public static Tensor<long> Range(long start, long limit, long delta)
    {
        if (delta == 0L) throw new ArgumentException(nameof(delta));
        // Counts beyond int.MaxValue are unrepresentable in int32 dims: fail
        // here instead of overflowing the narrowing cast below (which throws
        // OutOfMemoryException on huge allocates, a fatal error the node
        // boundary cannot convert).
        double longQuotient = ((double)limit - start) / delta;
        if (longQuotient > int.MaxValue) throw new ArgumentException("Range count exceeds the maximum tensor length.", nameof(limit));
        int count = Math.Max((int)Math.Ceiling(longQuotient), 0);
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
        // Same unrepresentable-count guard as the long kernel above.
        double intQuotient = ((double)limit - start) / delta;
        if (intQuotient > int.MaxValue) throw new ArgumentException("Range count exceeds the maximum tensor length.", nameof(limit));
        int count = Math.Max((int)Math.Ceiling(intQuotient), 0);
        var output = new DenseTensor<int>(count);
        var span = output.Buffer.Span;
        for (int i = 0; i < count; i++) span[i] = start + i * delta;
        return output;
    }

    /// <summary>
    /// Generates start, start plus delta, and so on, stopping before limit. Zero delta throws.
    /// Emitted values always fit int16 for valid inputs (count derives from limit),
    /// so the narrowing cast is exact, not a wrap decision.
    /// </summary>
    public static Tensor<short> Range(short start, short limit, short delta)
    {
        if (delta == 0) throw new ArgumentException(nameof(delta));
        int count = Math.Max((int)Math.Ceiling(((double)limit - start) / delta), 0);
        var output = new DenseTensor<short>(count);
        var span = output.Buffer.Span;
        for (int i = 0; i < count; i++) span[i] = unchecked((short)(start + i * delta));
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
    /// Repeats the input repeats[i] times along each dimension.
    /// The repeats rank must match the input rank and every repeat must be non-negative.
    /// </summary>
    public static Tensor<uint> Tile(Tensor<uint> x, int[] repeats) => TileCore(x, repeats);

    /// <summary>
    /// Repeats the input repeats[i] times along each dimension.
    /// The repeats rank must match the input rank and every repeat must be non-negative.
    /// </summary>
    public static Tensor<ulong> Tile(Tensor<ulong> x, int[] repeats) => TileCore(x, repeats);

    /// <summary>
    /// Repeats the input repeats[i] times along each dimension.
    /// The repeats rank must match the input rank and every repeat must be non-negative.
    /// </summary>
    public static Tensor<bool> Tile(Tensor<bool> x, int[] repeats) => TileCore(x, repeats);

    /// <summary>
    /// Repeats the input repeats[i] times along each dimension.
    /// The repeats rank must match the input rank and every repeat must be non-negative.
    /// </summary>
    public static Tensor<sbyte> Tile(Tensor<sbyte> x, int[] repeats) => TileCore(x, repeats);

    /// <summary>
    /// Repeats the input repeats[i] times along each dimension.
    /// The repeats rank must match the input rank and every repeat must be non-negative.
    /// </summary>
    public static Tensor<byte> Tile(Tensor<byte> x, int[] repeats) => TileCore(x, repeats);

    /// <summary>
    /// Repeats the input repeats[i] times along each dimension.
    /// The repeats rank must match the input rank and every repeat must be non-negative.
    /// </summary>
    public static Tensor<short> Tile(Tensor<short> x, int[] repeats) => TileCore(x, repeats);

    /// <summary>
    /// Repeats the input repeats[i] times along each dimension.
    /// The repeats rank must match the input rank and every repeat must be non-negative.
    /// </summary>
    public static Tensor<ushort> Tile(Tensor<ushort> x, int[] repeats) => TileCore(x, repeats);

    /// <summary>
    /// Repeats the input repeats[i] times along each dimension.
    /// The repeats rank must match the input rank and every repeat must be non-negative.
    /// </summary>
    public static Tensor<Half> Tile(Tensor<Half> x, int[] repeats) => TileCore(x, repeats);

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
                newSize *= 0;
            }
            else
            {
                newShapeDims.Add(Convert.ToInt32(shape[i]));
                newSize *= Convert.ToInt32(shape[i]);
            }
        }
        if (unknownDim != -1)
        {
            if (newSize == 0)
            {
                // Zero-size -1 inference: a literal zero (or a copied zero)
                // makes the known product zero. ORT 1.29 resolves from nonzero
                // extents on both sides for empty inputs and rejects otherwise.
                if (input.Length != 0)
                {
                    throw new ArgumentException(nameof(shape), $"The input tensor cannot be reshaped to the requested shape. Input shape:{input.PrintShape()}, requested shape:{newShapeDims.Print()}");
                }
                newShapeDims[unknownDim] = NonZeroVolume(input.Dimensions.ToArray()) / NonZeroVolume(newShapeDims);
            }
            else
            {
                newShapeDims[unknownDim] = Convert.ToInt32(input.Length / newSize);
            }
            newSize *= newShapeDims[unknownDim];
        }

        static int NonZeroVolume(System.Collections.Generic.IEnumerable<int> dims)
        {
            int volume = 1;
            foreach (var d in dims) if (d != 0 && d != -1) volume *= d;
            return volume;
        }

        if (newSize != input.Length)
        {
            throw new ArgumentException(nameof(shape), $"The input tensor cannot be reshaped to the requested shape. Input shape:{input.PrintShape()}, requested shape:{newShapeDims.Print()}");
        }

        return input.Reshape(newShapeDims.ToArray());
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
            if (perm.Any(q => q < 0))
            {
                throw new ArgumentException(nameof(perm), "The permutation axes must be non-negative.");
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
            if (perm.Any(q => q < 0)) throw new ArgumentException(nameof(perm), "The permutation axes must be non-negative.");
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
        if (ReferenceEquals(destination, data) || TensorAlias.SharesBackingMemory(destination, data)) throw new ArgumentException(nameof(destination), "Destination must not alias the input tensor: permutation is not an in-place operation.");
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

    public static Tensor<T> Concat(Tensor<T> x, Tensor<T> y, int axis)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (x.Rank != y.Rank) throw new ArgumentException(nameof(y), "The rank of each tensor in a concat operation must be the same.");
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(x.Rank, axis);
        if (axis < 0 || axis >= x.Rank) throw new ArgumentException(nameof(axis), "The concat axis is out of range.");
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
        // A single input is the identity (ORT 1.29); only the empty list is rejected.
        if (inputs.Length < 1) throw new ArgumentException(nameof(inputs), "At least one tensor must be specified for the concat operation.");
        if (!inputs.All(i => i.Rank == inputs[0].Rank)) throw new ArgumentException(nameof(inputs), $"Each input tensor in a concat operation must be of the same rank.");
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(inputs[0].Rank, axis);
        if (axis < 0 || axis >= inputs[0].Rank) throw new ArgumentException(nameof(axis), "The concat axis is out of range.");
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
        if (steps is not null && steps.Any(s => s == 0)) throw new ArgumentException(nameof(steps), "Slice steps must be non-zero.");
        
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
        if (!ArrayUtilities.CheckNoRepeatedDims(axes.ToArray())) throw new ArgumentException(nameof(axes), "Slice axes must be distinct.");
        if (axes.Any(a => a < 0 || a >= data.Rank)) throw new ArgumentException(nameof(axes), "Slice axes must index dimensions of the data tensor.");
        if (steps is null)
        {
            steps = Tensor<int>.Ones(length);
        }
       
        start = start.Select((s, i) => ArrayUtilities.Clamp(ArrayUtilities.HandleNegativeAxisOrIndex(data.Dimensions[axes[i]], s), 0, data.Dimensions[axes[i]])).ToArray().ToTensor<int>();
        var stepsArr = steps.ToArray();
        var endsRaw = ends.ToArray();
        // Negative steps: an end below -dim (after axis normalization) means
        // "run past index 0" (verified against ORT 1.29: [8,-100,-2] yields
        // [8,6,4,2,0]); clamping it to 0 would drop the final element, so it
        // is omitted and the -1 default applies downstream.
        int?[] endStops = new int?[length];
        for (int k = 0; k < length; k++)
        {
            int dim = data.Dimensions[axes[k]];
            int n = ArrayUtilities.HandleNegativeAxisOrIndex(dim, endsRaw[k]);
            endStops[k] = stepsArr[k] < 0 && n < 0 ? null : (int?)ArrayUtilities.Clamp(n, 0, dim);
        }
        SliceIndex[] indices = new SliceIndex[data.Rank];
        for (int i = 0; i < data.Rank; i++) 
        {
            indices[i] = axes.Contains(i) ? new SliceIndex(start[axes.IndexOf(i)], endStops[axes.IndexOf(i)], steps[axes.IndexOf(i)]) : new SliceIndex(0, data.dimensions[i]);
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
}
