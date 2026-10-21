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
        var outDims = new int[x.Rank];
        for (int i = 0; i < x.Rank; i++)
        {
            if (repeats[i] < 0) throw new ArgumentException("Repeats must be non-negative.", nameof(repeats));
            long dimension = (long)x.Dimensions[i] * repeats[i];
            if (dimension > int.MaxValue) throw new ArgumentException("Repeats exceeds the maximum tensor dimension.", nameof(repeats));
            outDims[i] = (int)dimension;
        }
        var output = new DenseTensor<U>((ReadOnlySpan<int>)outDims);
        var xd = x.ToDenseTensor();
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
            throw new ArgumentException("Shape tensors must be of rank 1.", nameof(shape));
        }
        if (shape.Any(v => v < -1))
        {
            throw new ArgumentException($"A shape dimension cannot be < -1, got {shape.First(v => v < -1)}.", nameof(shape));
        }
        if (shape.Count(v => v == -1) > 1)
        {
            throw new ArgumentException($"At most 1 shape dimension can be -1.", nameof(shape));
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
                    throw new ArgumentException($"The input tensor cannot be reshaped to the requested shape. Input shape:{input.PrintShape()}, requested shape:{newShapeDims.Print()}", nameof(shape));
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
            throw new ArgumentException($"The input tensor cannot be reshaped to the requested shape. Input shape:{input.PrintShape()}, requested shape:{newShapeDims.Print()}", nameof(shape));
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
                throw new ArgumentException($"The size of the permutation array must be the rank of the tensor: {rank}.", nameof(perm));
            }
            if (!perm.All(q => q < rank))
            {
                throw new ArgumentException($"The permuted dimension {perm.First(q => q >= rank)} exceeds the number of dimensions in the tensor.", nameof(perm));
            }
            if (perm.Any(q => q < 0))
            {
                throw new ArgumentException("The permutation axes must be non-negative.", nameof(perm));
            }
            if (!ArrayUtilities.CheckNoRepeatedDims(perm))
            {
                throw new ArgumentException("The permutation array has a repeated dimension.", nameof(perm));
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
    /// Shared transpose copy: contiguous trailing runs copy whole lines and the
    /// axis-1-to-last float faces use matrix tiles; other shapes keep the odometer over
    /// the destination with source stride math, so no iterator objects or
    /// per-element virtual dispatch remain. The source is densified once up front;
    /// the destination must be standard.
    /// </summary>
    static void TransposeInto(Tensor<T> data, DenseTensor<T> destination, int[] perm)
    {
        // Empty tensors have no faces; do not flatten their potentially large axes.
        if (destination.Length == 0) return;
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
        // Fast path: a trailing output run that advances source bytes by one
        // copies whole lines (head-split attention permutes are 360 line copies
        // here); every other shape keeps the element loop below by construction.
        long runBytes = 1;
        int runAxes = 0;
        for (int d = rank - 1; d >= 0; d--)
        {
            if (map[d] != runBytes) break;
            runBytes *= destDims[d];
            runAxes++;
        }
        if (runAxes > 0 && runBytes > 1)
        {
            int block = (int)runBytes;
            int outer = (int)destination.Length / block;
            int outerRank = rank - runAxes;
            for (int o = 0; o < outer; o++)
            {
                int srcOff = 0;
                int rem = o;
                for (int d = outerRank - 1; d >= 0; d--)
                {
                    int c = rem % destDims[d];
                    rem /= destDims[d];
                    srcOff += c * map[d];
                }
                xs.Slice(srcOff, block).CopyTo(ds.Slice(o * block, block));
            }
            return;
        }
        // Moving axis 1 to the end is a matrix transpose per batch. Use the
        // existing contiguous-load tile when both axes contain a complete tile.
        if (rank == 3 && perm[0] == 0 && perm[1] == 2 && perm[2] == 1
            && typeof(T) == typeof(float) && AblationSwitches.EnableVectorTransposeFaces && Avx.IsSupported
            && xd.Dimensions[1] >= 8 && xd.Dimensions[2] >= 8 && HasStandardStrides(xd))
        {
            unsafe
            {
                using var source = xd.Buffer.Pin();
                using var target = destination.Buffer.Pin();
                transpose_unsafe_shuffle8x8_lastTwoAxes(xd.Dimensions[0], 1, xd.Dimensions[1], xd.Dimensions[2],
                    (float*)source.Pointer, (float*)target.Pointer);
            }
            return;
        }
        // [B,H,S,D] -> [B,S,D,H] collapses to [B,H,S*D] -> [B,S*D,H].
        // Nonempty dense storage ensures the flattened suffix fits in an int.
        if (rank == 4 && perm[0] == 0 && perm[1] == 2 && perm[2] == 3 && perm[3] == 1 && HasStandardStrides(xd))
        {
            int dimB = xd.Dimensions[0], dimH = xd.Dimensions[1], dimS = xd.Dimensions[2], dimD = xd.Dimensions[3];
            if (typeof(T) == typeof(float) && AblationSwitches.EnableVectorTransposeFaces && Avx.IsSupported)
            {
                unsafe
                {
                    using var source = xd.Buffer.Pin();
                    using var target = destination.Buffer.Pin();
                    if (dimH >= 8 && dimS * dimD >= 8)
                        transpose_unsafe_shuffle8x8_lastTwoAxes(dimB, 1, dimH, dimS * dimD, (float*)source.Pointer, (float*)target.Pointer);
                    else
                        transpose_unsafe_vector8_headMerge(dimB, dimS, dimH, dimD, (float*)source.Pointer, (float*)target.Pointer);
                }
                return;
            }
            for (int b = 0; b < dimB; b++)
                for (int i = 0; i < dimS; i++)
                    for (int jj = 0; jj < dimD; jj++)
                        for (int k = 0; k < dimH; k++)
                            ds[((b * dimS + i) * dimD + jj) * dimH + k] = xs[((b * dimH + k) * dimS + i) * dimD + jj];
            return;
        }
        // Fast path: the 4D last-two-axes swap (0,1,3,2) is one 2D rotation per
        // batch-and-head position, tiled directly instead of odometer-stepped.
        // Each output is written once from its source element, so this is
        // bit-identical to the generic path by construction.
        if (rank == 4 && perm[0] == 0 && perm[1] == 1 && perm[2] == 3 && perm[3] == 2 && HasStandardStrides(xd))
        {
            int dimB = xd.Dimensions[0], dimH = xd.Dimensions[1], dimS = xd.Dimensions[2], dimD = xd.Dimensions[3];
            if (typeof(T) == typeof(float) && AblationSwitches.EnableVectorTransposeFaces && Avx.IsSupported)
            {
                unsafe
                {
                    using var source = xd.Buffer.Pin();
                    using var target = destination.Buffer.Pin();
                    transpose_unsafe_shuffle8x8_lastTwoAxes(dimB, dimH, dimS, dimD, (float*)source.Pointer, (float*)target.Pointer);
                }
                return;
            }
            const int Tile = 8;
            for (int b = 0; b < dimB; b++)
                for (int h = 0; h < dimH; h++)
                    for (int i = 0; i < dimS; i += Tile)
                        for (int j = 0; j < dimD; j += Tile)
                            for (int ii = i; ii < System.Math.Min(i + Tile, dimS); ii++)
                                for (int jj = j; jj < System.Math.Min(j + Tile, dimD); jj++)
                                    ds[((b * dimH + h) * dimD + jj) * dimS + ii] = xs[((b * dimH + h) * dimS + ii) * dimD + jj];
            return;
        }
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
        var r = DenseTensor<T>.OfShape(TransposedShapeCore(data.Dimensions, p));
        TransposeInto(data, r, p);
        return r;
    }

    /// <summary>Computes the transposed shape without moving elements.</summary>
    public static int[] TransposedShape(ReadOnlySpan<int> dimensions, int[]? perm) =>
        TransposedShapeCore(dimensions, NormalizeTransposePerm(dimensions.Length, perm));

    static int[] TransposedShapeCore(ReadOnlySpan<int> dimensions, int[] perm)
    {
        var shape = new int[dimensions.Length];
        for (int i = 0; i < perm.Length; i++) shape[i] = dimensions[perm[i]];
        return shape;
    }

    /// <summary>Writes the transposed tensor into an existing dense destination.</summary>
    public static Tensor<T> Transpose(Tensor<T> data, DenseTensor<T> destination, int[]? perm)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        StartOpStage(OpStage.ValidateArguments);
        var p = NormalizeTransposePerm(data.Rank, perm);
        if (!destination.Dimensions.SequenceEqual(TransposedShapeCore(data.Dimensions, p))) throw new ArgumentException("Destination shape must match the transposed shape.", nameof(destination));
        if (!HasStandardStrides(destination)) throw new ArgumentException("Destination must have standard row-major strides.", nameof(destination));
        if (ReferenceEquals(destination, data) || TensorAlias.SharesBackingMemory(destination, data)) throw new ArgumentException("Destination must not alias the input tensor: permutation is not an in-place operation.", nameof(destination));
        StartOpStage(OpStage.Copy);
        TransposeInto(data, destination, p);
        return destination;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]  
    public static Tensor<T> Gather(Tensor<T> data, Tensor<int> indices, int? _axis)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (data.Rank == 0) throw new ArgumentException("Cannot gather from a tensor of rank 0.", nameof (data));

        StartOpStage(OpStage.CalculateIndices);
        var axis = _axis.HasValue ? ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, _axis.Value) : 0;    
        if (axis < 0 || axis >= data.Rank)
        {
            throw new ArgumentException($"The specified axis {_axis} exceeds the number of dimensions in the tensor.", nameof(_axis));
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
        if (output.Length == 0) return output;
        StartOpStage(OpStage.Copy);
        if (data is DenseTensor<T> denseData && HasStandardStrides(denseData)
            && indices is DenseTensor<int> denseIndices && HasStandardStrides(denseIndices))
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
            for (int j = 0; j < rows.Length; j++) rows[j] = NormalizeGatherIndex(indexSpan[j], axisDim, nameof(indices));
            for (int o = 0; o < outer; o++)
            {
                for (int j = 0; j < rows.Length; j++)
                {
                    srcSpan.Slice((o * axisDim + rows[j]) * inner, inner).CopyTo(dstSpan.Slice((o * rows.Length + j) * inner, inner));
                }
            }
            return output;
        }
        var sourceCoordinates = new int[data.Rank];
        var indexCoordinates = new int[indices.Rank];
        int position = 0;
        foreach (var di in output.GetDimensionsIterator())
        {
            di.AsSpan(0, axis).CopyTo(sourceCoordinates);
            di.AsSpan(axis, indices.Rank).CopyTo(indexCoordinates);
            sourceCoordinates[axis] = NormalizeGatherIndex(indices[indexCoordinates], data.dimensions[axis], nameof(indices));
            di.AsSpan(axis + indices.Rank, data.Rank - axis - 1).CopyTo(sourceCoordinates.AsSpan(axis + 1));
            output.SetValue(position++, data[sourceCoordinates]);
        }
        return output;  
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static int NormalizeGatherIndex(int index, int dimension, string parameterName)
    {
        int normalized = ArrayUtilities.HandleNegativeAxisOrIndex(dimension, index);
        if (normalized < 0 || normalized >= dimension)
            throw new ArgumentOutOfRangeException(parameterName, index, "Gather index is outside the selected axis.");
        return normalized;
    }

    public static Tensor<T> Concat(Tensor<T> x, Tensor<T> y, int axis)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (x.Rank != y.Rank) throw new ArgumentException("The rank of each tensor in a concat operation must be the same.", nameof(y));
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(x.Rank, axis);
        if (axis < 0 || axis >= x.Rank) throw new ArgumentException("The concat axis is out of range.", nameof(axis));
        for (int i = 0; i < x.Rank; i++)
        {
            if (i == axis) continue;
            if (x.dimensions[i] != y.dimensions[i])
            {
                throw new ArgumentException("The dimensions of each tensor in a concat operation must be the same, with the exception of the axis dimension.", nameof(y));
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
        => Concat(inputs, axis, null);

    internal static int[] ConcatShape(Tensor<T>[] inputs, ref int axis)
    {
        StartOpStage(OpStage.ValidateArguments);
        // A single input is the identity (ORT 1.29); only the empty list is rejected.
        if (inputs.Length < 1) throw new ArgumentException("At least one tensor must be specified for the concat operation.", nameof(inputs));
        if (!inputs.All(i => i.Rank == inputs[0].Rank)) throw new ArgumentException($"Each input tensor in a concat operation must be of the same rank.", nameof(inputs));
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(inputs[0].Rank, axis);
        if (axis < 0 || axis >= inputs[0].Rank) throw new ArgumentException("The concat axis is out of range.", nameof(axis));
        int normalizedAxis = axis;
        if (!inputs.All(i => i.dimensions.Select((d, n) => n == normalizedAxis ? 0 : d - inputs[0].dimensions[n]).All(s => s == 0)))
            throw new ArgumentException("The dimensions of each tensor in a concat operation must be the same, with the exception of the axis dimension.", nameof(inputs));
        var shape = inputs[0].dimensions.Copy();
        shape[axis] = 0;
        foreach (var input in inputs) shape[axis] += input.dimensions[axis];
        return shape;
    }

    internal static Tensor<T> Concat(Tensor<T>[] inputs, int axis, DenseTensor<T>? destination)
    {
        var shape = ConcatShape(inputs, ref axis);
        if (destination is not null && (!destination.Dimensions.SequenceEqual(shape) || !HasStandardStrides(destination)))
            throw new ArgumentException("Destination must be contiguous and match the concatenated shape.", nameof(destination));

        StartOpStage(OpStage.Copy);
        var output = destination ?? DenseTensor<T>.OfShape(shape);
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
        if (data.Rank == 0) throw new ArgumentException("Cannot slice a tensor of rank 0.", nameof(data));
        if (start.Rank != 1) throw new ArgumentException("The rank of the start tensor must be 1.", nameof(start));
        if (start.Length > data.Rank) throw new ArgumentException("The length of the start tensor must be less-than or equal to the rank of the data tensor.", nameof(start));
        if (ends.Rank != 1) throw new ArgumentException("The rank of the end tensor must be 1", nameof(ends));
        if (start.Length != ends.Length) throw new ArgumentException("The end tensor must be the same length as the start tensor.", nameof(ends));
        if (axes is not null && (axes.Rank != 1 || axes.Length != start.Length)) throw new ArgumentException("The axes tensor must be a rank 1 tensor with the same length as the start tensor.", nameof(axes));
        if (steps is not null && (steps.Rank != 1 || steps.Length != start.Length)) throw new ArgumentException("The steps tensor must be a rank 1 tensor with the same length as the start tensor.", nameof(steps));
        if (steps is not null && steps.Any(s => s == 0)) throw new ArgumentException("Slice steps must be non-zero.", nameof(steps));
        
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
        if (!ArrayUtilities.CheckNoRepeatedDims(axes.ToArray())) throw new ArgumentException("Slice axes must be distinct.", nameof(axes));
        if (axes.Any(a => a < 0 || a >= data.Rank)) throw new ArgumentException("Slice axes must index dimensions of the data tensor.", nameof(axes));
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
        var shape = ArrayUtilities.UnsqueezedShape(data.Dimensions, axes);
        StartOpStage(OpStage.CalculateIndices);
        return data.Reshape(shape);
    }
}
