namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

public class TensorSlice<T> : Tensor<T> where T : unmanaged 
{
    #region Constructors
    public TensorSlice(Tensor<T> parent, SliceIndex[] indices)
        : base((ReadOnlySpan<int>)SliceDims(parent, indices, out var expanded), parent.IsReversedStride)
    {
        this.parent = parent;
        this.slices = expanded.Select((i, n) => i.ToSliceDef(parent.dimensions[n])).ToArray();
    }
    static int[] SliceDims(Tensor<T> parent, SliceIndex[] indices, out SliceIndex[] expanded)
    {
        expanded = parent.ExpandEllipsis(indices);
        return parent.SliceAxes(expanded);
    }
    #endregion

    #region Methods

    #region Tensor<T> methods
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public override T GetValue(int index) => parent.GetValue(ParentOffset(index));

    /// <summary>
    /// Translates a linear index of this view to a parent storage offset using
    /// a method-local span: coordinates never escape, reduced dimensions are
    /// re-inserted as zeros, and every write is capacity-checked.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    int ParentOffset(int index)
    {
        if (strides.Length > parent.Rank) throw new ArgumentOutOfRangeException(nameof(index), "Too many stride dimensions for the parent rank.");
        Span<int> coords = parent.Rank < ArrayUtilities.StackallocMax ? stackalloc int[parent.Rank] : new int[parent.Rank];

        if (strides.Length == 1)
            coords[0] = index;

        int counter = index;

        int stride;
        for (int i = 0; i < strides.Length; i++)
        {
            unchecked
            {
                stride = strides[i];
                if (stride == 0)
                {
                    coords[i] = 0;
                }
                else
                {
                    coords[i] = counter / stride;
                    counter -= coords[i] * stride;
                }
            }
        }

        return GetOffsetCore(coords, strides.Length);
    }

    /// <summary>
    /// Shared offset translation: inserts reduced dimensions into count user
    /// coordinates, then maps through slice starts, steps and parent strides.
    /// Both coordinate-length contracts funnel here after bounds checks.
    /// </summary>
    int GetOffsetCore(Span<int> coords, int count)
    {
        var orig_ndim = parent.Rank;
        if (orig_ndim > Rank && orig_ndim > count)
        {
            // fill in reduced dimensions in the provided coordinates
            for (int i = 0; i < parent.Rank; i++)
            {
                if (i >= slices.Length) break;
                var slice = slices[i];
                if (slice.IsIndex)
                {
                    if (count >= parent.Rank) throw new ArgumentOutOfRangeException(nameof(coords), "Too many coordinates for the parent rank.");
                    for (int j = count; j > i; j--) coords[j] = coords[j - 1];
                    coords[i] = 0;
                    count++;
                }
                if (count == orig_ndim)
                    break;
            }
        }

        var orig_strides = parent.strides;
        int offset = 0;

        for (int i = 0; i < count; i++)
        {
            if (slices.Length <= i)
            {
                offset += orig_strides[i] * coords[i];
                continue;
            }

            var slice = slices[i];
            var start = slice.Start;
            if (slice.IsIndex)
                offset += orig_strides[i] * start; // the coord is irrelevant for index-slices (they are reduced dimensions)
            else
                offset += orig_strides[i] * (start + coords[i] * slice.Step);
        }

        return offset;
    }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public override void SetValue(int index, T value) => parent.SetValue(ParentOffset(index), value);

    /// <summary>
    /// Obtains the value at the specified indices
    /// </summary>
    /// <param name="indices">A span integers that represent the indices specifying the position of the element to get.</param>
    /// <returns>The value at the specified position in this Tensor.</returns>
    
    public override T this[ReadOnlySpan<int> indices]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        get
        {
            if (indices.Length == 1 && Rank == 0 && indices[0] == 0)
            {
                return GetValue(0);
            }
            var idx = GetOffset(indices);
            return parent.GetValue(idx);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        set
        {
            if (indices.Length == 1 && Rank == 0 && indices[0] == 0)
            {
                SetValue(0, value);
                return;
            }
            var idx = GetOffset(indices);
            parent.SetValue(idx, value);
        }
    }

   
    public override Tensor<T> Clone() => ToDenseTensor();

    public override Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions) => new DenseTensor<TResult>(dimensions, this.IsReversedStride);
    
    public override Tensor<T> InsertDim(int dim) => Clone().InsertDim(dim);

    public override Tensor<T> RemoveDim(int dim) => Clone().RemoveDim(dim);

    public override Tensor<T> Reshape(ReadOnlySpan<int> dimensions) => Clone().Reshape(dimensions);

    public override BroadcastedTensor<T> BroadcastDim(int dim, int size) => Clone().BroadcastDim(dim, size);
    #endregion

    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
    public int GetOffset(params int[] indices)
    {
        if (indices.Length > parent.Dimensions.Length)
            throw new ArgumentOutOfRangeException(nameof(indices), $"select has too many coordinates for this shape");
        Span<int> coords = parent.Rank < ArrayUtilities.StackallocMax ? stackalloc int[parent.Rank] : new int[parent.Rank];
        indices.CopyTo(coords);
        return GetOffsetCore(coords, indices.Length);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
    public int GetOffset(ReadOnlySpan<int> indices)
    {
        // Bounded buffer instead of a pointer list: more indices than this view
        // holds are rejected before anything is written.
        if (indices.Length > this.Rank) throw new ArgumentOutOfRangeException(nameof(indices), $"Too many coordinates for tensor rank {this.Rank}.");
        Span<int> coords = parent.Rank < ArrayUtilities.StackallocMax ? stackalloc int[parent.Rank] : new int[parent.Rank];
        indices.CopyTo(coords);
        return GetOffsetCore(coords, indices.Length);
    }
    #endregion

    #region Fields
    internal Tensor<T> parent;
    internal SliceDef[] slices = Array.Empty<SliceDef>();
    #endregion
}

