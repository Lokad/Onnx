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
    public TensorSlice(Tensor<T> parent, SliceIndex[] indices) : base((ReadOnlySpan<int>) parent.SliceAxes(parent.ExpandEllipsis(indices)), parent.IsReversedStride)
    {
        this.parent = parent;
        this.slices = parent.ExpandEllipsis(indices).Select((i, n) => i.ToSliceDef(parent.dimensions[n])).ToArray(); 
    }
    #endregion

    #region Methods

    #region Tensor<T> methods
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public unsafe override T GetValue(int index)
    {
        int* coords = stackalloc int[parent.Rank];

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
       
        var _coords = new UnsafeFixedSizeList<int>(coords, parent.Rank, strides.Length);
        int offset;

        var orig_ndim = parent.Rank;
        if (orig_ndim > this.Rank && orig_ndim > _coords.Count)
        {
            // fill in reduced dimensions in the provided coordinates 
            for (int i = 0; i < parent.Rank; i++)
            {
                var slice = slices[i];
                if (slice.IsIndex)
                    _coords.Insert(i, 0);
               
            }
        }

        var orig_strides = parent.strides;
        //var orig_dims = vi.OriginalShape.dimensions;
        offset = 0;

        for (int i = 0; i < _coords.Count; i++)
        {
            // note: we can refrain from bounds checking here, because we should not allow negative indices at all, this should be checked higher up though.
            //var coord = coords[i];
            //var dim = orig_dims[i];
            //if (coord < -dim || coord >= dim)
            //    throw new ArgumentException($"index {coord} is out of bounds for axis {i} with a size of {dim}");
            //if (coord < 0)
            //    coord = dim + coord;
            if (slices.Length <= i)
            {
                offset += orig_strides[i] * _coords[i];
                continue;
            }

            var slice = slices[i];
            var start = slice.Start;
            if (slice.IsIndex)
                offset += orig_strides[i] * start; // the coord is irrelevant for index-slices (they are reduced dimensions)
            else
                offset += orig_strides[i] * (start + _coords[i] * slice.Step);
        }

        return parent.GetValue(offset);

        //var indices = GetCoordinates(index);
        //var idx = GetOffset(indices);
        //return parent.GetValue(idx);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public unsafe override void SetValue(int index, T value)
    {
        int* coords = stackalloc int[parent.Rank];

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

        var _coords = new UnsafeFixedSizeList<int>(coords, parent.Rank, strides.Length);
        int offset;

        var orig_ndim = parent.Rank;
        if (orig_ndim > this.Rank && orig_ndim > _coords.Count)
        {
            // fill in reduced dimensions in the provided coordinates 
            for (int i = 0; i < parent.Rank; i++)
            {
                var slice = slices[i];
                if (slice.IsIndex)
                    _coords.Insert(i, 0);

            }
        }

        var orig_strides = parent.strides;
        //var orig_dims = vi.OriginalShape.dimensions;
        offset = 0;

        for (int i = 0; i < _coords.Count; i++)
        {
            // note: we can refrain from bounds checking here, because we should not allow negative indices at all, this should be checked higher up though.
            //var coord = coords[i];
            //var dim = orig_dims[i];
            //if (coord < -dim || coord >= dim)
            //    throw new ArgumentException($"index {coord} is out of bounds for axis {i} with a size of {dim}");
            //if (coord < 0)
            //    coord = dim + coord;
            if (slices.Length <= i)
            {
                offset += orig_strides[i] * _coords[i];
                continue;
            }

            var slice = slices[i];
            var start = slice.Start;
            if (slice.IsIndex)
                offset += orig_strides[i] * start; // the coord is irrelevant for index-slices (they are reduced dimensions)
            else
                offset += orig_strides[i] * (start + _coords[i] * slice.Step);
        }

        parent.SetValue(offset, value);
    }

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
            var idx = GetOffsetUnsafe(indices);
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
            var idx = GetOffsetUnsafe(indices);
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
        int offset;

        var coords = new List<int>(indices);
        if (indices.Length > parent.Dimensions.Length)
            throw new ArgumentOutOfRangeException(nameof(indices), $"select has too many coordinates for this shape");
        var orig_ndim = parent.Rank;
        if (orig_ndim > this.Rank && orig_ndim > indices.Length)
        {
            // fill in reduced dimensions in the provided coordinates 
            for (int i = 0; i < parent.Rank; i++)
            {
                var slice = slices[i];
                if (slice.IsIndex)
                    coords.Insert(i, 0);
                if (coords.Count == orig_ndim)
                    break;
            }
        }

        var orig_strides = parent.strides;
        //var orig_dims = vi.OriginalShape.dimensions;
        offset = 0;
    
        for (int i = 0; i < coords.Count; i++)
        {
            // note: we can refrain from bounds checking here, because we should not allow negative indices at all, this should be checked higher up though.
            //var coord = coords[i];
            //var dim = orig_dims[i];
            //if (coord < -dim || coord >= dim)
            //    throw new ArgumentException($"index {coord} is out of bounds for axis {i} with a size of {dim}");
            //if (coord < 0)
            //    coord = dim + coord;
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

    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
    public unsafe int GetOffsetUnsafe(ReadOnlySpan<int> indices)
    {
        int offset;
        var coordsptr = stackalloc int[parent.Rank]; 
        var coords = new UnsafeFixedSizeList<int>(coordsptr, parent.Rank);
        coords.AddRange(indices);
        var orig_ndim = parent.Rank;
        if (orig_ndim > this.Rank && orig_ndim > indices.Length)
        {
            // fill in reduced dimensions in the provided coordinates 
            for (int i = 0; i < parent.Rank; i++)
            {
                var slice = slices[i];
                if (slice.IsIndex)
                    coords.Insert(i, 0);
                if (coords.Count == orig_ndim)
                    break;
            }
        }

        var orig_strides = parent.strides;
        //var orig_dims = vi.OriginalShape.dimensions;
        offset = 0;

        for (int i = 0; i < coords.Count; i++)
        {
            // note: we can refrain from bounds checking here, because we should not allow negative indices at all, this should be checked higher up though.
            //var coord = coords[i];
            //var dim = orig_dims[i];
            //if (coord < -dim || coord >= dim)
            //    throw new ArgumentException($"index {coord} is out of bounds for axis {i} with a size of {dim}");
            //if (coord < 0)
            //    coord = dim + coord;
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
    #endregion

    #region Fields
    internal Tensor<T> parent;
    internal SliceDef[] slices = Array.Empty<SliceDef>();
    #endregion
}

