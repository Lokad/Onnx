﻿namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

public class BroadcastedTensor<T> : Tensor<T> where T :  unmanaged
{
    #region Constructor
    public BroadcastedTensor(Tensor<T> source, ReadOnlySpan<int> dimensions, int[] broadcastedDims) : 
        base(dimensions, false)
    {
        if (broadcastedDims.Length == 0)
        {
            throw new ArgumentException(nameof(broadcastedDims), "The number of broadcasted dimensions cannot be 0.");
        }
        if (broadcastedDims.Length > dimensions.Length) 
        { 
            throw new ArgumentException(nameof(broadcastedDims), "The number of broadcasted dimensions cannot be more than the number of source dimensions.");
        }
        this.source = source;
        this.broadcastedDims = broadcastedDims;
        this.effectiveStrides = source.strides.Copy();
        for (int i = 0; i < dimensions.Length; i++)
        {
            if (Array.IndexOf(broadcastedDims, i) != -1)
            {
                effectiveStrides[i] = 0;   
            }
        }  
    }
    #endregion

    #region Methods

    #region Tensor<T> members
    public override T this[ReadOnlySpan<int> indices]
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        get => this.source.GetValue(ArrayUtilities.GetIndex(effectiveStrides, indices));

        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        set => this.source.SetValue(ArrayUtilities.GetIndex(effectiveStrides, indices), value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public override T GetValue(int index)
    {
        int[] indices = new int[this.Rank];
        ArrayUtilities.GetIndices(strides, IsReversedStride, index, indices);
        return source.GetValue(ArrayUtilities.GetIndex(effectiveStrides, indices));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public override void SetValue(int index, T value)
    {
        int[] indices = new int[this.Rank];
        ArrayUtilities.GetIndices(strides, IsReversedStride, index, indices);
        this.source.SetValue(ArrayUtilities.GetIndex(effectiveStrides, indices), value);
    }

    public override Tensor<T> Clone() => new BroadcastedTensor<T>(source, dimensions, broadcastedDims);
        
    public override Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions) => new DenseTensor<TResult>(dimensions);  

    public override Tensor<T> Reshape(ReadOnlySpan<int> dims)
    {
        throw new NotSupportedException();
    }
   
    public override Tensor<T> InsertDim(int dim)
    {
        if (dim >= Rank) throw new ArgumentException(nameof(dim));
        var dims = this.dimensions.ToList();
        dims.Insert(dim, 1);
        var bdims = broadcastedDims.Copy();
        for(int i = 0; i < bdims.Length; i++)
        {
            if (bdims[i] >= dim)
            {
                bdims[i] += 1;
            }
        }
        return new BroadcastedTensor<T>(source.InsertDim(dim), dims.ToArray(), bdims);
    }

    public override Tensor<T> RemoveDim(int dim)
    {
        if (dim >= Rank) throw new ArgumentException(nameof(dim));
        if (dimensions[dim] != 1) throw new ArgumentException(nameof(dim), $"Can only remove a dimension of size 1. Dimension {dim} has size {dimensions[dim]}.");
        var dims = dimensions.ToList();
        dims.RemoveAt(dim);
        var bdims = broadcastedDims.Copy();
        for (int i = 0; i < bdims.Length; i++)
        {
            if (bdims[i] >= dim)
            {
                bdims[i] -= 1;
            }
        }
        return new BroadcastedTensor<T>(source.RemoveDim(dim), dims.ToArray(), bdims);
    }

    public override BroadcastedTensor<T> BroadcastDim(int dim, int size)
    {
        if (dim >= Rank)
        {
            throw new ArgumentException($"The specified dimension {dim} exceeds the tensor rank.");
        }
        else if (dimensions[dim] != 1)
        {
            throw new ArgumentException($"Dimension {dim} must be of size 1 to broadcast.");
        }
        else
        {
            dimensions[dim] = size;
            return new BroadcastedTensor<T>(source, dimensions, broadcastedDims.Append(dim).ToArray());
        }
    }
    #endregion

    #endregion

    #region Fields
    public readonly Tensor<T> source;
    public int[] broadcastedDims;
    public int[] effectiveStrides;
    #endregion
}

