namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;

public class BroadcastedTensor<T> : Tensor<T> where T :  struct
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
    }
    #endregion

    #region Methods

    #region Tensor<T> members
    public override T GetValue(int index)
    {
        if (index >= Length) throw new IndexOutOfRangeException();
        int[] indices = new int[this.Rank];
        ArrayUtilities.GetIndices(strides, IsReversedStride, index, indices);
        return source.GetValue(ArrayUtilities.GetIndex(source.strides, indices, broadcastedDims));
    }

    public override void SetValue(int index, T value)
    {
        if (index >= Length) throw new IndexOutOfRangeException();
        int[] indices = new int[this.Rank];
        ArrayUtilities.GetIndices(strides, IsReversedStride, index, indices);
        this.source.SetValue(ArrayUtilities.GetIndex(source.strides, indices, broadcastedDims), value);
    }

    public override Tensor<T> Clone() => new BroadcastedTensor<T>(source, dimensions, broadcastedDims);
        
    public override Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions) => new DenseTensor<TResult>(dimensions);  

    public override Tensor<T> Reshape(ReadOnlySpan<int> dims)
    {
        throw new NotSupportedException();
    }
    #endregion

    public override Tensor<T> InsertDim(int dim)
    {
        if (dim >= Rank) throw new ArgumentException(nameof(dim));
        var dims = this.dimensions.ToList();
        dims.Insert(dim, 1);
        return new BroadcastedTensor<T>(source, dims.ToArray(), broadcastedDims);
    }

    public override Tensor<T> RemoveDim(int dim)
    {
        if (dim >= Rank) throw new ArgumentException(nameof(dim));
        if (dimensions[dim] != 1) throw new ArgumentException(nameof(dim), $"Can only remove a dimension of size 1. Dimension {dim} has size {dimensions[dim]}.");
        var dims = dimensions.ToList();
        dims.RemoveAt(dim);
        return new BroadcastedTensor<T>(source, dims.ToArray(), broadcastedDims);
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

    public static Tensor<T>[] PadSame(Tensor<T> a, Tensor<T> b)
    {
        if (a.Dimensions.Length < b.Dimensions.Length)
        {
            return PadSame(a.PadLeft(), b);
        }
        else if (b.Dimensions.Length < a.Dimensions.Length)
        {
            return PadSame(a, b.PadLeft());
        }
        else return new[] { a, b };
    }
    #endregion

    #region Fields
    public readonly Tensor<T> source;
    public int[] broadcastedDims;
    #endregion
}

