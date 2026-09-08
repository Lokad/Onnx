namespace Lokad.Onnx;

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

    /// <summary>
    /// Index decomposition without per-element coordinate arrays: this view is
    /// always standard row-major, so digits come straight from the flat index.
    /// </summary>
    int ToSourceOffset(int index)
    {
        int rem = index;
        int offset = 0;
        for (int d = Rank - 1; d >= 0; d--)
        {
            int dim = dimensions[d];
            int coord = dim == 0 ? 0 : rem % dim;
            rem = dim == 0 ? rem : rem / dim;
            offset += coord * effectiveStrides[d];
        }
        return offset;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public override T GetValue(int index) => source.GetValue(ToSourceOffset(index));

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public override void SetValue(int index, T value) => this.source.SetValue(ToSourceOffset(index), value);

    /// <summary>
    /// Copies values into new backing storage, per the clone contract.
    /// Use <see cref="BroadcastDim"/> to create sharing views instead.
    /// </summary>
    public override Tensor<T> Clone() => ToDenseTensor();
        
    public override Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions) => new DenseTensor<TResult>(dimensions);  

    /// <summary>
    /// Materializes the broadcast view with block copies over contiguous source
    /// runs instead of the per-element indexed base implementation, which
    /// allocates an index array per element. Falls back to the base path when
    /// the source is not a standard row-major dense tensor or the positional
    /// dimension mapping does not hold.
    /// </summary>
    public override DenseTensor<T> ToDenseTensor()
    {
        var src = source as DenseTensor<T> ?? source.ToDenseTensor();
        if (src.IsReversedStride) return base.ToDenseTensor();
        int rank = Rank;
        var dims = Dimensions.ToArray();
        var sd = src.Dimensions.ToArray();
        if (sd.Length != rank) return base.ToDenseTensor();
        var srcStrides = ArrayUtilities.GetStrides(sd);
        var estride = new int[rank];
        for (int d = 0; d < rank; d++)
        {
            if (Array.IndexOf(broadcastedDims, d) != -1) estride[d] = 0;
            else
            {
                if (sd[d] != dims[d]) return base.ToDenseTensor();
                estride[d] = srcStrides[d];
            }
        }
        var output = new DenseTensor<T>(dimensions);
        var dst = output.Buffer.Span;
        var sbuf = src.Buffer.Span;
        int run = 1;
        int rd = rank - 1;
        while (rd >= 0)
        {
            if (dims[rd] == 1) { rd--; continue; }
            if (estride[rd] != run) break;
            run *= dims[rd];
            rd--;
        }
        Span<int> pos = stackalloc int[rank];
        int srcPos = 0;
        int dstPos = 0;
        int total = 1;
        foreach (var dd in dims) total *= dd;
        int blocks = total / run;
        for (int b = 0; b < blocks; b++)
        {
            sbuf.Slice(srcPos, run).CopyTo(dst.Slice(dstPos, run));
            dstPos += run;
            for (int d = rd; d >= 0; d--)
            {
                pos[d]++;
                srcPos += estride[d];
                if (pos[d] < dims[d]) break;
                pos[d] = 0;
                srcPos -= estride[d] * dims[d];
            }
        }
        return output;
    }
    public override Tensor<T> Reshape(ReadOnlySpan<int> dims)
    {
            return ToDenseTensor().Reshape(dims);
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
            var dims = (int[])dimensions.Clone();
            dims[dim] = size;
            return new BroadcastedTensor<T>(source, dims, broadcastedDims.Append(dim).ToArray());
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

