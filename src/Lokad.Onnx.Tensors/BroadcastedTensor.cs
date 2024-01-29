using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Versioning;

namespace Lokad.Onnx
{
    public class BroadcastedTensor<T> : Tensor<T> where T :  struct
    {
        #region Constructor
        public BroadcastedTensor(Tensor<T> source, ReadOnlySpan<int> dimensions, int[] broadcastedStrides, bool reverseStride = false) : 
            base(dimensions, reverseStride, broadcastedStrides)
        {
            this.source = source;
            this.sourceStrides = source.strides.ToArray();  
        }
        #endregion

        #region Properties
        //public Memory<T> Buffer => memory;
        #endregion

        #region Methods

        #region Tensor<T> members
        public override T GetValue(int index)
        {
            if (index >= Length) throw new IndexOutOfRangeException();
            var idx = ArrayUtilities.TransformIndexByStrides(index, sourceStrides, IsReversedStride, strides);
            return source.GetValue(idx);
        }

        public override void SetValue(int index, T value)
        {
            if (index >= Length) throw new IndexOutOfRangeException();
            var idx = ArrayUtilities.TransformIndexByStrides(index, sourceStrides, IsReversedStride, strides);
            this.source.SetValue(idx, value);
        }

        public override Tensor<T> Clone() => new BroadcastedTensor<T>(source, dimensions, strides, IsReversedStride);
        
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
            var bstrides = strides.ToList();
            bstrides.Insert(dim, 0);    
            return new BroadcastedTensor<T>(source, dims.ToArray(), strides.ToArray(), IsReversedStride);
        }

        public override Tensor<T> RemoveDim(int dim)
        {
            if (dim >= Rank) throw new ArgumentException(nameof(dim));
            var dims = dimensions.ToList();
            dims.RemoveAt(dim);
            var bstrides = strides.ToList();
            bstrides.RemoveAt(dim);
            return new BroadcastedTensor<T>(source, dims.ToArray(), strides.ToArray(), IsReversedStride);
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
                strides[dim] = 0;
                return new BroadcastedTensor<T>(source, dimensions, strides, IsReversedStride);
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
        public int[] sourceStrides;
        #endregion
    }
}
