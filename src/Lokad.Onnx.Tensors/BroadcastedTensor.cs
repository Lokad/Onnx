using System;
using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx
{
    public class BroadcastedTensor<T> : Tensor<T>
    {
        #region Constructor
        public BroadcastedTensor(Memory<T> memory, ReadOnlySpan<int> dimensions, int[] broadcastedStrides, bool reverseStride = false) : 
            base(dimensions, reverseStride)
        {
            this.memory = memory;
            this.sourceStrides = strides;
            this.broadcastedStrides = broadcastedStrides;
        }
        #endregion

        #region Properties
        public Memory<T> Buffer => memory;
        #endregion

        #region Methods

        #region Tensor<T> members
        public override T GetValue(int index)
        {
            var idx = ArrayUtilities.TransformIndexByStrides(index, this.sourceStrides, IsReversedStride, broadcastedStrides);
            return memory.Span[idx];
        }

        public override void SetValue(int index, T value)
        {
            var idx = ArrayUtilities.TransformIndexByStrides(index, this.sourceStrides, IsReversedStride, broadcastedStrides);
            memory.Span[idx] = value;
        }

        public override Tensor<T> Clone()
        {
            // create copy
            return new BroadcastedTensor<T>(new Memory<T>(Buffer.ToArray()), dimensions, broadcastedStrides, IsReversedStride);
        }

        public override Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions)
        {
            throw new NotSupportedException();
        }

        public override Tensor<T> Reshape(ReadOnlySpan<int> dims)
        {
            throw new NotSupportedException();
        }
        #endregion

        public override BroadcastedTensor<T> PadLeft() => new BroadcastedTensor<T>(Buffer, dimensions.Prepend(1).ToArray(), strides.Prepend(0).ToArray(), IsReversedStride);

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
                broadcastedStrides[dim] = 0;
                return new BroadcastedTensor<T>(memory, dimensions, broadcastedStrides, IsReversedStride);
            }
        }

        public static BroadcastedTensor<T>[] PadSame(BroadcastedTensor<T> a, BroadcastedTensor<T> b)
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
        public readonly Memory<T> memory;
        public readonly int[] sourceStrides;
        public readonly int[] broadcastedStrides;
        #endregion
    }
}
