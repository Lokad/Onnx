using Lokad.Onnx;
using System;
using System.Collections;
using System.Linq;
using System.Text;

namespace Lokad.Onnx
{
    public interface ITensor : IEnumerable
    {
        string Name { get; set; }

        TensorElementType ElementType { get; }
        
        Type PrimitiveType {get; }
        
        int[] Dims { get; }

        int Rank { get; }

        ITensor Clone();

        ITensor CloneEmpty();

        Array ToArray();

        object this[params int[] indices]
        {
            get;
            set;
        }

        ITensor this[params object[] indices]
        {
            get;
            set;
        }

        object GetValue(int index);

        void SetValue(int index, object? value); 

        static void ThrowIfDifferentElementTypes(params ITensor[] tensors) => Array.ForEach(tensors, tensor =>
        {
            if (tensor.ElementType != tensors[0].ElementType) throw new ArgumentException("All tensors must have the same element type.");
        });
        
        static ITensor[] Broadcast(INumericTensor inA, INumericTensor inB)
        {
            return Broadcast(inA, inB, out var left, out var right)
                ? new[] { left, right } : Array.Empty<ITensor>();
        }

        static bool Broadcast(INumericTensor x, INumericTensor y, out ITensor bx, out ITensor by)
        {
            bool compatible = TryBroadcast(x, x.Dims, y, y.Dims, out var left, out var right);
            bx = left;
            by = right;
            return compatible;
        }

        internal static bool TryBroadcast(INumericTensor x, ReadOnlySpan<int> xShape,
            INumericTensor y, ReadOnlySpan<int> yShape, out INumericTensor bx, out INumericTensor by)
        {
            bx = x;
            by = y;
            if (xShape.SequenceEqual(yShape)) return true;
            if (!ArrayUtilities.TryBroadcastShape(xShape, yShape, out var shape)) return false;
            bx = BroadcastView(x, xShape, shape);
            by = BroadcastView(y, yShape, shape);
            return true;
        }

        internal static INumericTensor BroadcastView(INumericTensor input, ReadOnlySpan<int> dimensions, int[] shape)
        {
            int offset = shape.Length - dimensions.Length;
            var result = input;
            for (int i = 0; i < offset; i++) result = result.InsertDim(0);
            for (int i = 0; i < shape.Length; i++)
                if ((i < offset ? 1 : dimensions[i - offset]) != shape[i])
                    result = result.BroadcastDim(i, shape[i]);
            return result;
        }

        ITensor Softmax()
        {
            if (Rank != 1)
            {
                throw new InvalidOperationException("The Softmax method is only valid for vectors.");
            }
            switch (ElementType)
            {
                case TensorElementType.Float: return Tensor<float>.Softmax((Tensor<float>)this, -1, null, 13);
                case TensorElementType.Double: return Tensor<double>.Softmax((Tensor<double>)this, -1, null, 13);
                default: throw new NotSupportedException($"The Softmax method is not supported for type {ElementType}.");
            }
        }

        string PrintShape();

        long Length { get; }

        string PrintData(bool includeWhitespace);

        string TensorNameDesc() => $"{Name}:{ElementType.ToString().ToLowerInvariant()}:{string.Join("x",Dims.Select(d => d.ToString()))}";

        ITensor Cast<U>() where U : unmanaged
        {
            Profiler.StartOpStage(OpStage.Cast);
            var info = TensorBase.GetTypeInfo(typeof(U));
            if (info is null) throw new NotSupportedException($"Cast to {typeof(U).Name} is not supported.");
            return CastOps.Cast(this, info.ElementType);
        }

        ITensor ConvertToInt32() 
        {
            Profiler.StartOpStage(OpStage.Cast);
            if (this.ElementType != TensorElementType.Int64) throw new ArgumentException();
            Tensor<long> input = (Tensor<long>) this;
            DenseTensor<int> output = DenseTensor<int>.OfShape(Dims);
            if (input is DenseTensor<long> dense && !dense.IsReversedStride)
            {
                var source = dense.Buffer.Span;
                var destination = output.Buffer.Span;
                for (int i = 0; i < source.Length; i++)
                    destination[i] = Convert.ToInt32(source[i]);
            }
            else
            {
                // Flatten logical coordinates, not a view's or reversed tensor's storage.
                int position = 0;
                foreach (var coordinate in input.GetDimensionsIterator())
                    output.SetValue(position++, Convert.ToInt32(input[coordinate]));
            }
            return output;
        }

}

    /// <summary>
    /// Numeric shape capability for dense-compatible tensors. Sequences are
    /// tensors for binding and validation but never implement this
    /// interface: every member below is meaningless without numeric storage.
    /// </summary>
    public interface INumericTensor : ITensor
    {
        INumericTensor CloneEmpty<U>() where U : unmanaged;

        INumericTensor Reshape(params int[] shape);

        INumericTensor Slice(string indices);

        INumericTensor InsertDim(int dim);

        INumericTensor RemoveDim(int dim);

        INumericTensor BroadcastDim(int dim, int size);

        INumericTensor ToDenseTensor();

        INumericTensor PadLeft() => InsertDim(0);

        /// <summary>
        /// Inserts size-1 dimensions at the given axes.
        /// Axes index the output tensor, so negatives resolve against Rank plus the inserted count.
        /// </summary>
        INumericTensor Unsqueeze(int[] axes)
        {
            Profiler.StartOpStage(OpStage.ValidateArguments);
            var shape = ArrayUtilities.UnsqueezedShape(Dims, axes);
            Profiler.StartOpStage(OpStage.CalculateIndices);
            return Reshape(shape);
        }
    }
}

public class TensorInputShapeException : ArgumentException
{
    public ITensor Input { get; set; }
    public int[] Shape { get; set; }

    public string Name { get; set; }
    public TensorInputShapeException(string paramName, int[] shape, ITensor input) : base($"The input parameter {paramName} has shape {input.PrintShape()} but is required to be {shape.Print()}.", paramName)
    {
        Name = paramName;
        Shape = shape;
        Input = input;
    }
}
