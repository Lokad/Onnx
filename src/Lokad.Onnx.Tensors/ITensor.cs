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

        ITensor CloneEmpty<U>() where U : struct;
        
        ITensor Reshape(params int[] shape);

        ITensor Slice(string indices);

        ITensor InsertDim(int dim);

        ITensor RemoveDim(int dim);

        ITensor BroadcastDim(int dim, int size);

        ITensor ToDenseTensor();

        Array ToArray();

        ITensor PadLeft() => InsertDim(0);
        
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

        void SetValue(int index, object value); 

        static void ThrowIfDifferentElementTypes(params ITensor[] tensors) => Array.ForEach(tensors, tensor =>
        {
            if (tensor.ElementType != tensors[0].ElementType) throw new ArgumentException("All tensors must have the same element type.");
        });
        
        static ITensor[] Broadcast(ITensor inA, ITensor inB)
        {
            var broadcastRank = Math.Max(inA.Rank, inB.Rank);
            var outA = inA.Clone();
            var outB = inB.Clone();
            for (var i = 0; i < broadcastRank; i++)
            {
                var idxA = i - broadcastRank + inA.Rank;
                var idxB = i - broadcastRank + inB.Rank;
                if (i < broadcastRank - inA.Rank)
                {
                    outA = outA.InsertDim(i);
                    outA = outA.BroadcastDim(i, inB.Dims[idxB]);
                }
                else if (i < broadcastRank - inB.Rank)
                {
                    outB = outB.InsertDim(i);
                    outB = outB.BroadcastDim(i, inA.Dims[idxA]);
                }
                else if (inA.Dims[idxA] == inB.Dims[idxB])
                {
                }
                else if (inA.Dims[idxA] == 1)
                {
                    outA = outA.BroadcastDim(i, inB.Dims[idxB]);
                }
                else if (inB.Dims[idxB] == 1)
                {
                    outB = outB.BroadcastDim(i, inA.Dims[idxA]);
                }
                else
                {
                    return Array.Empty<ITensor>();
                    //return OpResult.Failure(OpType.Broadcast, $"Trying to broadcast incompatible shapes: {inA.Dimensions.ToArray()} and {inB.Dimensions.ToArray()}");
                }
            }
            return new[] { outA, outB };
        }

        static bool Broadcast(ITensor x, ITensor y, out ITensor bx, out ITensor by)
        {
            var b = Broadcast(x, y);
            if (b.Length == 0)
            {
                bx = x;
                by = y;
                return false;
            }
            else
            {
                bx = b[0];
                by = b[1];  
                return true;
            }
        }

        ITensor Softmax()
        {
            if (Rank != 1)
            {
                throw new InvalidOperationException("The Softmax method is only valid for vectors.");
            }
            switch (ElementType)
            {
                case TensorElementType.Float: return Tensor<float>.Softmax((Tensor<float>)this);
                case TensorElementType.Double: return Tensor<double>.Softmax((Tensor<double>)this);
                default: throw new NotSupportedException($"The Softmax method is not supported for type {ElementType}.");
            }
        }

        string PrintShape();

        long Length { get; }

        string PrintData(bool includeWhitespace = true);

        string TensorNameDesc() => $"{Name}:{ElementType.ToString().ToLower()}:{string.Join("x",Dims.Select(d => d.ToString()))}";

        ITensor Cast<U>() where U : struct
        {
            ITensor output = CloneEmpty<U>();
            for (int i = 0; i < Length; i++)
            {
                output.SetValue(i, (U)Convert.ChangeType(GetValue(i), typeof(U)));
            }
            return output;
        }
    }
}

public class TensorInputShapeException : ArgumentException
{
    public ITensor Input { get; set; }
    public int[] Shape { get; set; }

    public string Name { get; set; }
    public TensorInputShapeException(string paramName, int[] shape, ITensor input) : base(paramName, $"The input parameter {paramName} has shape {input.PrintShape()} but is required to be {shape.Print()}.")
    {
        Name = paramName;
        Shape = shape;
        Input = input;
    }
}
