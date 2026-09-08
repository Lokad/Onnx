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

        ITensor CloneEmpty<U>() where U : unmanaged;
        
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

        void SetValue(int index, object? value); 

        static void ThrowIfDifferentElementTypes(params ITensor[] tensors) => Array.ForEach(tensors, tensor =>
        {
            if (tensor.ElementType != tensors[0].ElementType) throw new ArgumentException("All tensors must have the same element type.");
        });
        
        static ITensor[] Broadcast(ITensor inA, ITensor inB)
        {
            if (inA.Rank == 0 && inB.Rank != 0)
            {
                var _A = inB.CloneEmpty();
                for (int i = 0; i < _A.Length; i++)
                {
                    _A.SetValue(i, inA.GetValue(0));
                }
                return new ITensor[2] { _A, inB };
            }
            else if (inB.Rank == 0 && inA.Rank != 0)
            {
                var _B = inA.CloneEmpty();
                for (int i = 0; i < _B.Length; i++)
                {
                    _B.SetValue(i, inB.GetValue(0));
                }
                return new ITensor[2] { inA, _B };
            }

            var broadcastRank = Math.Max(inA.Rank, inB.Rank);
            var outA = inA;
            var outB = inB;
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
                case TensorElementType.Float: return Tensor<float>.Softmax((Tensor<float>)this, -1, null, 13);
                case TensorElementType.Double: return Tensor<double>.Softmax((Tensor<double>)this, -1, null, 13);
                default: throw new NotSupportedException($"The Softmax method is not supported for type {ElementType}.");
            }
        }

        string PrintShape();

        long Length { get; }

        string PrintData(bool includeWhitespace);

        string TensorNameDesc() => $"{Name}:{ElementType.ToString().ToLower()}:{string.Join("x",Dims.Select(d => d.ToString()))}";

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
            for (int i = 0; i < Length; i++)
            {
                output.SetValue(i, (Convert.ToInt32(input.GetValue(i))));
            }
            return output;
        }

        /// <summary>
        /// Inserts size-1 dimensions at the given axes.
        /// Axes index the output tensor, so negatives resolve against Rank plus the inserted count.
        /// </summary>
        ITensor Unsqueeze(int[] axes)
        {
            Profiler.StartOpStage(OpStage.ValidateArguments);    
            
            Profiler.StartOpStage(OpStage.CalculateIndices);
            axes = axes.Select(a => ArrayUtilities.HandleNegativeAxisOrIndex(Rank + axes.Length, a)).ToArray();
            if (!ArrayUtilities.CheckNoRepeatedDims(axes)) throw new ArgumentException(nameof(axes), "axes contains a repeated dimension.");
            if (axes.Any(a => a < 0 || a > (Rank + axes.Length) - 1)) throw new ArgumentException(nameof(axes), "Each specified axis must be a dimension of the output tensor.");
            var newshape = new int[axes.Length + Rank];
            for (int i = 0; i < axes.Length; i++)
            {
                newshape[axes[i]] = 1;
            }
            var e = Dims.GetEnumerator();
            for (int i = 0; i < newshape.Length; i++)
            {
                if (newshape[i] == 0)
                {
                    if (!e.MoveNext()) throw new InvalidOperationException("Out of dimensions.");
                    newshape[i] = (int)e.Current;
                }
            }
            return Reshape(newshape.ToArray());
        }

        Tensor<T> AsTensor<T>() where T : unmanaged => (Tensor<T>) this;
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
