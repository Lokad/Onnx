using System;
using System.Collections;
using System.Linq;

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

        ITensor Reshape(params int[] shape);

        ITensor Slice(string indices);

        ITensor InsertDim(int dim);
        
        ITensor BroadcastDim(int dim, int size);

        ITensor ToDenseTensor();

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
                    outA = outA.PadLeft();
                    outA = outA.BroadcastDim(0, inB.Dims[idxB]);
                }
                else if (i < broadcastRank - inB.Rank)
                {
                    outB = outB.PadLeft();
                    outB = outB.BroadcastDim(0, inA.Dims[idxA]);
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
    }
}
