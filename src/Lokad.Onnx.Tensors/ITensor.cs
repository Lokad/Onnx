using System;

namespace Lokad.Onnx
{
    public interface ITensor
    {
        string Name { get; set; }

        TensorElementType ElementType { get; }
        
        Type PrimitiveType {get; }
        
        int[] Dims { get; }

        int Rank { get; }

        ITensor Clone();

        ITensor Reshape(int[] shape);

        ITensor InsertDim(int dim);
        
        ITensor PadLeft();

        ITensor BroadcastDim(int dim, int size);

        ITensor ToBroadcastedTensor();

        ITensor ToDenseTensor();

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
