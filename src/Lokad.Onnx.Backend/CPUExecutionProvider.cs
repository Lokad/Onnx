namespace Lokad.Onnx.Backend;

extern alias OnnxSharp;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

public enum ExecutionProvider
{
    CPU
}

public class CPUExecutionProvider
{
    public static Dictionary<OpType, int[]> SupportedOps { get; } = new Dictionary<OpType, int[]>();
    
    public bool SupportsOp(OpType op, int version) => SupportedOps.ContainsKey(op) && SupportedOps[op].Any(v => v == version);
    
    public static OpResult Squeeze(int version, ITensor input, ITensor? axes = null)
    {
        Tensor<long>? _axes = null;
        if (axes is not null)
        {
            if (axes.Dims.Length != 1)
            {
                return OpResult.Failure(OpType.Squeeze, $"The axes tensor  {axes.Name} must have dimension 1.");
            }
            else if (axes.ElementType != TensorElementType.Int64)
            {
                return OpResult.WrongInputParameterType(OpType.Squeeze, TensorElementType.Int64, axes);
            }
            else
            {
                _axes = (Tensor<long>)axes;
            }
        }
        
        switch (input.ElementType)
        {
            case TensorElementType.Bool: return Squeeze((Tensor<bool>)input, _axes);
            case TensorElementType.Int8: return Squeeze((Tensor<byte>)input, _axes);
            case TensorElementType.UInt8: return Squeeze((Tensor<sbyte>)input, _axes);
            case TensorElementType.Int16: return Squeeze((Tensor<short>)input, _axes);
            case TensorElementType.UInt16: return Squeeze((Tensor<ushort>)input, _axes);
            case TensorElementType.Int32: return Squeeze((Tensor<int>)input, _axes);
            case TensorElementType.UInt32: return Squeeze((Tensor<uint>)input, _axes);
            case TensorElementType.Int64: return Squeeze((Tensor<long>)input, _axes);
            case TensorElementType.UInt64: return Squeeze((Tensor<ulong>)input, _axes);
            case TensorElementType.Float: return Squeeze((Tensor<float>)input, _axes);
            case TensorElementType.Double: return Squeeze((Tensor<double>)input, _axes);
            case TensorElementType.Float16: return Squeeze((Tensor<Float16>)input, _axes);
            case TensorElementType.BFloat16: return Squeeze((Tensor<BFloat16>)input, _axes);
            case TensorElementType.Complex64: return Squeeze((Tensor<System.Numerics.Complex>)input, _axes);
            default: return OpResult.NotSupported(OpType.Squeeze);
        }
    }

    public static OpResult Squeeze<T>(Tensor<T> input, Tensor<long>? axes = null)
    {
        long[] dims = (axes is not null) ? axes.ToArray() : Enumerable.Range(0, input.Dimensions.Length - 1).Cast<long>().ToArray();
        List<int> squeezedDims = new List<int>();
        for (int i = 0; i < dims.Length; i++)
        {
            var a = TensorUtil.HandleNegativeAxis((int)dims[i], input.Dimensions.Length);
            if (input.Dimensions[a] == 1)
            {
                squeezedDims.Add(a);
            }

        }
        return OpResult.Success(OpType.Squeeze, new[] { input.Reshape(squeezedDims.ToArray()) });
    }

    public static OpResult Broadcast<T>(DenseTensor<T> inA, DenseTensor<T> inB)
    {
        var broadcastRank = Math.Max(inA.Rank, inB.Rank);
        var outA = inA.ToBroadcastedTensor();
        var outB = inB.ToBroadcastedTensor();
        for (var i = 0; i < broadcastRank; i++)
        {
            var idxA = i - broadcastRank + inA.Rank;
            var idxB = i - broadcastRank + inB.Rank;
            if (i < broadcastRank - inA.Rank)
            {
                outA = outA.PadLeft();
                outA = outA.BroadcastDim(0, inB.Dimensions[idxB]);
            }
            else if (i < broadcastRank - inB.Rank)
            {
                outB = outB.PadLeft();
                outB = outB.BroadcastDim(0, inA.Dimensions[idxA]);
            }
            else if (inA.Dimensions[idxA] == inB.Dimensions[idxB])
            {
            }
            else if (inA.Dimensions[idxA] == 1)
            {
                outA = outA.BroadcastDim(i, inB.Dimensions[idxB]);
            }
            else if (inB.Dimensions[idxB] == 1)
            {
                outB = outB.BroadcastDim(i, inA.Dimensions[idxA]);
            }
            else
            {
                return OpResult.Failure(OpType.Broadcast, $"Trying to broadcast incompatible shapes: {inA.Dimensions.ToArray()} and {inB.Dimensions.ToArray()}");
            }
        }
        return OpResult.Success(OpType.Broadcast, new[] { outA, outB });
    }
}

