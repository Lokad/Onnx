namespace Lokad.Onnx.Backend;

extern alias OnnxSharp;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Versioning;

using static OpResult;
public enum ExecutionProvider
{
    CPU
}

[RequiresPreviewFeatures]
public class CPUExecutionProvider
{
    public static Dictionary<OpType, int[]> SupportedOps { get; } = new Dictionary<OpType, int[]>();
    
    public bool SupportsOp(OpType op, int version) => SupportedOps.ContainsKey(op) && SupportedOps[op].Any(v => v == version);
    
    public static OpResult Reshape(ITensor? input, ITensor? shape, bool? allowZero = null)
    {
        var op = OpType.Reshape;
        if (input is null) return MissingInput(op, nameof(input));
        if (shape is null) return MissingInput(op, nameof(shape));
        if (shape.ElementType != TensorElementType.Int64) return WrongInputType(op, nameof(shape), TensorElementType.Int64, shape);
        switch (input.ElementType)
        {
            case TensorElementType.Bool: return Reshape((Tensor<bool>)input, (Tensor<long>) shape, allowZero);
            case TensorElementType.Int8: return Reshape((Tensor<byte>)input, (Tensor<long>) shape, allowZero);
            case TensorElementType.UInt8: return Reshape((Tensor<sbyte>)input, (Tensor<long>) shape, allowZero);
            case TensorElementType.Int16: return Reshape((Tensor<short>)input, (Tensor<long>) shape, allowZero);
            case TensorElementType.UInt16: return Reshape((Tensor<ushort>)input, (Tensor<long>) shape, allowZero);
            case TensorElementType.Int32: return Reshape((Tensor<int>)input, (Tensor<long>) shape, allowZero);
            case TensorElementType.UInt32: return Reshape((Tensor<uint>)input, (Tensor<long>) shape, allowZero);
            case TensorElementType.Int64: return Reshape((Tensor<long>)input, (Tensor<long>) shape, allowZero);
            case TensorElementType.UInt64: return Reshape((Tensor<ulong>)input, (Tensor<long>) shape, allowZero);
            case TensorElementType.Float: return Reshape((Tensor<float>)input, (Tensor<long>) shape, allowZero);
            case TensorElementType.Double: return Reshape((Tensor<double>)input, (Tensor<long>) shape, allowZero);
            case TensorElementType.Float16: return Reshape((Tensor<Half>)input, (Tensor<long>) shape, allowZero);
            case TensorElementType.BFloat16: return Reshape((Tensor<BFloat16>)input, (Tensor<long>) shape, allowZero);
            case TensorElementType.Complex64: return Reshape((Tensor<System.Numerics.Complex>)input, (Tensor<long>) shape, allowZero);
            default: return OpResult.NotSupported(OpType.Squeeze);
        }
    }

    public static OpResult Reshape<T>(Tensor<T> input, Tensor<long> shape, bool? allowZero = null) where T : struct 
        => Success(OpType.Reshape, Tensor<T>.Reshape(input, shape, allowZero ?? false));
    public static OpResult Squeeze(int version, ITensor input, ITensor? axes = null)
    {
        Tensor<long>? shape = null;
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
                shape = (Tensor<long>)axes;
            }
        }
        
        switch (input.ElementType)
        {
            case TensorElementType.Bool: return Squeeze((Tensor<bool>)input, shape);
            case TensorElementType.Int8: return Squeeze((Tensor<byte>)input, shape);
            case TensorElementType.UInt8: return Squeeze((Tensor<sbyte>)input, shape);
            case TensorElementType.Int16: return Squeeze((Tensor<short>)input, shape);
            case TensorElementType.UInt16: return Squeeze((Tensor<ushort>)input, shape);
            case TensorElementType.Int32: return Squeeze((Tensor<int>)input, shape);
            case TensorElementType.UInt32: return Squeeze((Tensor<uint>)input, shape);
            case TensorElementType.Int64: return Squeeze((Tensor<long>)input, shape);
            case TensorElementType.UInt64: return Squeeze((Tensor<ulong>)input, shape);
            case TensorElementType.Float: return Squeeze((Tensor<float>)input, shape);
            case TensorElementType.Double: return Squeeze((Tensor<double>)input, shape);
            case TensorElementType.Float16: return Squeeze((Tensor<Half>)input, shape);
            case TensorElementType.BFloat16: return Squeeze((Tensor<BFloat16>)input, shape);
            case TensorElementType.Complex64: return Squeeze((Tensor<System.Numerics.Complex>)input, shape);
            default: return OpResult.NotSupported(OpType.Squeeze);
        }
    }

    public static OpResult Squeeze<T>(Tensor<T> input, Tensor<long>? axes = null) where T: struct
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

    public static OpResult Broadcast<T>(DenseTensor<T> inA, DenseTensor<T> inB) where T: struct, INumber<T>
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

