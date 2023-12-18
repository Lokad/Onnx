namespace Lokad.Onnx.Backend;

using System;
using System.Collections.Generic;
using System.Linq;


public enum ExecutionProvider
{
    CPU
}
    
public class CPUExecutionProvider
{
    public static OpResult Squeeze(ITensor input, ITensor? axes = null)
    {
        if (axes is not null)
        {
            if (axes.Dimensions.Length != 1)
            {
                return OpResult.Failure(OpType.Squeeze, $"The axes tensor  {axes.Name} must have dimension 1.");
            }
            else if (axes.ElementType != TensorElementType.Int32)
            {
                return OpResult.WrongInputParameterType(OpType.Squeeze, TensorElementType.Int64, axes);
            }
        }

        switch (input.ElementType)
        {
            case TensorElementType.Bool: return Squeeze((Tensor<bool>)input, axes);
            case TensorElementType.Int8: return Squeeze((Tensor<byte>)input, axes);
            case TensorElementType.UInt8: return Squeeze((Tensor<sbyte>)input, axes);
            case TensorElementType.Int16: return Squeeze((Tensor<short>)input, axes);
            case TensorElementType.UInt16: return Squeeze((Tensor<ushort>)input, axes);
            case TensorElementType.Int32: return Squeeze((Tensor<int>)input, axes);
            case TensorElementType.UInt32: return Squeeze((Tensor<uint>)input, axes);
            case TensorElementType.Int64: return Squeeze((Tensor<long>)input, axes);
            case TensorElementType.UInt64: return Squeeze((Tensor<ulong>)input, axes);
            case TensorElementType.Float: return Squeeze((Tensor<float>)input, axes);
            case TensorElementType.Double: return Squeeze((Tensor<double>)input, axes);
            case TensorElementType.Float16: return Squeeze((Tensor<Float16>)input, axes);
            case TensorElementType.BFloat16: return Squeeze((Tensor<BFloat16>)input, axes);
            case TensorElementType.Complex64: return Squeeze((Tensor<System.Numerics.Complex>)input, axes);
            default: return OpResult.NotSupported(OpType.Squeeze);
        }
    }

    public static OpResult Squeeze<T>(Tensor<T> input, Tensor<long>? axes=null)
    {
        long[] dims = (axes is not null) ? axes.ToArray() : Enumerable.Range(0, input.Dimensions.Length - 1).Cast<long>().ToArray();
        List<int> squeezedDims = new List<int>();
        for(int i = 0; i < dims.Length; i++) 
        {
            var a = TensorUtil.HandleNegativeAxis((int) dims[i], input.Dimensions.Length);
            if (input.Dimensions[a] == 1)
            {
                squeezedDims.Add(a);
            }

        }
        return OpResult.Success(OpType.Squeeze, input.Reshape_(squeezedDims.ToArray()));
    }
}

