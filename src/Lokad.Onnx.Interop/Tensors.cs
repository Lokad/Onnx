using System;
using System.Reflection.Metadata.Ecma335;

namespace Lokad.Onnx.Interop;

public class Tensors
{
    public static ITensor MakeTensor<T>(int[] dims) where T : struct => new DenseTensor<T>(dims);

    public static ITensor ARange(int start, int end, int step = 1) => Tensor<int>.Arange(start, end, step);

    public static ITensor Add(ITensor x, ITensor y)
    {
        ITensor.ThrowIfDifferentElementTypes(x, y);
        switch (x.ElementType)
        {
            case TensorElementType.Int8: return Tensor<byte>.Add((Tensor<byte>) x, (Tensor<byte>) y);
            case TensorElementType.UInt8: return Tensor<sbyte>.Add((Tensor<sbyte>)x, (Tensor<sbyte>)y);
            case TensorElementType.Int16: return Tensor<short>.Add((Tensor<short>)x, (Tensor<short>)y);
            case TensorElementType.UInt16: return Tensor<ushort>.Add((Tensor<ushort>)x, (Tensor<ushort>)y);
            case TensorElementType.Int32: return Tensor<int>.Add((Tensor<int>)x, (Tensor<int>)y);
            case TensorElementType.UInt32: return Tensor<uint>.Add((Tensor<uint>)x, (Tensor<uint>)y);
            case TensorElementType.Int64: return Tensor<long>.Add((Tensor<long>)x, (Tensor<long>)y);
            case TensorElementType.UInt64: return Tensor<ulong>.Add((Tensor<ulong>)x, (Tensor<ulong>)y);
            case TensorElementType.Float: return Tensor<float>.Add((Tensor<float>)x, (Tensor<float>)y);
            case TensorElementType.Double: return Tensor<double>.Add((Tensor<double>)x, (Tensor<double>)y);
            default: throw new NotSupportedException();
        }
    }
}

