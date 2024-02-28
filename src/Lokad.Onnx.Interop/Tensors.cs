using System;
using System.Reflection.Metadata.Ecma335;

namespace Lokad.Onnx.Interop;

public class Tensors
{
    public static ITensor MakeTensor<T>(int[] dims) where T : struct => new DenseTensor<T>(dims);

    public static ITensor Ones<T>(int[] dims) where T : struct => Tensor<T>.Ones(dims);

    public static ITensor ARange(int start, int end, int step = 1) => Tensor<int>.Arange(start, end, step);

    public static ITensor Add(ITensor x, object _y)
    {
        switch (_y)
        {
            case ITensor y:
                ITensor.ThrowIfDifferentElementTypes(x, y);
                switch (x.ElementType)
                {
                    case TensorElementType.Int32: return Tensor<int>.Add((Tensor<int>)x, (Tensor<int>)y);
                    case TensorElementType.Float: return Tensor<float>.Add((Tensor<float>)x, (Tensor<float>)y);
                    case TensorElementType.Double: return Tensor<double>.Add((Tensor<double>)x, (Tensor<double>)y);
                    default: throw new NotSupportedException();
                }
            default:
                switch (x.ElementType)
                {
                    case TensorElementType.Int32: return Tensor<int>.Add((Tensor<int>)x, Convert.ToInt32(_y));
                    case TensorElementType.Float: return Tensor<float>.Add((Tensor<float>)x, Convert.ToSingle(_y));
                    case TensorElementType.Double: return Tensor<double>.Add((Tensor<double>)x, Convert.ToDouble(_y));
                    default: throw new NotSupportedException();
                }
        }
    }

    public static ITensor MatMul(ITensor x, ITensor y)
    {
        ITensor.ThrowIfDifferentElementTypes(x, y);
        switch (x.ElementType)
        {
            case TensorElementType.Int32: return Tensor<int>.MatMul((Tensor<int>)x, (Tensor<int>)y);
            case TensorElementType.Float: return Tensor<float>.MatMul((Tensor<float>)x, (Tensor<float>)y);
            case TensorElementType.Double: return Tensor<double>.MatMul((Tensor<double>)x, (Tensor<double>)y);
            default: throw new NotSupportedException();
        }
    }
    
}

