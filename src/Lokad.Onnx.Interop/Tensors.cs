﻿using System;
using System.Reflection.Metadata.Ecma335;

namespace Lokad.Onnx.Interop;

public class Tensors
{
    public static ITensor MakeTensor<T>(Array data) where T : unmanaged => DenseTensor<T>.OfValues(data);
    
    public static ITensor MakeTensor<T>(T[] data, int[] dims) where T : unmanaged => new DenseTensor<T>(data, dims);

    public static ITensor MakeEmptyTensor<T>(int[] dims) where T : unmanaged => new DenseTensor<T>(dims);

    public static ITensor MakeScalar<T>(T val) where T : unmanaged => DenseTensor<T>.Scalar(val);

    public static ITensor Ones<T>(int[] dims) where T : unmanaged => Tensor<T>.Ones(dims);

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

