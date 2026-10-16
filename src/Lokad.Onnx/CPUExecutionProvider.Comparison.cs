namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;
using static OpResult;

public partial class CPUExecutionProvider
{
    /// <summary>Ordered elementwise comparison with ONNX multidirectional broadcasting.</summary>
    public static OpResult LessOrEqual(ITensor? a, ITensor? b, ExecutionOptions? options)
    {
        const OpType op = OpType.LessOrEqual;
        if (a is null) return MissingInput(op, nameof(a));
        if (b is null) return MissingInput(op, nameof(b));
        (options ?? ExecutionOptions.Default).Validated();
        if (a.ElementType != b.ElementType)
            return WrongInputType(op, nameof(b), "Input tensors must be of the same type.", b);
        Profiler.StartOpStage(OpStage.Math);
        return a.ElementType switch
        {
            TensorElementType.Float => Success(op, OrderedLessOrEqual((Tensor<float>)a, (Tensor<float>)b, static (x, y) => x <= y)),
            TensorElementType.Double => Success(op, OrderedLessOrEqual((Tensor<double>)a, (Tensor<double>)b, static (x, y) => x <= y)),
            TensorElementType.Int8 => Success(op, IntegerLessOrEqual((Tensor<sbyte>)a, (Tensor<sbyte>)b)),
            TensorElementType.UInt8 => Success(op, IntegerLessOrEqual((Tensor<byte>)a, (Tensor<byte>)b)),
            TensorElementType.Int16 => Success(op, IntegerLessOrEqual((Tensor<short>)a, (Tensor<short>)b)),
            TensorElementType.UInt16 => Success(op, IntegerLessOrEqual((Tensor<ushort>)a, (Tensor<ushort>)b)),
            TensorElementType.Int32 => Success(op, IntegerLessOrEqual((Tensor<int>)a, (Tensor<int>)b)),
            TensorElementType.UInt32 => Success(op, IntegerLessOrEqual((Tensor<uint>)a, (Tensor<uint>)b)),
            TensorElementType.Int64 => Success(op, IntegerLessOrEqual((Tensor<long>)a, (Tensor<long>)b)),
            TensorElementType.UInt64 => Success(op, IntegerLessOrEqual((Tensor<ulong>)a, (Tensor<ulong>)b)),
            _ => InputTypeNotSupported(op, nameof(a), a),
        };
    }

    static Tensor<bool> IntegerLessOrEqual<T>(Tensor<T> a, Tensor<T> b) where T : unmanaged =>
        OrderedLessOrEqual(a, b, static (x, y) => Comparer<T>.Default.Compare(x, y) <= 0);

    static Tensor<bool> OrderedLessOrEqual<T>(Tensor<T> a, Tensor<T> b, Func<T, T, bool> compare) where T : unmanaged
    {
        if (!Tensor<T>.Broadcast(a, b, out var left, out var right))
            throw new ArgumentException("Inputs are not broadcastable.");
        var result = DenseTensor<bool>.OfShape(left.Dimensions.ToArray());
        for (int i = 0; i < result.Length; i++)
            result.SetValue(i, compare(left.GetValue(i), right.GetValue(i)));
        return result;
    }
}
