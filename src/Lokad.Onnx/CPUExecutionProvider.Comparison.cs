namespace Lokad.Onnx;

using System;
using System.Numerics;
using static OpResult;

public partial class CPUExecutionProvider
{
    /// <summary>Ordered comparisons with ONNX multidirectional broadcasting and NaN semantics.</summary>
    public static OpResult LessOrEqual(ITensor? a, ITensor? b, ExecutionOptions? options) =>
        OrderedComparison(OpType.LessOrEqual, a, b, options);

    public static OpResult Greater(ITensor? a, ITensor? b, ExecutionOptions? options) =>
        OrderedComparison(OpType.Greater, a, b, options);

    public static OpResult GreaterOrEqual(ITensor? a, ITensor? b, ExecutionOptions? options) =>
        OrderedComparison(OpType.GreaterOrEqual, a, b, options);

    static OpResult OrderedComparison(OpType op, ITensor? a, ITensor? b, ExecutionOptions? options)
    {
        if (a is null) return MissingInput(op, nameof(a));
        if (b is null) return MissingInput(op, nameof(b));
        var opts = (options ?? ExecutionOptions.Default).Validated();
        if (a.ElementType != b.ElementType)
            return WrongInputType(op, nameof(b), "Input tensors must be of the same type.", b);
        Profiler.StartOpStage(OpStage.Math);
        if (opts.Tensor.UseSimd && Vector.IsHardwareAccelerated && a is Tensor<float> single && b is Tensor<float> scalar)
        {
            var fast = RightScalarComparison(single, scalar, op);
            if (fast is not null) return Success(op, fast);
        }
        return a.ElementType switch
        {
            TensorElementType.Float => Success(op, OrderedCompare((Tensor<float>)a, (Tensor<float>)b, op)),
            TensorElementType.Double => Success(op, OrderedCompare((Tensor<double>)a, (Tensor<double>)b, op)),
            TensorElementType.Int8 => Success(op, OrderedCompare((Tensor<sbyte>)a, (Tensor<sbyte>)b, op)),
            TensorElementType.UInt8 => Success(op, OrderedCompare((Tensor<byte>)a, (Tensor<byte>)b, op)),
            TensorElementType.Int16 => Success(op, OrderedCompare((Tensor<short>)a, (Tensor<short>)b, op)),
            TensorElementType.UInt16 => Success(op, OrderedCompare((Tensor<ushort>)a, (Tensor<ushort>)b, op)),
            TensorElementType.Int32 => Success(op, OrderedCompare((Tensor<int>)a, (Tensor<int>)b, op)),
            TensorElementType.UInt32 => Success(op, OrderedCompare((Tensor<uint>)a, (Tensor<uint>)b, op)),
            TensorElementType.Int64 => Success(op, OrderedCompare((Tensor<long>)a, (Tensor<long>)b, op)),
            TensorElementType.UInt64 => Success(op, OrderedCompare((Tensor<ulong>)a, (Tensor<ulong>)b, op)),
            _ => InputTypeNotSupported(op, nameof(a), a),
        };
    }

    static DenseTensor<bool>? RightScalarComparison(Tensor<float> input, Tensor<float> scalar, OpType op)
    {
        if (input.Length < 256 || scalar.Length != 1 || scalar.Rank > input.Rank) return null;
        var data = input.ToRowMajor();
        var result = new DenseTensor<bool>(input.Dimensions.ToArray());
        var values = data.Buffer.Span;
        var target = result.Buffer.Span;
        float limit = scalar.GetLogicalValue(0);
        switch (op)
        {
            case OpType.Less: for (int i = 0; i < values.Length; i++) target[i] = values[i] < limit; break;
            case OpType.LessOrEqual: for (int i = 0; i < values.Length; i++) target[i] = values[i] <= limit; break;
            case OpType.Greater: for (int i = 0; i < values.Length; i++) target[i] = values[i] > limit; break;
            case OpType.GreaterOrEqual: for (int i = 0; i < values.Length; i++) target[i] = values[i] >= limit; break;
            default: throw new ArgumentOutOfRangeException(nameof(op));
        }
        return result;
    }

    static Tensor<bool> OrderedCompare<T>(Tensor<T> a, Tensor<T> b, OpType op) where T : unmanaged, INumber<T>
    {
        if (!Tensor<T>.Broadcast(a, b, out var left, out var right))
            throw new ArgumentException("Inputs are not broadcastable.");
        Func<T, T, bool> compare = op switch
        {
            OpType.Greater => static (x, y) => x > y,
            OpType.GreaterOrEqual => static (x, y) => x >= y,
            _ => static (x, y) => x <= y,
        };
        var result = DenseTensor<bool>.OfShape(left.Dimensions.ToArray());
        for (int i = 0; i < result.Length; i++)
            result.SetValue(i, compare(left.GetLogicalValue(i), right.GetLogicalValue(i)));
        return result;
    }
}
