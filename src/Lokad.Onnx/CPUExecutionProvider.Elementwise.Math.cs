namespace Lokad.Onnx;

using System;
using System.Numerics;
using static OpResult;

public partial class CPUExecutionProvider
{
    public static OpResult Exp(ITensor? input, ExecutionOptions? options)
    {
        if (input is DenseTensor<float> dense && input.GetType() == typeof(DenseTensor<float>)
            && !dense.IsReversedStride && dense.Buffer.Length == dense.Length
            && dense.strides.SequenceEqual(ArrayUtilities.GetStrides(dense.dimensions)))
        {
            var tensor = (options ?? ExecutionOptions.Default).Validated().Tensor;
            if (tensor.UseSimd && tensor.UseIntrinsics && Vector.IsHardwareAccelerated
                && tensor.MaxDegreeOfParallelism == 1)
            {
                Profiler.StartOpStage(OpStage.Math);
                return Success(OpType.Exp, dense.VectorizedApply(Vector.Exp, MathF.Exp, tensor));
            }
        }
        return FloatingMap(OpType.Exp, input, MathF.Exp, Math.Exp, options);
    }

    public static OpResult Round(ITensor? input, ExecutionOptions? options) =>
        FloatingMap(OpType.Round, input, MathF.Round, Math.Round, options);

    static OpResult FloatingMap(OpType op, ITensor? input, Func<float, float> single,
        Func<double, double> twice, ExecutionOptions? options)
    {
        if (input is null) return MissingInput(op, nameof(input));
        (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Math);
        return input switch
        {
            Tensor<float> values => Success(op, MapValues(values, single)),
            Tensor<double> values => Success(op, MapValues(values, twice)),
            _ => InputTypeNotSupported(op, nameof(input), input),
        };
    }

    static DenseTensor<TResult> MapValues<T, TResult>(Tensor<T> input, Func<T, TResult> map)
        where T : unmanaged where TResult : unmanaged
    {
        var result = new DenseTensor<TResult>(input.Dimensions.ToArray());
        for (int i = 0; i < result.Length; i++) result.SetValue(i, map(input.GetLogicalValue(i)));
        return result;
    }

    public static OpResult IsNaN(ITensor? input, ExecutionOptions? options)
    {
        const OpType op = OpType.IsNaN;
        if (input is null) return MissingInput(op, nameof(input));
        (options ?? ExecutionOptions.Default).Validated();
        return input switch
        {
            Tensor<float> values => Success(op, MapValues(values, float.IsNaN)),
            Tensor<double> values => Success(op, MapValues(values, double.IsNaN)),
            Tensor<Half> values => Success(op, MapValues(values, Half.IsNaN)),
            _ => InputTypeNotSupported(op, nameof(input), input),
        };
    }

    public static OpResult Min(ITensor[] inputs, ExecutionOptions? options)
    {
        const OpType op = OpType.Min;
        if (inputs is null || inputs.Length == 0) return MissingInput(op, nameof(inputs));
        if (inputs[0] is null) return MissingInput(op, nameof(inputs));
        foreach (var input in inputs)
        {
            if (input is null) return MissingInput(op, nameof(inputs));
            if (input.ElementType != inputs[0].ElementType)
                return WrongInputType(op, nameof(inputs), inputs[0].ElementType, input);
        }
        var opts = (options ?? ExecutionOptions.Default).Validated();
        return inputs[0].ElementType switch
        {
            TensorElementType.Float => Success(op, MinValues<float>(inputs, opts)),
            TensorElementType.Double => Success(op, MinValues<double>(inputs, opts)),
            TensorElementType.Int32 => Success(op, MinValues<int>(inputs, opts)),
            TensorElementType.Int64 => Success(op, MinValues<long>(inputs, opts)),
            _ => InputTypeNotSupported(op, nameof(inputs), inputs[0]),
        };
    }

    static Tensor<T> MinValues<T>(ITensor[] inputs, ExecutionOptions options) where T : unmanaged, INumber<T>
    {
        Tensor<T> result = new DenseTensor<T>(((Tensor<T>)inputs[0]).ToArray(), inputs[0].Dims);
        for (int i = 1; i < inputs.Length; i++)
            result = result.BroadcastApply((Tensor<T>)inputs[i],
                static (a, b) => T.IsNaN(a) ? a : T.IsNaN(b) ? b : a < b ? a : b, options.Tensor);
        return result;
    }
}
