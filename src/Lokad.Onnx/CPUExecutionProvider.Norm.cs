namespace Lokad.Onnx;


using System;

using static OpResult;
public partial class CPUExecutionProvider
{

    public static OpResult RotaryEmbedding(ITensor? x, ITensor? cos, ITensor? sin, int? half, int? axis, int? concatAxis, ExecutionOptions? options, TensorBufferPool? pool)
    {
        var op = OpType.RotaryEmbedding;
        if (x is null) return MissingInput(op, nameof(x));
        if (cos is null) return MissingInput(op, nameof(cos));
        if (sin is null) return MissingInput(op, nameof(sin));
        if (half is null) return MissingInput(op, nameof(half));
        (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Math);
        if (x.ElementType != TensorElementType.Float) return InputTypeNotSupported(op, nameof(x), x);
        if (cos.ElementType != TensorElementType.Float) return InputTypeNotSupported(op, nameof(cos), cos);
        if (sin.ElementType != TensorElementType.Float) return InputTypeNotSupported(op, nameof(sin), sin);
        var fx = (Tensor<float>)x;
        var rented = pool is null ? null : new DenseTensor<float>(new Memory<float>(pool.Rent<float>((int)fx.Length)), fx.Dimensions.ToArray());
        if (rented is null) return Success(op, Tensor<float>.RotaryEmbedding(fx, (Tensor<float>)cos, (Tensor<float>)sin, half.Value, axis ?? -1, concatAxis ?? -1));
        return Success(op, Tensor<float>.RotaryEmbedding(fx, (Tensor<float>)cos, (Tensor<float>)sin, rented, half.Value, axis ?? -1, concatAxis ?? -1));
    }
    public static OpResult LayerNormalization(ITensor? x, ITensor? scale, ITensor? bias, int? axis, float? epsilon, int? stashType, int outputCount, ExecutionOptions? options, TensorBufferPool? pool)
    {
        var op = OpType.LayerNormalization;
        if (x is null) return MissingInput(op, nameof(x));
        if (scale is null) return MissingInput(op, nameof(scale));
        int stash = stashType ?? 1;
        if (stash != 1) return AttributeNotSupported(op, "stash_type", stash.ToString(), "Only stash_type 1 (32-bit float stage-one compute) is supported.");
        if (outputCount < 1 || outputCount > 3) return Failure(op, "LayerNormalization declares " + outputCount + " outputs; 1 (Y) to 3 (Y, Mean, InvStdDev) are supported.");
        var opts = (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Math);
        int ax = axis ?? -1;
        float eps = epsilon ?? 1e-5f;
        switch (x.ElementType)
        {
            case TensorElementType.Float:
            {
                if (scale.ElementType != TensorElementType.Float) return WrongInputType(op, nameof(scale), TensorElementType.Float, scale);
                if (bias is not null && bias.ElementType != TensorElementType.Float) return WrongInputType(op, nameof(bias), TensorElementType.Float, bias);
                var fx = (Tensor<float>)x;
                var destination = pool is null ? null : new DenseTensor<float>(new Memory<float>(pool.Rent<float>((int)fx.Length)), fx.Dimensions);
                var result = Tensor<float>.LayerNormalizationCore(fx, (Tensor<float>)scale, (Tensor<float>?)bias, destination, ax, eps, outputCount, opts.Tensor);
                return outputCount switch
                {
                    1 => Success(op, result.Output),
                    2 => Success(op, result.Output, result.Mean!),
                    _ => Success(op, result.Output, result.Mean!, result.InvStdDev!)
                };
            }
            case TensorElementType.Double:
            {
                if (scale.ElementType != TensorElementType.Double) return WrongInputType(op, nameof(scale), TensorElementType.Double, scale);
                if (bias is not null && bias.ElementType != TensorElementType.Double) return WrongInputType(op, nameof(bias), TensorElementType.Double, bias);
                var dx = (Tensor<double>)x;
                var result = Tensor<double>.LayerNormalizationCore(dx, (Tensor<double>)scale, (Tensor<double>?)bias, null, ax, eps, outputCount);
                return outputCount switch
                {
                    1 => Success(op, result.Output),
                    2 => Success(op, result.Output, result.Mean!),
                    _ => Success(op, result.Output, result.Mean!, result.InvStdDev!)
                };
            }
            default: return InputTypeNotSupported(op, nameof(x), x);
        }
    }
}
