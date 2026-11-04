namespace Lokad.Onnx;

using System;
using System.Numerics;
using static OpResult;

public partial class CPUExecutionProvider
{
    internal static OpResult RandomNormalLike(ITensor? input, int? dtype, float? mean, float? scale,
        float? seed, NormalRandomState state, ExecutionOptions? options)
    {
        const OpType op = OpType.RandomNormalLike;
        if (input is null) return MissingInput(op, nameof(input));
        (options ?? ExecutionOptions.Default).Validated();
        var type = dtype.HasValue ? (TensorElementType)dtype.Value : input.ElementType;
        if (type is not TensorElementType.Float and not TensorElementType.Double)
            return AttributeNotSupported(op, nameof(dtype), ((int)type).ToString(), "Only float32 and float64 output is supported.");
        if (!float.IsFinite(mean ?? 0f)) return AttributeNotSupported(op, nameof(mean), mean.ToString()!, "Mean must be finite.");
        if (!float.IsFinite(scale ?? 1f) || scale < 0f) return AttributeNotSupported(op, nameof(scale), scale.ToString()!, "Scale must be finite and nonnegative.");
        if (seed.HasValue && !float.IsFinite(seed.Value)) return AttributeNotSupported(op, nameof(seed), seed.ToString()!, "Seed must be finite.");
        int? bits = seed.HasValue ? (seed.Value == 0f ? 0 : BitConverter.SingleToInt32Bits(seed.Value)) : null;
        Profiler.StartOpStage(OpStage.Math);
        return type == TensorElementType.Float
            ? Success(op, NormalValues<float>(input.Dims, mean ?? 0f, scale ?? 1f, bits, state))
            : Success(op, NormalValues<double>(input.Dims, mean ?? 0f, scale ?? 1f, bits, state));
    }

    static DenseTensor<T> NormalValues<T>(int[] shape, double mean, double scale, int? seed,
        NormalRandomState state) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var result = new DenseTensor<T>(shape);
        lock (state)
        {
            state.SetSeed(seed);
            var values = result.Buffer.Span;
            for (int i = 0; i < values.Length; i++) values[i] = T.CreateTruncating(mean + scale * state.Next());
        }
        return result;
    }
}
