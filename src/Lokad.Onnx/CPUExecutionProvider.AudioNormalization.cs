namespace Lokad.Onnx;

using System;
using System.Linq;
using System.Numerics;
using static OpResult;

public partial class CPUExecutionProvider
{
    /// <summary>Normalize each float32 [N, C, spatial...] slice independently.</summary>
    public static OpResult InstanceNorm(ITensor? input, ITensor? scale, ITensor? bias, float? epsilon, ExecutionOptions? options)
    {
        var op = OpType.InstanceNormalization;
        if (input is null) return MissingInput(op, nameof(input));
        if (scale is null) return MissingInput(op, nameof(scale));
        if (bias is null) return MissingInput(op, nameof(bias));
        var opts = (options ?? ExecutionOptions.Default).Validated();
        if (input.ElementType != TensorElementType.Float) return InputTypeNotSupported(op, nameof(input), input);
        if (scale.ElementType != TensorElementType.Float) return WrongInputType(op, nameof(scale), TensorElementType.Float, scale);
        if (bias.ElementType != TensorElementType.Float) return WrongInputType(op, nameof(bias), TensorElementType.Float, bias);
        if (input.Rank < 3) return WrongInputShape(op, nameof(input), input, "InstanceNormalization requires rank >= 3 [N, C, spatial...].");
        int channels = input.Dims[1];
        if (scale.Rank != 1 || scale.Dims[0] != channels) return WrongInputShape(op, nameof(scale), scale, "scale must have shape [C].");
        if (bias.Rank != 1 || bias.Dims[0] != channels) return WrongInputShape(op, nameof(bias), bias, "bias must have shape [C].");
        var x = Tensor<float>.RequireContiguous((Tensor<float>)input, nameof(input), opts.Tensor.CopyReporter);
        var s = Tensor<float>.RequireContiguous((Tensor<float>)scale, nameof(scale), opts.Tensor.CopyReporter);
        var b = Tensor<float>.RequireContiguous((Tensor<float>)bias, nameof(bias), opts.Tensor.CopyReporter);
        var output = new DenseTensor<float>(x.Dimensions.ToArray());
        if (x.Length == 0) return Success(op, output);
        int spatial = 1;
        for (int d = 2; d < x.Rank; d++) spatial = checked(spatial * x.Dimensions[d]);
        float eps = epsilon ?? 1e-5f;
        var xs = x.Buffer.Span;
        var ys = output.Buffer.Span;
        Profiler.StartOpStage(OpStage.Math);
        for (int start = 0, channel = 0; start < xs.Length; start += spatial, channel = (channel + 1) % channels)
        {
            float mean = 0;
            for (int k = 0; k < spatial; k++) mean += xs[start + k];
            mean /= spatial;
            float variance = 0;
            for (int k = 0; k < spatial; k++)
            {
                float delta = xs[start + k] - mean;
                variance += delta * delta;
            }
            // Match ORT CPU's float32 statistics and folded affine transform.
            // A centered double transform can differ materially at large offsets.
            float channelScale = (1f / MathF.Sqrt(variance / spatial + eps)) * s.Buffer.Span[channel];
            float channelShift = b.Buffer.Span[channel] - mean * channelScale;
            for (int k = 0; k < spatial; k++) ys[start + k] = xs[start + k] * channelScale + channelShift;
        }
        return Success(op, output);
    }

    public static OpResult LeakyRelu(ITensor? input, float? alpha, ExecutionOptions? options)
    {
        var op = OpType.LeakyRelu;
        if (input is null) return MissingInput(op, nameof(input));
        var opts = (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Math);
        return input.ElementType switch
        {
            TensorElementType.Float => Success(op, LeakyReluTyped((Tensor<float>)input, alpha ?? .01f, opts.Tensor.CopyReporter)),
            TensorElementType.Double => Success(op, LeakyReluTyped((Tensor<double>)input, (double)(alpha ?? .01f), opts.Tensor.CopyReporter)),
            _ => InputTypeNotSupported(op, nameof(input), input)
        };
    }

    static DenseTensor<T> LeakyReluTyped<T>(Tensor<T> input, T alpha, ICopyAccountant? copy) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var x = Tensor<T>.RequireContiguous(input, nameof(input), copy);
        var output = new DenseTensor<T>(x.Dimensions.ToArray());
        var xs = x.Buffer.Span;
        var ys = output.Buffer.Span;
        for (int i = 0; i < xs.Length; i++) ys[i] = xs[i] < T.Zero ? alpha * xs[i] : xs[i];
        return output;
    }

    public static OpResult LogSoftmax(ITensor? input, int? axis, ExecutionOptions? options, TensorBufferPool? pool, int opsetVersion)
    {
        var op = OpType.LogSoftmax;
        if (input is null) return MissingInput(op, nameof(input));
        var opts = (options ?? ExecutionOptions.Default).Validated();
        int a = axis ?? (opsetVersion < 13 ? 1 : -1);
        if (a < 0) a += input.Rank;
        if (a < 0 || a >= input.Rank) return Failure(op, "axis must identify an input dimension.");
        Profiler.StartOpStage(OpStage.Math);
        return input.ElementType switch
        {
            TensorElementType.Float => Success(op, LogSoftmaxTyped((Tensor<float>)input, a, opsetVersion, opts.Tensor.CopyReporter, pool)),
            TensorElementType.Double => Success(op, LogSoftmaxTyped((Tensor<double>)input, a, opsetVersion, opts.Tensor.CopyReporter, pool)),
            _ => InputTypeNotSupported(op, nameof(input), input)
        };
    }

    static DenseTensor<T> LogSoftmaxTyped<T>(Tensor<T> input, int axis, int version, ICopyAccountant? copy, TensorBufferPool? pool)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var x = Tensor<T>.RequireContiguous(input, nameof(input), copy);
        var output = pool is null ? new DenseTensor<T>(x.Dimensions.ToArray())
            : new DenseTensor<T>(new Memory<T>(pool.Rent<T>(checked((int)x.Length))), x.Dimensions.ToArray());
        if (x.Length == 0) return output;
        int width = x.Dimensions[axis], inner = 1;
        // Before opset 13, axis starts a flattened suffix. Later versions
        // reduce that dimension alone, preserving every other coordinate.
        for (int d = axis + 1; d < x.Rank; d++)
        {
            if (version < 13) width = checked(width * x.Dimensions[d]);
            else inner = checked(inner * x.Dimensions[d]);
        }
        var xs = x.Buffer.Span;
        var ys = output.Buffer.Span;
        int block = checked(width * inner);
        for (int start = 0; start < xs.Length; start += block)
        for (int offset = 0; offset < inner; offset++)
        {
            int first = start + offset;
            T max = T.NegativeInfinity;
            for (int j = 0; j < width; j++) max = T.Max(max, xs[first + j * inner]);
            T sum = T.Zero;
            for (int j = 0; j < width; j++) sum += T.Exp(xs[first + j * inner] - max);
            // ORT's double CPU path uses fmax(sum, 1e-20f), which selects the
            // finite operand for NaN. The float MLAS path propagates NaN.
            if (typeof(T) == typeof(double)) sum = T.MaxNumber(sum, T.CreateChecked(1e-20f));
            T logSum = T.Log(sum);
            for (int j = 0; j < width; j++)
            {
                int index = first + j * inner;
                // Taking log(softmax(x)) instead would underflow large logits.
                ys[index] = xs[index] - max - logSum;
            }
        }
        return output;
    }
}
