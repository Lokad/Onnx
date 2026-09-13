namespace Lokad.Onnx;


using System;
using System.Collections.Generic;
using System.Linq;

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
    /// <summary>
    /// Instance normalization over the spatial dimensions per (N, C) slice:
    /// mean/variance pooling followed by the per-channel scale and bias.
    /// Float32 only, matching the ORT CPU implemented surface; every output
    /// element is assigned through a fixed scalar path.
    /// </summary>
    public static OpResult InstanceNorm(ITensor? X, ITensor? scale, ITensor? bias, float? epsilon, ExecutionOptions? options)
    {
        var op = OpType.InstanceNormalization;
        if (X is null) return MissingInput(op, nameof(X));
        if (scale is null) return MissingInput(op, nameof(scale));
        if (bias is null) return MissingInput(op, nameof(bias));
        var opts = (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Math);
        if (X.ElementType != TensorElementType.Float) return InputTypeNotSupported(op, nameof(X), X, "Only float32 instance normalization is supported.");
        if (scale.ElementType != TensorElementType.Float) return WrongInputType(op, nameof(scale), TensorElementType.Float, scale);
        if (bias.ElementType != TensorElementType.Float) return WrongInputType(op, nameof(bias), TensorElementType.Float, bias);
        if (X.Rank < 2) return WrongInputShape(op, nameof(X), X, "InstanceNormalization requires rank 2 or more [N, C, ...].");
        int channels = X.Dims[1];
        if (scale.Rank != 1 || scale.Dims[0] != channels)
            return WrongInputShape(op, nameof(scale), scale, "scale must be [C].");
        if (bias.Rank != 1 || bias.Dims[0] != channels)
            return WrongInputShape(op, nameof(bias), bias, "bias must be [C].");
        float eps = epsilon ?? 1e-5f;
        // Coordinate reads below assume standard row-major strides.
        var xd = Tensor<float>.RequireContiguous((Tensor<float>)X, nameof(X), opts.Tensor.CopyReporter);
        var sd = Tensor<float>.RequireContiguous((Tensor<float>)scale, nameof(scale), opts.Tensor.CopyReporter);
        var bd = Tensor<float>.RequireContiguous((Tensor<float>)bias, nameof(bias), opts.Tensor.CopyReporter);
        var xs = xd.Buffer.Span;
        var ss = sd.Buffer.Span;
        var bs = bd.Buffer.Span;
        var dims = xd.Dimensions.ToArray();
        var y = DenseTensor<float>.OfShape(dims);
        var ys = y.Buffer.Span;
        int spatial = 1;
        for (int d = 2; d < dims.Length; d++) spatial *= dims[d];
        int batch = dims[0];
        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < channels; c++)
            {
                int baseOff = (n * channels + c) * spatial;
                double sum = 0;
                for (int k = 0; k < spatial; k++) sum += xs[baseOff + k];
                double mean = spatial == 0 ? 0 : sum / spatial;
                double var = 0;
                for (int k = 0; k < spatial; k++)
                {
                    double d = xs[baseOff + k] - mean;
                    var += d * d;
                }
                var /= spatial == 0 ? 1 : spatial;
                float inv = (float)(1.0 / Math.Sqrt(var + eps));
                float sc = ss[c];
                float bc = bs[c];
                for (int k = 0; k < spatial; k++) ys[baseOff + k] = (xs[baseOff + k] - (float)mean) * inv * sc + bc;
            }
        }
        return Success(op, y);
    }

    public static OpResult LayerNormalization(ITensor? x, ITensor? scale, ITensor? bias, int? axis, float? epsilon, int? stashType, int outputCount, ExecutionOptions? options, TensorBufferPool? pool)
    {
        var op = OpType.LayerNormalization;
        if (x is null) return MissingInput(op, nameof(x));
        if (scale is null) return MissingInput(op, nameof(scale));
        int stash = stashType ?? 1;
        if (stash != 1) return AttributeNotSupported(op, "stash_type", stash.ToString(), "Only stash_type 1 (32-bit float stage-one compute) is supported.");
        if (outputCount < 1 || outputCount > 3) return Failure(op, "LayerNormalization declares " + outputCount + " outputs; 1 (Y) to 3 (Y, Mean, InvStdDev) are supported.");
        (options ?? ExecutionOptions.Default).Validated();
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
                var fb = (Tensor<float>?)bias;
                Tensor<float> y;
                if (pool is null) y = Tensor<float>.LayerNormalization(fx, (Tensor<float>)scale, fb, ax, eps);
                else
                {
                    var rented = new DenseTensor<float>(new Memory<float>(pool.Rent<float>((int)fx.Length)), fx.Dimensions.ToArray());
                    y = Tensor<float>.LayerNormalization(fx, (Tensor<float>)scale, fb, rented, ax, eps);
                }
                if (outputCount == 1) return Success(op, y);
                var stats = LayerNormStats(fx.ToDenseTensor(), ax, eps);
                if (outputCount == 2) return Success(op, y, stats.Mean);
                return Success(op, y, stats.Mean, stats.InvStdDev);
            }
            case TensorElementType.Double:
            {
                if (scale.ElementType != TensorElementType.Double) return WrongInputType(op, nameof(scale), TensorElementType.Double, scale);
                if (bias is not null && bias.ElementType != TensorElementType.Double) return WrongInputType(op, nameof(bias), TensorElementType.Double, bias);
                var dx = (Tensor<double>)x;
                var y = Tensor<double>.LayerNormalization(dx, (Tensor<double>)scale, (Tensor<double>?)bias, ax, eps);
                if (outputCount == 1) return Success(op, y);
                var stats = LayerNormStats(dx.ToDenseTensor(), ax, eps);
                if (outputCount == 2) return Success(op, y, stats.Mean);
                return Success(op, y, stats.Mean, stats.InvStdDev);
            }
            default: return InputTypeNotSupported(op, nameof(x), x);
        }
    }

    readonly struct LayerNormStatTensors
    {
        public readonly DenseTensor<float> Mean;
        public readonly DenseTensor<float> InvStdDev;
        public LayerNormStatTensors(DenseTensor<float> mean, DenseTensor<float> invStdDev)
        {
            Mean = mean;
            InvStdDev = invStdDev;
        }
    }

    static LayerNormStatTensors LayerNormStats(DenseTensor<float> xd, int axis, double epsilon)
    {
        int rank = xd.Rank;
        int a = axis < 0 ? axis + rank : axis;
        var dims = xd.Dimensions.ToArray();
        var statDims = new int[rank];
        int block = 1;
        for (int d = 0; d < rank; d++)
        {
            if (d < a) statDims[d] = dims[d];
            else { statDims[d] = 1; block *= dims[d]; }
        }
        int outer = (int)(xd.Length / block);
        var xs = xd.Buffer.Span;
        var mean = new DenseTensor<float>(statDims);
        var inv = new DenseTensor<float>(statDims);
        var ms = mean.Buffer.Span;
        var vs = inv.Buffer.Span;
        for (int o = 0; o < outer; o++)
        {
            double m = 0.0;
            for (int i = 0; i < block; i++) m += xs[o * block + i];
            m /= block;
            double v = 0.0;
            for (int i = 0; i < block; i++) { double dd = xs[o * block + i] - m; v += dd * dd; }
            v /= block;
            ms[o] = (float)m;
            vs[o] = (float)(1.0 / Math.Sqrt(v + epsilon));
        }
        return new LayerNormStatTensors(mean, inv);
    }

    static LayerNormStatTensors LayerNormStats(DenseTensor<double> xd, int axis, double epsilon)
    {
        int rank = xd.Rank;
        int a = axis < 0 ? axis + rank : axis;
        var dims = xd.Dimensions.ToArray();
        var statDims = new int[rank];
        int block = 1;
        for (int d = 0; d < rank; d++)
        {
            if (d < a) statDims[d] = dims[d];
            else { statDims[d] = 1; block *= dims[d]; }
        }
        int outer = (int)(xd.Length / block);
        var xs = xd.Buffer.Span;
        var mean = new DenseTensor<float>(statDims);
        var inv = new DenseTensor<float>(statDims);
        var ms = mean.Buffer.Span;
        var vs = inv.Buffer.Span;
        for (int o = 0; o < outer; o++)
        {
            double m = 0.0;
            for (int i = 0; i < block; i++) m += xs[o * block + i];
            m /= block;
            double v = 0.0;
            for (int i = 0; i < block; i++) { double dd = xs[o * block + i] - m; v += dd * dd; }
            v /= block;
            ms[o] = (float)m;
            vs[o] = (float)(1.0 / Math.Sqrt(v + epsilon));
        }
        return new LayerNormStatTensors(mean, inv);
    }
}
