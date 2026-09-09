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
