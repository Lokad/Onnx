namespace Lokad.Onnx;


using System;
using System.Collections.Generic;
using System.Linq;

using static OpResult;
public partial class CPUExecutionProvider
{

    /// <summary>2D convolution, lowered through the shared matrix dispatcher.</summary>
    /// <summary>2D convolution, lowered through the shared matrix dispatcher.</summary>
    public static OpResult Conv(ITensor? X, ITensor? W, ITensor? B, string? auto_pad, int[]? dilations, int? group, int[]? kernel_shape, int[]? pads, int[]? strides, ExecutionOptions? options) =>
        ConvCore(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides, options, false);

    /// <summary>2D convolution with a fused ReLU epilogue for graph-fused Conv+Relu pairs.</summary>
    public static OpResult Conv(ITensor? X, ITensor? W, ITensor? B, string? auto_pad, int[]? dilations, int? group, int[]? kernel_shape, int[]? pads, int[]? strides, ExecutionOptions? options, bool fuseRelu) =>
        ConvCore(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides, options, fuseRelu);

    private static OpResult ConvCore(ITensor? X, ITensor? W, ITensor? B, string? auto_pad, int[]? dilations, int? group, int[]? kernel_shape, int[]? pads, int[]? strides, ExecutionOptions? options, bool fuseRelu)
    {
        var op = OpType.Conv;
        if (X is null) return MissingInput(op, nameof(X));
        if (W is null) return MissingInput(op, nameof(W));
        if (W.ElementType != X.ElementType) return WrongInputType(op, nameof(W), X.ElementType, W, "The weights tensor must be the same type as the input tensor.");
        if (B is not null && B.ElementType != X.ElementType) return WrongInputType(op, nameof(B), X.ElementType, B, "The bias tensor must be the same type as the input tensor.");
        if (X.Rank == 3)
        {
            // One-dimensional convolution rides the two-dimensional
            // machinery: normalize [N,C,L] inputs to [N,C,L,1] (and likewise
            // weights plus every spatial attribute), run the proven 2D path,
            // then drop the trailing size-1 dimension. Unsqueeze aliases
            // storage, so only the final squeeze may densify.
            if (W.Rank != 3)
            {
                return WrongInputShape(op, nameof(W), 3, W);
            }
            if (kernel_shape is not null && kernel_shape.Length != 1)
            {
                return AttributeNotSupported(op, "kernel_shape", kernel_shape.Print(), "One-dimensional kernel_shape must hold one value.");
            }
            if (strides is not null && strides.Length != 1)
            {
                return AttributeNotSupported(op, "strides", strides.Print(), "One-dimensional strides must hold one value.");
            }
            if (dilations is not null && dilations.Length != 1)
            {
                return AttributeNotSupported(op, "dilations", dilations.Print(), "One-dimensional dilations must hold one value.");
            }
            if (pads is not null && pads.Length != 2)
            {
                return AttributeNotSupported(op, "pads", pads.Print(), "One-dimensional pads must hold two values [begin, end].");
            }
            if (X.ElementType == TensorElementType.Float
                && (string.IsNullOrEmpty(auto_pad) || auto_pad == "NOTSET")
                && Tensor<float>.TryConvDepthwise1D((Tensor<float>)X, (Tensor<float>)W, (Tensor<float>?)B, group ?? 1, pads, kernel_shape, strides, dilations, (options ?? ExecutionOptions.Default).Validated().Tensor, fuseRelu, out var dwConv)
                && dwConv is not null)
            {
                return Success(op, dwConv);
            }
            var xu = Unsqueeze(X, new[] { 3 }, options);
            if (xu.Status != OpStatus.Success) return xu;
            var wu = Unsqueeze(W, new[] { 3 }, options);
            if (wu.Status != OpStatus.Success) return wu;
            int[]? ks4 = kernel_shape is null ? null : new[] { kernel_shape[0], 1 };
            int[]? st4 = strides is null ? null : new[] { strides[0], 1 };
            int[]? di4 = dilations is null ? null : new[] { dilations[0], 1 };
            int[]? pa4 = pads is null ? null : new[] { pads[0], 0, pads[1], 0 };
            var c = ConvCore(xu.Outputs[0], wu.Outputs[0], B, auto_pad, di4, group, ks4, pa4, st4, options, fuseRelu);
            if (c.Status != OpStatus.Success) return c;
            var s = Squeeze(c.Outputs[0], new DenseTensor<long>(new long[] { 3 }, new[] { 1 }), options);
            s.Op = op;
            return s;
        }
        if (X.Rank != 4 && X.Rank != 3)
        {
            return WrongInputShape(op, nameof(X), 4, X);
        }
        if (W.Rank != 4 && W.Rank != 3)
        {
            return WrongInputShape(op, nameof(W), 4, W);
        }
        MathOps.PadType? padmode;
        if (string.IsNullOrEmpty(auto_pad) || auto_pad == "NOTSET") padmode = null;
        else if (auto_pad == "VALID") padmode = MathOps.PadType.Valid;
        else if (auto_pad == "SAME_UPPER") padmode = MathOps.PadType.SameUpper;
        else if (auto_pad == "SAME_LOWER") padmode = MathOps.PadType.SameLower;
        else return AttributeNotSupported(op, "auto_pad", auto_pad, "auto_pad must be NOTSET, VALID, SAME_UPPER, or SAME_LOWER.");
        // ORT 1.29 load-fails a Conv carrying both pads and auto_pad, even
        // when every pad is zero; refuse the combination descriptively.
        if (!string.IsNullOrEmpty(auto_pad) && auto_pad != "NOTSET" && pads is not null)
        {
            return AttributeNotSupported(op, "pads", pads.Print(), "pads must not be specified together with an automatic padding mode.");
        }
        if (pads is not null && pads.Length != 4)
        {
            return AttributeNotSupported(op, "pads", pads.Print(), "Explicit pads must have four values [begin_h, begin_w, end_h, end_w].");
        }
        var opts = (options ?? ExecutionOptions.Default).Validated();
        switch (X.ElementType)
        {
            case TensorElementType.Float:
            {
                var bias = B is null ? null : (Tensor<float>)B;
                if (padmode is null)
                {
                    return Success(op, Tensor<float>.Conv2D((Tensor<float>)X, (Tensor<float>)W, group ?? 1, pads ?? new int[] { 0, 0, 0, 0 }, bias, kernel_shape, strides, dilations, opts.Tensor, fuseRelu));
                }
                return Success(op, Tensor<float>.Conv2D((Tensor<float>)X, (Tensor<float>)W, group ?? 1, padmode.Value, null, bias, kernel_shape, strides, dilations, opts.Tensor, fuseRelu));
            }
            case TensorElementType.Double:
            {
                var biasd = B is null ? null : (Tensor<double>)B;
                if (padmode is null)
                {
                    return Success(op, Tensor<double>.Conv2D((Tensor<double>)X, (Tensor<double>)W, group ?? 1, pads ?? new int[] { 0, 0, 0, 0 }, biasd, kernel_shape, strides, dilations, opts.Tensor, fuseRelu));
                }
                return Success(op, Tensor<double>.Conv2D((Tensor<double>)X, (Tensor<double>)W, group ?? 1, padmode.Value, null, biasd, kernel_shape, strides, dilations, opts.Tensor, fuseRelu));
            }
            default:
                return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult MaxPool(ITensor? X, string? auto_pad, int? ceil_mode, int[]? dilations, int[]? kernel_shape, int[]? pads, int? storage_order, int[]? strides, ExecutionOptions? options)
    {
        var op = OpType.MaxPool;
        if (X is null) return MissingInput(op, nameof(X));
        (options ?? ExecutionOptions.Default).Validated();
        if (X.Rank == 3)
        {
            // One-dimensional pooling rides the two-dimensional machinery
            // exactly like one-dimensional convolution above.
            if (kernel_shape is not null && kernel_shape.Length != 1)
            {
                return AttributeNotSupported(op, "kernel_shape", kernel_shape.Print(), "One-dimensional kernel_shape must hold one value.");
            }
            if (strides is not null && strides.Length != 1)
            {
                return AttributeNotSupported(op, "strides", strides.Print(), "One-dimensional strides must hold one value.");
            }
            if (dilations is not null && dilations.Length != 1)
            {
                return AttributeNotSupported(op, "dilations", dilations.Print(), "One-dimensional dilations must hold one value.");
            }
            if (pads is not null && pads.Length != 2)
            {
                return AttributeNotSupported(op, "pads", pads.Print(), "One-dimensional pads must hold two values [begin, end].");
            }
            var xu = Unsqueeze(X, new[] { 3 }, options);
            if (xu.Status != OpStatus.Success) return xu;
            int[]? ks4 = kernel_shape is null ? null : new[] { kernel_shape[0], 1 };
            int[]? st4 = strides is null ? null : new[] { strides[0], 1 };
            int[]? di4 = dilations is null ? null : new[] { dilations[0], 1 };
            int[]? pa4 = pads is null ? null : new[] { pads[0], 0, pads[1], 0 };
            var m = MaxPool(xu.Outputs[0], auto_pad, ceil_mode, di4, ks4, pa4, storage_order, st4, options);
            if (m.Status != OpStatus.Success) return m;
            var s = Squeeze(m.Outputs[0], new DenseTensor<long>(new long[] { 3 }, new[] { 1 }), options);
            s.Op = op;
            return s;
        }
        if (X.Rank != 4 && X.Rank != 3)
        {
            return WrongInputShape(op, nameof(X), 4, X);
        }
        MathOps.PadType? padmode;
        if (string.IsNullOrEmpty(auto_pad) || auto_pad == "NOTSET") padmode = null;
        else if (auto_pad == "VALID") padmode = MathOps.PadType.Valid;
        else if (auto_pad == "SAME_UPPER") padmode = MathOps.PadType.SameUpper;
        else if (auto_pad == "SAME_LOWER") padmode = MathOps.PadType.SameLower;
        else return AttributeNotSupported(op, "auto_pad", auto_pad, "auto_pad must be NOTSET, VALID, SAME_UPPER, or SAME_LOWER.");
        // ORT 1.29 ignores explicit pads when auto_pad is set on MaxPool
        // (probed values match auto_pad alone), so only validate their shape.
        if (pads is not null && pads.Length != 4)
        {
            return AttributeNotSupported(op, "pads", pads.Print(), "Explicit pads must have four values [begin_h, begin_w, end_h, end_w].");
        }
        if (kernel_shape is null) return MissingAttribute(op, nameof(kernel_shape), null);
        int ceil = ceil_mode ?? 0;
        if (ceil != 0 && ceil != 1) return AttributeNotSupported(op, "ceil_mode", ceil.ToString(), "ceil_mode must be 0 or 1.");
        int order = storage_order ?? 0;
        if (order != 0) return AttributeNotSupported(op, "storage_order", order.ToString(), "Only row-major storage order is supported because the optional Indices output is not supported.");
        switch (X.ElementType)
        {
            case TensorElementType.Float:
            {
                if (padmode is null)
                {
                    return Success(op, Tensor<float>.MaxPool2D((Tensor<float>) X, kernel_shape, pads ?? new int[] { 0, 0, 0, 0 }, strides, dilations, ceil == 1));
                }
                return Success(op, Tensor<float>.MaxPool2D((Tensor<float>) X, kernel_shape, padmode.Value, null, strides, dilations, ceil == 1));
            }
            case TensorElementType.Double:
            {
                if (padmode is null)
                {
                    return Success(op, Tensor<double>.MaxPool2D((Tensor<double>)X, kernel_shape, pads ?? new int[] { 0, 0, 0, 0 }, strides, dilations, ceil == 1));
                }
                return Success(op, Tensor<double>.MaxPool2D((Tensor<double>)X, kernel_shape, padmode.Value, null, strides, dilations, ceil == 1));
            }
            default:
                return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult GlobalAveragePool(ITensor? X, ExecutionOptions? options)
    {
        var op = OpType.GlobalAveragePool;
        if (X is null) return MissingInput(op, nameof(X));
        if (X.Rank < 3) return WrongInputShape(op, nameof(X), X, "GlobalAveragePool requires an input of rank 3 or more (NxCxD1..Dn).");
        if (X.Dims.Contains(0) && X.Dims[0] != 0) return WrongInputShape(op, nameof(X), X, "GlobalAveragePool supports an empty batch only; other zero extents are not supported.");
        var axes = new int[X.Rank - 2];
        for (int i = 0; i < axes.Length; i++) axes[i] = i + 2;
        var axesTensor = DenseTensor<int>.OfValues(axes);
        var opts = (options ?? ExecutionOptions.Default).Validated();
        switch (X.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.ReduceMean((Tensor<float>)X, axesTensor, true, false, opts.Tensor));
            case TensorElementType.Double: return Success(op, Tensor<double>.ReduceMean((Tensor<double>)X, axesTensor, true, false, opts.Tensor));
            default: return InputTypeNotSupported(op, nameof(X), X);
        }
    }
}
