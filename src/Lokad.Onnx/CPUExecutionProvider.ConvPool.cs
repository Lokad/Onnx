namespace Lokad.Onnx;


using System;
using System.Collections.Generic;
using System.Linq;

using static OpResult;
public partial class CPUExecutionProvider
{

    /// <summary>1D or 2D convolution, lowered through the shared matrix dispatcher.</summary>
    public static OpResult Conv(ITensor? X, ITensor? W, ITensor? B, string? auto_pad, int[]? dilations, int? group, int[]? kernel_shape, int[]? pads, int[]? strides, ExecutionOptions? options)
    {
        var op = OpType.Conv;
        if (X is null) return MissingInput(op, nameof(X));
        if (W is null) return MissingInput(op, nameof(W));
        if (W.ElementType != X.ElementType) return WrongInputType(op, nameof(W), X.ElementType, W, "The weights tensor must be the same type as the input tensor.");
        if (B is not null && B.ElementType != X.ElementType) return WrongInputType(op, nameof(B), X.ElementType, B, "The bias tensor must be the same type as the input tensor.");
        if (X.Rank == 3)
        {
            // Adapted from voice 13e98bd, using the final branch's width-first
            // layout. [N,C,L] -> [N,C,1,L] keeps the long axis contiguous.
            // Preserve the existing 2D planner's padding/group/type validation.
            if (W.Rank != 3) return WrongInputShape(op, nameof(W), 3, W);
            if (kernel_shape is not null && kernel_shape.Length != 1)
                return AttributeNotSupported(op, "kernel_shape", kernel_shape.Print(), "One-dimensional kernel_shape must hold one value.");
            if (strides is not null && strides.Length != 1)
                return AttributeNotSupported(op, "strides", strides.Print(), "One-dimensional strides must hold one value.");
            if (dilations is not null && dilations.Length != 1)
                return AttributeNotSupported(op, "dilations", dilations.Print(), "One-dimensional dilations must hold one value.");
            if (pads is not null && pads.Length != 2)
                return AttributeNotSupported(op, "pads", pads.Print(), "One-dimensional pads must hold two values [begin, end].");
            var xu = Unsqueeze(X, new[] { 2 }, options);
            if (xu.Status != OpStatus.Success) return xu;
            var wu = Unsqueeze(W, new[] { 2 }, options);
            if (wu.Status != OpStatus.Success) return wu;
            int[]? ks4 = kernel_shape is null ? null : new[] { 1, kernel_shape[0] };
            int[]? st4 = strides is null ? null : new[] { 1, strides[0] };
            int[]? di4 = dilations is null ? null : new[] { 1, dilations[0] };
            int[]? pa4 = pads is null ? null : new[] { 0, pads[0], 0, pads[1] };
            var conv = Conv(xu.Outputs[0], wu.Outputs[0], B, auto_pad, di4, group, ks4, pa4, st4, options);
            if (conv.Status != OpStatus.Success) return conv;
            var squeezed = Squeeze(conv.Outputs[0], DenseTensor<long>.OfValues(new long[] { 2 }), options);
            squeezed.Op = op;
            return squeezed;
        }
        if (X.Rank != 4)
        {
            return WrongInputShape(op, nameof(X), 4, X);
        }
        if (W.Rank != 4)
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
                    return Success(op, Tensor<float>.Conv2D((Tensor<float>)X, (Tensor<float>)W, group ?? 1, pads ?? new int[] { 0, 0, 0, 0 }, bias, kernel_shape, strides, dilations, opts.Tensor));
                }
                return Success(op, Tensor<float>.Conv2D((Tensor<float>)X, (Tensor<float>)W, group ?? 1, padmode.Value, null, bias, kernel_shape, strides, dilations, opts.Tensor));
            }
            case TensorElementType.Double:
            {
                var biasd = B is null ? null : (Tensor<double>)B;
                if (padmode is null)
                {
                    return Success(op, Tensor<double>.Conv2D((Tensor<double>)X, (Tensor<double>)W, group ?? 1, pads ?? new int[] { 0, 0, 0, 0 }, biasd, kernel_shape, strides, dilations, opts.Tensor));
                }
                return Success(op, Tensor<double>.Conv2D((Tensor<double>)X, (Tensor<double>)W, group ?? 1, padmode.Value, null, biasd, kernel_shape, strides, dilations, opts.Tensor));
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
            // A singleton width preserves the existing 2D pooling semantics;
            // optional indices/storage_order retain the existing restrictions.
            if (kernel_shape is not null && kernel_shape.Length != 1)
                return AttributeNotSupported(op, "kernel_shape", kernel_shape.Print(), "One-dimensional kernel_shape must hold one value.");
            if (strides is not null && strides.Length != 1)
                return AttributeNotSupported(op, "strides", strides.Print(), "One-dimensional strides must hold one value.");
            if (dilations is not null && dilations.Length != 1)
                return AttributeNotSupported(op, "dilations", dilations.Print(), "One-dimensional dilations must hold one value.");
            if (pads is not null && pads.Length != 2)
                return AttributeNotSupported(op, "pads", pads.Print(), "One-dimensional pads must hold two values [begin, end].");
            var xu = Unsqueeze(X, new[] { 3 }, options);
            if (xu.Status != OpStatus.Success) return xu;
            int[]? ks4 = kernel_shape is null ? null : new[] { kernel_shape[0], 1 };
            int[]? st4 = strides is null ? null : new[] { strides[0], 1 };
            int[]? di4 = dilations is null ? null : new[] { dilations[0], 1 };
            int[]? pa4 = pads is null ? null : new[] { pads[0], 0, pads[1], 0 };
            var pool = MaxPool(xu.Outputs[0], auto_pad, ceil_mode, di4, ks4, pa4, storage_order, st4, options);
            if (pool.Status != OpStatus.Success) return pool;
            var squeezed = Squeeze(pool.Outputs[0], DenseTensor<long>.OfValues(new long[] { 3 }), options);
            squeezed.Op = op;
            return squeezed;
        }
        if (X.Rank != 4)
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
