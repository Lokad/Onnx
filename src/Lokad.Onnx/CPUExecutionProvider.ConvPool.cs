namespace Lokad.Onnx;


using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

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
    /// <summary>2D convolution with a fused residual Add and trailing Relu for graph-fused Conv+Add+Relu triples.</summary>
    /// <remarks>The residual tensor rides as a fourth node input so lifetime analysis
    /// keeps the skip value alive; the plain convolution runs first and the residual
    /// plus Relu apply as one in-place pass over the owned output (bit-identical to the
    /// unfused chain). Overlapping or non-array storage, shape mismatch, and non-float
    /// dtypes fall back to separate broadcast-capable Add and Relu passes.</remarks>
    public static OpResult Conv(ITensor? X, ITensor? W, ITensor? B, ITensor? residual, string? auto_pad, int[]? dilations, int? group, int[]? kernel_shape, int[]? pads, int[]? strides, ExecutionOptions? options, TensorBufferPool? pool) =>
        ConvCoreResidual(X, W, B, residual, auto_pad, dilations, group, kernel_shape, pads, strides, options, pool);

    static OpResult ConvCoreResidual(ITensor? X, ITensor? W, ITensor? B, ITensor? residual, string? auto_pad, int[]? dilations, int? group, int[]? kernel_shape, int[]? pads, int[]? strides, ExecutionOptions? options, TensorBufferPool? pool)
    {
        var plain = ConvCore(X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides, options, false);
        if (plain.Status != OpStatus.Success) return plain;
        if (residual is null) return plain;
        var convOut = plain.Outputs[0];
        if (convOut is DenseTensor<float> convDense && residual is DenseTensor<float> residualDense)
        {
            var dst = convDense.Buffer;
            var src = residualDense.Buffer;
            if (dst.Length == convDense.Length && src.Length == residualDense.Length && dst.Length == src.Length
                && convDense.IsReversedStride == false && residualDense.IsReversedStride == false
                && convDense.Strides.SequenceEqual(ArrayUtilities.GetStrides(convDense.Dimensions))
                && residualDense.Strides.SequenceEqual(ArrayUtilities.GetStrides(residualDense.Dimensions))
                && TryGetInPlaceSlices(dst, src, out var dSeg, out var sSeg) && dSeg.Array is not null && sSeg.Array is not null)
            {
                var opts = (options ?? ExecutionOptions.Default).Validated();
                AddResidualReluInPlace(dSeg.Array, dSeg.Offset, sSeg.Array, sSeg.Offset, dst.Length, opts.Tensor);
                return plain;
            }
        }
        var added = Add(convOut, residual, options, pool);
        if (added.Status != OpStatus.Success) return added;
        return Relu(added.Outputs[0], options);
    }

    static bool TryGetInPlaceSlices(Memory<float> dst, Memory<float> src, out ArraySegment<float> dSeg, out ArraySegment<float> sSeg)
    {
        dSeg = default;
        sSeg = default;
        if (!MemoryMarshal.TryGetArray(dst, out dSeg) || dSeg.Array is null) return false;
        if (!MemoryMarshal.TryGetArray(src, out sSeg) || sSeg.Array is null) return false;
        if (ReferenceEquals(dSeg.Array, sSeg.Array))
        {
            int dEnd = dSeg.Offset + dSeg.Count;
            int sEnd = sSeg.Offset + sSeg.Count;
            if (dSeg.Offset < sEnd && sSeg.Offset < dEnd) return false;
        }
        return true;
    }

    static void AddResidualReluInPlace(float[] dst, int dOff, float[] src, int sOff, int n, TensorExecutionOptions tensorOpts)
    {
        unsafe
        {
            fixed (float* dBase = dst, sBase = src)
            {
                float* d = dBase + dOff;
                float* s = sBase + sOff;
                if (tensorOpts.UseSimd && tensorOpts.UseIntrinsics && Avx512F.IsSupported && Fma.IsSupported)
                {
                    var zero = Vector512<float>.Zero;
                    int full = n & ~(Vector512<float>.Count - 1);
                    int i = 0;
                    for (; i < full; i += Vector512<float>.Count)
                    {
                        var v = Avx512F.Add(Vector512.LoadUnsafe(ref d[i]), Vector512.LoadUnsafe(ref s[i]));
                        var keep = Vector512.GreaterThan(v, zero) | Vector512.Equals(v, zero) | ~Vector512.Equals(v, v);
                        Vector512.ConditionalSelect(keep, v, zero).StoreUnsafe(ref d[i]);
                    }
                    for (; i < n; i++)
                    {
                        float v = d[i] + s[i];
                        d[i] = v < 0f ? 0f : v;
                    }
                    return;
                }
                if (tensorOpts.UseSimd && tensorOpts.UseIntrinsics && Avx.IsSupported && Fma.IsSupported)
                {
                    var zero = Vector256<float>.Zero;
                    int full = n & ~(Vector256<float>.Count - 1);
                    int i = 0;
                    for (; i < full; i += Vector256<float>.Count)
                    {
                        var v = Avx.Add(Vector256.LoadUnsafe(ref d[i]), Vector256.LoadUnsafe(ref s[i]));
                        var keep = Vector256.GreaterThan(v, zero) | Vector256.Equals(v, zero) | ~Vector256.Equals(v, v);
                        Vector256.ConditionalSelect(keep, v, zero).StoreUnsafe(ref d[i]);
                    }
                    for (; i < n; i++)
                    {
                        float v = d[i] + s[i];
                        d[i] = v < 0f ? 0f : v;
                    }
                    return;
                }
                for (int i = 0; i < n; i++)
                {
                    float v = d[i] + s[i];
                    d[i] = v < 0f ? 0f : v;
                }
            }
        }
    }



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
            // machinery width-first: normalize [N,C,L] inputs to [N,C,1,L]
            // (and likewise weights plus every spatial attribute), run the
            // proven 2D path, then drop the size-1 height axis. The long
            // axis stays contiguous, so the patch gather copies runs
            // instead of striding; Unsqueeze aliases storage, so only the
            // final squeeze may densify.
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
            var xu = Unsqueeze(X, new[] { 2 }, options);
            if (xu.Status != OpStatus.Success) return xu;
            var wu = Unsqueeze(W, new[] { 2 }, options);
            if (wu.Status != OpStatus.Success) return wu;
            int[]? ks4 = kernel_shape is null ? null : new[] { 1, kernel_shape[0] };
            int[]? st4 = strides is null ? null : new[] { 1, strides[0] };
            int[]? di4 = dilations is null ? null : new[] { 1, dilations[0] };
            int[]? pa4 = pads is null ? null : new[] { 0, pads[0], 0, pads[1] };
            var c = ConvCore(xu.Outputs[0], wu.Outputs[0], B, auto_pad, di4, group, ks4, pa4, st4, options, fuseRelu);
            if (c.Status != OpStatus.Success) return c;
            var s = Squeeze(c.Outputs[0], new DenseTensor<long>(new long[] { 2 }, new[] { 1 }), options);
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
                    if (Tensor<float>.TryConvDepthwise2D((Tensor<float>)X, (Tensor<float>)W, bias, group ?? 1, pads, kernel_shape, strides, dilations, opts.Tensor, fuseRelu, out var dwConv2D) && dwConv2D is not null)
                    {
                        return Success(op, dwConv2D);
                    }
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
            // with a trailing singleton axis (unlike convolution above,
            // which runs width-first); pooling semantics are unchanged.
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
