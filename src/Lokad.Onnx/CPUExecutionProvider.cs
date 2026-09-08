namespace Lokad.Onnx;


using System;
using System.Collections.Generic;
using System.Linq;

using static OpResult;

public enum ExecutionProvider
{
    CPU
}

public enum OptimizationMode
{
    Speed,
    Memory
}

public class CPUExecutionProvider
{
    public static IReadOnlyList<OpType> SupportedOps { get; } = new List<OpType>()
    {
        OpType.Reshape,
        OpType.Add,
        OpType.Div,
        OpType.Sub,
        OpType.Mul,
        OpType.Pow,
        OpType.Conv,
        OpType.Relu,
        OpType.MaxPool,
        OpType.MatMul,
        OpType.Sqrt,
        OpType.Erf,
        OpType.Transpose,
        OpType.Constant,
        OpType.Cast,
        OpType.Concat,
        OpType.Shape,
        OpType.Gather,
        OpType.Slice,
        OpType.Equal,
        OpType.Where,
        OpType.Expand,
        OpType.Resize,
        OpType.Unsqueeze,
        OpType.ReduceSum,
        OpType.ReduceMean,
        OpType.ReduceMax,
        OpType.Softmax,
        OpType.Abs,
        OpType.Cos,
        OpType.Sin,
        OpType.Neg,
        OpType.Gelu,
        OpType.Squeeze,
        OpType.Range,
        OpType.Tile,
        OpType.LayerNormalization,
        OpType.SplitToSequence,
        OpType.SequenceAt,
        OpType.RotaryEmbedding,
        OpType.Gemm,
        OpType.Tanh,
        OpType.Split,
        OpType.Less,
        OpType.ConstantOfShape,
        OpType.GlobalAveragePool,
    };

    public static bool SupportsOp(OpType op) => SupportedOps.Contains(op);

    public static bool IsStandardDomain(string? domain) =>
        string.IsNullOrEmpty(domain) || domain == "ai.onnx";

    public static bool SupportsNode(Node node) =>
        node.Op != OpType.Unknown && IsStandardDomain(node.Domain) && SupportedOps.Contains(node.Op);

    public static string DescribeNode(Node node)
    {
        var domain = string.IsNullOrEmpty(node.Domain) ? "ai.onnx" : node.Domain;
        var op = string.IsNullOrEmpty(node.OpTypeName) ? node.Op.ToString() : node.OpTypeName;
        return domain + ":" + op + ":" + node.OpsetVersion + (node.IsFused ? " (fused)" : "");
    }

    public static OpResult Reshape(ITensor? input, ITensor? shape, bool? allow_zero, ExecutionOptions? options)
    {
        var op = OpType.Reshape;
        if (input is null) return MissingInput(op, nameof(input));
        if (shape is null) return MissingInput(op, nameof(shape));
        (options ?? ExecutionOptions.Default).Validated();
        if (shape.ElementType != TensorElementType.Int64) return WrongInputType(op, nameof(shape), TensorElementType.Int64, shape);
        switch (input.ElementType)
        {
            case TensorElementType.Bool: return Success(op, Tensor<bool>.Reshape((Tensor<bool>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.Int8: return Success(op, Tensor<sbyte>.Reshape((Tensor<sbyte>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.UInt8: return Success(op, Tensor<byte>.Reshape((Tensor<byte>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.Int16: return Success(op, Tensor<short>.Reshape((Tensor<short>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.UInt16: return Success(op, Tensor<ushort>.Reshape((Tensor<ushort>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Reshape((Tensor<int>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.UInt32: return Success(op, Tensor<uint>.Reshape((Tensor<uint>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Reshape((Tensor<long>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.UInt64: return Success(op, Tensor<ulong>.Reshape((Tensor<ulong>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.Float: return Success(op, Tensor<float>.Reshape((Tensor<float>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.Double: return Success(op, Tensor<double>.Reshape((Tensor<double>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.Float16: return Success(op, Tensor<Float16>.Reshape((Tensor<Float16>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.BFloat16: return Success(op, Tensor<BFloat16>.Reshape((Tensor<BFloat16>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.Complex64: return Success(op, Tensor<System.Numerics.Complex>.Reshape((Tensor<System.Numerics.Complex>)input, (Tensor<long>)shape, allow_zero ?? false));
            default: return NotSupported(op);
        }
    }

    public static OpResult Add(ITensor? A, ITensor? B, ExecutionOptions? options, TensorBufferPool? pool)
    {
        var op = OpType.Add;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }

        Profiler.StartOpStage(OpStage.Broadcast);
        // Shapes are dtype-independent: one pure shape check keeps the
        // CannotBroadcast diagnostic while kernels read the originals directly.
        if (!Tensor<int>.BroadcastShape(A.Dims, B.Dims, out _))
        {
            return CannotBroadcast(op, A, B);
        }

        var opts = (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Math);
        switch (A.ElementType)
        {
            case TensorElementType.UInt8: return Success(op, ((Tensor<byte>)A).BroadcastApply<AddBroadcast<byte>>((Tensor<byte>)B, opts.Tensor));
            case TensorElementType.Int32: return Success(op, ((Tensor<int>)A).BroadcastApply<AddBroadcast<int>>((Tensor<int>)B, opts.Tensor));
            case TensorElementType.Int64: return Success(op, ((Tensor<long>)A).BroadcastApply<AddBroadcast<long>>((Tensor<long>)B, opts.Tensor));
            case TensorElementType.Float:
            {
                var fa = (Tensor<float>)A;
                var fb = (Tensor<float>)B;
                if (!Tensor<float>.BroadcastShape(fa.Dimensions, fb.Dimensions, out var shape)) return CannotBroadcast(op, A, B);
                if (pool is null) return Success(op, fa.BroadcastApply<AddBroadcast<float>>(fb, opts.Tensor));
                int flat = 1;
                foreach (var d in shape) flat *= d;
                var rented = new DenseTensor<float>(new Memory<float>(pool.Rent<float>(flat)), shape);
                return Success(op, fa.BroadcastApply<AddBroadcast<float>>(fb, rented, opts.Tensor));
            }
            case TensorElementType.Double: return Success(op, ((Tensor<double>)A).BroadcastApply<AddBroadcast<double>>((Tensor<double>)B, opts.Tensor));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Sub(ITensor? A, ITensor? B, ExecutionOptions? options)
    {
        var op = OpType.Sub;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }
        Profiler.StartOpStage(OpStage.Broadcast);
        // Shapes are dtype-independent: one pure shape check keeps the
        // CannotBroadcast diagnostic while kernels read the originals directly.
        if (!Tensor<int>.BroadcastShape(A.Dims, B.Dims, out _))
        {
            return CannotBroadcast(op, A, B);
        }

        var opts = (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Math);
        switch (A.ElementType)
        {
            case TensorElementType.UInt8: return Success(op, ((Tensor<byte>)A).BroadcastApply<SubtractBroadcast<byte>>((Tensor<byte>)B, opts.Tensor));
            case TensorElementType.Int32: return Success(op, ((Tensor<int>)A).BroadcastApply<SubtractBroadcast<int>>((Tensor<int>)B, opts.Tensor));
            case TensorElementType.Int64: return Success(op, ((Tensor<long>)A).BroadcastApply<SubtractBroadcast<long>>((Tensor<long>)B, opts.Tensor));
            case TensorElementType.Float: return Success(op, ((Tensor<float>)A).BroadcastApply<SubtractBroadcast<float>>((Tensor<float>)B, opts.Tensor));
            case TensorElementType.Double: return Success(op, ((Tensor<double>)A).BroadcastApply<SubtractBroadcast<double>>((Tensor<double>)B, opts.Tensor));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Mul(ITensor? A, ITensor? B, ExecutionOptions? options, TensorBufferPool? pool)
    {
        var op = OpType.Mul;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }
        Profiler.StartOpStage(OpStage.Broadcast);
        // Shapes are dtype-independent: one pure shape check keeps the
        // CannotBroadcast diagnostic while kernels read the originals directly.
        if (!Tensor<int>.BroadcastShape(A.Dims, B.Dims, out _))
        {
            return CannotBroadcast(op, A, B);
        }

        var opts = (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Math);
        switch (A.ElementType)
        {
            case TensorElementType.UInt8: return Success(op, ((Tensor<byte>)A).BroadcastApply<MultiplyBroadcast<byte>>((Tensor<byte>)B, opts.Tensor));
            case TensorElementType.Int32: return Success(op, ((Tensor<int>)A).BroadcastApply<MultiplyBroadcast<int>>((Tensor<int>)B, opts.Tensor));
            case TensorElementType.Int64: return Success(op, ((Tensor<long>)A).BroadcastApply<MultiplyBroadcast<long>>((Tensor<long>)B, opts.Tensor));
            case TensorElementType.Float:
            {
                var fa = (Tensor<float>)A;
                var fb = (Tensor<float>)B;
                if (!Tensor<float>.BroadcastShape(fa.Dimensions, fb.Dimensions, out var shape)) return CannotBroadcast(op, A, B);
                if (pool is null) return Success(op, fa.BroadcastApply<MultiplyBroadcast<float>>(fb, opts.Tensor));
                int flat = 1;
                foreach (var d in shape) flat *= d;
                var rented = new DenseTensor<float>(new Memory<float>(pool.Rent<float>(flat)), shape);
                return Success(op, fa.BroadcastApply<MultiplyBroadcast<float>>(fb, rented, opts.Tensor));
            }
            case TensorElementType.Double: return Success(op, ((Tensor<double>)A).BroadcastApply<MultiplyBroadcast<double>>((Tensor<double>)B, opts.Tensor));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Div(ITensor? A, ITensor? B, ExecutionOptions? options, TensorBufferPool? pool)
    {
        var op = OpType.Div;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }
        Profiler.StartOpStage(OpStage.Broadcast);
        // Shapes are dtype-independent: one pure shape check keeps the
        // CannotBroadcast diagnostic while kernels read the originals directly.
        if (!Tensor<int>.BroadcastShape(A.Dims, B.Dims, out _))
        {
            return CannotBroadcast(op, A, B);
        }

        var opts = (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Math);
        switch (A.ElementType)
        {
            case TensorElementType.UInt8: return Success(op, ((Tensor<byte>)A).BroadcastApply<DivideBroadcast<byte>>((Tensor<byte>)B, opts.Tensor));
            case TensorElementType.Int32: return Success(op, ((Tensor<int>)A).BroadcastApply<DivideBroadcast<int>>((Tensor<int>)B, opts.Tensor));
            case TensorElementType.Int64: return Success(op, ((Tensor<long>)A).BroadcastApply<DivideBroadcast<long>>((Tensor<long>)B, opts.Tensor));
            case TensorElementType.Float:
            {
                var fa = (Tensor<float>)A;
                var fb = (Tensor<float>)B;
                if (!Tensor<float>.BroadcastShape(fa.Dimensions, fb.Dimensions, out var shape)) return CannotBroadcast(op, A, B);
                if (pool is null) return Success(op, fa.BroadcastApply<DivideBroadcast<float>>(fb, opts.Tensor));
                int flat = 1;
                foreach (var d in shape) flat *= d;
                var rented = new DenseTensor<float>(new Memory<float>(pool.Rent<float>(flat)), shape);
                return Success(op, fa.BroadcastApply<DivideBroadcast<float>>(fb, rented, opts.Tensor));
            }
            case TensorElementType.Double: return Success(op, ((Tensor<double>)A).BroadcastApply<DivideBroadcast<double>>((Tensor<double>)B, opts.Tensor));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Pow(ITensor? A, ITensor? B, ExecutionOptions? options)
    {
        var op = OpType.Pow;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }
        Profiler.StartOpStage(OpStage.Broadcast);
        // Shapes are dtype-independent: one pure shape check keeps the
        // CannotBroadcast diagnostic while kernels read the originals directly.
        if (!Tensor<int>.BroadcastShape(A.Dims, B.Dims, out _))
        {
            return CannotBroadcast(op, A, B);
        }

        var opts = (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Math);
        switch (A.ElementType)
        {
            case TensorElementType.Float: return Success(op, ((Tensor<float>)A).BroadcastApply((Tensor<float>)B, MathF.Pow, opts.Tensor));
            case TensorElementType.Double: return Success(op, ((Tensor<double>)A).BroadcastApply((Tensor<double>)B, Math.Pow, opts.Tensor));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }
    /// <summary>2D convolution, lowered through the shared matrix dispatcher.</summary>
    public static OpResult Conv(ITensor? X, ITensor? W, ITensor? B, string? auto_pad, int[]? dilations, int? group, int[]? kernel_shape, int[]? pads, int[]? strides, ExecutionOptions? options)
    {
        var op = OpType.Conv;
        if (X is null) return MissingInput(op, nameof(X));
        if (W is null) return MissingInput(op, nameof(W));
        if (W.ElementType != X.ElementType) return WrongInputType(op, nameof(W), X.ElementType, W, "The weights tensor must be the same type as the input tensor.");
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
        if (!string.IsNullOrEmpty(auto_pad) && auto_pad != "NOTSET" && pads is not null && pads.Any(p => p != 0))
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

    public static OpResult Relu(ITensor? X, ExecutionOptions? options)
    {
        var op = OpType.Relu;
        if (X is null) return MissingInput(op, nameof(X));
        var opts = (options ?? ExecutionOptions.Default).Validated();
        if (opts.Optimization == OptimizationMode.Speed)
        {
            Profiler.StartOpStage(OpStage.Copy);
            X = X.ToDenseTensor();
        }
        Profiler.StartOpStage(OpStage.Math);
        switch (X.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Relu((Tensor<float>)X));
            case TensorElementType.Double: return Success(op, Tensor<double>.Relu((Tensor<double>)X));
            default: return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult MaxPool(ITensor? X, string? auto_pad, int? ceil_mode, int[]? dilations, int[]? kernel_shape, int[]? pads, int? storage_order, int[]? strides, ExecutionOptions? options)
    {
        var op = OpType.MaxPool;
        if (X is null) return MissingInput(op, nameof(X));
        (options ?? ExecutionOptions.Default).Validated();
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
        if (!string.IsNullOrEmpty(auto_pad) && auto_pad != "NOTSET" && pads is not null && pads.Any(p => p != 0))
        {
            return AttributeNotSupported(op, "pads", pads.Print(), "pads must not be specified together with an automatic padding mode.");
        }
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

    public static OpResult MatMul(ITensor? A, ITensor? B, ExecutionOptions? options, TensorBufferPool? pool)
    {
        var op = OpType.MatMul;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));

        var opts = (options ?? ExecutionOptions.Default).Validated();
        if (opts.Optimization == OptimizationMode.Speed)
        {
            Profiler.StartOpStage(OpStage.Copy);
            A = A.ToDenseTensor();
            B = B.ToDenseTensor();
        }
        switch (A.ElementType)
        {
            case TensorElementType.Int32: return Success(op, Tensor<int>.MatMul((Tensor<int>)A, (Tensor<int>)B, opts.Tensor));
            case TensorElementType.Float: return Success(op, Tensor<float>.MatMul((Tensor<float>)A, (Tensor<float>)B, opts.Tensor, pool));
            case TensorElementType.Double: return Success(op, Tensor<double>.MatMul((Tensor<double>)A, (Tensor<double>)B, opts.Tensor));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Gemm(ITensor? A, ITensor? B, ITensor? C, float alpha, float beta, ExecutionOptions? options, int transA, int transB)
    {
        var op = OpType.Gemm;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        var opts = (options ?? ExecutionOptions.Default).Validated();
        if (transA != 0 && transA != 1) return AttributeNotSupported(op, "transA", transA.ToString(), "transA must be 0 or 1.");
        if (transB != 0 && transB != 1) return AttributeNotSupported(op, "transB", transB.ToString(), "transB must be 0 or 1.");
        if (A.ElementType != B.ElementType) return WrongInputType(op, nameof(B), A.ElementType, B, "Gemm inputs A and B must have the same element type.");
        if (A.Rank != 2) return WrongInputShape(op, nameof(A), 2, A);
        if (B.Rank != 2) return WrongInputShape(op, nameof(B), 2, B);
        if (C is not null && C.ElementType != A.ElementType) return WrongInputType(op, nameof(C), A.ElementType, C, "Gemm input C must have the same element type as A and B.");
        int m = transA == 1 ? A.Dims[1] : A.Dims[0];
        int kA = transA == 1 ? A.Dims[0] : A.Dims[1];
        int kB = transB == 1 ? B.Dims[1] : B.Dims[0];
        int n = transB == 1 ? B.Dims[0] : B.Dims[1];
        if (kA != kB) return WrongInputShape(op, nameof(B), B, "Gemm inner dimensions disagree after transpose.");
        int k = kA;
        if (C is not null && C.Length > 1)
        {
            bool okc = (C.Rank == 1 && C.Dims[0] == n)
                || (C.Rank == 2 && ((C.Dims[0] == 1 && C.Dims[1] == n) || (C.Dims[0] == m && C.Dims[1] == 1) || (C.Dims[0] == m && C.Dims[1] == n)));
            if (!okc) return WrongInputShape(op, nameof(C), C, "Gemm bias C must be a scalar, [N], [1,N], [M,1], or [M,N].");
        }

        /// <summary>
        /// Writes alpha * (a @ b) + beta * c into one owned destination: the product
        /// lands directly in the result through the shared overwriting entry, and the
        /// scale/bias pass runs only when it can change anything.
        /// </summary>
        static Tensor<float> GemmFloat(Tensor<float> a, Tensor<float> b, Tensor<float>? c, int m, int k, int n, float alpha, float beta, TensorExecutionOptions tensorOptions)
        {
            var y = DenseTensor<float>.OfShape(m, n);
            Tensor<float>.MatMul2D(a, b, y, tensorOptions);
            if (alpha == 1f && (c is null || beta == 0f)) return y;
            var ys = y.Buffer.Span;
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float cb = 0f;
                    if (c is not null && beta != 0f)
                    {
                        if (c.Length == 1) cb = c.GetValue(0);
                        else if (c.Rank == 1) cb = c.GetValue(j);
                        else cb = c.GetValue((c.Dimensions[0] == 1 ? 0 : i) * c.Dimensions[1] + (c.Dimensions[1] == 1 ? 0 : j));
                    }
                    ys[i * n + j] = alpha * ys[i * n + j] + beta * cb;
                }
            }
            return y;
        }

        /// <summary>
        /// Writes alpha * (a @ b) + beta * c into one owned destination: the product
        /// lands directly in the result through the shared overwriting entry, and the
        /// scale/bias pass runs only when it can change anything.
        /// </summary>
        static Tensor<double> GemmDouble(Tensor<double> a, Tensor<double> b, Tensor<double>? c, int m, int k, int n, float alpha, float beta, TensorExecutionOptions tensorOptions)
        {
            var y = DenseTensor<double>.OfShape(m, n);
            Tensor<double>.MatMul2D(a, b, y, tensorOptions);
            if (alpha == 1f && (c is null || beta == 0f)) return y;
            var ys = y.Buffer.Span;
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double cb = 0.0;
                    if (c is not null && beta != 0f)
                    {
                        if (c.Length == 1) cb = c.GetValue(0);
                        else if (c.Rank == 1) cb = c.GetValue(j);
                        else cb = c.GetValue((c.Dimensions[0] == 1 ? 0 : i) * c.Dimensions[1] + (c.Dimensions[1] == 1 ? 0 : j));
                    }
                    ys[i * n + j] = alpha * ys[i * n + j] + beta * cb;
                }
            }
            return y;
        }
        switch (A.ElementType)
        {
            case TensorElementType.Float:
            {
                var ea = transA == 1 ? Tensor<float>.Transpose((Tensor<float>)A, null) : (Tensor<float>)A;
                var eb = transB == 1 ? Tensor<float>.Transpose((Tensor<float>)B, null) : (Tensor<float>)B;
                return Success(op, GemmFloat(ea, eb, (Tensor<float>?)C, m, k, n, alpha, beta, opts.Tensor));
            }
            case TensorElementType.Double:
            {
                var ea = transA == 1 ? Tensor<double>.Transpose((Tensor<double>)A, null) : (Tensor<double>)A;
                var eb = transB == 1 ? Tensor<double>.Transpose((Tensor<double>)B, null) : (Tensor<double>)B;
                return Success(op, GemmDouble(ea, eb, (Tensor<double>?)C, m, k, n, alpha, beta, opts.Tensor));
            }
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Sqrt(ITensor? A, ExecutionOptions? options)
    {
        var op = OpType.Sqrt;
        if (A is null) return MissingInput(op, nameof(A));
        var opts = (options ?? ExecutionOptions.Default).Validated();
        if (opts.Optimization == OptimizationMode.Speed)
        {
            Profiler.StartOpStage(OpStage.Copy);
            A = A.ToDenseTensor();
        }
        Profiler.StartOpStage(OpStage.Math);
        switch (A.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Sqrt((Tensor<float>)A, opts.Tensor));
            case TensorElementType.Double: return Success(op, Tensor<double>.Sqrt((Tensor<double>)A, opts.Tensor));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Erf(ITensor? X, ExecutionOptions? options, TensorBufferPool? pool)
    {
        var op = OpType.Erf;
        if (X is null) return MissingInput(op, nameof(X));
        Profiler.StartOpStage(OpStage.Math);
        var opts = (options ?? ExecutionOptions.Default).Validated();
        switch (X.ElementType)
        {
            case TensorElementType.Float:
            {
                var tensorOptions = opts.Tensor;
                var fx = (Tensor<float>)X;
                if (pool is null) return Success(op, Tensor<float>.Erf(fx, tensorOptions));
                var rented = new DenseTensor<float>(new Memory<float>(pool.Rent<float>((int)fx.Length)), fx.Dimensions.ToArray());
                return Success(op, Tensor<float>.Erf(fx, rented, tensorOptions));
            }
            case TensorElementType.Double: return Success(op, Tensor<double>.Erf((Tensor<double>)X));
            default: return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult Transpose(ITensor? data, int[]? perm, ExecutionOptions? options, TensorBufferPool? pool)
    {
        var op = OpType.Transpose;
        if (data is null) return MissingInput(op, nameof(data));
        (options ?? ExecutionOptions.Default).Validated();
        switch (data.ElementType)
        {
            case TensorElementType.Bool: return Success(op, Tensor<bool>.Transpose((Tensor<bool>)data, perm));
            case TensorElementType.Int8: return Success(op, Tensor<sbyte>.Transpose((Tensor<sbyte>)data, perm));
            case TensorElementType.UInt8: return Success(op, Tensor<byte>.Transpose((Tensor<byte>)data, perm));
            case TensorElementType.Int16: return Success(op, Tensor<short>.Transpose((Tensor<short>)data, perm));
            case TensorElementType.UInt16: return Success(op, Tensor<ushort>.Transpose((Tensor<ushort>)data, perm));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Transpose((Tensor<int>)data, perm));
            case TensorElementType.UInt32: return Success(op, Tensor<uint>.Transpose((Tensor<uint>)data, perm));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Transpose((Tensor<long>)data, perm));
            case TensorElementType.UInt64: return Success(op, Tensor<ulong>.Transpose((Tensor<ulong>)data, perm));
            case TensorElementType.Float:
            {
                var fx = (Tensor<float>)data;
                if (pool is null) return Success(op, Tensor<float>.Transpose(fx, perm));
                var dims = Tensor<float>.TransposedShape(fx.Dimensions, perm);
                var rented = new DenseTensor<float>(new Memory<float>(pool.Rent<float>((int)fx.Length)), dims);
                return Success(op, Tensor<float>.Transpose(fx, rented, perm));
            }
            case TensorElementType.Double: return Success(op, Tensor<double>.Transpose((Tensor<double>)data, perm));
            case TensorElementType.Float16: return Success(op, Tensor<Float16>.Transpose((Tensor<Float16>)data, perm));
            case TensorElementType.BFloat16: return Success(op, Tensor<BFloat16>.Transpose((Tensor<BFloat16>)data, perm));
            case TensorElementType.Complex64: return Success(op, Tensor<System.Numerics.Complex>.Transpose((Tensor<System.Numerics.Complex>)data, perm));
            default: return NotSupported(op);
        }
    }

    public static OpResult Constant(object? value, ExecutionOptions? options)
    {
        var op = OpType.Constant;
        if (value is null) return MissingAttribute(op, nameof(value), null);
        (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Copy);
        ITensor CloneConstantTensor(ITensor source)
        {
            if (source is TensorSequence seq) return seq.Clone();
            var sourceType = source.GetType();
            if (sourceType.IsGenericType && sourceType.GetGenericTypeDefinition() == typeof(DenseTensor<>)) return source.Clone();
            try
            {
                var dense = source.ToDenseTensor();
                if (!ReferenceEquals(dense, source)) return dense;
            }
            catch
            {
            }
            return source.Clone();
        }
        switch (value)
        {
            case ITensor t: return Success(op, CloneConstantTensor(t));
            case float f: return Success(op, DenseTensor<float>.Scalar(f));
            case float[] fa: return Success(op, DenseTensor<float>.OfValues(fa));
            case int i: return Success(op, DenseTensor<int>.Scalar(i));
            case int[] ia: return Success(op, DenseTensor<int>.OfValues(ia));
            case long l: return Success(op, DenseTensor<long>.Scalar(l));
            case long[] la: return Success(op, DenseTensor<long>.OfValues(la));
            default: return NotSupported(op);
        }
    }

    public static OpResult ConstantOfShape(ITensor? shape, ITensor? value, ExecutionOptions? options)
    {
        var op = OpType.ConstantOfShape;
        if (shape is null) return MissingInput(op, nameof(shape));
        (options ?? ExecutionOptions.Default).Validated();
        int[] dims;
        if (shape.ElementType == TensorElementType.Int64) dims = ((Tensor<long>)shape).ToArray().Select(v => checked((int)v)).ToArray();
        else if (shape.ElementType == TensorElementType.Int32) dims = ((Tensor<int>)shape).ToArray();
        else return InputTypeNotSupported(op, nameof(shape), shape);
        if (dims.Any(d => d < 0)) return WrongInputShape(op, nameof(shape), shape, "ConstantOfShape dimensions must be non-negative.");
        value ??= DenseTensor<float>.OfValues(new float[] { 0f });
        if (value.Length != 1) return WrongInputShape(op, nameof(value), value, "ConstantOfShape value must hold a single element.");
        switch (value.ElementType)
        {
            case TensorElementType.Float: { var y = DenseTensor<float>.OfShape(dims); y.Fill(((Tensor<float>)value).GetValue(0)); return Success(op, y); }
            case TensorElementType.Double: { var y = DenseTensor<double>.OfShape(dims); y.Fill(((Tensor<double>)value).GetValue(0)); return Success(op, y); }
            case TensorElementType.Int32: { var y = DenseTensor<int>.OfShape(dims); y.Fill(((Tensor<int>)value).GetValue(0)); return Success(op, y); }
            case TensorElementType.Int64: { var y = DenseTensor<long>.OfShape(dims); y.Fill(((Tensor<long>)value).GetValue(0)); return Success(op, y); }
            case TensorElementType.Bool: { var y = DenseTensor<bool>.OfShape(dims); y.Fill(((Tensor<bool>)value).GetValue(0)); return Success(op, y); }
            default: return InputTypeNotSupported(op, nameof(value), value);
        }
    }

    public static OpResult Cast(ITensor? input, TensorElementType to, ExecutionOptions? options)
    {
        var op = OpType.Cast;
        if (input is null) return MissingInput(op, nameof(input));
        (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Copy);
        switch (to)
        {
            case TensorElementType.Bool: return Success(op, input.Cast<bool>());
            case TensorElementType.Int8: return Success(op, input.Cast<sbyte>());
            case TensorElementType.UInt8: return Success(op, input.Cast<byte>());
            case TensorElementType.Int16: return Success(op, input.Cast<short>());
            case TensorElementType.UInt16: return Success(op, input.Cast<ushort>());
            case TensorElementType.Int32: return Success(op, input.Cast<int>());
            case TensorElementType.UInt32: return Success(op, input.Cast<uint>());
            case TensorElementType.Int64: return Success(op, input.Cast<long>());
            case TensorElementType.UInt64: return Success(op, input.Cast<ulong>());
            case TensorElementType.Float: return Success(op, input.Cast<float>());
            case TensorElementType.Double: return Success(op, input.Cast<double>());
            //case TensorElementType.Float16: return Success(op, input.Cast<Float16>());
            //case TensorElementType.BFloat16: return Success(op, input.Cast<BFloat16>());
            //case TensorElementType.Complex64: return Success(op, input.Cast<System.Numerics.Complex>());
            default: return AttributeNotSupported(op, "to", to.ToString(), null);

        }
    }
    public static OpResult Concat(ITensor[]? inputs, int? _axis, ExecutionOptions? options)
    {
        var op = OpType.Concat;
        if (inputs is null) return MissingInput(op, nameof(inputs));
        (options ?? ExecutionOptions.Default).Validated();
        if (!inputs.All(i => i.ElementType == inputs[0].ElementType)) return WrongInputType(op, nameof(inputs), inputs[0].ElementType, inputs.First(i => i.ElementType == inputs[0].ElementType), "All tensors in a concat operation must have the same type.");
        var axis = _axis.HasValue ? _axis.Value :  0;
        switch (inputs[0].ElementType) 
        {
            case TensorElementType.Bool: return Success(op, Tensor<bool>.Concat(inputs.CastA<Tensor<bool>>(), axis));
            case TensorElementType.Int8: return Success(op, Tensor<sbyte>.Concat(inputs.CastA<Tensor<sbyte>>(), axis));
            case TensorElementType.UInt8: return Success(op, Tensor<byte>.Concat(inputs.CastA<Tensor<byte>>(), axis));
            case TensorElementType.Int16: return Success(op, Tensor<short>.Concat(inputs.CastA<Tensor<short>>(), axis));
            case TensorElementType.UInt16: return Success(op, Tensor<ushort>.Concat(inputs.CastA<Tensor<ushort>>(), axis));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Concat(inputs.CastA<Tensor<int>>(), axis));
            case TensorElementType.UInt32: return Success(op, Tensor<uint>.Concat(inputs.CastA<Tensor<uint>>(), axis));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Concat(inputs.CastA<Tensor<long>>(), axis));
            case TensorElementType.UInt64: return Success(op, Tensor<ulong>.Concat(inputs.CastA<Tensor<ulong>>(), axis));
            case TensorElementType.Float: return Success(op, Tensor<float>.Concat(inputs.CastA<Tensor<float>>(), axis));
            case TensorElementType.Double: return Success(op, Tensor<double>.Concat(inputs.CastA<Tensor<double>>(), axis));
            case TensorElementType.Float16: return Success(op, Tensor<Float16>.Concat(inputs.CastA<Tensor<Float16>>(), axis));
            case TensorElementType.BFloat16: return Success(op, Tensor<BFloat16>.Concat(inputs.CastA<Tensor<BFloat16>>(), axis));
            case TensorElementType.Complex64: return Success(op, Tensor<System.Numerics.Complex>.Concat(inputs.CastA<Tensor<System.Numerics.Complex>>(), axis));
            default: return InputTypeNotSupported(op, "inputs", inputs[0]);
        }
    }

    public static OpResult Shape(ITensor? data, int? _start, int? _end, ExecutionOptions? options)
    {
        var op = OpType.Shape;
        if (data is null) return MissingInput(op, nameof(data));
        (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.CalculateIndices);
        var start = ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, _start.HasValue ? _start.Value : 0);
        var end = ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, _end.HasValue ? _end.Value : data.Rank);
        start = ArrayUtilities.Clamp(start, 0, data.Rank);
        end = ArrayUtilities.Clamp(end, 0, data.Rank);  
        var _shape = data.Dims.Convert<int, long>()[start..end];
        return Success(op, DenseTensor<long>.OfValues(_shape));
    }

    public static OpResult Gather(ITensor? data, ITensor? indices, int? axis, ExecutionOptions? options) 
    {
        var op = OpType.Gather;
        if (data is null) return MissingInput(op, nameof(data));
        if (indices is null) return MissingInput(op, nameof(indices));
        (options ?? ExecutionOptions.Default).Validated();
        if (indices.Rank > data.Rank) return WrongInputShape(op, nameof(indices), data.Rank, indices);
        
        if (indices.ElementType == TensorElementType.Int64)
        {
            indices = indices.ConvertToInt32();
        }
        
        switch (data.ElementType)
        {
            case TensorElementType.Bool: return Success(op, Tensor<bool>.Gather((Tensor<bool>)data, (Tensor<int>)  indices, axis));
            case TensorElementType.Int8: return Success(op, Tensor<sbyte>.Gather((Tensor<sbyte>)data, (Tensor<int>)  indices, axis));
            case TensorElementType.UInt8: return Success(op, Tensor<byte>.Gather((Tensor<byte>)data, (Tensor<int>)  indices, axis));
            case TensorElementType.Int16: return Success(op, Tensor<short>.Gather((Tensor<short>)data, (Tensor<int>)  indices, axis));
            case TensorElementType.UInt16: return Success(op, Tensor<ushort>.Gather((Tensor<ushort>)data, (Tensor<int>)  indices, axis));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Gather((Tensor<int>)data, (Tensor<int>)  indices, axis));
            case TensorElementType.UInt32: return Success(op, Tensor<uint>.Gather((Tensor<uint>)data, (Tensor<int>)  indices, axis));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Gather((Tensor<long>)data, (Tensor<int>)  indices, axis));
            case TensorElementType.UInt64: return Success(op, Tensor<ulong>.Gather((Tensor<ulong>)data, (Tensor<int>)  indices, axis));
            case TensorElementType.Float: return Success(op, Tensor<float>.Gather((Tensor<float>)data, (Tensor<int>)  indices, axis));
            case TensorElementType.Double: return Success(op, Tensor<double>.Gather((Tensor<double>)data, (Tensor<int>)  indices, axis));
            case TensorElementType.Float16: return Success(op, Tensor<Float16>.Gather((Tensor<Float16>)data, (Tensor<int>)  indices, axis));
            case TensorElementType.BFloat16: return Success(op, Tensor<BFloat16>.Gather((Tensor<BFloat16>)data, (Tensor<int>)  indices, axis));
            case TensorElementType.Complex64: return Success(op, Tensor<System.Numerics.Complex>.Gather((Tensor<System.Numerics.Complex>)data, (Tensor<int>)  indices, axis));
            default: return NotSupported(op);
        }
    }

    public static OpResult Slice(ITensor? data, ITensor? starts, ITensor? ends, ITensor? axes, ITensor? steps, ExecutionOptions? options)
    {
        var op = OpType.Slice;
        if (data is null) return MissingInput(op, nameof(data));
        if (starts is null) return MissingInput(op, nameof(starts));
        if (ends is null) return MissingInput(op, nameof(ends));
        (options ?? ExecutionOptions.Default).Validated();
        if (data.Rank == 0) return WrongInputShape(op, nameof(data), data, "Cannot slice a tensor of rank 0");
        if (starts.Rank != 1) return WrongInputShape(op, nameof(starts), starts, "The rank of the starts tensor must be 1");
        if (starts.Length > data.Rank) return WrongInputShape(op, nameof(starts), starts, "The length of the starts tensor must be less-than or equal to the rank of the data tensor.");
        if (ends.Rank != 1) return WrongInputShape(op, nameof(ends), ends, "The rank of the ends tensor must be 1");
        if (starts.Length != ends.Length) return WrongInputShape(op, nameof(ends), ends, "The ends tensor must be the same length as the start tensor.");
        if (axes is not null && (axes.Rank != 1 || axes.Length != starts.Length)) return WrongInputShape(op, nameof(axes), axes, "The axes tensor must be a rank 1 tensor with the same length as the start tensor.");
        if (steps is not null && (steps.Rank != 1 || steps.Length != starts.Length)) return WrongInputShape(op, nameof(steps), steps, "The steps tensor must be a rank 1 tensor with the same length as the start tensor.");
        
        if (starts.ElementType == TensorElementType.Int64)
        {
            starts = ToInt32Saturating(starts);
        }

        if (ends.ElementType == TensorElementType.Int64)
        {
            ends = ToInt32Saturating(ends);
        }

        if (axes is not null && axes.ElementType == TensorElementType.Int64)
        {
            axes = ToInt32Saturating(axes);
        }

        if (steps is not null && steps.ElementType == TensorElementType.Int64)
        {
            steps = ToInt32Saturating(steps);
        }

        switch (data.ElementType)
        {
            case TensorElementType.Bool: return Success(op, Tensor<bool>.Slice((Tensor<bool>)data, (Tensor<int>) starts, (Tensor<int>) ends, (Tensor<int>?) axes, (Tensor<int>?) steps));
            case TensorElementType.Int8: return Success(op, Tensor<sbyte>.Slice((Tensor<sbyte>)data, (Tensor<int>) starts, (Tensor<int>) ends, (Tensor<int>?) axes, (Tensor<int>?) steps));
            case TensorElementType.UInt8: return Success(op, Tensor<byte>.Slice((Tensor<byte>)data, (Tensor<int>) starts, (Tensor<int>) ends, (Tensor<int>?) axes, (Tensor<int>?) steps));
            case TensorElementType.Int16: return Success(op, Tensor<short>.Slice((Tensor<short>)data, (Tensor<int>) starts, (Tensor<int>) ends, (Tensor<int>?) axes, (Tensor<int>?) steps));
            case TensorElementType.UInt16: return Success(op, Tensor<ushort>.Slice((Tensor<ushort>)data, (Tensor<int>) starts, (Tensor<int>) ends, (Tensor<int>?) axes, (Tensor<int>?) steps));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Slice((Tensor<int>)data, (Tensor<int>) starts, (Tensor<int>) ends, (Tensor<int>?) axes, (Tensor<int>?) steps));
            case TensorElementType.UInt32: return Success(op, Tensor<uint>.Slice((Tensor<uint>)data, (Tensor<int>) starts, (Tensor<int>) ends, (Tensor<int>?) axes, (Tensor<int>?) steps));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Slice((Tensor<long>)data, (Tensor<int>) starts, (Tensor<int>) ends, (Tensor<int>?) axes, (Tensor<int>?) steps));
            case TensorElementType.UInt64: return Success(op, Tensor<ulong>.Slice((Tensor<ulong>)data, (Tensor<int>) starts, (Tensor<int>) ends, (Tensor<int>?) axes, (Tensor<int>?) steps));
            case TensorElementType.Float: return Success(op, Tensor<float>.Slice((Tensor<float>)data, (Tensor<int>) starts, (Tensor<int>) ends, (Tensor<int>?) axes, (Tensor<int>?) steps));
            case TensorElementType.Double: return Success(op, Tensor<double>.Slice((Tensor<double>)data, (Tensor<int>) starts, (Tensor<int>) ends, (Tensor<int>?) axes, (Tensor<int>?) steps));
            case TensorElementType.Float16: return Success(op, Tensor<Float16>.Slice((Tensor<Float16>)data, (Tensor<int>) starts, (Tensor<int>) ends, (Tensor<int>?) axes, (Tensor<int>?) steps));
            case TensorElementType.BFloat16: return Success(op, Tensor<BFloat16>.Slice((Tensor<BFloat16>)data, (Tensor<int>) starts, (Tensor<int>) ends, (Tensor<int>?) axes, (Tensor<int>?) steps));
            case TensorElementType.Complex64: return Success(op, Tensor<System.Numerics.Complex>.Slice((Tensor<System.Numerics.Complex>)data, (Tensor<int>) starts, (Tensor<int>) ends, (Tensor<int>?) axes, (Tensor<int>?) steps));
            default: return NotSupported(op);
        }
    }

    public static OpResult Split(ITensor? data, ITensor? split, int? _axis, int[]? _split, int? _numOutputs, ExecutionOptions? options, int? _outputCount)
    {
        var op = OpType.Split;
        if (data is null) return MissingInput(op, nameof(data));
        (options ?? ExecutionOptions.Default).Validated();
        int axis = _axis ?? 0;
        if (axis < 0) axis += data.Rank;
        if (axis < 0 || axis >= data.Rank) return WrongInputShape(op, nameof(data), data, "Split axis is out of range.");
        int dim = data.Dims[axis];
        // The split input must be a rank-1 vector (a scalar is rejected by the
        // native engine) and is mutually exclusive with num_outputs, even when
        // wired but empty. An empty split vector or attribute means "unspecified".
        if (split is not null && split.Rank != 1) return WrongInputShape(op, "split", split, "The split tensor must be a rank-1 vector tensor.");
        if (split is not null && _numOutputs.HasValue) return Failure(op, "If 'num_outputs' is specified, the 'split' input should not be provided.");
        int[] sizes;
        if (split is not null && split.Length > 0)
        {
            if (split.ElementType == TensorElementType.Int64) sizes = ((Tensor<long>)split).ToArray().Select(v => checked((int)v)).ToArray();
            else if (split.ElementType == TensorElementType.Int32) sizes = ((Tensor<int>)split).ToArray();
            else return InputTypeNotSupported(op, "split", split);
        }
        else if (_numOutputs.HasValue)
        {
            // Split-18 distribution: ceil-sized leading outputs, remainder in the
            // last one (verified against ORT 1.29: 5/2 -> [3,2], 5/3 -> [2,2,1]).
            // A non-positive last part fails the sum check below, as natively.
            int n = _numOutputs.Value;
            if (n <= 0) return WrongInputShape(op, "split", data, "Split num_outputs must be positive.");
            int s = (dim + n - 1) / n;
            sizes = new int[n];
            for (int i = 0; i < n - 1; i++) sizes[i] = s;
            int last = dim - s * (n - 1);
            sizes[n - 1] = last > 0 ? last : s;
        }
        else if (_split is not null && _split.Length > 0) sizes = _split;
        else if (_outputCount.HasValue && _outputCount.Value > 0 && dim % _outputCount.Value == 0) sizes = Enumerable.Repeat(dim / _outputCount.Value, _outputCount.Value).ToArray();
        else return MissingAttribute(op, "split", "Split needs the split input, the split attribute, num_outputs, or an evenly divisible node output count.");
        if (sizes.Any(z => z < 0) || sizes.Sum() != dim) return WrongInputShape(op, "split", data, "Split sizes must be non-negative and sum to the axis dimension.");
        var inDims = data.Dims.ToArray();
        int inner = 1;
        for (int i = axis + 1; i < data.Rank; i++) inner *= inDims[i];
        int outer = 1;
        for (int i = 0; i < axis; i++) outer *= inDims[i];
            static DenseTensor<T> SplitPart<T>(System.ReadOnlySpan<T> src, int[] partDims, int outer, int inner, int dim, int start, int size) where T : unmanaged
            {
                var dst = DenseTensor<T>.OfShape(partDims);
                ArrayUtilities.CopyAxisChunks(src, dim, dst.Buffer.Span, size, outer, inner, start, 0, size);
                return dst;
            }
        var outputs = new ITensor[sizes.Length];
        switch (data.ElementType)
        {
            case TensorElementType.Float:
            {
                var dd = ((Tensor<float>)data).ToDenseTensor();
                var span = dd.Buffer.Span;
                int start = 0;
                for (int p = 0; p < sizes.Length; p++)
                {
                    var partDims = (int[])inDims.Clone();
                    partDims[axis] = sizes[p];
                    outputs[p] = SplitPart(span, partDims, outer, inner, dim, start, sizes[p]);
                    start += sizes[p];
                }
                break;
            }
            case TensorElementType.Double:
            {
                var dd = ((Tensor<double>)data).ToDenseTensor();
                var span = dd.Buffer.Span;
                int start = 0;
                for (int p = 0; p < sizes.Length; p++)
                {
                    var partDims = (int[])inDims.Clone();
                    partDims[axis] = sizes[p];
                    outputs[p] = SplitPart(span, partDims, outer, inner, dim, start, sizes[p]);
                    start += sizes[p];
                }
                break;
            }
            case TensorElementType.Int32:
            {
                var dd = ((Tensor<int>)data).ToDenseTensor();
                var span = dd.Buffer.Span;
                int start = 0;
                for (int p = 0; p < sizes.Length; p++)
                {
                    var partDims = (int[])inDims.Clone();
                    partDims[axis] = sizes[p];
                    outputs[p] = SplitPart(span, partDims, outer, inner, dim, start, sizes[p]);
                    start += sizes[p];
                }
                break;
            }
            case TensorElementType.Int64:
            {
                var dd = ((Tensor<long>)data).ToDenseTensor();
                var span = dd.Buffer.Span;
                int start = 0;
                for (int p = 0; p < sizes.Length; p++)
                {
                    var partDims = (int[])inDims.Clone();
                    partDims[axis] = sizes[p];
                    outputs[p] = SplitPart(span, partDims, outer, inner, dim, start, sizes[p]);
                    start += sizes[p];
                }
                break;
            }
            default: return InputTypeNotSupported(op, nameof(data), data);
        }
        return Success(op, outputs);
    }

    public static OpResult Equal(ITensor? A, ITensor? B, ExecutionOptions? options)
    {
        var op = OpType.Equal;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        (options ?? ExecutionOptions.Default).Validated();
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }
        switch (A.ElementType)
        {
            case TensorElementType.Bool: return Success(op, Tensor<bool>.Equal((Tensor<bool>)A, (Tensor<bool>)B));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Equal((Tensor<int>)A, (Tensor<int>)B));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Equal((Tensor<long>)A, (Tensor<long>)B));
            case TensorElementType.Float: return Success(op, Tensor<float>.Equal((Tensor<float>)A, (Tensor<float>)B));
            case TensorElementType.Double: return Success(op, Tensor<double>.Equal((Tensor<double>)A, (Tensor<double>)B));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Less(ITensor? A, ITensor? B, ExecutionOptions? options)
    {
        var op = OpType.Less;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        (options ?? ExecutionOptions.Default).Validated();
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }
        switch (A.ElementType)
        {
            case TensorElementType.Int32: return Success(op, Tensor<int>.Less((Tensor<int>)A, (Tensor<int>)B));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Less((Tensor<long>)A, (Tensor<long>)B));
            case TensorElementType.Float: return Success(op, Tensor<float>.Less((Tensor<float>)A, (Tensor<float>)B));
            case TensorElementType.Double: return Success(op, Tensor<double>.Less((Tensor<double>)A, (Tensor<double>)B));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Where(ITensor? condition, ITensor? X, ITensor? Y, ExecutionOptions? options)
    {
        var op = OpType.Where;
        if (condition is null) return MissingInput(op, nameof(condition));
        if (X is null) return MissingInput(op, nameof(X));
        if (Y is null) return MissingInput(op, nameof(Y));
        (options ?? ExecutionOptions.Default).Validated();
        if (condition.ElementType != TensorElementType.Bool) return WrongInputType(op, nameof(condition), TensorElementType.Bool, condition);
        if (X.ElementType != Y.ElementType)
        {
            return WrongInputType(op, nameof(Y), "Input tensors must be of the same type.", Y);
        }
        switch (X.ElementType)
        {
            case TensorElementType.Bool: return Success(op, Tensor<bool>.Where((Tensor<bool>)condition, (Tensor<bool>)X, (Tensor<bool>)Y));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Where((Tensor<bool>)condition, (Tensor<int>)X, (Tensor<int>)Y));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Where((Tensor<bool>)condition, (Tensor<long>)X, (Tensor<long>)Y));
            case TensorElementType.Float: return Success(op, Tensor<float>.Where((Tensor<bool>)condition, (Tensor<float>)X, (Tensor<float>)Y));
            case TensorElementType.Double: return Success(op, Tensor<double>.Where((Tensor<bool>)condition, (Tensor<double>)X, (Tensor<double>)Y));
            default: return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult Expand(ITensor? data, ITensor? shape, ExecutionOptions? options)
    {
        var op = OpType.Expand;
        if (data is null) return MissingInput(op, nameof(data));
        if (shape is null) return MissingInput(op, nameof(shape));
        (options ?? ExecutionOptions.Default).Validated();
        if (shape.ElementType == TensorElementType.Int64)
        {
            shape = shape.ConvertToInt32();
        }
        if (shape.ElementType != TensorElementType.Int32) return WrongInputType(op, nameof(shape), TensorElementType.Int32, shape);

        var targetShape = ((Tensor<int>)shape).ToArray();
        switch (data.ElementType)
        {
            case TensorElementType.Bool: return Success(op, Tensor<bool>.Expand((Tensor<bool>)data, targetShape));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Expand((Tensor<int>)data, targetShape));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Expand((Tensor<long>)data, targetShape));
            case TensorElementType.Float: return Success(op, Tensor<float>.Expand((Tensor<float>)data, targetShape));
            case TensorElementType.Double: return Success(op, Tensor<double>.Expand((Tensor<double>)data, targetShape));
            default: return InputTypeNotSupported(op, nameof(data), data);
        }
    }

    public static OpResult Resize(ITensor? X, ITensor? roi, ITensor? scales, ITensor? sizes,
        string? mode, string? coordinateTransformationMode, string? nearestMode, float? cubicCoeffA, float? extrapolationValue, ExecutionOptions? options)
    {
        var op = OpType.Resize;

        /// <summary>
        /// Derives Resize output sizes from per-axis scales with floor semantics
        /// (verified against ORT 1.29: 5 * 1.5 scales to 7). Returns null for
        /// non-finite scales or negative results.
        /// </summary>
        int[]? ResizeSizesFromScales(int[] dims, double[] scales)
        {
            var sizes = new int[dims.Length];
            for (int i = 0; i < dims.Length; i++)
            {
                double v = dims[i] * scales[i];
                if (double.IsNaN(v) || double.IsInfinity(v) || v < 0.0) return null;
                double f = Math.Floor(v);
                if (f > int.MaxValue) return null;
                sizes[i] = (int)f;
            }
            return sizes;
        }
        if (X is null) return MissingInput(op, nameof(X));
        if (sizes is null && scales is null) return MissingInput(op, nameof(sizes));
        if (sizes is not null && sizes.ElementType == TensorElementType.Int64)
        {
            sizes = sizes.ConvertToInt32();
        }
        if (sizes is not null && sizes.ElementType != TensorElementType.Int32)
        {
            return WrongInputType(op, nameof(sizes), TensorElementType.Int32, sizes);
        }

        int[]? targetSizes;
        double[]? trueScales = null;
        if (sizes is not null)
        {
            targetSizes = ((Tensor<int>)sizes).ToArray();
        }
        else
        {
            if (scales is null) return MissingInput(op, nameof(scales));
            if (scales.ElementType != TensorElementType.Float && scales.ElementType != TensorElementType.Double)
            {
                return WrongInputType(op, nameof(scales), "Scales must be float or double.", scales);
            }
            var dims = X.Dims;
            var scaleArray = scales.ElementType == TensorElementType.Float
                ? ((Tensor<float>)scales).ToArray().Select(s => (double)s).ToArray()
                : ((Tensor<double>)scales).ToArray();
            if (scaleArray.Length != dims.Length)
            {
                return WrongInputShape(op, nameof(scales), dims.Length, scales);
            }
            targetSizes = ResizeSizesFromScales(dims.ToArray(), scaleArray);
            if (targetSizes is null) return WrongInputShape(op, nameof(scales), scales, "Resize scales must be finite and produce non-negative output sizes.");
            trueScales = scaleArray;
        }
        // Boundary validation: only the resamplers 4D-NCHW scope is supported,
        // with N/C unchanged (N/C tiling is out of scope); ROI is accepted and
        // ignored for the supported coordinate modes and extrapolation_value is
        // unused in range (both verified against ORT 1.29).
        if (X.Rank != 4) return WrongInputShape(op, nameof(X), X, "Resize currently supports only 4D tensors (NCHW).");
        if (targetSizes.Length != 4) return WrongInputShape(op, nameof(sizes), X, "Resize sizes must have one entry per input dimension.");
        if (targetSizes[0] != X.Dims[0] || targetSizes[1] != X.Dims[1]) return WrongInputShape(op, nameof(sizes), X, "Resize currently requires N and C dimensions to remain unchanged.");
        if (targetSizes.Any(z => z < 0)) return WrongInputShape(op, nameof(sizes), X, "Resize sizes must be non-negative.");

        MathOps.ResizeMode resizeMode;
        switch (mode ?? "nearest")
        {
            case "nearest": resizeMode = MathOps.ResizeMode.Nearest; break;
            case "linear": resizeMode = MathOps.ResizeMode.Linear; break;
            case "cubic": resizeMode = MathOps.ResizeMode.Cubic; break;
            default: return Failure(op, $"Resize mode {mode} is not supported.");
        }
        MathOps.ResizeCoordinateTransformation ctm;
        switch (coordinateTransformationMode ?? "half_pixel")
        {
            case "half_pixel": ctm = MathOps.ResizeCoordinateTransformation.HalfPixel; break;
            case "align_corners": ctm = MathOps.ResizeCoordinateTransformation.AlignCorners; break;
            case "asymmetric": ctm = MathOps.ResizeCoordinateTransformation.Asymmetric; break;
            default: return Failure(op, $"Resize coordinate_transformation_mode {coordinateTransformationMode} is not supported.");
        }
        var nm = MathOps.ResizeNearestMode.RoundPreferFloor;
        if (resizeMode == MathOps.ResizeMode.Nearest)
        {
            switch (nearestMode ?? "round_prefer_floor")
            {
                case "floor": nm = MathOps.ResizeNearestMode.Floor; break;
                case "ceil": nm = MathOps.ResizeNearestMode.Ceil; break;
                case "round_prefer_floor": nm = MathOps.ResizeNearestMode.RoundPreferFloor; break;
                case "round_prefer_ceil": nm = MathOps.ResizeNearestMode.RoundPreferCeil; break;
                default: return Failure(op, $"Resize nearest_mode {nearestMode} is not supported.");
            }
        }
        var cubicA = cubicCoeffA ?? -0.75f;

        switch (X.ElementType)
        {
            case TensorElementType.Float:
                return Success(op, Tensor<float>.Resize((Tensor<float>)X, targetSizes, resizeMode, ctm, nm, cubicA, trueScales));
            case TensorElementType.Double:
                return Success(op, Tensor<double>.Resize((Tensor<double>)X, targetSizes, resizeMode, ctm, nm, cubicA, trueScales));
            default:
                return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult Unsqueeze(ITensor? data, ITensor? axes, ExecutionOptions? options)
    {
        var op = OpType.Unsqueeze;
        if (data is null) return MissingInput(op, nameof(data));
        if (axes is null) return MissingInput(op, nameof(axes));
        (options ?? ExecutionOptions.Default).Validated();
        if (axes.ElementType == TensorElementType.Int64)
        {
            axes = ToInt32Saturating(axes);
        }
        var _axes = ((Tensor<int>) axes).ToArray();
        return Success(op, data.Unsqueeze(_axes));
    }

    public static OpResult Unsqueeze(ITensor? data, int[] axes, ExecutionOptions? options)
    {
        var op = OpType.Unsqueeze;
        if (data is null) return MissingInput(op, nameof(data));
        if (axes is null) return MissingInput(op, nameof(axes));
        (options ?? ExecutionOptions.Default).Validated();
        return Success(op, data.Unsqueeze(axes));
    }

    public static OpResult ReduceSum(ITensor? data, ITensor? axes, int? _keep_dims, int? noop_with_empty_axes, ExecutionOptions? options)
    {
        var op = OpType.ReduceSum;
        if (data is null) return MissingInput(op, nameof(data));
        (options ?? ExecutionOptions.Default).Validated();
        if (axes is not null && axes.ElementType == TensorElementType.Int64)
        {
            axes = axes.Cast<int>();
        }
        var keepDims = _keep_dims.HasValue ? Convert.ToBoolean(_keep_dims.Value) : true;
        var noopWithEmptyAxes = noop_with_empty_axes.HasValue ? Convert.ToBoolean(noop_with_empty_axes.Value) : false;
        switch (data.ElementType)
        {
            case TensorElementType.Int32: return Success(op, Tensor<int>.ReduceSum((Tensor<int>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes));
            case TensorElementType.Float: return Success(op, Tensor<float>.ReduceSum((Tensor<float>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes));
            case TensorElementType.Double: return Success(op, Tensor<double>.ReduceSum((Tensor<double>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes));
            default: return NotSupported(op);
        }
    }

    public static OpResult ReduceMean(ITensor? data, ITensor? axes, int? _keep_dims, int? noop_with_empty_axes, ExecutionOptions? options)
    {
        var op = OpType.ReduceMean;
        if (data is null) return MissingInput(op, nameof(data));
        if (axes is not null && axes.Rank != 1) return WrongInputShape(op, nameof(axes), 1, axes);
        if (axes is not null && axes.ElementType == TensorElementType.Int64)
        {
            axes = axes.Cast<int>();
        }
        var keepDims = _keep_dims.HasValue ? Convert.ToBoolean(_keep_dims.Value) : true;
        var noopWithEmptyAxes = noop_with_empty_axes.HasValue ? Convert.ToBoolean(noop_with_empty_axes.Value) : false;
        var opts = (options ?? ExecutionOptions.Default).Validated();
        switch (data.ElementType)
        {
            case TensorElementType.Int32: return Success(op, Tensor<int>.ReduceMean((Tensor<int>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes, opts.Tensor));
            case TensorElementType.Float: return Success(op, Tensor<float>.ReduceMean((Tensor<float>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes, opts.Tensor));
            case TensorElementType.Double: return Success(op, Tensor<double>.ReduceMean((Tensor<double>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes, opts.Tensor));
            default: return NotSupported(op);
        }
    }

    public static OpResult ReduceMax(ITensor? data, ITensor? axes, int? _keep_dims, int? noop_with_empty_axes, ExecutionOptions? options)
    {
        var op = OpType.ReduceMax;
        if (data is null) return MissingInput(op, nameof(data));
        if (axes is not null && axes.Rank != 1) return WrongInputShape(op, nameof(axes), 1, axes);
        (options ?? ExecutionOptions.Default).Validated();
        if (axes is not null && axes.ElementType == TensorElementType.Int64)
        {
            axes = axes.Cast<int>();
        }
        var keepDims = _keep_dims.HasValue ? Convert.ToBoolean(_keep_dims.Value) : true;
        var noopWithEmptyAxes = noop_with_empty_axes.HasValue ? Convert.ToBoolean(noop_with_empty_axes.Value) : false;
        switch (data.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.ReduceMax((Tensor<float>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes));
            case TensorElementType.Double: return Success(op, Tensor<double>.ReduceMax((Tensor<double>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes));
            default: return NotSupported(op);
        }
    }

    public static OpResult Softmax(ITensor? input, int? _axis, ExecutionOptions? options, TensorBufferPool? pool, int opsetVersion)
    {
        var op = OpType.Softmax;
        if (input is null) return MissingInput(op, nameof(input));
        var axis = _axis.HasValue ? _axis.Value : (opsetVersion < 13 ? 1 : -1);
        var opts = (options ?? ExecutionOptions.Default).Validated();
        var tensorOptions = opts.Tensor;
        if (opts.Optimization == OptimizationMode.Speed)
        {
            input = input.ToDenseTensor();
        }
        switch (input.ElementType)
        {
            case TensorElementType.Float:
            {
                var fx = (Tensor<float>)input;
                if (pool is null) return Success(op, Tensor<float>.Softmax(fx, axis, tensorOptions, opsetVersion));
                var rented = new DenseTensor<float>(new Memory<float>(pool.Rent<float>((int)fx.Length)), fx.Dimensions.ToArray());
                return Success(op, Tensor<float>.Softmax(fx, rented, axis, tensorOptions, opsetVersion));
            }
            case TensorElementType.Double: return Success(op, Tensor<double>.Softmax((Tensor<double>) input, axis, tensorOptions, opsetVersion));
            default: return InputTypeNotSupported(op, nameof(input), input);
        }
    }


    static Tensor<int> ToInt32Saturating(ITensor data)
    {
        if (data is Tensor<int> i) return i;
        if (data is Tensor<long> l)
        {
            var values = l.ToArray();
            var clamped = new int[values.Length];
            for (int k = 0; k < values.Length; k++) clamped[k] = values[k] > int.MaxValue ? int.MaxValue : values[k] < int.MinValue ? int.MinValue : (int)values[k];
            return DenseTensor<int>.OfValues(clamped);
        }
        throw new ArgumentException("Expected int32/int64 tensor.");
    }

    static int[] ToIntArray(ITensor data, string name)
    {
        if (data is Tensor<long> l) return l.ToArray().Select(v => (int)v).ToArray();
        if (data is Tensor<int> i) return i.ToArray();
        throw new ArgumentException($"Expected int32/int64 tensor for {name}.");
    }

    static long ToInt64Scalar(ITensor data, string name)
    {
        if (data is Tensor<long> l) return l.ToArray()[0];
        if (data is Tensor<int> i) return i.ToArray()[0];
        throw new ArgumentException($"Expected int32/int64 scalar tensor for {name}.");
    }

    public static OpResult Abs(ITensor? X, ExecutionOptions? options)
    {
        var op = OpType.Abs;
        if (X is null) return MissingInput(op, nameof(X));
        (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Math);
        switch (X.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Abs((Tensor<float>)X));
            case TensorElementType.Double: return Success(op, Tensor<double>.Abs((Tensor<double>)X));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Abs((Tensor<int>)X));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Abs((Tensor<long>)X));
            default: return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult Cos(ITensor? X, ExecutionOptions? options)
    {
        var op = OpType.Cos;
        if (X is null) return MissingInput(op, nameof(X));
        Profiler.StartOpStage(OpStage.Math);
        var opts = (options ?? ExecutionOptions.Default).Validated();
        switch (X.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Cos((Tensor<float>)X, opts.Tensor));
            case TensorElementType.Double: return Success(op, Tensor<double>.Cos((Tensor<double>)X, opts.Tensor));
            default: return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult Sin(ITensor? X, ExecutionOptions? options)
    {
        var op = OpType.Sin;
        if (X is null) return MissingInput(op, nameof(X));
        Profiler.StartOpStage(OpStage.Math);
        var opts = (options ?? ExecutionOptions.Default).Validated();
        switch (X.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Sin((Tensor<float>)X, opts.Tensor));
            case TensorElementType.Double: return Success(op, Tensor<double>.Sin((Tensor<double>)X, opts.Tensor));
            default: return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    /// <summary>Hyperbolic tangent. This operator runs a fixed scalar path and ignores execution modes.</summary>
    public static OpResult Tanh(ITensor? X, ExecutionOptions? options)
    {
        var op = OpType.Tanh;
        if (X is null) return MissingInput(op, nameof(X));
        (options ?? ExecutionOptions.Default).Validated();
        switch (X.ElementType)
        {
            case TensorElementType.Float:
            {
                var x = ((Tensor<float>)X).ToDenseTensor();
                var y = DenseTensor<float>.OfShape(x.Dimensions.ToArray());
                var xs = x.Buffer.Span;
                var ys = y.Buffer.Span;
                for (int i = 0; i < xs.Length; i++) ys[i] = MathF.Tanh(xs[i]);
                return Success(op, y);
            }
            case TensorElementType.Double:
            {
                var x = ((Tensor<double>)X).ToDenseTensor();
                var y = DenseTensor<double>.OfShape(x.Dimensions.ToArray());
                var xs = x.Buffer.Span;
                var ys = y.Buffer.Span;
                for (int i = 0; i < xs.Length; i++) ys[i] = Math.Tanh(xs[i]);
                return Success(op, y);
            }
            default: return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult Neg(ITensor? X, ExecutionOptions? options)
    {
        var op = OpType.Neg;
        if (X is null) return MissingInput(op, nameof(X));
        Profiler.StartOpStage(OpStage.Math);
        var opts = (options ?? ExecutionOptions.Default).Validated();
        switch (X.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Negate((Tensor<float>)X, opts.Tensor));
            case TensorElementType.Double: return Success(op, Tensor<double>.Negate((Tensor<double>)X, opts.Tensor));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Negate((Tensor<int>)X, opts.Tensor));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Negate((Tensor<long>)X, opts.Tensor));
            default: return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult Gelu(ITensor? X, string? approximate, ExecutionOptions? options, TensorBufferPool? pool)
    {
        var op = OpType.Gelu;
        if (X is null) return MissingInput(op, nameof(X));
        if (approximate is not null && approximate != "none") return AttributeNotSupported(op, nameof(approximate), approximate, null);
        Profiler.StartOpStage(OpStage.Math);
        var opts = (options ?? ExecutionOptions.Default).Validated();
        switch (X.ElementType)
        {
            case TensorElementType.Float:
            {
                var tensorOptions = opts.Tensor;
                var fx = (Tensor<float>)X;
                if (pool is null) return Success(op, Tensor<float>.Gelu(fx, tensorOptions));
                var rented = new DenseTensor<float>(new Memory<float>(pool.Rent<float>((int)fx.Length)), fx.Dimensions.ToArray());
                return Success(op, Tensor<float>.Gelu(fx, rented, tensorOptions));
            }
            case TensorElementType.Double: return Success(op, Tensor<double>.Gelu((Tensor<double>)X, opts.Tensor));
            default: return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    /// <summary>
    /// Removes size-1 dimensions; a null axes removes every size-1 dimension.
    /// Axes index the input tensor; squeezing a dimension whose size is not 1 fails.
    /// </summary>
    public static OpResult Squeeze(ITensor? data, ITensor? axes, ExecutionOptions? options)
    {
        var op = OpType.Squeeze;
        if (data is null) return MissingInput(op, nameof(data));
        (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Math);
        int[] dims = data.Dims.ToArray();
        // Absent or explicitly empty axes squeeze every size-1 dimension; a
        // scalar axes tensor is rejected (verified against ORT 1.29).
        int[]? ax = null;
        if (axes is not null)
        {
            if (axes.Rank != 1) return WrongInputShape(op, nameof(axes), axes, "The axes tensor must be a rank-1 vector tensor.");
            ax = ToIntArray(axes, nameof(axes)).Select(a => a < 0 ? a + data.Rank : a).ToArray();
        }
        if (ax is null || ax.Length == 0)
        {
            dims = dims.Where(d => d != 1).ToArray();
        }
        else
        {
            foreach (var a in ax)
            {
                if (a < 0 || a >= data.Rank) return WrongInputShape(op, nameof(axes), data, $"Axis {a} is out of range.");
                if (dims[a] != 1) return WrongInputShape(op, nameof(axes), data, $"Axis {a} has size {dims[a]}, only size-1 dimensions can be squeezed.");
            }
            dims = dims.Where((d, i) => !ax.Contains(i)).ToArray();
        }
        return Success(op, data.Reshape(dims).ToDenseTensor());
    }

    /// <summary>
    /// Dispatches scalar start, limit, and delta to the matching dtype Range kernel.
    /// </summary>
    public static OpResult Range(ITensor? start, ITensor? limit, ITensor? delta, ExecutionOptions? options)
    {
        var op = OpType.Range;
        if (start is null) return MissingInput(op, nameof(start));
        if (limit is null) return MissingInput(op, nameof(limit));
        if (delta is null) return MissingInput(op, nameof(delta));
        (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Math);
        switch (start.ElementType)
        {
            case TensorElementType.Float:
                return Success(op, Tensor<float>.Range(Convert.ToSingle(start.GetValue(0)), Convert.ToSingle(limit.GetValue(0)), Convert.ToSingle(delta.GetValue(0))));
            case TensorElementType.Double:
                return Success(op, Tensor<double>.Range(Convert.ToDouble(start.GetValue(0)), Convert.ToDouble(limit.GetValue(0)), Convert.ToDouble(delta.GetValue(0))));
            case TensorElementType.Int64:
                return Success(op, Tensor<long>.Range(Convert.ToInt64(start.GetValue(0)), Convert.ToInt64(limit.GetValue(0)), Convert.ToInt64(delta.GetValue(0))));
            case TensorElementType.Int32:
                return Success(op, Tensor<int>.Range(Convert.ToInt32(start.GetValue(0)), Convert.ToInt32(limit.GetValue(0)), Convert.ToInt32(delta.GetValue(0))));
            default: return InputTypeNotSupported(op, nameof(start), start);
        }
    }

    /// <summary>
    /// Dispatches data and int32/int64 repeats to the matching dtype Tile kernel.
    /// </summary>
    public static OpResult Tile(ITensor? data, ITensor? repeats, ExecutionOptions? options)
    {
        var op = OpType.Tile;
        if (data is null) return MissingInput(op, nameof(data));
        if (repeats is null) return MissingInput(op, nameof(repeats));
        (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Math);
        var reps = ToIntArray(repeats, nameof(repeats));
        switch (data.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Tile((Tensor<float>)data, reps));
            case TensorElementType.Double: return Success(op, Tensor<double>.Tile((Tensor<double>)data, reps));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Tile((Tensor<int>)data, reps));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Tile((Tensor<long>)data, reps));
            default: return InputTypeNotSupported(op, nameof(data), data);
        }
    }

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
    public static OpResult LayerNormalization(ITensor? x, ITensor? scale, ITensor? bias, int? axis, float? epsilon, ExecutionOptions? options, TensorBufferPool? pool, Dictionary<string, object>? attributes)
    {
        var op = OpType.LayerNormalization;
        if (x is null) return MissingInput(op, nameof(x));
        if (scale is null) return MissingInput(op, nameof(scale));
        if (attributes is not null && attributes.ContainsKey("stash_type")) return AttributeNotSupported(op, "stash_type", "stash_type output selection is not supported.", null);
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
                if (pool is null) return Success(op, Tensor<float>.LayerNormalization(fx, (Tensor<float>)scale, fb, ax, eps));
                var rented = new DenseTensor<float>(new Memory<float>(pool.Rent<float>((int)fx.Length)), fx.Dimensions.ToArray());
                return Success(op, Tensor<float>.LayerNormalization(fx, (Tensor<float>)scale, fb, rented, ax, eps));
            }
            case TensorElementType.Double:
                if (scale.ElementType != TensorElementType.Double) return WrongInputType(op, nameof(scale), TensorElementType.Double, scale);
                if (bias is not null && bias.ElementType != TensorElementType.Double) return WrongInputType(op, nameof(bias), TensorElementType.Double, bias);
                return Success(op, Tensor<double>.LayerNormalization((Tensor<double>)x, (Tensor<double>)scale, (Tensor<double>?)bias, ax, eps));
            default: return InputTypeNotSupported(op, nameof(x), x);
        }
    }

    /// <summary>
    /// Splits the input along the axis into the given sizes and returns a TensorSequence.
    /// Keepdims 1 preserves the rank; keepdims 0 drops the split axis and needs every size to be 1.
    /// </summary>
    public static OpResult SplitToSequence(ITensor? input, ITensor? split, int? axis, int? keepdims, ExecutionOptions? options)
    {
        var op = OpType.SplitToSequence;
        if (input is null) return MissingInput(op, nameof(input));
        (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Math);
        int rank = input.Rank;
        int ax = axis.HasValue ? (axis.Value < 0 ? axis.Value + rank : axis.Value) : 0;
        if (ax < 0 || ax >= rank) return WrongInputShape(op, nameof(axis), input, $"Axis {axis} is out of range.");
        int dim = input.Dims[ax];
        bool keep = (keepdims ?? 1) == 1;
        // Absent split behaves as scalar chunk size 1 (verified against ORT 1.29:
        // one size-1 piece per axis element). A scalar split is a chunk size
        // (last chunk may be partial); a vector split lists explicit sizes and
        // keeps the axis regardless of keepdims.
        int[] sizes;
        bool squeezePieces = false;
        if (split is null || split.Rank == 0)
        {
            long chunk = split is null ? 1L : ToInt64Scalar(split, nameof(split));
            if (chunk <= 0L) return WrongInputShape(op, nameof(split), input, "Split chunk size must be positive.");
            var parts = new List<int>();
            long rem = dim;
            while (rem > 0L) { int take = rem < chunk ? (int)rem : (int)chunk; parts.Add(take); rem -= take; }
            if (parts.Count == 0) parts.Add(0);
            sizes = parts.ToArray();
            squeezePieces = !keep;
        }
        else if (split.Rank == 1)
        {
            sizes = ToIntArray(split, nameof(split));
            if (sizes.Any(z => z < 0)) return WrongInputShape(op, nameof(split), input, "Split sizes must be non-negative.");
            if (sizes.Sum() != dim) return WrongInputShape(op, nameof(split), input, "Split sizes must sum to the split dimension.");
        }
        else return WrongInputShape(op, nameof(split), split, "The split tensor must be a scalar or a rank-1 vector tensor.");
        // Densify once up front: every chunk below then shares this input
        // instead of copying a non-dense source per piece.
        ITensor? dense = input.ElementType switch
        {
            TensorElementType.Float => ((Tensor<float>)input).ToDenseTensor(),
            TensorElementType.Double => ((Tensor<double>)input).ToDenseTensor(),
            TensorElementType.Int32 => ((Tensor<int>)input).ToDenseTensor(),
            TensorElementType.Int64 => ((Tensor<long>)input).ToDenseTensor(),
            _ => null,
        };
        if (dense is null) return InputTypeNotSupported(op, nameof(input), input);
        var items = new List<ITensor>();
        int start = 0;
        foreach (var length in sizes)
        {
            if (start + length > dim) return WrongInputShape(op, nameof(split), input, "Split sizes exceed the split dimension.");
            ITensor? chunk = dense.ElementType switch
            {
                TensorElementType.Float => Tensor<float>.ChunkCopy((Tensor<float>)dense, ax, start, length),
                TensorElementType.Double => Tensor<double>.ChunkCopy((Tensor<double>)dense, ax, start, length),
                TensorElementType.Int32 => Tensor<int>.ChunkCopy((Tensor<int>)dense, ax, start, length),
                TensorElementType.Int64 => Tensor<long>.ChunkCopy((Tensor<long>)dense, ax, start, length),
                _ => null,
            };
            if (chunk is null) return InputTypeNotSupported(op, nameof(input), input);
            if (squeezePieces)
            {
                // Dropping a non-singleton axis cannot preserve the element
                // count, so the native engine fails there too (ORT 1.29).
                if (chunk.Dims[ax] != 1) return WrongInputShape(op, nameof(split), input, "Cannot drop a non-singleton split axis with keepdims=0.");
                chunk = chunk.Reshape(chunk.Dims.Where((d, i) => i != ax).ToArray()).ToDenseTensor();
            }
            items.Add(chunk);
            start += length;
        }
        return Success(op, new TensorSequence(items));
    }

    /// <summary>
    /// Returns the sequence element at the scalar index; negative indices count from the end.
    /// </summary>
    public static OpResult SequenceAt(ITensor? sequence, ITensor? index, ExecutionOptions? options)
    {
        var op = OpType.SequenceAt;
        if (sequence is null) return MissingInput(op, nameof(sequence));
        if (index is null) return MissingInput(op, nameof(index));
        (options ?? ExecutionOptions.Default).Validated();
        if (sequence is not TensorSequence seq) return WrongInputType(op, nameof(sequence), "Input must be a sequence.", sequence);
        Profiler.StartOpStage(OpStage.Math);
        long position = ToInt64Scalar(index, nameof(index));
        if (position < 0) position += seq.Items.Count;
        if (position < 0 || position >= seq.Items.Count) return WrongInputShape(op, nameof(index), index, "Sequence index is out of range.");
        return Success(op, seq.Items[(int)position]);
    }
}




