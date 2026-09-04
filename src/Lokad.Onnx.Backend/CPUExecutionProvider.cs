namespace Lokad.Onnx;

extern alias OnnxSharp;

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

public class CPUExecutionProvider : Runtime
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
    };

    public static OptimizationMode OptimizationMode { get; set; } = OptimizationMode.Speed;

    public static bool SupportsOp(OpType op) => SupportedOps.Contains(op);

    public static OpResult Reshape(ITensor? input, ITensor? shape, bool? allow_zero = null, ExecutionOptions? options = null)
    {
        var op = OpType.Reshape;
        if (input is null) return MissingInput(op, nameof(input));
        if (shape is null) return MissingInput(op, nameof(shape));
        if (shape.ElementType != TensorElementType.Int64) return WrongInputType(op, nameof(shape), TensorElementType.Int64, shape);
        switch (input.ElementType)
        {
            case TensorElementType.Bool: return Success(op, Tensor<bool>.Reshape((Tensor<bool>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.Int8: return Success(op, Tensor<byte>.Reshape((Tensor<byte>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.UInt8: return Success(op, Tensor<sbyte>.Reshape((Tensor<sbyte>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.Int16: return Success(op, Tensor<short>.Reshape((Tensor<short>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.UInt16: return Success(op, Tensor<ushort>.Reshape((Tensor<ushort>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Reshape((Tensor<int>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.UInt32: return Success(op, Tensor<uint>.Reshape((Tensor<uint>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Reshape((Tensor<long>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.UInt64: return Success(op, Tensor<ulong>.Reshape((Tensor<ulong>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.Float: return Success(op, Tensor<float>.Reshape((Tensor<float>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.Double: return Success(op, Tensor<double>.Reshape((Tensor<double>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.Float16: return Success(op, Tensor<Half>.Reshape((Tensor<Half>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.BFloat16: return Success(op, Tensor<BFloat16>.Reshape((Tensor<BFloat16>)input, (Tensor<long>)shape, allow_zero ?? false));
            case TensorElementType.Complex64: return Success(op, Tensor<System.Numerics.Complex>.Reshape((Tensor<System.Numerics.Complex>)input, (Tensor<long>)shape, allow_zero ?? false));
            default: return NotSupported(op);
        }
    }

    public static OpResult Add(ITensor? A, ITensor? B, ExecutionOptions? options = null)
    {
        var op = OpType.Add;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }

        Profiler.StartOpStage(OpStage.Broadcast);
        if (!ITensor.Broadcast(A, B, out var bA, out var bB))
        {
            return CannotBroadcast(op, A, B);
        }
        if ((options ?? ExecutionOptions.Default).Optimization == OptimizationMode.Speed)
        {
            Profiler.StartOpStage(OpStage.Copy);
            bA = bA.ToDenseTensor();
            bB = bB.ToDenseTensor();    
        }
        Profiler.StartOpStage(OpStage.Math);
        switch (A.ElementType)
        {
            case TensorElementType.UInt8: return Success(op, Tensor<byte>.Add((Tensor<byte>)bA, (Tensor<byte>)bB));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Add((Tensor<int>)bA, (Tensor<int>)bB));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Add((Tensor<long>)bA, (Tensor<long>)bB));
            case TensorElementType.Float: return Success(op, Tensor<float>.Add((Tensor<float>)bA, (Tensor<float>)bB));
            case TensorElementType.Double: return Success(op, Tensor<double>.Add((Tensor<double>)bA, (Tensor<double>)bB));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Sub(ITensor? A, ITensor? B, ExecutionOptions? options = null)
    {
        var op = OpType.Sub;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }
        Profiler.StartOpStage(OpStage.Broadcast);
        if (!ITensor.Broadcast(A, B, out var bA, out var bB))
        {
            return CannotBroadcast(op, A, B);
        }
        if ((options ?? ExecutionOptions.Default).Optimization == OptimizationMode.Speed)
        {
            Profiler.StartOpStage(OpStage.Copy);
            bA = bA.ToDenseTensor();
            bB = bB.ToDenseTensor();
        }
        Profiler.StartOpStage(OpStage.Math);
        switch (A.ElementType)
        {
            case TensorElementType.UInt8: return Success(op, Tensor<byte>.Subtract((Tensor<byte>)bA, (Tensor<byte>)bB));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Subtract((Tensor<int>)bA, (Tensor<int>)bB));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Subtract((Tensor<long>)bA, (Tensor<long>)bB));
            case TensorElementType.Float: return Success(op, Tensor<float>.Subtract((Tensor<float>)bA, (Tensor<float>)bB));
            case TensorElementType.Double: return Success(op, Tensor<double>.Subtract((Tensor<double>)bA, (Tensor<double>)bB));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Mul(ITensor? A, ITensor? B, ExecutionOptions? options = null)
    {
        var op = OpType.Mul;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }
        Profiler.StartOpStage(OpStage.Broadcast);
        if (!ITensor.Broadcast(A, B, out var bA, out var bB))
        {
            return CannotBroadcast(op, A, B);
        }

        if ((options ?? ExecutionOptions.Default).Optimization == OptimizationMode.Speed)
        {
            Profiler.StartOpStage(OpStage.Copy); 
            bA = bA.ToDenseTensor();
            bB = bB.ToDenseTensor();
        }
        Profiler.StartOpStage(OpStage.Math);     
        switch (A.ElementType)
        {
            case TensorElementType.UInt8: return Success(op, Tensor<byte>.Multiply((Tensor<byte>)bA, (Tensor<byte>)bB));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Multiply((Tensor<int>)bA, (Tensor<int>)bB));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Multiply((Tensor<long>)bA, (Tensor<long>)bB));
            case TensorElementType.Float: return Success(op, Tensor<float>.Multiply((Tensor<float>)bA, (Tensor<float>)bB));
            case TensorElementType.Double: return Success(op, Tensor<double>.Multiply((Tensor<double>)bA, (Tensor<double>)bB));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Div(ITensor? A, ITensor? B, ExecutionOptions? options = null)
    {
        var op = OpType.Div;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }
        Profiler.StartOpStage(OpStage.Broadcast);
        if (!ITensor.Broadcast(A, B, out var bA, out var bB))
        {
            return CannotBroadcast(op, A, B);
        }

        if ((options ?? ExecutionOptions.Default).Optimization == OptimizationMode.Speed)
        {
            Profiler.StartOpStage(OpStage.Copy);
            bA = bA.ToDenseTensor();
            bB = bB.ToDenseTensor();
        }
        Profiler.StartOpStage(OpStage.Math);
        switch (A.ElementType)
        {
            case TensorElementType.UInt8: return Success(op, Tensor<byte>.Divide((Tensor<byte>)bA, (Tensor<byte>)bB));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Divide((Tensor<int>)bA, (Tensor<int>)bB));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Divide((Tensor<long>)bA, (Tensor<long>)bB));
            case TensorElementType.Float: return Success(op, Tensor<float>.Divide((Tensor<float>)bA, (Tensor<float>)bB));
            case TensorElementType.Double: return Success(op, Tensor<double>.Divide((Tensor<double>)bA, (Tensor<double>)bB));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Pow(ITensor? A, ITensor? B, ExecutionOptions? options = null)
    {
        var op = OpType.Pow;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }
        Profiler.StartOpStage(OpStage.Broadcast);    
        if (!ITensor.Broadcast(A, B, out var bA, out var bB))
        {
            return CannotBroadcast(op, A, B);
        }

        if ((options ?? ExecutionOptions.Default).Optimization == OptimizationMode.Speed)
        {
            Profiler.StartOpStage(OpStage.Copy);
            bA = bA.ToDenseTensor();
            bB = bB.ToDenseTensor();
        }
        Profiler.StartOpStage(OpStage.Math);
        switch (A.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Pow((Tensor<float>)bA, (Tensor<float>)bB));
            case TensorElementType.Double: return Success(op, Tensor<double>.Pow((Tensor<double>)bA, (Tensor<double>)bB));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }
    public static OpResult Conv(ITensor? X, ITensor? W, ITensor? B, string? auto_pad = null, int[]? dilations = null, int? group = null, int[]? kernel_shape = null, int[]? pads = null, int[]? strides = null, ExecutionOptions? options = null)
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
        var padmode = MathOps.PadType.Valid;
        int? padvalue = null;
        if (!string.IsNullOrEmpty(auto_pad))
        {
            switch (auto_pad)
            {
                case "VALID":
                    padmode = MathOps.PadType.Valid;
                    break;
                case "SAME_UPPER":
                    padmode = MathOps.PadType.SameUpper;
                    break;
                case "SAME_LOWER":
                    padmode = MathOps.PadType.SameLower;
                    break;
                case "NOTSET":
                    padmode = MathOps.PadType.Value;
                    if (pads is null)
                    {
                        return MissingAttribute(op, nameof(pads), "When auto_pad is NOTSET pads must be specified");
                    }
                    else if (!pads.All(p => p == pads[0]))
                    {
                        return AttributeNotSupported(op, "pads", pads.Print(), "Asymmetric padding is not supported.");
                    }
                    padvalue = pads[0];
                    break;
            }
        }
        switch (X.ElementType)
        {
            case TensorElementType.Float:
                var bias = B is null ? null : (Tensor<float>)B;
                return Success(op, Tensor<float>.Conv2D((Tensor<float>)X, (Tensor<float>)W, group ?? 1, padmode, padvalue, bias, kernel_shape, strides, dilations));
            case TensorElementType.Double:
                var biasd = B is null ? null : (Tensor<double>)B;
                return Success(op, Tensor<double>.Conv2D((Tensor<double>)X, (Tensor<double>)W, group ?? 1, padmode, padvalue, biasd, kernel_shape, strides, dilations));
            default:
                return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult Relu(ITensor? X, ExecutionOptions? options = null)
    {
        var op = OpType.Relu;
        if (X is null) return MissingInput(op, nameof(X));
        if ((options ?? ExecutionOptions.Default).Optimization == OptimizationMode.Speed)
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

    public static OpResult MaxPool(ITensor? X, string? auto_pad = null, int? ceil_mode = null, int[]? dilations = null, int[]? kernel_shape = null, int[]? pads = null, int? storage_order = null, int[]? strides = null, ExecutionOptions? options = null)
    {
        var op = OpType.MaxPool;
        if (X is null) return MissingInput(op, nameof(X));
        if (X.Rank != 4)
        {
            return WrongInputShape(op, nameof(X), 4, X);
        }
        var padmode = MathOps.PadType.Valid;
        int? padvalue = null;
        if (!string.IsNullOrEmpty(auto_pad))
        {
            switch (auto_pad)
            {
                case "VALID":
                    padmode = MathOps.PadType.Valid;
                    break;
                case "SAME_UPPER":
                    padmode = MathOps.PadType.SameUpper;
                    break;
                case "SAME_LOWER":
                    padmode = MathOps.PadType.SameLower;
                    break;
                case "NOTSET":
                    padmode = MathOps.PadType.Value;
                    if (pads is null)
                    {
                        return MissingAttribute(op, nameof(pads), "When auto_pad is NOTSET pads must be specified");
                    }
                    else if (!pads.All(p => p == pads[0]))
                    {
                        return AttributeNotSupported(op, "pads", pads.Print(), "Asymmetric padding is not supported.");
                    }
                    padvalue = pads?[0] ?? 0;
                    break;
            }
        }
        switch (X.ElementType)
        {
            case TensorElementType.Float:
                return Success(op, Tensor<float>.MaxPool2D((Tensor<float>) X, kernel_shape, padmode, padvalue, strides, dilations));
            case TensorElementType.Double:
                return Success(op, Tensor<double>.MaxPool2D((Tensor<double>)X, kernel_shape, padmode, padvalue, strides, dilations));
            default:
                return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult MatMul(ITensor? A, ITensor? B, ExecutionOptions? options = null)
    {
        var op = OpType.MatMul;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));

        if ((options ?? ExecutionOptions.Default).Optimization == OptimizationMode.Speed)
        {
            Profiler.StartOpStage(OpStage.Copy);
            A = A.ToDenseTensor();
            B = B.ToDenseTensor();
        }
        switch (A.ElementType)
        {
            case TensorElementType.Int32: return Success(op, Tensor<int>.MatMul((Tensor<int>)A, (Tensor<int>)B));
            case TensorElementType.Float: return Success(op, Tensor<float>.MatMul((Tensor<float>)A, (Tensor<float>)B, (options ?? ExecutionOptions.Default).Tensor));
            case TensorElementType.Double: return Success(op, Tensor<double>.MatMul((Tensor<double>)A, (Tensor<double>)B));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Sqrt(ITensor? A, ExecutionOptions? options = null)
    {
        var op = OpType.Sqrt;
        if (A is null) return MissingInput(op, nameof(A));
        if ((options ?? ExecutionOptions.Default).Optimization == OptimizationMode.Speed)
        {
            Profiler.StartOpStage(OpStage.Copy);
            A = A.ToDenseTensor();
        }
        Profiler.StartOpStage(OpStage.Math);
        switch (A.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Sqrt((Tensor<float>)A));
            case TensorElementType.Double: return Success(op, Tensor<double>.Sqrt((Tensor<double>)A));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Erf(ITensor? X, ExecutionOptions? options = null)
    {
        var op = OpType.Erf;
        if (X is null) return MissingInput(op, nameof(X));
        Profiler.StartOpStage(OpStage.Math);
        switch (X.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Erf((Tensor<float>)X));
            case TensorElementType.Double: return Success(op, Tensor<double>.Erf((Tensor<double>)X));
            default: return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult Transpose(ITensor? data, int[]? perm = null, ExecutionOptions? options = null)
    {
        var op = OpType.Transpose;
        if (data is null) return MissingInput(op, nameof(data));
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
            case TensorElementType.Float: return Success(op, Tensor<float>.Transpose((Tensor<float>)data, perm));
            case TensorElementType.Double: return Success(op, Tensor<double>.Transpose((Tensor<double>)data, perm));
            case TensorElementType.Float16: return Success(op, Tensor<Half>.Transpose((Tensor<Half>)data, perm));
            case TensorElementType.BFloat16: return Success(op, Tensor<BFloat16>.Transpose((Tensor<BFloat16>)data, perm));
            case TensorElementType.Complex64: return Success(op, Tensor<System.Numerics.Complex>.Transpose((Tensor<System.Numerics.Complex>)data, perm));
            default: return NotSupported(op);
        }
    }

    public static OpResult Constant(object? value, ExecutionOptions? options = null)
    {
        var op = OpType.Constant;
        if (value is null) return MissingAttribute(op, nameof(value));
        Profiler.StartOpStage(OpStage.Copy);
        switch (value)
        {
            case ITensor t: return Success(op, t);
            case float f: return Success(op, DenseTensor<float>.Scalar(f));
            case float[] fa: return Success(op, DenseTensor<float>.OfValues(fa));
            case int i: return Success(op, DenseTensor<int>.Scalar(i));
            case int[] ia: return Success(op, DenseTensor<int>.OfValues(ia));
            default: return NotSupported(op);
        }
    }

    public static OpResult Cast(ITensor? input, long to, ExecutionOptions? options = null)
    {
        var op = OpType.Cast;
        if (input is null) return MissingInput(op, nameof(input));
        Profiler.StartOpStage(OpStage.Copy);
        var type = (TensorElementType)to;
        switch (type)
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
            //case TensorElementType.Float16: return Success(op, input.Cast<Half>());
            //case TensorElementType.BFloat16: return Success(op, input.Cast<BFloat16>());
            //case TensorElementType.Complex64: return Success(op, input.Cast<System.Numerics.Complex>());
            default: return AttributeNotSupported(op, "to", to.ToString());

        }
    }
    public static OpResult Concat(ITensor[]? inputs, int? _axis, ExecutionOptions? options = null)
    {
        var op = OpType.Concat;
        if (inputs is null) return MissingInput(op, nameof(inputs));
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
            case TensorElementType.Float16: return Success(op, Tensor<Half>.Concat(inputs.CastA<Tensor<Half>>(), axis));
            case TensorElementType.BFloat16: return Success(op, Tensor<BFloat16>.Concat(inputs.CastA<Tensor<BFloat16>>(), axis));
            case TensorElementType.Complex64: return Success(op, Tensor<System.Numerics.Complex>.Concat(inputs.CastA<Tensor<System.Numerics.Complex>>(), axis));
            default: return InputTypeNotSupported(op, "inputs", inputs[0]);
        }
    }

    public static OpResult Shape(ITensor? data, int? _start = null, int? _end = null, ExecutionOptions? options = null)
    {
        var op = OpType.Shape;
        if (data is null) return MissingInput(op, nameof(data));
        Profiler.StartOpStage(OpStage.CalculateIndices);
        var start = ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, _start.HasValue ? _start.Value : 0);
        var end = ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, _end.HasValue ? _end.Value : data.Rank);
        start = ArrayUtilities.Clamp(start, 0, data.Rank);
        end = ArrayUtilities.Clamp(end, 0, data.Rank);  
        var _shape = data.Dims.Convert<int, long>()[start..end];
        return Success(op, DenseTensor<long>.OfValues(_shape));
    }

    public static OpResult Gather(ITensor? data, ITensor? indices, int? axis = null, ExecutionOptions? options = null) 
    {
        var op = OpType.Gather;
        if (data is null) return MissingInput(op, nameof(data));
        if (indices is null) return MissingInput(op, nameof(indices));
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
            case TensorElementType.Float16: return Success(op, Tensor<Half>.Gather((Tensor<Half>)data, (Tensor<int>)  indices, axis));
            case TensorElementType.BFloat16: return Success(op, Tensor<BFloat16>.Gather((Tensor<BFloat16>)data, (Tensor<int>)  indices, axis));
            case TensorElementType.Complex64: return Success(op, Tensor<System.Numerics.Complex>.Gather((Tensor<System.Numerics.Complex>)data, (Tensor<int>)  indices, axis));
            default: return NotSupported(op);
        }
    }

    public static OpResult Slice(ITensor? data, ITensor? starts, ITensor? ends, ITensor? axes, ITensor? steps, ExecutionOptions? options = null)
    {
        var op = OpType.Slice;
        if (data is null) return MissingInput(op, nameof(data));
        if (starts is null) return MissingInput(op, nameof(starts));
        if (ends is null) return MissingInput(op, nameof(ends));
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
            case TensorElementType.Float16: return Success(op, Tensor<Half>.Slice((Tensor<Half>)data, (Tensor<int>) starts, (Tensor<int>) ends, (Tensor<int>?) axes, (Tensor<int>?) steps));
            case TensorElementType.BFloat16: return Success(op, Tensor<BFloat16>.Slice((Tensor<BFloat16>)data, (Tensor<int>) starts, (Tensor<int>) ends, (Tensor<int>?) axes, (Tensor<int>?) steps));
            case TensorElementType.Complex64: return Success(op, Tensor<System.Numerics.Complex>.Slice((Tensor<System.Numerics.Complex>)data, (Tensor<int>) starts, (Tensor<int>) ends, (Tensor<int>?) axes, (Tensor<int>?) steps));
            default: return NotSupported(op);
        }
    }

    public static OpResult Equal(ITensor? A, ITensor? B, ExecutionOptions? options = null)
    {
        var op = OpType.Equal;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
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

    public static OpResult Where(ITensor? condition, ITensor? X, ITensor? Y, ExecutionOptions? options = null)
    {
        var op = OpType.Where;
        if (condition is null) return MissingInput(op, nameof(condition));
        if (X is null) return MissingInput(op, nameof(X));
        if (Y is null) return MissingInput(op, nameof(Y));
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

    public static OpResult Expand(ITensor? data, ITensor? shape, ExecutionOptions? options = null)
    {
        var op = OpType.Expand;
        if (data is null) return MissingInput(op, nameof(data));
        if (shape is null) return MissingInput(op, nameof(shape));
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
        string? mode, string? coordinateTransformationMode, string? nearestMode, float? cubicCoeffA, float? extrapolationValue, ExecutionOptions? options = null)
    {
        var op = OpType.Resize;
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

        int[] targetSizes;
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
            targetSizes = dims.Select((d, i) => Convert.ToInt32(Math.Round(d * scaleArray[i]))).ToArray();
        }

        var resizeMode = mode ?? "nearest";
        var ctm = coordinateTransformationMode ?? "half_pixel";
        var nm = nearestMode ?? "round_prefer_floor";
        var cubicA = cubicCoeffA ?? -0.75f;

        switch (X.ElementType)
        {
            case TensorElementType.Float:
                return Success(op, Tensor<float>.Resize((Tensor<float>)X, targetSizes, resizeMode, ctm, nm, cubicA));
            case TensorElementType.Double:
                return Success(op, Tensor<double>.Resize((Tensor<double>)X, targetSizes, resizeMode, ctm, nm, cubicA));
            default:
                return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult Unsqueeze(ITensor? data, ITensor? axes, ExecutionOptions? options = null)
    {
        var op = OpType.Unsqueeze;
        if (data is null) return MissingInput(op, nameof(data));
        if (axes is null) return MissingInput(op, nameof(axes));
        if (axes.ElementType == TensorElementType.Int64)
        {
            axes = ToInt32Saturating(axes);
        }
        var _axes = ((Tensor<int>) axes).ToArray();
        return Success(op, data.Unsqueeze(_axes));
    }

    public static OpResult Unsqueeze(ITensor? data, int[] axes, ExecutionOptions? options = null)
    {
        var op = OpType.Unsqueeze;
        if (data is null) return MissingInput(op, nameof(data));
        if (axes is null) return MissingInput(op, nameof(axes));
        return Success(op, data.Unsqueeze(axes));
    }

    public static OpResult ReduceSum(ITensor? data, ITensor? axes, int? _keep_dims, int? noop_with_empty_axes, ExecutionOptions? options = null)
    {
        var op = OpType.ReduceSum;
        if (data is null) return MissingInput(op, nameof(data));
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

    public static OpResult ReduceMean(ITensor? data, ITensor? axes, int? _keep_dims, int? noop_with_empty_axes, ExecutionOptions? options = null)
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
        switch (data.ElementType)
        {
            case TensorElementType.Int32: return Success(op, Tensor<int>.ReduceMean((Tensor<int>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes));
            case TensorElementType.Float: return Success(op, Tensor<float>.ReduceMean((Tensor<float>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes));
            case TensorElementType.Double: return Success(op, Tensor<double>.ReduceMean((Tensor<double>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes));
            default: return NotSupported(op);
        }
    }

    public static OpResult ReduceMax(ITensor? data, ITensor? axes, int? _keep_dims, ExecutionOptions? options = null)
    {
        var op = OpType.ReduceMax;
        if (data is null) return MissingInput(op, nameof(data));
        if (axes is not null && axes.Rank != 1) return WrongInputShape(op, nameof(axes), 1, axes);
        if (axes is not null && axes.ElementType == TensorElementType.Int64)
        {
            axes = axes.Cast<int>();
        }
        var keepDims = _keep_dims.HasValue ? Convert.ToBoolean(_keep_dims.Value) : true;
        switch (data.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.ReduceMax((Tensor<float>)data, (Tensor<int>?)axes, keepDims));
            case TensorElementType.Double: return Success(op, Tensor<double>.ReduceMax((Tensor<double>)data, (Tensor<int>?)axes, keepDims));
            default: return NotSupported(op);
        }
    }

    public static OpResult Softmax(ITensor? input, int? _axis, ExecutionOptions? options = null)
    {
        var op = OpType.Softmax;
        if (input is null) return MissingInput(op, nameof(input));
        var axis = _axis.HasValue ? _axis.Value : -1;
        if ((options ?? ExecutionOptions.Default).Optimization == OptimizationMode.Speed)
        {
            input = input.ToDenseTensor();
        }
        switch (input.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Softmax((Tensor<float>) input, axis));
            case TensorElementType.Double: return Success(op, Tensor<double>.Softmax((Tensor<double>) input, axis));
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

    public static OpResult Abs(ITensor? X, ExecutionOptions? options = null)
    {
        var op = OpType.Abs;
        if (X is null) return MissingInput(op, nameof(X));
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

    public static OpResult Cos(ITensor? X, ExecutionOptions? options = null)
    {
        var op = OpType.Cos;
        if (X is null) return MissingInput(op, nameof(X));
        Profiler.StartOpStage(OpStage.Math);
        switch (X.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Cos((Tensor<float>)X));
            case TensorElementType.Double: return Success(op, Tensor<double>.Cos((Tensor<double>)X));
            default: return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult Sin(ITensor? X, ExecutionOptions? options = null)
    {
        var op = OpType.Sin;
        if (X is null) return MissingInput(op, nameof(X));
        Profiler.StartOpStage(OpStage.Math);
        switch (X.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Sin((Tensor<float>)X));
            case TensorElementType.Double: return Success(op, Tensor<double>.Sin((Tensor<double>)X));
            default: return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult Neg(ITensor? X, ExecutionOptions? options = null)
    {
        var op = OpType.Neg;
        if (X is null) return MissingInput(op, nameof(X));
        Profiler.StartOpStage(OpStage.Math);
        switch (X.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Negate((Tensor<float>)X));
            case TensorElementType.Double: return Success(op, Tensor<double>.Negate((Tensor<double>)X));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Negate((Tensor<int>)X));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Negate((Tensor<long>)X));
            default: return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult Gelu(ITensor? X, string? approximate = null, ExecutionOptions? options = null)
    {
        var op = OpType.Gelu;
        if (X is null) return MissingInput(op, nameof(X));
        if (approximate is not null && approximate != "none") return AttributeNotSupported(op, nameof(approximate), approximate);
        Profiler.StartOpStage(OpStage.Math);
        switch (X.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Gelu((Tensor<float>)X));
            case TensorElementType.Double: return Success(op, Tensor<double>.Gelu((Tensor<double>)X));
            default: return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    /// <summary>
    /// Removes size-1 dimensions; a null axes removes every size-1 dimension.
    /// Axes index the input tensor; squeezing a dimension whose size is not 1 fails.
    /// </summary>
    public static OpResult Squeeze(ITensor? data, ITensor? axes = null, ExecutionOptions? options = null)
    {
        var op = OpType.Squeeze;
        if (data is null) return MissingInput(op, nameof(data));
        Profiler.StartOpStage(OpStage.Math);
        int[] dims = data.Dims.ToArray();
        if (axes is null)
        {
            dims = dims.Where(d => d != 1).ToArray();
        }
        else
        {
            var ax = ToIntArray(axes, nameof(axes)).Select(a => a < 0 ? a + data.Rank : a).ToArray();
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
    public static OpResult Range(ITensor? start, ITensor? limit, ITensor? delta, ExecutionOptions? options = null)
    {
        var op = OpType.Range;
        if (start is null) return MissingInput(op, nameof(start));
        if (limit is null) return MissingInput(op, nameof(limit));
        if (delta is null) return MissingInput(op, nameof(delta));
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
    public static OpResult Tile(ITensor? data, ITensor? repeats, ExecutionOptions? options = null)
    {
        var op = OpType.Tile;
        if (data is null) return MissingInput(op, nameof(data));
        if (repeats is null) return MissingInput(op, nameof(repeats));
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

    public static OpResult LayerNormalization(ITensor? x, ITensor? scale, ITensor? bias, int? axis = null, float? epsilon = null, ExecutionOptions? options = null)
    {
        var op = OpType.LayerNormalization;
        if (x is null) return MissingInput(op, nameof(x));
        if (scale is null) return MissingInput(op, nameof(scale));
        Profiler.StartOpStage(OpStage.Math);
        int ax = axis ?? -1;
        float eps = epsilon ?? 1e-5f;
        switch (x.ElementType)
        {
            case TensorElementType.Float:
                return Success(op, Tensor<float>.LayerNormalization((Tensor<float>)x, (Tensor<float>)scale, bias as Tensor<float>, ax, eps));
            case TensorElementType.Double:
                return Success(op, Tensor<double>.LayerNormalization((Tensor<double>)x, (Tensor<double>)scale, bias as Tensor<double>, ax, eps));
            default: return InputTypeNotSupported(op, nameof(x), x);
        }
    }

    /// <summary>
    /// Splits the input along the axis into the given sizes and returns a TensorSequence.
    /// Keepdims 1 preserves the rank; keepdims 0 drops the split axis and needs every size to be 1.
    /// </summary>
    public static OpResult SplitToSequence(ITensor? input, ITensor? split, int? axis = null, int? keepdims = null, ExecutionOptions? options = null)
    {
        var op = OpType.SplitToSequence;
        if (input is null) return MissingInput(op, nameof(input));
        if (split is null) return MissingInput(op, nameof(split));
        Profiler.StartOpStage(OpStage.Math);
        int rank = input.Rank;
        int ax = axis.HasValue ? (axis.Value < 0 ? axis.Value + rank : axis.Value) : 0;
        if (ax < 0 || ax >= rank) return WrongInputShape(op, nameof(axis), input, $"Axis {axis} is out of range.");
        int[] sizes = ToIntArray(split, nameof(split));
        int dim = input.Dims[ax];
        if (sizes.Sum() != dim) return WrongInputShape(op, nameof(split), input, "Split sizes must sum to the split dimension.");
        bool keep = (keepdims ?? 1) == 1;
        var items = new List<ITensor>();
        int start = 0;
        foreach (var length in sizes)
        {
            if (start + length > dim) return WrongInputShape(op, nameof(split), input, "Split sizes exceed the split dimension.");
            ITensor? chunk = input.ElementType switch
            {
                TensorElementType.Float => Tensor<float>.ChunkCopy((Tensor<float>)input, ax, start, length),
                TensorElementType.Double => Tensor<double>.ChunkCopy((Tensor<double>)input, ax, start, length),
                TensorElementType.Int32 => Tensor<int>.ChunkCopy((Tensor<int>)input, ax, start, length),
                TensorElementType.Int64 => Tensor<long>.ChunkCopy((Tensor<long>)input, ax, start, length),
                _ => null,
            };
            if (chunk is null) return InputTypeNotSupported(op, nameof(input), input);
            if (!keep) chunk = chunk.Reshape(chunk.Dims.Where((d, i) => i != ax).ToArray()).ToDenseTensor();
            items.Add(chunk);
            start += length;
        }
        return Success(op, new TensorSequence(items));
    }

    /// <summary>
    /// Returns the sequence element at the scalar index; negative indices count from the end.
    /// </summary>
    public static OpResult SequenceAt(ITensor? sequence, ITensor? index, ExecutionOptions? options = null)
    {
        var op = OpType.SequenceAt;
        if (sequence is null) return MissingInput(op, nameof(sequence));
        if (index is null) return MissingInput(op, nameof(index));
        if (sequence is not TensorSequence seq) return WrongInputType(op, nameof(sequence), "Input must be a sequence.", sequence);
        Profiler.StartOpStage(OpStage.Math);
        long position = ToInt64Scalar(index, nameof(index));
        if (position < 0) position += seq.Items.Count;
        if (position < 0 || position >= seq.Items.Count) return WrongInputShape(op, nameof(index), index, "Sequence index is out of range.");
        return Success(op, seq.Items[(int)position]);
    }
}


