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

public class CPUExecutionProvider : Runtime
{
    public static List<OpType> SupportedOps { get; } = new List<OpType>()
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
        OpType.Unsqueeze,
        OpType.ReduceSum,
        OpType.ReduceMean,
        OpType.ReduceMax,
        OpType.Softmax,
    };

    public static bool SupportsOp(OpType op) => SupportedOps.Contains(op);

    public static OpResult Reshape(ITensor? input, ITensor? shape, bool? allow_zero = null)
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
            default: return NotSupported(OpType.Reshape);
        }
    }

    public static OpResult Add(ITensor? A, ITensor? B)
    {
        var op = OpType.Add;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }
        if (!ITensor.Broadcast(A, B, out var bA, out var bB))
        {
            return CannotBroadcast(op, A, B);
        }
        switch (A.ElementType)
        {
            case TensorElementType.UInt8: return Success(op, Tensor<byte>.Add((Tensor<byte>)bA, (Tensor<byte>)bB));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Add((Tensor<int>)bA, (Tensor<int>)bB));
            case TensorElementType.Float: return Success(op, Tensor<float>.Add((Tensor<float>)bA, (Tensor<float>)bB));
            case TensorElementType.Double: return Success(op, Tensor<double>.Add((Tensor<double>)bA, (Tensor<double>)bB));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Sub(ITensor? A, ITensor? B)
    {
        var op = OpType.Sub;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }
        if (!ITensor.Broadcast(A, B, out var bA, out var bB))
        {
            return CannotBroadcast(op, A, B);
        }
        switch (A.ElementType)
        {
            case TensorElementType.UInt8: return Success(op, Tensor<byte>.Subtract((Tensor<byte>)bA, (Tensor<byte>)bB));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Subtract((Tensor<int>)bA, (Tensor<int>)bB));
            case TensorElementType.Float: return Success(op, Tensor<float>.Subtract((Tensor<float>)bA, (Tensor<float>)bB));
            case TensorElementType.Double: return Success(op, Tensor<double>.Subtract((Tensor<double>)bA, (Tensor<double>)bB));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Mul(ITensor? A, ITensor? B)
    {
        var op = OpType.Mul;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }
        if (!ITensor.Broadcast(A, B, out var bA, out var bB))
        {
            return CannotBroadcast(op, A, B);
        }
        switch (A.ElementType)
        {
            case TensorElementType.UInt8: return Success(op, Tensor<byte>.Multiply((Tensor<byte>)bA, (Tensor<byte>)bB));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Multiply((Tensor<int>)bA, (Tensor<int>)bB));
            case TensorElementType.Float: return Success(op, Tensor<float>.Multiply((Tensor<float>)bA, (Tensor<float>)bB));
            case TensorElementType.Double: return Success(op, Tensor<double>.Multiply((Tensor<double>)bA, (Tensor<double>)bB));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Div(ITensor? A, ITensor? B)
    {
        var op = OpType.Div;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }
        if (!ITensor.Broadcast(A, B, out var bA, out var bB))
        {
            return CannotBroadcast(op, A, B);
        }
        switch (A.ElementType)
        {
            case TensorElementType.UInt8: return Success(op, Tensor<byte>.Divide((Tensor<byte>)bA, (Tensor<byte>)bB));
            case TensorElementType.Float: return Success(op, Tensor<float>.Divide((Tensor<float>)bA, (Tensor<float>)bB));
            case TensorElementType.Double: return Success(op, Tensor<double>.Divide((Tensor<double>)bA, (Tensor<double>)bB));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Pow(ITensor? A, ITensor? B)
    {
        var op = OpType.Pow;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        if (A.ElementType != B.ElementType)
        {
            return WrongInputType(op, nameof(B), "Input tensors must be of the same type.", B);
        }
        if (!ITensor.Broadcast(A, B, out var bA, out var bB))
        {
            return CannotBroadcast(op, A, B);
        }
        switch (A.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Pow((Tensor<float>)bA, (Tensor<float>)bB));
            case TensorElementType.Double: return Success(op, Tensor<double>.Pow((Tensor<double>)bA, (Tensor<double>)bB));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }
    public static OpResult Conv(ITensor? X, ITensor? W, ITensor? B, string? auto_pad = null, int[]? dilations = null, int? group = null, int[]? kernel_shape = null, int[]? pads = null, int[]? strides = null)
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

    public static OpResult Relu(ITensor? X)
    {
        var op = OpType.Relu;
        if (X is null) return MissingInput(op, nameof(X));
        switch (X.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Relu((Tensor<float>)X));
            case TensorElementType.Double: return Success(op, Tensor<double>.Relu((Tensor<double>)X));
            default: return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult MaxPool(ITensor? X, string? auto_pad = null, int? ceil_mode = null, int[]? dilations = null, int[]? kernel_shape = null, int[]? pads = null, int? storage_order = null, int[]? strides = null)
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

    public static OpResult MatMul(ITensor? A, ITensor? B)
    {
        var op = OpType.MatMul;
        if (A is null) return MissingInput(op, nameof(A));
        if (B is null) return MissingInput(op, nameof(B));
        switch (A.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.MatMul((Tensor<float>)A, (Tensor<float>)B));
            case TensorElementType.Double: return Success(op, Tensor<double>.MatMul((Tensor<double>)A, (Tensor<double>)B));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Sqrt(ITensor? A)
    {
        var op = OpType.Sqrt;
        if (A is null) return MissingInput(op, nameof(A));
        switch (A.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Sqrt((Tensor<float>)A));
            case TensorElementType.Double: return Success(op, Tensor<double>.Sqrt((Tensor<double>)A));
            default: return InputTypeNotSupported(op, nameof(A), A);
        }
    }

    public static OpResult Squeeze<T>(Tensor<T> input, Tensor<long>? axes = null) where T : struct
    {
        long[] dims = (axes is not null) ? axes.ToArray() : Enumerable.Range(0, input.Dimensions.Length - 1).Cast<long>().ToArray();
        List<int> squeezedDims = new List<int>();
        for (int i = 0; i < dims.Length; i++)
        {
            var a = TensorUtil.HandleNegativeAxis((int)dims[i], input.Dimensions.Length);
            if (input.Dimensions[a] == 1)
            {
                squeezedDims.Add(a);
            }

        }
        return OpResult.Success(OpType.Squeeze, new[] { input.Reshape(squeezedDims.ToArray()) });
    }

    public static OpResult Squeeze(int version, ITensor input, ITensor? axes = null)
    {
        Tensor<long>? shape = null;
        if (axes is not null)
        {
            if (axes.Dims.Length != 1)
            {
                return OpResult.Failure(OpType.Transpose, $"The axes tensor  {axes.Name} must have dimension 1.");
            }
            else if (axes.ElementType != TensorElementType.Int64)
            {
                return OpResult.WrongInputType(OpType.Transpose, nameof(axes), TensorElementType.Int64, axes);
            }
            else
            {
                shape = (Tensor<long>)axes;
            }
        }

        switch (input.ElementType)
        {
            case TensorElementType.Bool: return Squeeze((Tensor<bool>)input, (Tensor<long>?) shape);
            case TensorElementType.Int8: return Squeeze((Tensor<byte>)input, shape);
            case TensorElementType.UInt8: return Squeeze((Tensor<sbyte>)input, shape);
            case TensorElementType.Int16: return Squeeze((Tensor<short>)input, shape);
            case TensorElementType.UInt16: return Squeeze((Tensor<ushort>)input, shape);
            case TensorElementType.Int32: return Squeeze((Tensor<int>)input, shape);
            case TensorElementType.UInt32: return Squeeze((Tensor<uint>)input, shape);
            case TensorElementType.Int64: return Squeeze((Tensor<long>)input, shape);
            case TensorElementType.UInt64: return Squeeze((Tensor<ulong>)input, shape);
            case TensorElementType.Float: return Squeeze((Tensor<float>)input, shape);
            case TensorElementType.Double: return Squeeze((Tensor<double>)input, shape);
            case TensorElementType.Float16: return Squeeze((Tensor<Half>)input, shape);
            case TensorElementType.BFloat16: return Squeeze((Tensor<BFloat16>)input, shape);
            case TensorElementType.Complex64: return Squeeze((Tensor<System.Numerics.Complex>)input, shape);
            default: return OpResult.NotSupported(OpType.Squeeze);
        }
    }

    public static OpResult Erf(ITensor? X)
    {
        var op = OpType.Erf;
        if (X is null) return MissingInput(op, nameof(X));
        switch (X.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Erf((Tensor<float>)X));
            case TensorElementType.Double: return Success(op, Tensor<double>.Erf((Tensor<double>)X));
            default: return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult Softmax(ITensor? X)
    {
        var op = OpType.Softmax;
        if (X is null) return MissingInput(op, nameof(X));
        switch (X.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Erf((Tensor<float>)X));
            case TensorElementType.Double: return Success(op, Tensor<double>.Erf((Tensor<double>)X));
            default: return InputTypeNotSupported(op, nameof(X), X);
        }
    }

    public static OpResult Transpose(ITensor? data, int[]? perm = null)
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

    public static OpResult Constant(object? value)
    {
        var op = OpType.Constant;
        if (value is null) return MissingAttribute(op, nameof(value));
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

    public static OpResult Cast(ITensor? input, long to)
    {
        var op = OpType.Cast;
        if (input is null) return MissingInput(op, nameof(input));
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
    public static OpResult Concat(ITensor[]? inputs, int? _axis)
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

    public static OpResult Shape(ITensor? data, int? _start = null, int? _end = null)
    {
        var op = OpType.Shape;
        if (data is null) return MissingInput(op, nameof(data));
        var start = ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, _start.HasValue ? _start.Value : 0);
        var end = ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, _end.HasValue ? _end.Value : data.Rank);
        start = ArrayUtilities.Clamp(start, 0, data.Rank);
        end = ArrayUtilities.Clamp(end, 0, data.Rank);  
        var _shape = data.Dims.Convert<int, long>()[start..end];
        return Success(op, DenseTensor<long>.OfValues(_shape));
    }

    public static OpResult Gather(ITensor? data, ITensor? indices, int? axis = null) 
    {
        var op = OpType.Gather;
        if (data is null) return MissingInput(op, nameof(data));
        if (indices is null) return MissingInput(op, nameof(indices));
        if (indices.Rank > data.Rank) return WrongInputShape(op, nameof(indices), data.Rank, indices);
        
        if (indices.ElementType == TensorElementType.Int64)
        {
            indices = indices.Cast<int>();
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

    public static OpResult Slice(ITensor? data, ITensor? starts, ITensor? ends, ITensor? axes, ITensor? steps)
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
            starts = starts.Cast<int>();
        }

        if (ends.ElementType == TensorElementType.Int64)
        {
            ends = ends.Cast<int>();
        }

        if (axes is not null && axes.ElementType == TensorElementType.Int64)
        {
            axes = axes.Cast<int>();
        }

        if (steps is not null && steps.ElementType == TensorElementType.Int64)
        {
            steps = steps.Cast<int>();
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

    public static OpResult Unsqueeze(ITensor? data, ITensor? axes)
    {
        var op = OpType.Unsqueeze;
        if (data is null) return MissingInput(op, nameof(data));
        if (axes is null) return MissingInput(op, nameof(axes));
        if (axes.ElementType == TensorElementType.Int64)
        {
            axes = axes.Cast<int>();
        }
        var _axes = ((Tensor<int>) axes).ToArray();
        return Success(op, data.Unsqueeze(_axes));
    }

    public static OpResult ReduceSum(ITensor? data, ITensor? axes, int? _keep_dims, int? noop_with_empty_axes)
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

    public static OpResult ReduceMean(ITensor? data, ITensor? axes, int? _keep_dims, int? noop_with_empty_axes)
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

    public static OpResult ReduceMax(ITensor? data, ITensor? axes, int? _keep_dims)
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

    public static OpResult Softmax(ITensor? input, int? _axis)
    {
        var op = OpType.Softmax;
        if (input is null) return MissingInput(op, nameof(input));
        var axis = _axis.HasValue ? _axis.Value : -1;
        switch (input.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Softmax((Tensor<float>) input, axis));
            case TensorElementType.Double: return Success(op, Tensor<double>.Softmax((Tensor<double>) input, axis));
            default: return InputTypeNotSupported(op, nameof(input), input);
        }
    }
}


