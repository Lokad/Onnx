namespace Lokad.Onnx.Backend;

extern alias OnnxSharp;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Versioning;

using static OpResult;
public enum ExecutionProvider
{
    CPU
}

[RequiresPreviewFeatures]
public class CPUExecutionProvider
{
    public static Dictionary<OpType, int[]> SupportedOps { get; } = new Dictionary<OpType, int[]>()
    {
        {OpType.Add, new [] {16} }
    };

    public bool SupportsOp(OpType op, int version) => SupportedOps.ContainsKey(op) && SupportedOps[op].Any(v => v == version);

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
            case TensorElementType.Int8: return Success(op, Tensor<byte>.Add((Tensor<byte>)bA, (Tensor<byte>)bB));
            case TensorElementType.UInt8: return Success(op, Tensor<sbyte>.Add((Tensor<sbyte>)bA, (Tensor<sbyte>)bB));
            case TensorElementType.Int16: return Success(op, Tensor<short>.Add((Tensor<short>)bA, (Tensor<short>)bB));
            case TensorElementType.UInt16: return Success(op, Tensor<ushort>.Add((Tensor<ushort>)bA, (Tensor<ushort>)bB));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Add((Tensor<int>)bA, (Tensor<int>)bB));
            case TensorElementType.UInt32: return Success(op, Tensor<uint>.Add((Tensor<uint>)bA, (Tensor<uint>)bB));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Add((Tensor<long>)bA, (Tensor<long>)bB));
            case TensorElementType.UInt64: return Success(op, Tensor<ulong>.Add((Tensor<ulong>)bA, (Tensor<ulong>)bB));
            case TensorElementType.Float: return Success(op, Tensor<float>.Add((Tensor<float>)bA, (Tensor<float>)bB));
            case TensorElementType.Double: return Success(op, Tensor<double>.Add((Tensor<double>)bA, (Tensor<double>)bB));
            case TensorElementType.Float16: return Success(op, Tensor<Half>.Add((Tensor<Half>)bA, (Tensor<Half>)bB));
            default: return InputTypeNotSupported(OpType.Add, nameof(A), A);
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
            case TensorElementType.Int8: return Success(op, Tensor<byte>.Relu((Tensor<byte>)X));
            case TensorElementType.UInt8: return Success(op, Tensor<sbyte>.Relu((Tensor<sbyte>)X));
            case TensorElementType.Int16: return Success(op, Tensor<short>.Relu((Tensor<short>)X));
            case TensorElementType.UInt16: return Success(op, Tensor<ushort>.Relu((Tensor<ushort>)X));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Relu((Tensor<int>)X));
            case TensorElementType.UInt32: return Success(op, Tensor<uint>.Relu((Tensor<uint>)X));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Relu((Tensor<long>)X));
            case TensorElementType.UInt64: return Success(op, Tensor<ulong>.Relu((Tensor<ulong>)X));
            case TensorElementType.Float: return Success(op, Tensor<float>.Relu((Tensor<float>)X));
            case TensorElementType.Double: return Success(op, Tensor<double>.Relu((Tensor<double>)X));
            case TensorElementType.Float16: return Success(op, Tensor<Half>.Relu((Tensor<Half>)X));
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
            case TensorElementType.Int8: return Success(op, Tensor<byte>.MatMul((Tensor<byte>)A, (Tensor<byte>)B));
            case TensorElementType.UInt8: return Success(op, Tensor<sbyte>.MatMul((Tensor<sbyte>)A, (Tensor<sbyte>)B));
            case TensorElementType.Int16: return Success(op, Tensor<short>.MatMul((Tensor<short>)A, (Tensor<short>)B));
            case TensorElementType.UInt16: return Success(op, Tensor<ushort>.MatMul((Tensor<ushort>)A, (Tensor<ushort>)B));
            case TensorElementType.Int32: return Success(op, Tensor<int>.MatMul((Tensor<int>)A, (Tensor<int>)B));
            case TensorElementType.UInt32: return Success(op, Tensor<uint>.MatMul((Tensor<uint>)A, (Tensor<uint>)B));
            case TensorElementType.Int64: return Success(op, Tensor<long>.MatMul((Tensor<long>)A, (Tensor<long>)B));
            case TensorElementType.UInt64: return Success(op, Tensor<ulong>.MatMul((Tensor<ulong>)A, (Tensor<ulong>)B));
            case TensorElementType.Float: return Success(op, Tensor<float>.MatMul((Tensor<float>)A, (Tensor<float>)B));
            case TensorElementType.Double: return Success(op, Tensor<double>.MatMul((Tensor<double>)A, (Tensor<double>)B));
            case TensorElementType.Float16: return Success(op, Tensor<Half>.MatMul((Tensor<Half>)A, (Tensor<Half>)B));
            default: return InputTypeNotSupported(OpType.Add, nameof(A), A);
        }
    }
    public static OpResult Squeeze(int version, ITensor input, ITensor? axes = null)
    {
        Tensor<long>? shape = null;
        if (axes is not null)
        {
            if (axes.Dims.Length != 1)
            {
                return OpResult.Failure(OpType.Squeeze, $"The axes tensor  {axes.Name} must have dimension 1.");
            }
            else if (axes.ElementType != TensorElementType.Int64)
            {
                return OpResult.WrongInputType(OpType.Squeeze, nameof(axes), TensorElementType.Int64, axes);
            }
            else
            {
                shape = (Tensor<long>)axes;
            }
        }

        switch (input.ElementType)
        {
            case TensorElementType.Bool: return Squeeze((Tensor<bool>)input, shape);
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

    public static OpResult Broadcast<T>(DenseTensor<T> inA, DenseTensor<T> inB) where T : struct, INumber<T>
    {
        var broadcastRank = Math.Max(inA.Rank, inB.Rank);
        var outA = inA.Clone();
        var outB = inB.Clone();
        for (var i = 0; i < broadcastRank; i++)
        {
            var idxA = i - broadcastRank + inA.Rank;
            var idxB = i - broadcastRank + inB.Rank;
            if (i < broadcastRank - inA.Rank)
            {
                outA = outA.PadLeft();
                outA = outA.BroadcastDim(0, inB.Dimensions[idxB]);
            }
            else if (i < broadcastRank - inB.Rank)
            {
                outB = outB.PadLeft();
                outB = outB.BroadcastDim(0, inA.Dimensions[idxA]);
            }
            else if (inA.Dimensions[idxA] == inB.Dimensions[idxB])
            {
            }
            else if (inA.Dimensions[idxA] == 1)
            {
                outA = outA.BroadcastDim(i, inB.Dimensions[idxB]);
            }
            else if (inB.Dimensions[idxB] == 1)
            {
                outB = outB.BroadcastDim(i, inA.Dimensions[idxA]);
            }
            else
            {
                return OpResult.Failure(OpType.Broadcast, $"Trying to broadcast incompatible shapes: {inA.Dimensions.ToArray()} and {inB.Dimensions.ToArray()}");
            }
        }
        return OpResult.Success(OpType.Broadcast, new[] { outA, outB });
    }
}

