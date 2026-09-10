namespace Lokad.Onnx;


using System;
using System.Collections.Generic;
using System.Linq;

using static OpResult;
public partial class CPUExecutionProvider
{

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
            case TensorElementType.Float16: return Success(op, Tensor<Half>.Reshape((Tensor<Half>)input, (Tensor<long>)shape, allow_zero ?? false));
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
            case TensorElementType.Int8: return Success(op, ((Tensor<sbyte>)A).BroadcastApply<AddBroadcast<sbyte>>((Tensor<sbyte>)B, opts.Tensor));
            case TensorElementType.Int16: return Success(op, ((Tensor<short>)A).BroadcastApply<AddBroadcast<short>>((Tensor<short>)B, opts.Tensor));
            case TensorElementType.UInt16: return Success(op, ((Tensor<ushort>)A).BroadcastApply<AddBroadcast<ushort>>((Tensor<ushort>)B, opts.Tensor));
            case TensorElementType.UInt8: return Success(op, ((Tensor<byte>)A).BroadcastApply<AddBroadcast<byte>>((Tensor<byte>)B, opts.Tensor));
            case TensorElementType.Int32: return Success(op, ((Tensor<int>)A).BroadcastApply<AddBroadcast<int>>((Tensor<int>)B, opts.Tensor));
            case TensorElementType.Int64: return Success(op, ((Tensor<long>)A).BroadcastApply<AddBroadcast<long>>((Tensor<long>)B, opts.Tensor));
            case TensorElementType.UInt32: return Success(op, ((Tensor<uint>)A).BroadcastApply<AddBroadcast<uint>>((Tensor<uint>)B, opts.Tensor));
            case TensorElementType.UInt64: return Success(op, ((Tensor<ulong>)A).BroadcastApply<AddBroadcast<ulong>>((Tensor<ulong>)B, opts.Tensor));
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
            case TensorElementType.Int8: return Success(op, ((Tensor<sbyte>)A).BroadcastApply<SubtractBroadcast<sbyte>>((Tensor<sbyte>)B, opts.Tensor));
            case TensorElementType.Int16: return Success(op, ((Tensor<short>)A).BroadcastApply<SubtractBroadcast<short>>((Tensor<short>)B, opts.Tensor));
            case TensorElementType.UInt16: return Success(op, ((Tensor<ushort>)A).BroadcastApply<SubtractBroadcast<ushort>>((Tensor<ushort>)B, opts.Tensor));
            case TensorElementType.UInt8: return Success(op, ((Tensor<byte>)A).BroadcastApply<SubtractBroadcast<byte>>((Tensor<byte>)B, opts.Tensor));
            case TensorElementType.Int32: return Success(op, ((Tensor<int>)A).BroadcastApply<SubtractBroadcast<int>>((Tensor<int>)B, opts.Tensor));
            case TensorElementType.Int64: return Success(op, ((Tensor<long>)A).BroadcastApply<SubtractBroadcast<long>>((Tensor<long>)B, opts.Tensor));
            case TensorElementType.UInt32: return Success(op, ((Tensor<uint>)A).BroadcastApply<SubtractBroadcast<uint>>((Tensor<uint>)B, opts.Tensor));
            case TensorElementType.UInt64: return Success(op, ((Tensor<ulong>)A).BroadcastApply<SubtractBroadcast<ulong>>((Tensor<ulong>)B, opts.Tensor));
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
            case TensorElementType.Int8: return Success(op, ((Tensor<sbyte>)A).BroadcastApply<MultiplyBroadcast<sbyte>>((Tensor<sbyte>)B, opts.Tensor));
            case TensorElementType.Int16: return Success(op, ((Tensor<short>)A).BroadcastApply<MultiplyBroadcast<short>>((Tensor<short>)B, opts.Tensor));
            case TensorElementType.UInt16: return Success(op, ((Tensor<ushort>)A).BroadcastApply<MultiplyBroadcast<ushort>>((Tensor<ushort>)B, opts.Tensor));
            case TensorElementType.UInt8: return Success(op, ((Tensor<byte>)A).BroadcastApply<MultiplyBroadcast<byte>>((Tensor<byte>)B, opts.Tensor));
            case TensorElementType.Int32: return Success(op, ((Tensor<int>)A).BroadcastApply<MultiplyBroadcast<int>>((Tensor<int>)B, opts.Tensor));
            case TensorElementType.Int64: return Success(op, ((Tensor<long>)A).BroadcastApply<MultiplyBroadcast<long>>((Tensor<long>)B, opts.Tensor));
            case TensorElementType.UInt32: return Success(op, ((Tensor<uint>)A).BroadcastApply<MultiplyBroadcast<uint>>((Tensor<uint>)B, opts.Tensor));
            case TensorElementType.UInt64: return Success(op, ((Tensor<ulong>)A).BroadcastApply<MultiplyBroadcast<ulong>>((Tensor<ulong>)B, opts.Tensor));
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
            case TensorElementType.Int8: return Success(op, ((Tensor<sbyte>)A).BroadcastApply<DivideBroadcast<sbyte>>((Tensor<sbyte>)B, opts.Tensor));
            case TensorElementType.Int16: return Success(op, ((Tensor<short>)A).BroadcastApply<DivideBroadcast<short>>((Tensor<short>)B, opts.Tensor));
            case TensorElementType.UInt16: return Success(op, ((Tensor<ushort>)A).BroadcastApply<DivideBroadcast<ushort>>((Tensor<ushort>)B, opts.Tensor));
            case TensorElementType.UInt8: return Success(op, ((Tensor<byte>)A).BroadcastApply<DivideBroadcast<byte>>((Tensor<byte>)B, opts.Tensor));
            case TensorElementType.Int32: return Success(op, ((Tensor<int>)A).BroadcastApply<DivideBroadcast<int>>((Tensor<int>)B, opts.Tensor));
            case TensorElementType.Int64: return Success(op, ((Tensor<long>)A).BroadcastApply<DivideBroadcast<long>>((Tensor<long>)B, opts.Tensor));
            case TensorElementType.UInt32: return Success(op, ((Tensor<uint>)A).BroadcastApply<DivideBroadcast<uint>>((Tensor<uint>)B, opts.Tensor));
            case TensorElementType.UInt64: return Success(op, ((Tensor<ulong>)A).BroadcastApply<DivideBroadcast<ulong>>((Tensor<ulong>)B, opts.Tensor));
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

    /// <summary>
    /// Integer Pow matching ORT 1.29: computed float-mediated as (int)pow((double)a, (double)b).
    /// Fractional results truncate; NaN, out-of-range, and infinite results yield
    /// int.MinValue like the native conversion (unlike .NET saturating casts).
    /// </summary>
    static int IntPow(int a, int b)
    {
        double d = System.Math.Pow(a, b);
        if (double.IsNaN(d) || d >= 2147483648.0 || d < -2147483648.0) return int.MinValue;
        return (int)d;
    }

    /// <summary>
    /// Integer Pow matching ORT 1.29: computed float-mediated as (long)pow((double)a, (double)b).
    /// Fractional results truncate; NaN, out-of-range, and infinite results yield
    /// long.MinValue like the native conversion (unlike .NET saturating casts).
    /// </summary>
    static long LongPow(long a, long b)
    {
        double d = System.Math.Pow(a, b);
        if (double.IsNaN(d) || d >= 9223372036854775808.0 || d < -9223372036854775808.0) return long.MinValue;
        return (long)d;
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
            case TensorElementType.Int32: return Success(op, ((Tensor<int>)A).BroadcastApply((Tensor<int>)B, IntPow, opts.Tensor));
            case TensorElementType.Int64: return Success(op, ((Tensor<long>)A).BroadcastApply((Tensor<long>)B, LongPow, opts.Tensor));
            default: return InputTypeNotSupported(op, nameof(A), A);
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
            X = ((INumericTensor)X).ToDenseTensor();
        }
        Profiler.StartOpStage(OpStage.Math);
        switch (X.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Relu((Tensor<float>)X));
            case TensorElementType.Double: return Success(op, Tensor<double>.Relu((Tensor<double>)X));
            default: return InputTypeNotSupported(op, nameof(X), X);
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
            A = ((INumericTensor)A).ToDenseTensor();
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
            case TensorElementType.Int8: return Success(op, Tensor<sbyte>.Equal((Tensor<sbyte>)A, (Tensor<sbyte>)B));
            case TensorElementType.UInt8: return Success(op, Tensor<byte>.Equal((Tensor<byte>)A, (Tensor<byte>)B));
            case TensorElementType.Int16: return Success(op, Tensor<short>.Equal((Tensor<short>)A, (Tensor<short>)B));
            case TensorElementType.UInt16: return Success(op, Tensor<ushort>.Equal((Tensor<ushort>)A, (Tensor<ushort>)B));
            case TensorElementType.UInt32: return Success(op, Tensor<uint>.Equal((Tensor<uint>)A, (Tensor<uint>)B));
            case TensorElementType.UInt64: return Success(op, Tensor<ulong>.Equal((Tensor<ulong>)A, (Tensor<ulong>)B));
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
            case TensorElementType.Int8: return Success(op, Tensor<sbyte>.Less((Tensor<sbyte>)A, (Tensor<sbyte>)B));
            case TensorElementType.UInt8: return Success(op, Tensor<byte>.Less((Tensor<byte>)A, (Tensor<byte>)B));
            case TensorElementType.Int16: return Success(op, Tensor<short>.Less((Tensor<short>)A, (Tensor<short>)B));
            case TensorElementType.UInt16: return Success(op, Tensor<ushort>.Less((Tensor<ushort>)A, (Tensor<ushort>)B));
            case TensorElementType.UInt32: return Success(op, Tensor<uint>.Less((Tensor<uint>)A, (Tensor<uint>)B));
            case TensorElementType.UInt64: return Success(op, Tensor<ulong>.Less((Tensor<ulong>)A, (Tensor<ulong>)B));
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
            case TensorElementType.UInt32: return Success(op, Tensor<uint>.Where((Tensor<bool>)condition, (Tensor<uint>)X, (Tensor<uint>)Y));
            case TensorElementType.UInt64: return Success(op, Tensor<ulong>.Where((Tensor<bool>)condition, (Tensor<ulong>)X, (Tensor<ulong>)Y));
            case TensorElementType.Float: return Success(op, Tensor<float>.Where((Tensor<bool>)condition, (Tensor<float>)X, (Tensor<float>)Y));
            case TensorElementType.Double: return Success(op, Tensor<double>.Where((Tensor<bool>)condition, (Tensor<double>)X, (Tensor<double>)Y));
            default: return InputTypeNotSupported(op, nameof(X), X);
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
            input = ((INumericTensor)input).ToDenseTensor();
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
            case TensorElementType.UInt32: return Success(op, Tensor<uint>.Abs((Tensor<uint>)X));
            case TensorElementType.UInt64: return Success(op, Tensor<ulong>.Abs((Tensor<ulong>)X));
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
}
