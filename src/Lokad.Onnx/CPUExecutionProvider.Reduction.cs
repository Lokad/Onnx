namespace Lokad.Onnx;


using System;
using System.Collections.Generic;
using System.Linq;

using static OpResult;
public partial class CPUExecutionProvider
{

    /// <summary>Squares elements before summing; an empty-axis no-op still squares.</summary>
    public static OpResult ReduceSumSquare(ITensor? data, ITensor? axes, int? keepDims, int? noopWithEmptyAxes, ExecutionOptions? options)
    {
        var op = OpType.ReduceSumSquare;
        if (data is null) return MissingInput(op, nameof(data));
        (options ?? ExecutionOptions.Default).Validated();
        if (axes is not null && axes.Rank != 1) return WrongInputShape(op, nameof(axes), 1, axes);
        if (axes is not null && axes.ElementType is not (TensorElementType.Int32 or TensorElementType.Int64))
            return WrongInputType(op, nameof(axes), "Axes must be int32 or int64.", axes);
        if (axes?.ElementType == TensorElementType.Int64) axes = ToInt32Saturating(axes);
        var dimensions = (Tensor<int>?)axes;
        bool keep = keepDims is null || keepDims != 0, noop = noopWithEmptyAxes is not null && noopWithEmptyAxes != 0;
        // Validate before allocating the elementwise temporary.
        ReductionPlan.Create(data.Rank, dimensions, keep, noop);
        return data.ElementType switch
        {
            TensorElementType.Float => Success(op, Tensor<float>.ReduceSum(SquareForReduction((Tensor<float>)data), dimensions, keep, noop)),
            TensorElementType.Double => Success(op, Tensor<double>.ReduceSum(SquareForReduction((Tensor<double>)data), dimensions, keep, noop)),
            TensorElementType.Int32 => Success(op, IntegerSumSquare((Tensor<int>)data, dimensions, keep, noop)),
            TensorElementType.Int64 => Success(op, IntegerSumSquare((Tensor<long>)data, dimensions, keep, noop)),
            _ => InputTypeNotSupported(op, nameof(data), data)
        };
    }

    static DenseTensor<T> IntegerSumSquare<T>(Tensor<T> data, Tensor<int>? axes, bool keep, bool noop)
        where T : unmanaged, System.Numerics.IBinaryInteger<T>
    {
        // ORT's integer SumSquare aggregator squares/accumulates in double,
        // then saturates the final integer. Squaring in T would wrap first.
        var input = data.ToArray();
        var squared = new double[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            double value = double.CreateChecked(input[i]);
            squared[i] = value * value;
        }
        var sums = Tensor<double>.ReduceSum(new DenseTensor<double>(squared, data.Dimensions.ToArray()), axes, keep, noop);
        var values = sums.ToArray();
        var output = new T[values.Length];
        for (int i = 0; i < values.Length; i++) output[i] = T.CreateSaturating(values[i]);
        return new DenseTensor<T>(output, sums.Dimensions.ToArray());
    }

    static DenseTensor<T> SquareForReduction<T>(Tensor<T> data) where T : unmanaged, System.Numerics.INumber<T>
    {
        Profiler.StartOpStage(OpStage.Math);
        var values = data.ToArray();
        for (int i = 0; i < values.Length; i++) values[i] = unchecked(values[i] * values[i]);
        return new DenseTensor<T>(values, data.Dimensions.ToArray());
    }

    public static OpResult ReduceSum(ITensor? data, ITensor? axes, int? _keep_dims, int? noop_with_empty_axes, ExecutionOptions? options)
    {
        var op = OpType.ReduceSum;
        if (data is null) return MissingInput(op, nameof(data));
        (options ?? ExecutionOptions.Default).Validated();
        if (axes is not null && axes.ElementType == TensorElementType.Int64)
        {
            // Saturate before narrowing: Cast<int> wrapped 2^40 into a valid
            // axis; the plan range-check below then fails descriptively.
            axes = ToInt32Saturating(axes);
        }
        if (axes is not null && axes.ElementType != TensorElementType.Int32 && axes.ElementType != TensorElementType.Int64) return WrongInputType(op, nameof(axes), "The axes tensor must be int32 or int64.", axes);
        var keepDims = _keep_dims.HasValue ? Convert.ToBoolean(_keep_dims.Value) : true;
        var noopWithEmptyAxes = noop_with_empty_axes.HasValue ? Convert.ToBoolean(noop_with_empty_axes.Value) : false;
        switch (data.ElementType)
        {
            case TensorElementType.Int32: return Success(op, Tensor<int>.ReduceSum((Tensor<int>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes));
            case TensorElementType.Int64: return Success(op, Tensor<long>.ReduceSum((Tensor<long>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes));
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
            // Saturate before narrowing: Cast<int> wrapped 2^40 into a valid
            // axis; the plan range-check below then fails descriptively.
            axes = ToInt32Saturating(axes);
        }
        if (axes is not null && axes.ElementType != TensorElementType.Int32 && axes.ElementType != TensorElementType.Int64) return WrongInputType(op, nameof(axes), "The axes tensor must be int32 or int64.", axes);
        var keepDims = _keep_dims.HasValue ? Convert.ToBoolean(_keep_dims.Value) : true;
        var noopWithEmptyAxes = noop_with_empty_axes.HasValue ? Convert.ToBoolean(noop_with_empty_axes.Value) : false;
        var opts = (options ?? ExecutionOptions.Default).Validated();
        switch (data.ElementType)
        {
            case TensorElementType.Int32: return Success(op, Tensor<int>.ReduceMean((Tensor<int>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes, opts.Tensor));
            case TensorElementType.Int64: return Success(op, Tensor<long>.ReduceMean((Tensor<long>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes, opts.Tensor));
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
            // Saturate before narrowing: Cast<int> wrapped 2^40 into a valid
            // axis; the plan range-check below then fails descriptively.
            axes = ToInt32Saturating(axes);
        }
        if (axes is not null && axes.ElementType != TensorElementType.Int32 && axes.ElementType != TensorElementType.Int64) return WrongInputType(op, nameof(axes), "The axes tensor must be int32 or int64.", axes);
        var keepDims = _keep_dims.HasValue ? Convert.ToBoolean(_keep_dims.Value) : true;
        var noopWithEmptyAxes = noop_with_empty_axes.HasValue ? Convert.ToBoolean(noop_with_empty_axes.Value) : false;
        switch (data.ElementType)
        {
            case TensorElementType.Int8: return Success(op, Tensor<sbyte>.ReduceMax((Tensor<sbyte>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes));
            case TensorElementType.UInt8: return Success(op, Tensor<byte>.ReduceMax((Tensor<byte>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes));
            case TensorElementType.Int32: return Success(op, Tensor<int>.ReduceMax((Tensor<int>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes));
            case TensorElementType.Int64: return Success(op, Tensor<long>.ReduceMax((Tensor<long>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes));
            case TensorElementType.Float: return Success(op, Tensor<float>.ReduceMax((Tensor<float>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes));
            case TensorElementType.Double: return Success(op, Tensor<double>.ReduceMax((Tensor<double>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes));
            default: return NotSupported(op);
        }
    }
}
