namespace Lokad.Onnx;


using System;
using System.Collections.Generic;
using System.Linq;

using static OpResult;
public partial class CPUExecutionProvider
{

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
            case TensorElementType.Float: return Success(op, Tensor<float>.ReduceMax((Tensor<float>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes));
            case TensorElementType.Double: return Success(op, Tensor<double>.ReduceMax((Tensor<double>)data, (Tensor<int>?)axes, keepDims, noopWithEmptyAxes));
            default: return NotSupported(op);
        }
    }
}
