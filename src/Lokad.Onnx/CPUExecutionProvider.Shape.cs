namespace Lokad.Onnx;


using System;
using System.Collections.Generic;
using System.Linq;

using static OpResult;
public partial class CPUExecutionProvider
{

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
            case TensorElementType.Float16: return Success(op, Tensor<Half>.Transpose((Tensor<Half>)data, perm));
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
                var dense = ((INumericTensor)source).ToDenseTensor();
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
            case TensorElementType.UInt32: { var y = DenseTensor<uint>.OfShape(dims); y.Fill(((Tensor<uint>)value).GetValue(0)); return Success(op, y); }
            case TensorElementType.UInt64: { var y = DenseTensor<ulong>.OfShape(dims); y.Fill(((Tensor<ulong>)value).GetValue(0)); return Success(op, y); }
            case TensorElementType.Int8: { var y = DenseTensor<sbyte>.OfShape(dims); y.Fill(((Tensor<sbyte>)value).GetValue(0)); return Success(op, y); }
            case TensorElementType.UInt8: { var y = DenseTensor<byte>.OfShape(dims); y.Fill(((Tensor<byte>)value).GetValue(0)); return Success(op, y); }
            case TensorElementType.Int16: { var y = DenseTensor<short>.OfShape(dims); y.Fill(((Tensor<short>)value).GetValue(0)); return Success(op, y); }
            case TensorElementType.UInt16: { var y = DenseTensor<ushort>.OfShape(dims); y.Fill(((Tensor<ushort>)value).GetValue(0)); return Success(op, y); }
            case TensorElementType.Bool: { var y = DenseTensor<bool>.OfShape(dims); y.Fill(((Tensor<bool>)value).GetValue(0)); return Success(op, y); }
            case TensorElementType.Float16: { var y = DenseTensor<Half>.OfShape(dims); y.Fill(((Tensor<Half>)value).GetValue(0)); return Success(op, y); }
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
            //case TensorElementType.Float16: return Success(op, input.Cast<Half>());
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
        if (!inputs.All(i => i.ElementType == inputs[0].ElementType)) return WrongInputType(op, nameof(inputs), inputs[0].ElementType, inputs.First(i => i.ElementType != inputs[0].ElementType), "All tensors in a concat operation must have the same type.");
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
        var full = data.Dims.Convert<int, long>();
        // Reversed slices yield an empty shape vector per the ONNX Shape contract.
        var _shape = start >= end ? Array.Empty<long>() : full[start..end];
        return Success(op, DenseTensor<long>.OfValues(_shape));
    }

    public static OpResult Gather(ITensor? data, ITensor? indices, int? axis, ExecutionOptions? options) 
    {
        var op = OpType.Gather;
        if (data is null) return MissingInput(op, nameof(data));
        if (indices is null) return MissingInput(op, nameof(indices));
        (options ?? ExecutionOptions.Default).Validated();
        if (indices.ElementType == TensorElementType.Int64)
        {
            // Checked conversion: out-of-range values throw instead of truncating.
            indices = indices.ConvertToInt32();
        }
        else if (indices.ElementType != TensorElementType.Int32)
        {
            return WrongInputShape(op, nameof(indices), indices, "Gather indices must be int32 or int64.");
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
        if (starts.ElementType != TensorElementType.Int32) return WrongInputType(op, nameof(starts), "The starts tensor must be int32 or int64.", starts);

        if (ends.ElementType == TensorElementType.Int64)
        {
            ends = ToInt32Saturating(ends);
        }
        if (ends.ElementType != TensorElementType.Int32) return WrongInputType(op, nameof(ends), "The ends tensor must be int32 or int64.", ends);

        if (axes is not null && axes.ElementType == TensorElementType.Int64)
        {
            axes = ToInt32Saturating(axes);
        }
        if (axes is not null && axes.ElementType != TensorElementType.Int32) return WrongInputType(op, nameof(axes), "The axes tensor must be int32 or int64.", axes);

        if (steps is not null && steps.ElementType == TensorElementType.Int64)
        {
            steps = ToInt32Saturating(steps);
        }
        if (steps is not null && steps.ElementType != TensorElementType.Int32) return WrongInputType(op, nameof(steps), "The steps tensor must be int32 or int64.", steps);

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
            case TensorElementType.UInt32:
            {
                var dd = ((Tensor<uint>)data).ToDenseTensor();
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
            case TensorElementType.Int8:
            {
                var dd = ((Tensor<sbyte>)data).ToDenseTensor();
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
            case TensorElementType.UInt8:
            {
                var dd = ((Tensor<byte>)data).ToDenseTensor();
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
            case TensorElementType.Int16:
            {
                var dd = ((Tensor<short>)data).ToDenseTensor();
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
            case TensorElementType.UInt16:
            {
                var dd = ((Tensor<ushort>)data).ToDenseTensor();
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
            case TensorElementType.UInt64:
            {
                var dd = ((Tensor<ulong>)data).ToDenseTensor();
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
            case TensorElementType.Bool:
            {
                var dd = ((Tensor<bool>)data).ToDenseTensor();
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
            case TensorElementType.Float16:
            {
                var dd = ((Tensor<Half>)data).ToDenseTensor();
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
            case TensorElementType.Float16: return Success(op, Tensor<Half>.Expand((Tensor<Half>)data, targetShape));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Expand((Tensor<int>)data, targetShape));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Expand((Tensor<long>)data, targetShape));
            case TensorElementType.UInt32: return Success(op, Tensor<uint>.Expand((Tensor<uint>)data, targetShape));
            case TensorElementType.UInt64: return Success(op, Tensor<ulong>.Expand((Tensor<ulong>)data, targetShape));
            case TensorElementType.Int8: return Success(op, Tensor<sbyte>.Expand((Tensor<sbyte>)data, targetShape));
            case TensorElementType.UInt8: return Success(op, Tensor<byte>.Expand((Tensor<byte>)data, targetShape));
            case TensorElementType.Int16: return Success(op, Tensor<short>.Expand((Tensor<short>)data, targetShape));
            case TensorElementType.UInt16: return Success(op, Tensor<ushort>.Expand((Tensor<ushort>)data, targetShape));
            case TensorElementType.Float: return Success(op, Tensor<float>.Expand((Tensor<float>)data, targetShape));
            case TensorElementType.Double: return Success(op, Tensor<double>.Expand((Tensor<double>)data, targetShape));
            default: return InputTypeNotSupported(op, nameof(data), data);
        }
    }

    public static OpResult Resize(ITensor? X, ITensor? roi, ITensor? scales, ITensor? sizes,
        string? mode, string? coordinateTransformationMode, string? nearestMode, float? cubicCoeffA, float? extrapolationValue, ExecutionOptions? options) =>
        Resize(X, roi, scales, sizes, mode, coordinateTransformationMode, nearestMode, cubicCoeffA, extrapolationValue, options, null, null, null, null);

    public static OpResult Resize(ITensor? X, ITensor? roi, ITensor? scales, ITensor? sizes,
        string? mode, string? coordinateTransformationMode, string? nearestMode, float? cubicCoeffA, float? extrapolationValue, ExecutionOptions? options,
        int? antialias, int[]? axes, int? excludeOutside, string? keepAspectRatioPolicy)
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
        (options ?? ExecutionOptions.Default).Validated();
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
        if (antialias is not null && antialias != 0) return AttributeNotSupported(op, nameof(antialias), antialias.Value.ToString(), "Resize antialiasing is not supported.");
        if (excludeOutside is not null && excludeOutside != 0) return AttributeNotSupported(op, nameof(excludeOutside), excludeOutside.Value.ToString(), "Resize exclude_outside is not supported.");
        if (keepAspectRatioPolicy is not null && keepAspectRatioPolicy != "stretch") return AttributeNotSupported(op, nameof(keepAspectRatioPolicy), keepAspectRatioPolicy, "Only the default stretch policy is supported.");
        if (axes is not null && !axes.SequenceEqual(Enumerable.Range(0, X.Rank))) return AttributeNotSupported(op, nameof(axes), string.Join(",", axes), "Only resizing all axes is supported.");

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
        if (axes.ElementType != TensorElementType.Int32) return WrongInputType(op, nameof(axes), "The axes tensor must be int32 or int64.", axes);
        var _axes = ((Tensor<int>) axes).ToArray();
        return Success(op, ((INumericTensor)data).Unsqueeze(_axes));
    }

    public static OpResult Unsqueeze(ITensor? data, int[] axes, ExecutionOptions? options)
    {
        var op = OpType.Unsqueeze;
        if (data is null) return MissingInput(op, nameof(data));
        if (axes is null) return MissingInput(op, nameof(axes));
        (options ?? ExecutionOptions.Default).Validated();
        return Success(op, ((INumericTensor)data).Unsqueeze(axes));
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
            if (axes.ElementType != TensorElementType.Int32 && axes.ElementType != TensorElementType.Int64) return WrongInputType(op, nameof(axes), "The axes tensor must be int32 or int64.", axes);
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
        return Success(op, ((INumericTensor)data).Reshape(dims).ToDenseTensor());
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
            case TensorElementType.Int16:
                return Success(op, Tensor<short>.Range(Convert.ToInt16(start.GetValue(0)), Convert.ToInt16(limit.GetValue(0)), Convert.ToInt16(delta.GetValue(0))));
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
        if (repeats.ElementType != TensorElementType.Int32 && repeats.ElementType != TensorElementType.Int64) return WrongInputType(op, nameof(repeats), "The repeats tensor must be int32 or int64.", repeats);
        var reps = ToIntArray(repeats, nameof(repeats));
        switch (data.ElementType)
        {
            case TensorElementType.Float: return Success(op, Tensor<float>.Tile((Tensor<float>)data, reps));
            case TensorElementType.Double: return Success(op, Tensor<double>.Tile((Tensor<double>)data, reps));
            case TensorElementType.Int32: return Success(op, Tensor<int>.Tile((Tensor<int>)data, reps));
            case TensorElementType.Int64: return Success(op, Tensor<long>.Tile((Tensor<long>)data, reps));
            case TensorElementType.UInt32: return Success(op, Tensor<uint>.Tile((Tensor<uint>)data, reps));
            case TensorElementType.UInt64: return Success(op, Tensor<ulong>.Tile((Tensor<ulong>)data, reps));
            case TensorElementType.Int8: return Success(op, Tensor<sbyte>.Tile((Tensor<sbyte>)data, reps));
            case TensorElementType.UInt8: return Success(op, Tensor<byte>.Tile((Tensor<byte>)data, reps));
            case TensorElementType.Int16: return Success(op, Tensor<short>.Tile((Tensor<short>)data, reps));
            case TensorElementType.UInt16: return Success(op, Tensor<ushort>.Tile((Tensor<ushort>)data, reps));
            case TensorElementType.Bool: return Success(op, Tensor<bool>.Tile((Tensor<bool>)data, reps));
            case TensorElementType.Float16: return Success(op, Tensor<Half>.Tile((Tensor<Half>)data, reps));
            default: return InputTypeNotSupported(op, nameof(data), data);
        }
    }
}
