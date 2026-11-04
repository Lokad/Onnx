namespace Lokad.Onnx;

using System;
using System.Numerics;
using static OpResult;

public partial class CPUExecutionProvider
{
    public static OpResult Flatten(ITensor? input, int? axis, ExecutionOptions? options)
    {
        const OpType op = OpType.Flatten;
        if (input is null) return MissingInput(op, nameof(input));
        int dim = axis ?? 1;
        if (dim < 0) dim += input.Rank;
        if (dim < 0 || dim > input.Rank)
            return AttributeNotSupported(op, nameof(axis), (axis ?? 1).ToString(), "Axis must be in [-rank, rank].");
        long left = 1, right = 1;
        for (int d = 0; d < input.Rank; d++)
            if (d < dim) left = TensorLimits.MultiplyDimensions(left, input.Dims[d], nameof(input));
            else right = TensorLimits.MultiplyDimensions(right, input.Dims[d], nameof(input));
        var result = Reshape(input, DenseTensor<long>.OfValues(new[] { left, right }), true, options);
        result.Op = op;
        return result;
    }

    public static OpResult GatherElements(ITensor? data, ITensor? indices, int? axis, ExecutionOptions? options)
    {
        const OpType op = OpType.GatherElements;
        if (data is null) return MissingInput(op, nameof(data));
        if (indices is null) return MissingInput(op, nameof(indices));
        (options ?? ExecutionOptions.Default).Validated();
        if (indices.ElementType != TensorElementType.Int32 && indices.ElementType != TensorElementType.Int64)
            return WrongInputType(op, nameof(indices), "Indices must be int32 or int64.", indices);
        if (data.Rank < 1 || indices.Rank != data.Rank)
            return WrongInputShape(op, nameof(indices), indices, "Data and indices must have equal, positive rank.");
        int dim = axis ?? 0;
        if (dim < 0) dim += data.Rank;
        if (dim < 0 || dim >= data.Rank)
            return AttributeNotSupported(op, nameof(axis), (axis ?? 0).ToString(), "Axis is outside the data rank.");
        for (int d = 0; d < data.Rank; d++)
            if (d != dim && indices.Dims[d] > data.Dims[d])
                return WrongInputShape(op, nameof(indices), indices, "A non-axis indices extent exceeds the data extent.");
        var positions = indices is Tensor<long> longs ? longs.ToArray() : Array.ConvertAll(((Tensor<int>)indices).ToArray(), x => (long)x);
        int extent = data.Dims[dim];
        foreach (long position in positions)
            if (position < -extent || position >= extent)
                return Failure(op, "GatherElements index is outside the axis extent.");
        return data switch
        {
            Tensor<float> t => Success(op, GatherElementValues(t, positions, indices.Dims, dim)),
            Tensor<double> t => Success(op, GatherElementValues(t, positions, indices.Dims, dim)),
            Tensor<long> t => Success(op, GatherElementValues(t, positions, indices.Dims, dim)),
            Tensor<int> t => Success(op, GatherElementValues(t, positions, indices.Dims, dim)),
            Tensor<bool> t => Success(op, GatherElementValues(t, positions, indices.Dims, dim)),
            Tensor<Half> t => Success(op, GatherElementValues(t, positions, indices.Dims, dim)),
            Tensor<sbyte> t => Success(op, GatherElementValues(t, positions, indices.Dims, dim)),
            Tensor<byte> t => Success(op, GatherElementValues(t, positions, indices.Dims, dim)),
            Tensor<short> t => Success(op, GatherElementValues(t, positions, indices.Dims, dim)),
            Tensor<ushort> t => Success(op, GatherElementValues(t, positions, indices.Dims, dim)),
            Tensor<uint> t => Success(op, GatherElementValues(t, positions, indices.Dims, dim)),
            Tensor<ulong> t => Success(op, GatherElementValues(t, positions, indices.Dims, dim)),
            _ => InputTypeNotSupported(op, nameof(data), data),
        };
    }

    static DenseTensor<T> GatherElementValues<T>(Tensor<T> data, long[] positions, int[] shape, int axis) where T : unmanaged
    {
        var result = new DenseTensor<T>(shape);
        for (int i = 0; i < positions.Length; i++)
        {
            int remaining = i, source = 0, stride = 1;
            for (int d = shape.Length - 1; d >= 0; d--)
            {
                int coordinate = remaining % shape[d];
                remaining /= shape[d];
                if (d == axis) coordinate = (int)(positions[i] < 0 ? positions[i] + data.Dimensions[d] : positions[i]);
                source += coordinate * stride;
                stride *= data.Dimensions[d];
            }
            result.SetValue(i, data.GetLogicalValue(source));
        }
        return result;
    }

    public static OpResult CumSum(ITensor? input, ITensor? axis, int? exclusive, int? reverse, ExecutionOptions? options)
    {
        const OpType op = OpType.CumSum;
        if (input is null) return MissingInput(op, nameof(input));
        if (axis is null) return MissingInput(op, nameof(axis));
        (options ?? ExecutionOptions.Default).Validated();
        if (exclusive is not null && exclusive != 0 && exclusive != 1)
            return AttributeNotSupported(op, nameof(exclusive), exclusive.ToString()!, "Expected 0 or 1.");
        if (reverse is not null && reverse != 0 && reverse != 1)
            return AttributeNotSupported(op, nameof(reverse), reverse.ToString()!, "Expected 0 or 1.");
        if (axis.Rank != 0) return WrongInputShape(op, nameof(axis), 0, axis);
        if (axis.ElementType != TensorElementType.Int32 && axis.ElementType != TensorElementType.Int64)
            return WrongInputType(op, nameof(axis), "Axis must be an int32 or int64 scalar.", axis);
        long dim = ToInt64Scalar(axis, nameof(axis));
        if (dim < 0) dim += input.Rank;
        if (dim < 0 || dim >= input.Rank) return Failure(op, "CumSum axis is outside the input rank.");
        return input switch
        {
            Tensor<float> t => Success(op, CumSumValues(t, (int)dim, exclusive == 1, reverse == 1)),
            Tensor<double> t => Success(op, CumSumValues(t, (int)dim, exclusive == 1, reverse == 1)),
            Tensor<int> t => Success(op, CumSumValues(t, (int)dim, exclusive == 1, reverse == 1)),
            Tensor<long> t => Success(op, CumSumValues(t, (int)dim, exclusive == 1, reverse == 1)),
            Tensor<uint> t => Success(op, CumSumValues(t, (int)dim, exclusive == 1, reverse == 1)),
            Tensor<ulong> t => Success(op, CumSumValues(t, (int)dim, exclusive == 1, reverse == 1)),
            _ => InputTypeNotSupported(op, nameof(input), input),
        };
    }

    static DenseTensor<T> CumSumValues<T>(Tensor<T> input, int axis, bool exclusive, bool reverse) where T : unmanaged, INumber<T>
    {
        var result = new DenseTensor<T>(input.Dimensions.ToArray());
        if (result.Length == 0) return result;
        var values = input.ToArray();
        int inner = 1;
        for (int d = axis + 1; d < input.Rank; d++) inner *= input.Dimensions[d];
        int length = input.Dimensions[axis], outer = values.Length / length / inner;
        for (int block = 0; block < outer; block++)
            for (int column = 0; column < inner; column++)
            {
                T sum = T.Zero;
                for (int step = 0; step < length; step++)
                {
                    int position = (block * length + (reverse ? length - 1 - step : step)) * inner + column;
                    if (exclusive) result.SetValue(position, sum);
                    sum += values[position];
                    if (!exclusive) result.SetValue(position, sum);
                }
            }
        return result;
    }
}
