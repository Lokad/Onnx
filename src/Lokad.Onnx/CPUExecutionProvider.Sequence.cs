namespace Lokad.Onnx;


using System;
using System.Collections.Generic;
using System.Linq;

using static OpResult;
public partial class CPUExecutionProvider
{

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
                chunk = ((INumericTensor)chunk).Reshape(chunk.Dims.Where((d, i) => i != ax).ToArray()).ToDenseTensor();
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
