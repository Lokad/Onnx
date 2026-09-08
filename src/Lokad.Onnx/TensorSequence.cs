using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx;

/// <summary>
/// Ordered ONNX sequence runtime value produced by SplitToSequence and
/// consumed by SequenceAt. It implements <see cref="ITensor"/> so sequences
/// flow through graph bindings, validation and ownership analysis like other
/// runtime values: Rank is structurally 1, Dims holds the item count and
/// ElementType is Sequence, all for validation and logging (copy Dims before
/// mutating; the array is fresh per call). Numeric tensor operations
/// (reshape, slice, broadcast, densify, typed clones) are unsupported by
/// design and throw, exactly like descriptor placeholders; use SequenceAt to
/// extract elements and per-element ops to compute on them.
/// </summary>
public sealed class TensorSequence : ITensor
{
    private readonly List<ITensor> items;

    public TensorSequence()
    {
        items = new List<ITensor>();
    }

    public TensorSequence(IEnumerable<ITensor> items)
    {
        this.items = new List<ITensor>(items);
    }

    public IReadOnlyList<ITensor> Items => items;

    public string Name { get; set; } = "";

    public TensorElementType ElementType => TensorElementType.Sequence;

    public Type PrimitiveType => typeof(ITensor);

    public int[] Dims => new int[] { items.Count };

    public int Rank => 1;

    public long Length => items.Count;

    public void Add(ITensor item) => items.Add(item);

    public ITensor Clone() => new TensorSequence(items.Select(i => i.Clone()));

    public ITensor CloneEmpty() => new TensorSequence();

    public ITensor CloneEmpty<U>() where U : unmanaged => throw new NotSupportedException("CloneEmpty<U> is not supported for sequences.");

    public ITensor Reshape(params int[] shape) => throw new NotSupportedException("Reshape is not supported for sequences.");

    public ITensor Slice(string indices) => throw new NotSupportedException("Slice is not supported for sequences.");

    public ITensor InsertDim(int dim) => throw new NotSupportedException("InsertDim is not supported for sequences.");

    public ITensor RemoveDim(int dim) => throw new NotSupportedException("RemoveDim is not supported for sequences.");

    public ITensor BroadcastDim(int dim, int size) => throw new NotSupportedException("BroadcastDim is not supported for sequences.");

    public ITensor ToDenseTensor() => throw new NotSupportedException("ToDenseTensor is not supported for sequences.");

    public Array ToArray() => items.ToArray();

    public object this[params int[] indices]
    {
        get => GetValue(indices.Length == 1 ? indices[0] : throw new ArgumentException("Sequence index must be a single integer.", nameof(indices)));
        set => SetValue(indices.Length == 1 ? indices[0] : throw new ArgumentException("Sequence index must be a single integer.", nameof(indices)), value);
    }

    public ITensor this[params object[] indices]
    {
        get => (ITensor)GetValue(Convert.ToInt32(indices[0]));
        set => SetValue(Convert.ToInt32(indices[0]), value);
    }

    public object GetValue(int index) => items[index];

    public void SetValue(int index, object? value)
    {
        if (value is null) throw new ArgumentNullException(nameof(value), "Tensor sequences hold non-null tensors.");
        items[index] = (ITensor)value;
    }

    public IEnumerator GetEnumerator() => items.GetEnumerator();

    public string PrintShape() => "seq[" + items.Count + "]";

    public string PrintData(bool includeWhitespace) => "seq[" + items.Count + "]";
}
