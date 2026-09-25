namespace Lokad.Onnx;

/// <summary>Logical float values with one owned payload in the existing 32-column panel layout.</summary>
internal sealed class OwnedPackedTensor : Tensor<float>
{
    internal readonly float[] PackedArray;
    internal readonly int Reduction, Columns;
    readonly int blocked;

    internal OwnedPackedTensor(float[] source, int reduction, int columns)
        : this(Pack(source, reduction, columns), reduction, columns, new[] { reduction, columns }) { }

    OwnedPackedTensor(float[] packed, int reduction, int columns, ReadOnlySpan<int> shape)
        : base(shape, false)
    {
        if (Length != packed.Length) throw new ArgumentException("Reshape must preserve the element count.", nameof(shape));
        PackedArray = packed; Reduction = reduction; Columns = columns; blocked = columns - columns % 32;
    }

    static unsafe float[] Pack(float[] source, int reduction, int columns)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (reduction < 1 || columns < 1) throw new ArgumentOutOfRangeException(nameof(reduction));
        if (checked(reduction * columns) != source.Length) throw new ArgumentException("Complete source matrix required.");
        var packed = new float[source.Length];
        fixed (float* original = source, destination = packed)
            MathOps.ShortWidePackPanelsB(reduction, columns, original, destination);
        return packed;
    }

    int PhysicalIndex(int index)
    {
        if ((uint)index >= (uint)PackedArray.Length) throw new ArgumentOutOfRangeException(nameof(index));
        int row = index / Columns, column = index % Columns;
        return column < blocked
            ? (column / 32) * Reduction * 32 + row * 32 + column % 32
            : blocked * Reduction + row * (Columns - blocked) + column - blocked;
    }

    int LogicalIndex(ReadOnlySpan<int> indices)
    {
        if (indices.Length != Rank) throw new ArgumentException("Coordinate rank must match the tensor.");
        int offset = 0;
        for (int i = 0; i < Rank; i++)
        {
            if ((uint)indices[i] >= (uint)dimensions[i]) throw new ArgumentOutOfRangeException(nameof(indices));
            offset += indices[i] * strides[i];
        }
        return offset;
    }

    public override float GetValue(int index) => PackedArray[PhysicalIndex(index)];
    public override void SetValue(int index, float value) => PackedArray[PhysicalIndex(index)] = value;
    public override float this[ReadOnlySpan<int> indices]
    {
        get => GetValue(LogicalIndex(indices));
        set => SetValue(LogicalIndex(indices), value);
    }

    public override DenseTensor<float> ToDenseTensor()
    {
        var dense = new DenseTensor<float>(Dimensions);
        var output = dense.Buffer.Span;
        for (int column = 0; column < blocked; column += 32)
            for (int row = 0; row < Reduction; row++)
                PackedArray.AsSpan(column * Reduction + row * 32, 32).CopyTo(output.Slice(row * Columns + column, 32));
        int tail = Columns - blocked;
        if (tail > 0)
            for (int row = 0; row < Reduction; row++)
                PackedArray.AsSpan(blocked * Reduction + row * tail, tail).CopyTo(output.Slice(row * Columns + blocked, tail));
        return dense;
    }

    public override Tensor<float> Clone() => ToDenseTensor();
    public override Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> shape) => new DenseTensor<TResult>(shape);
    public override Tensor<float> Reshape(ReadOnlySpan<int> shape) => new OwnedPackedTensor(PackedArray, Reduction, Columns, shape);

    // Alias protection covers the entire permuted payload, including arbitrary slices.
    internal static OwnedPackedTensor? FindStorage(Tensor<float> value) => value switch
    {
        OwnedPackedTensor packed => packed,
        BroadcastedTensor<float> broadcast => FindStorage(broadcast.source),
        TensorSlice<float> slice => FindStorage(slice.parent),
        _ => null
    };

    // Packed execution requires one complete original matrix shared across batches.
    internal static OwnedPackedTensor? Resolve(Tensor<float> value)
    {
        if (value.Rank < 2) return null;
        Tensor<float> core = value;
        while (core is BroadcastedTensor<float> broadcast)
        {
            var steps = broadcast.effectiveStrides;
            if (steps is null || steps.Length != core.Rank || steps[^1] != 1 || steps[^2] != core.Dimensions[^1]) return null;
            for (int i = 0; i < steps.Length - 2; i++) if (steps[i] != 0 && core.Dimensions[i] != 1) return null;
            core = broadcast.source;
        }
        if (core is not OwnedPackedTensor packed || core.Rank < 2
            || core.Dimensions[^2] != packed.Reduction || core.Dimensions[^1] != packed.Columns
            || value.Dimensions[^2] != packed.Reduction || value.Dimensions[^1] != packed.Columns) return null;
        for (int i = 0; i < core.Rank - 2; i++) if (core.Dimensions[i] != 1) return null;
        return packed;
    }
}
