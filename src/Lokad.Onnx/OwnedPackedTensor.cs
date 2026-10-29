namespace Lokad.Onnx;

/// <summary>Logical float values with one owned payload in the existing 32-column panel layout.</summary>
internal sealed class OwnedPackedTensor : Tensor<float>
{
    internal readonly float[] PackedArray;
    internal readonly int Reduction, Columns;
    readonly int blocked;
    readonly bool transposedLogical;
    readonly OwnedPackedTensor? physicalMatrix;
    internal bool IsTransposedLogical => transposedLogical;
    internal OwnedPackedTensor PhysicalMatrix => physicalMatrix ?? this;

    internal OwnedPackedTensor(float[] source, int reduction, int columns)
        : this(Pack(source, reduction, columns), reduction, columns, new[] { reduction, columns }) { }

    OwnedPackedTensor(float[] packed, int reduction, int columns, ReadOnlySpan<int> shape, bool transposedLogical = false)
        : base(shape, false)
    {
        if (Length != packed.Length) throw new ArgumentException("Reshape must preserve the element count.", nameof(shape));
        PackedArray = packed; Reduction = reduction; Columns = columns; blocked = columns - columns % 32;
        this.transposedLogical = transposedLogical;
        if (transposedLogical)
            physicalMatrix = new OwnedPackedTensor(packed, reduction, columns, new[] { reduction, columns });
    }

    // Replace [outputs,channels,1] pointwise weights with one payload packed as
    // [channels,outputs]. Logical reads/writes keep the original tensor order.
    internal static OwnedPackedTensor FromPointwise(float[] source, int channels, int outputs)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (channels < 1 || outputs < 1) throw new ArgumentOutOfRangeException(nameof(channels));
        if (checked(channels * outputs) != source.Length) throw new ArgumentException("Complete pointwise weights required.", nameof(source));
        var packed = new float[source.Length];
        for (int column = 0; column < outputs; column += 32)
        {
            int count = Math.Min(32, outputs - column);
            for (int row = 0; row < channels; row++)
                for (int lane = 0; lane < count; lane++)
                    packed[column * channels + row * count + lane] = source[(column + lane) * channels + row];
        }
        return new OwnedPackedTensor(packed, channels, outputs, new[] { outputs, channels, 1 }, true);
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
        int row = transposedLogical ? index % Reduction : index / Columns;
        int column = transposedLogical ? index / Reduction : index % Columns;
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
        if (transposedLogical)
        {
            for (int column = 0; column < Columns; column += 32)
            {
                int count = Math.Min(32, Columns - column);
                for (int row = 0; row < Reduction; row++)
                    for (int lane = 0; lane < count; lane++)
                        output[(column + lane) * Reduction + row] = PackedArray[column * Reduction + row * count + lane];
            }
            return dense;
        }
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
    public override Tensor<float> Reshape(ReadOnlySpan<int> shape) => new OwnedPackedTensor(PackedArray, Reduction, Columns, shape, transposedLogical);

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
        if (core is not OwnedPackedTensor packed || packed.transposedLogical || core.Rank < 2
            || core.Dimensions[^2] != packed.Reduction || core.Dimensions[^1] != packed.Columns
            || value.Dimensions[^2] != packed.Reduction || value.Dimensions[^1] != packed.Columns) return null;
        for (int i = 0; i < core.Rank - 2; i++) if (core.Dimensions[i] != 1) return null;
        return packed;
    }
}
