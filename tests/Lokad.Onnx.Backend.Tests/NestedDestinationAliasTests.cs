namespace Lokad.Onnx.Backend.Tests;

public class NestedDestinationAliasTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NestedSliceOverlap_IsRejectedBeforeWriting(bool transpose)
    {
        var values = new float[] { 9, 9, 9, 9, 1, 2, 3, 4 };
        var original = values.ToArray();
        var root = new DenseTensor<float>(values, new[] { 4, 2 });
        var source = root.Slice(new SliceIndex(2, 4), SliceIndex.All).Slice(SliceIndex.All, SliceIndex.All);
        var destination = new DenseTensor<float>(values.AsMemory(4, 4), new[] { 2, 2 });
        Assert.Throws<ArgumentException>(() => Write(source, destination, transpose));
        Assert.Equal(original, values);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NestedSliceDisjointDestination_RemainsUsable(bool transpose)
    {
        var values = new float[] { 9, 9, 1, 2, 3, 4, 9, 9, 0, 0, 0, 0 };
        var root = new DenseTensor<float>(values, new[] { 6, 2 });
        var source = root.Slice(new SliceIndex(1, 3), SliceIndex.All).Slice(SliceIndex.All, SliceIndex.All);
        var destination = new DenseTensor<float>(values.AsMemory(8, 4), new[] { 2, 2 });
        Write(source, destination, transpose);
        Assert.Equal(transpose ? new[] { 1f, 3f, 2f, 4f } : new[] { 1f, 2f, 3f, 4f }, destination.ToArray());
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, source.ToArray());
    }

    [Fact]
    public void UnknownLayout_CannotBypassOverlapProtection()
    {
        var values = new[] { 1f, 2f, 3f, 4f };
        var source = new HiddenTensor(values);
        var destination = new DenseTensor<float>(values, new[] { 2, 2 });
        Assert.Throws<ArgumentException>(() => Write(source, destination, false));
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, values);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NestedStorageIndices_ReferToRootValues(bool reversed)
    {
        var root = new DenseTensor<float>(Enumerable.Range(0, 8).Select(i => (float)i).ToArray(), new[] { 2, 4 }, reversed);
        var source = root.Slice(SliceIndex.All, new SliceIndex(1, 3)).Slice(SliceIndex.All, SliceIndex.All);
        CheckRootIndices(source);
        Assert.Equal(reversed ? new float[] { 2, 4, 3, 5 } : new float[] { 1, 2, 5, 6 },
            new[] { source[0, 0], source[0, 1], source[1, 0], source[1, 1] });
        var row = root.Slice(new SliceIndex(1, 2), new SliceIndex(1, 3));
        var broadcast = new BroadcastedTensor<float>(row, new[] { 2, 2 }, new[] { 0 });
        CheckRootIndices(broadcast.Slice(SliceIndex.All, SliceIndex.All));
        Assert.Equal(reversed ? new float[] { 3, 5, 3, 5 } : new float[] { 5, 6, 5, 6 },
            new[] { broadcast[0, 0], broadcast[0, 1], broadcast[1, 0], broadcast[1, 1] });
    }

    static void CheckRootIndices(Tensor<float> source)
    {
        for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
            var coordinates = new[] { i, j };
            Assert.Equal(source[coordinates], source.Storage.Span[source.GetStorageIndex(coordinates)]);
        }
    }

    static void Write(Tensor<float> source, DenseTensor<float> destination, bool transpose)
    {
        if (transpose) Tensor<float>.Transpose(source, destination, new[] { 1, 0 });
        else Tensor<float>.MatMul2D(source,
            DenseTensor<float>.OfValues(new float[,] { { 1, 0 }, { 0, 1 } }), destination, TensorExecutionOptions.Scalar);
    }

    sealed class HiddenTensor(float[] values) : Tensor<float>(new[] { 2, 2 }, false)
    {
        public override float GetValue(int index) => values[index];
        public override void SetValue(int index, float value) => values[index] = value;
        public override Tensor<float> Clone() => new DenseTensor<float>(values.ToArray(), new[] { 2, 2 });
        public override Tensor<T> CloneEmpty<T>(ReadOnlySpan<int> dimensions) => new DenseTensor<T>(dimensions);
        public override Tensor<float> Reshape(ReadOnlySpan<int> dimensions) => Clone().Reshape(dimensions);
    }
}
