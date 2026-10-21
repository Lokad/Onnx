using System;
using System.Linq;
using System.Numerics;

namespace Lokad.Onnx.Tensors.Tests;

public class NormalizationOverlapTests
{
    static DenseTensor<T> Copy<T>(Tensor<T> input) where T : unmanaged
        => new(input.ToArray(), input.Dimensions);

    static void CheckLayer<T>(int layout, Func<Tensor<T>, Tensor<T>, Tensor<T>, DenseTensor<T>, Tensor<T>> normalize) where T : unmanaged, INumber<T>
    {
        var storage = new T[32];
        int sourceStart = layout == 2 ? 1 : 0;
        int destinationStart = layout == 1 ? 1 : layout == 5 ? 16 : 0;
        var x = new DenseTensor<T>(storage.AsMemory(sourceStart, 8), new[] { 2, 4 });
        var destination = new DenseTensor<T>(storage.AsMemory(destinationStart, 8), new[] { 2, 4 });
        var scale = layout is 3 or 6 ? new DenseTensor<T>(storage.AsMemory(destinationStart, 4), new[] { 4 }) : new DenseTensor<T>(4);
        var bias = layout is 4 or 6 ? new DenseTensor<T>(storage.AsMemory(destinationStart + 4, 4), new[] { 4 }) : new DenseTensor<T>(4);
        for (int i = 0; i < 8; i++) x.SetValue(i, T.CreateChecked(1 << i));
        for (int i = 0; i < 4; i++) { scale.SetValue(i, T.CreateChecked(i + 1)); bias.SetValue(i, T.CreateChecked(i - 2)); }
        var expected = normalize(Copy(x), Copy(scale), Copy(bias), DenseTensor<T>.OfShape(2, 4)).ToArray();
        Assert.Equal(expected, normalize(x, scale, bias, destination).ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    public void LayerNormalization_PreservesOriginalOverlappingValues(int layout)
    {
        CheckLayer<float>(layout, (x, s, b, d) => Tensor<float>.LayerNormalization(x, s, b, d, 1, 1e-5f));
        CheckLayer<double>(layout, (x, s, b, d) => Tensor<double>.LayerNormalization(x, s, b, d, 1, 1e-5));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void RotaryEmbedding_PreservesOriginalOverlappingValues(int layout)
    {
        var storage = new float[16];
        var destination = new DenseTensor<float>(storage.AsMemory(layout == 1 ? 1 : 0, 8), new[] { 2, 4 });
        var x = layout is 0 or 1 or 4 ? new DenseTensor<float>(storage.AsMemory(0, 8), new[] { 2, 4 }) : DenseTensor<float>.OfShape(2, 4);
        var cos = layout is 2 or 4 ? new DenseTensor<float>(storage.AsMemory(0, 4), new[] { 4 }) : new DenseTensor<float>(4);
        var sin = layout is 3 or 4 ? new DenseTensor<float>(storage.AsMemory(4, 4), new[] { 4 }) : new DenseTensor<float>(4);
        for (int i = 0; i < 8; i++) x.SetValue(i, i + 1);
        for (int i = 0; i < 4; i++) { cos.SetValue(i, i + 1); sin.SetValue(i, i + 2); }
        var expected = Tensor<float>.RotaryEmbedding(Copy(x), Copy(cos), Copy(sin), 2, 1, 1).ToArray();
        Assert.Equal(expected, Tensor<float>.RotaryEmbedding(x, cos, sin, destination, 2, 1, 1).ToArray());
    }

    [Fact]
    public void InvalidNormalization_DoesNotMutateAliasedDestination()
    {
        var x = DenseTensor<float>.Ones(2, 4).ToDenseTensor();
        var before = x.ToArray();
        Assert.Throws<ArgumentException>(() => Tensor<float>.LayerNormalization(x, DenseTensor<float>.Ones(3), null, x, 1, 1e-5f));
        Assert.Throws<ArgumentException>(() => Tensor<float>.RotaryEmbedding(x, x, x, x, 2, 1, 0));
        Assert.Throws<ArgumentException>(() => Tensor<float>.RotaryEmbedding(x, DenseTensor<float>.Ones(3), DenseTensor<float>.Ones(3), x, 2, 1, 1));
        Assert.Equal(before, x.ToArray());
    }

    [Fact]
    public void ExactLayerNormalizationInPlace_DoesNotCopyPayload()
    {
        var x = DenseTensor<float>.Ones(128, 128).ToDenseTensor();
        var scale = DenseTensor<float>.Ones(128);
        for (int i = 0; i < 10; i++) Tensor<float>.LayerNormalization(x, scale, null, x, 1, 1e-5f);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10; i++) Tensor<float>.LayerNormalization(x, scale, null, x, 1, 1e-5f);
        Assert.True((GC.GetAllocatedBytesForCurrentThread() - before) / 10 < 8192);
    }
}
