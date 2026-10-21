namespace Lokad.Onnx.Tensors.Tests;

public class RangeGeneratorTests
{
    public static IEnumerable<object[]> IntegerCases => new[]
    {
        new object[] { 10, 0, -2, new[] { 10, 8, 6, 4, 2 } },
        new object[] { 10, 0, -3, new[] { 10, 7, 4, 1 } },
        new object[] { 1, 8, 3, new[] { 1, 4, 7 } },
        new object[] { 2, 4, -1, Array.Empty<int>() },
        new object[] { 4, 2, 1, Array.Empty<int>() },
        new object[] { 3, 3, -2, Array.Empty<int>() },
        new object[] { int.MaxValue, int.MinValue, int.MinValue, new[] { int.MaxValue, -1 } },
        new object[] { int.MinValue, int.MaxValue, int.MaxValue, new[] { int.MinValue, -1, int.MaxValue - 1 } },
    };

    [Theory]
    [MemberData(nameof(IntegerCases))]
    public void IntegerGeneratorsFollowTheSuppliedStartAndStep(int start, int limit, int step, int[] expected)
    {
        Assert.Equal(expected, Tensor<int>.Range(start, limit, step).ToArray());
        Assert.Equal(expected, Tensor<int>.Arange(start, limit, step).ToArray());
    }

    [Fact]
    public void FractionalDescendingGeneratorsStartAtStart()
    {
        Assert.Equal(new[] { 2f, 1.5f, 1f, .5f }, Tensor<float>.Arange(2f, 0f, -.5f).ToArray());
        Assert.Equal(new[] { .75f, .5f, .25f, 0f, -.25f }, Tensor<float>.Arange(.75f, -.5f, -.25f).ToArray());
        Assert.Empty(Tensor<float>.Arange(0f, 2f, -.5f).ToArray());
        Assert.Empty(Tensor<float>.Arange(2f, 0f, .5f).ToArray());
    }

    [Fact]
    public void FloatArangeKeepsTheQualifiedRangeFill()
    {
        var values = Tensor<float>.Arange(0f, 10f, .1f).ToArray();
        Assert.Equal(100, values.Length);
        Assert.Equal(9.900001525878906f, values[^1]);
        Assert.Equal(new[] { 0f, .1f, .2f, .3f }, Tensor<float>.Arange(0f, .3f, .1f).ToArray());
    }

    [Fact]
    public void ArangeUsesTheSameGeometryAndStorageLimits()
    {
        float bound = 1L << 48;
        float outside = MathF.BitIncrement(bound);
        Assert.ThrowsAny<ArgumentException>(() => Tensor<float>.Arange(outside, outside, 1f));
        Assert.ThrowsAny<ArgumentException>(() => Tensor<float>.Arange(0f, 1f, float.PositiveInfinity));
        Assert.ThrowsAny<ArgumentException>(() => Tensor<float>.Arange(0f, 1f, 0f));
        Assert.ThrowsAny<ArgumentException>(() => Tensor<int>.Arange(0, 1, 0));
        Assert.ThrowsAny<ArgumentException>(() => Tensor<int>.Arange(int.MinValue, int.MaxValue, 1));
        Assert.Empty(Tensor<float>.Arange(bound, bound, -1f).ToArray());
    }
}
