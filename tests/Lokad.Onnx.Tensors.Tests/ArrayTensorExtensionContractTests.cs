namespace Lokad.Onnx.Tensors.Tests;

public class ArrayTensorExtensionContractTests
{
    [Fact]
    public void Print_ShowsValuesAndSupportsEmptyAndNullElements()
    {
        Assert.Equal("[2, 3]", new[] { 2, 3 }.Print());
        Assert.Equal("[a, b]", new List<string> { "a", "b" }.Print());
        Assert.Equal("[]", Enumerable.Empty<int>().Print());
        Assert.Equal("[a, , b]", new string?[] { "a", null, "b" }.Print());
    }

    static IEnumerable<T> Once<T>(T[] values, Action disposed)
    {
        int starts = 0;
        return Enumerate();

        IEnumerable<T> Enumerate()
        {
            if (++starts != 1) throw new InvalidOperationException("Input enumerated twice.");
            try { foreach (var value in values) yield return value; }
            finally { disposed(); }
        }
    }

    [Theory]
    [InlineData(new int[] { 2, 3 }, "[2, 3]")]
    [InlineData(new int[] { }, "[]")]
    public void Print_EnumeratesAndDisposesIteratorOnce(int[] values, string expected)
    {
        int disposed = 0;
        Assert.Equal(expected, Once(values, () => disposed++).Print());
        Assert.Equal(1, disposed);
    }

    public static IEnumerable<object[]> Comparisons()
    {
        yield return new object[] { Array.Empty<double>(), Array.Empty<double>(), true };
        yield return new object[] { Array.Empty<double>(), new[] { 42.0 }, false };
        yield return new object[] { new[] { 42.0 }, Array.Empty<double>(), false };
        yield return new object[] { new[] { 1.0 }, new[] { 1.0, 100.0 }, false };
        yield return new object[] { new[] { 1.0, 100.0 }, new[] { 1.0 }, false };
        yield return new object[] { new[] { 1.0, 2.0 }, new[] { 1.0, 2.0 }, true };
        yield return new object[] { new[] { 1.0, 2.0 }, new[] { 1.0, 3.0 }, false };
        yield return new object[] { new[] { 1.23456 }, new[] { 1.23457 }, true };
        yield return new object[] { new[] { double.NaN }, new[] { double.NaN }, false };
        yield return new object[] { new[] { double.PositiveInfinity }, new[] { double.PositiveInfinity }, true };
        yield return new object[] { new[] { double.NegativeInfinity }, new[] { double.PositiveInfinity }, false };
        yield return new object[] { new[] { -0.0 }, new[] { +0.0 }, true };
    }

    [Theory]
    [MemberData(nameof(Comparisons))]
    public void AlmostEqual_ChecksValuesAndBothLengths(double[] expected, double[] actual, bool equal)
    {
        Assert.Equal(equal, expected.AlmostEqual(actual, 4));
        Assert.Equal(equal, expected.Select(x => (float)x).AlmostEqual(actual.Select(x => (float)x), 4));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AlmostEqual_EnumeratesAndDisposesInputsOnce(bool equal)
    {
        int expectedDisposed = 0, actualDisposed = 0;
        var expected = Once(new[] { 1.0, 2.0 }, () => expectedDisposed++);
        var actual = Once(equal ? new[] { 1.0, 2.0 } : new[] { 1.0 }, () => actualDisposed++);
        Assert.Equal(equal, expected.AlmostEqual(actual, 4));
        Assert.Equal(1, expectedDisposed);
        Assert.Equal(1, actualDisposed);
    }

    [Fact]
    public void AlmostEqual_StopsAndDisposesOnFirstMismatch()
    {
        int expectedDisposed = 0, actualDisposed = 0;
        IEnumerable<float> Mismatch(float value, Action disposed)
        {
            try
            {
                yield return value;
                throw new InvalidOperationException("Comparison read past the mismatch.");
            }
            finally { disposed(); }
        }
        var expected = Mismatch(1, () => expectedDisposed++);
        var actual = Mismatch(2, () => actualDisposed++);
        Assert.False(expected.AlmostEqual(actual, 4));
        Assert.Equal(1, expectedDisposed);
        Assert.Equal(1, actualDisposed);
    }

    [Fact]
    public void AlmostEqual_DisposesInputsWhenTheScalarComparatorThrows()
    {
        int expectedDisposed = 0, actualDisposed = 0;
        var expected = Once(new[] { 1.0 }, () => expectedDisposed++);
        var actual = Once(new[] { 1.0 }, () => actualDisposed++);
        Assert.Throws<ArgumentOutOfRangeException>(() => expected.AlmostEqual(actual, -1));
        Assert.Equal(1, expectedDisposed);
        Assert.Equal(1, actualDisposed);
    }
}
