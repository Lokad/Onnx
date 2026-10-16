namespace Lokad.Onnx.Backend.Tests;

using System.Reflection;
using System.Runtime.ExceptionServices;

public class EmbeddingGemma2Tests
{
    static object Call(string type, string method, object[] args)
    {
        var implementation = typeof(EmbeddingGemma2Embedder).Assembly.GetType("Lokad.Onnx." + type, true)!;
        try { return implementation.GetMethod(method, BindingFlags.NonPublic | BindingFlags.Static)!.Invoke(null, args)!; }
        catch (TargetInvocationException e) when (e.InnerException is not null)
        { ExceptionDispatchInfo.Capture(e.InnerException).Throw(); throw; }
    }

    static float[][] Pool(float[] values, int batch, int sequence, long[] mask, int dimensions, CancellationToken cancellation) =>
        (float[][])Call("EmbeddingGemma2Pooling", "Pool", new object[] {
            new DenseTensor<float>(values, new[] { batch, sequence, 768 }),
            new DenseTensor<long>(mask, new[] { batch, sequence }), dimensions, cancellation });

    [Theory]
    [InlineData(128)] [InlineData(256)] [InlineData(512)] [InlineData(768)]
    public void MaskedMeanTruncatesBeforeNormalizing_AndReturnsOwnedRows(int dimensions)
    {
        var values = new float[2 * 3 * 768];
        values[0] = 1; values[1] = 2; values[200] = 10;
        values[768] = 5; values[769] = 6; values[968] = 14;
        Array.Fill(values, float.NaN, 2 * 768, 768); // Excluded padding.
        values[3 * 768] = 8; values[3 * 768 + 1] = 6;
        var mask = new long[] { 1, 1, 0, 1, 0, 0 };
        float[] before = values.ToArray();
        var actual = Pool(values, 2, 3, mask, dimensions, CancellationToken.None);
        double norm = dimensions == 128 ? 5 : 13;
        Assert.Equal(2, actual.Length);
        Assert.All(actual, row => Assert.Equal(dimensions, row.Length));
        Assert.InRange(Math.Abs(actual[0][0] - 3 / norm), 0, 1e-7);
        Assert.InRange(Math.Abs(actual[0][1] - 4 / norm), 0, 1e-7);
        if (dimensions > 200) Assert.InRange(Math.Abs(actual[0][200] - 12 / norm), 0, 1e-7);
        Assert.InRange(Math.Abs(actual[1][0] - .8), 0, 1e-7);
        Assert.InRange(Math.Abs(actual[1][1] - .6), 0, 1e-7);
        Assert.All(actual, row => Assert.InRange(Math.Abs(row.Sum(x => (double)x * x) - 1), 0, 1e-6));
        Assert.Equal(before, values); Assert.Equal(new long[] { 1, 1, 0, 1, 0, 0 }, mask);
        actual[0][0] = 100;
        Assert.InRange(Math.Abs(actual[1][0] - .8), 0, 1e-7);
        var repeated = Pool(values, 2, 3, mask, dimensions, CancellationToken.None);
        Assert.InRange(Math.Abs(repeated[0][0] - 3 / norm), 0, 1e-7);
    }

    [Theory]
    [InlineData(float.NaN)] [InlineData(float.PositiveInfinity)] [InlineData(float.NegativeInfinity)]
    public void NonfiniteRetainedValuesFail(float bad)
    {
        var values = new float[768]; values[0] = bad;
        Assert.Throws<InvalidDataException>(() => Pool(values, 1, 1, new long[] { 1 }, 128, CancellationToken.None));
    }

    [Fact]
    public void EmptyOrNonbinaryMask_ZeroNorm_InvalidDimension_AndCancellationFail()
    {
        var values = new float[768]; values[0] = 1;
        Assert.Throws<InvalidDataException>(() => Pool(values, 1, 1, new long[] { 0 }, 128, CancellationToken.None));
        Assert.Throws<InvalidDataException>(() => Pool(values, 1, 1, new long[] { 2 }, 128, CancellationToken.None));
        Assert.Throws<InvalidDataException>(() => Pool(new float[768], 1, 1, new long[] { 1 }, 128, CancellationToken.None));
        Assert.Throws<ArgumentOutOfRangeException>(() => Pool(values, 1, 1, new long[] { 1 }, 100, CancellationToken.None));
        Assert.Throws<OperationCanceledException>(() => Pool(values, 1, 1, new long[] { 1 }, 128, new CancellationToken(true)));
    }

    [Fact]
    public void ShapeMismatchIsRejected()
    {
        Assert.Throws<InvalidDataException>(() => Call("EmbeddingGemma2Pooling", "Pool", new object[] {
            DenseTensor<float>.OfValues(new float[] { 1 }), DenseTensor<long>.OfValues(new long[] { 1 }), 128, CancellationToken.None }));
    }

    [Fact]
    public void PromptsAreExact_AndPreserveUserText()
    {
        Assert.Equal("task: search result | query:  café \t", Call("EmbeddingGemma2Inputs", "Query", new object[] { " café \t" }));
        Assert.Equal("title: none | text: ", Call("EmbeddingGemma2Inputs", "Document", new object[] { new EmbeddingGemma2Document("") }));
        Assert.Equal("title: none | text: test", Call("EmbeddingGemma2Inputs", "Document", new object[] { new EmbeddingGemma2Document("test", "") }));
        Assert.Equal("title:  X  | text:  a  b", Call("EmbeddingGemma2Inputs", "Document", new object[] { new EmbeddingGemma2Document(" a  b", " X ") }));
    }

    [Fact]
    public void InvalidOptionsFailBeforeLoadingAssets()
    {
        var invalid = new[] { new EmbeddingGemma2Options { MaximumTokens = 1 }, new EmbeddingGemma2Options { MaximumTokens = 8193 },
            new EmbeddingGemma2Options { MaximumBatchSize = 0 }, new EmbeddingGemma2Options { MaximumPaddedTokens = 511 },
            new EmbeddingGemma2Options { PreparedWeightBudgetBytes = -1 } };
        foreach (var options in invalid)
            Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingGemma2Embedder("missing-model-directory", options));
    }
}
