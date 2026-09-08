
namespace Lokad.Onnx.Backend.Tests;

[Collection("SequentialLogSink")]
public class TokenizerBatchTests
{
    static string AssetPath() =>
        ModelFixture.RequireModelOrSkip("e5 tokenizer", "models", "multilingual-e5-small", "sentencepiece.bpe.model");

    static Tensor<long> Part(ITensor[]? parts, int i)
    {
        Assert.NotNull(parts);
        return (Tensor<long>)parts![i];
    }

    [SkippableFact]
    public void EmptyBatch_ReturnsZeroShapedNamedTensors()
    {
        var parts = Text.RobertaTokenizeFromFile(System.Array.Empty<string>(), AssetPath());
        Assert.NotNull(parts);
        var ids = Part(parts, 0);
        var mask = Part(parts, 1);
        var types = Part(parts, 2);
        Assert.Equal(new int[] { 0, 0 }, ids.Dimensions.ToArray());
        Assert.Equal(new int[] { 0, 0 }, mask.Dimensions.ToArray());
        Assert.Equal(new int[] { 0, 0 }, types.Dimensions.ToArray());
        Assert.Equal("input_ids", ids.Name);
        Assert.Equal("attention_mask", mask.Name);
        Assert.Equal("token_type_ids", types.Name);
    }

    [SkippableFact]
    public void Batch_MatchesSerialSingles_Exactly()
    {
        string path = AssetPath();
        string[] texts = new[] { "Hello world", "query: What is the capital of France?", "a  b" };
        var batch = Text.RobertaTokenizeFromFile(texts, path);
        Assert.NotNull(batch);
        var singles = new System.Collections.Generic.List<long[]>();
        int max = 0;
        foreach (var s in texts)
        {
            var one = Text.RobertaTokenizeFromFile(s, path);
            Assert.NotNull(one);
            var arr = Part(one, 0).ToArray();
            singles.Add(arr);
            if (arr.Length > max) max = arr.Length;
        }
        var expectedIds = new long[texts.Length * max];
        var expectedMask = new long[texts.Length * max];
        var expectedTypes = new long[texts.Length * max];
        for (int i = 0; i < texts.Length; i++)
        {
            singles[i].CopyTo(expectedIds, i * max);
            for (int j = singles[i].Length; j < max; j++) expectedIds[i * max + j] = 1L;
            for (int j = 0; j < singles[i].Length; j++) expectedMask[i * max + j] = 1L;
        }
        Assert.Equal(expectedIds, Part(batch, 0).ToArray());
        Assert.Equal(expectedMask, Part(batch, 1).ToArray());
        Assert.Equal(expectedTypes, Part(batch, 2).ToArray());
        Assert.Equal(new int[] { texts.Length, max }, Part(batch, 0).Dimensions.ToArray());
    }

    [SkippableFact]
    public void Batch_Truncates_LikeSingle()
    {
        string path = AssetPath();
        string longText = string.Concat(System.Linq.Enumerable.Repeat("Hello world ", 400));
        var batch = Text.RobertaTokenizeFromFile(new[] { "Hi", longText }, path);
        Assert.NotNull(batch);
        var single = Text.RobertaTokenizeFromFile(longText, path);
        Assert.NotNull(single);
        var singleIds = Part(single, 0).ToArray();
        Assert.Equal(512, singleIds.Length);
        var batchIds = Part(batch, 0).ToArray();
        int max = Part(batch, 0).Dimensions[1];
        Assert.Equal(512, max);
        var tail = new long[512];
        System.Array.Copy(batchIds, max, tail, 0, 512);
        Assert.Equal(singleIds, tail);
    }
}
