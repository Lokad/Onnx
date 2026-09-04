using System.IO;

namespace Lokad.Onnx.Backend.Tests;

public class TokenizerTests
{
    static string AssetPath()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "models", "multilingual-e5-small", "sentencepiece.bpe.model");
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        throw new FileNotFoundException("e5 tokenizer asset not found under models/multilingual-e5-small.");
    }

    static long[] Ids(string text)
    {
        var parts = Text.RobertaTokenizeFromFile(text, AssetPath());
        Assert.NotNull(parts);
        return ((Tensor<long>)parts![0]).ToArray();
    }

    [Fact]
    public void HelloWorld_HasLiteralIds()
    {
        Assert.Equal(new long[] { 0, 35378, 8999, 2 }, Ids("Hello world"));
    }

    [Fact]
    public void MultiSpace_CollapsesToSingleSpaceIds()
    {
        Assert.Equal(new long[] { 0, 10, 876, 2 }, Ids("a  b"));
        Assert.Equal(Ids("a b"), Ids("a   b"));
    }

    [Fact]
    public void Batch_PadsToLongestWithMasks()
    {
        var parts = Text.RobertaTokenizeFromFile(new[] { "Hello world", "query: What is the capital of France?" }, AssetPath());
        Assert.NotNull(parts);
        var ids = (Tensor<long>)parts![0];
        var mask = (Tensor<long>)parts[1];
        var types = (Tensor<long>)parts[2];
        Assert.Equal(2, ids.Dimensions[0]);
        int n = ids.Dimensions[1];
        Assert.True(n > 4);
        Assert.Equal(ids.Dimensions[1], mask.Dimensions[1]);
        Assert.Equal(ids.Dimensions[1], types.Dimensions[1]);
        Assert.Equal(new long[] { 0, 35378, 8999, 2 }, new long[] { ids[0, 0], ids[0, 1], ids[0, 2], ids[0, 3] });
        Assert.Equal(1, ids[0, 4]);
        Assert.Equal(1, mask[0, 0]);
        Assert.Equal(0, mask[0, 4]);
        Assert.Equal(0, types[0, 0]);
        Assert.Equal(0, types[1, 0]);
    }

    [Fact]
    public void MissingModelPath_ThrowsWithoutCacheChange()
    {
        int before = Text.Tokenizers.Count;
        Assert.Throws<FileNotFoundException>(() => Text.RobertaTokenizeFromFile("Hello world", "no-such-tokenizer.bpe.model"));
        Assert.Equal(before, Text.Tokenizers.Count);
    }
}
