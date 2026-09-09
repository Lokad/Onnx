namespace Lokad.Onnx.Backend.Tests;

[Collection("ProcessState")]
public class BertTokenizerTests
{
    static string WriteVocab()
    {
        var tmp = Path.Combine(Path.GetTempPath(), "lokad-bert-" + Guid.NewGuid().ToString("N") + ".txt");
        File.WriteAllLines(tmp, new string[] { "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world" });
        return tmp;
    }

    [Fact]
    public void FileVocab_TokenizesDeterministically()
    {
        string tmp = WriteVocab();
        try
        {
            var first = Text.BertTokenizeFromFile("hello world", tmp);
            var second = Text.BertTokenizeFromFile("hello world", tmp);
            Assert.NotNull(first);
            Assert.NotNull(second);
            Assert.Equal(new long[] { 2, 5, 6, 3 }, ((Tensor<long>)first![0]).ToArray());
            Assert.Equal(((Tensor<long>)first![0]).ToArray(), ((Tensor<long>)second![0]).ToArray());
            Assert.Equal(new long[] { 1, 1, 1, 1 }, ((Tensor<long>)first![1]).ToArray());
        }
        finally
        {
            File.Delete(tmp);
        }
    }

    [Fact]
    public async Task ConcurrentFileTokenize_MatchesSerial()
    {
        string tmp = WriteVocab();
        try
        {
            var serial = Text.BertTokenizeFromFile("hello world", tmp);
            Assert.NotNull(serial);
            var expected = ((Tensor<long>)serial![0]).ToArray();
            var tasks = new Task<long[]>[8];
            for (int i = 0; i < tasks.Length; i++)
                tasks[i] = Task.Run(() =>
                {
                    var parts = Text.BertTokenizeFromFile("hello world", tmp);
                    Assert.NotNull(parts);
                    return ((Tensor<long>)parts![0]).ToArray();
                });
            var results = await Task.WhenAll(tasks);
            foreach (var ids in results) Assert.Equal(expected, ids);
        }
        finally
        {
            File.Delete(tmp);
        }
    }

    [Fact]
    public void TouchedVocab_ReloadsWithStableResults()
    {
        string tmp = WriteVocab();
        try
        {
            var first = Text.GetOrLoadBertFileTokenizer(tmp);
            var baseline = Text.BertTokenizeFromFile("hello world", tmp);
            Assert.NotNull(baseline);
            File.SetLastWriteTimeUtc(tmp, File.GetLastWriteTimeUtc(tmp).AddHours(1));
            var second = Text.GetOrLoadBertFileTokenizer(tmp);
            Assert.NotSame(first, second);
            var again = Text.BertTokenizeFromFile("hello world", tmp);
            Assert.NotNull(again);
            Assert.Equal(
                ((Tensor<long>)baseline![0]).ToArray(),
                ((Tensor<long>)again![0]).ToArray());
        }
        finally
        {
            File.Delete(tmp);
        }
    }

    [Fact]
    public void DeletedVocab_EvictsInsteadOfServingStale()
    {
        string tmp = WriteVocab();
        try
        {
            var first = Text.GetOrLoadBertFileTokenizer(tmp);
            File.Delete(tmp);
            Assert.Throws<FileNotFoundException>(() => Text.GetOrLoadBertFileTokenizer(tmp));
            File.WriteAllLines(tmp, new string[] { "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world" });
            var second = Text.GetOrLoadBertFileTokenizer(tmp);
            Assert.NotSame(first, second);
        }
        finally
        {
            if (File.Exists(tmp)) File.Delete(tmp);
        }
    }

    [Fact]
    public void MissingVocab_ThrowsWithoutFetching()
    {
        Assert.Throws<FileNotFoundException>(() => Text.LoadBertTokenizerFromFile("no-such-vocab.txt"));
        Assert.Throws<FileNotFoundException>(() => Text.BertTokenizeFromFile("hello world", "no-such-vocab.txt"));
    }
}
