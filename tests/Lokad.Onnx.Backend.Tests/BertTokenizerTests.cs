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

    static void WithVocabDirectory(Action<string, string> exercise)
    {
        string originalDirectory = Directory.GetCurrentDirectory();
        string directory = Path.GetFullPath(Path.Combine(Path.GetTempPath(), "lokad-bert-path-" + Guid.NewGuid().ToString("N")));
        string file = "vocab-" + Guid.NewGuid().ToString("N") + ".txt";
        Directory.CreateDirectory(directory);
        try { exercise(directory, file); }
        finally
        {
            Directory.SetCurrentDirectory(originalDirectory);
            Assert.True(directory.StartsWith(Path.GetFullPath(Path.GetTempPath()), StringComparison.Ordinal));
            Directory.Delete(directory, true);
        }
    }

    static void WriteVocab(string path, bool reversed = false)
    {
        File.WriteAllLines(path, new[] { "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            reversed ? "world" : "hello", reversed ? "hello" : "world" });
        File.SetLastWriteTimeUtc(path, new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc));
    }

    [Fact]
    public void RelativeVocabResolvesAgainstEachCallsDirectory()
    {
        WithVocabDirectory((directory, file) =>
        {
            string first = Directory.CreateDirectory(Path.Combine(directory, "first")).FullName;
            string second = Directory.CreateDirectory(Path.Combine(directory, "second")).FullName;
            WriteVocab(Path.Combine(first, file));
            WriteVocab(Path.Combine(second, file), reversed: true);
            Assert.Equal(new FileInfo(Path.Combine(first, file)).Length, new FileInfo(Path.Combine(second, file)).Length);
            Assert.Equal(File.GetLastWriteTimeUtc(Path.Combine(first, file)), File.GetLastWriteTimeUtc(Path.Combine(second, file)));

            Directory.SetCurrentDirectory(first);
            var firstTokenizer = Text.GetOrLoadBertFileTokenizer(file);
            Assert.Equal(new long[] { 2, 5, 3 }, ((Tensor<long>)Text.BertTokenizeFromFile("hello", file)![0]).ToArray());
            Directory.SetCurrentDirectory(second);
            Assert.Equal(new long[] { 2, 6, 3 }, ((Tensor<long>)Text.BertTokenizeFromFile("hello", file)![0]).ToArray());
            Assert.NotSame(firstTokenizer, Text.GetOrLoadBertFileTokenizer(file));
            Directory.SetCurrentDirectory(first);
            Assert.Same(firstTokenizer, Text.GetOrLoadBertFileTokenizer(file));
            Assert.Equal(new long[] { 2, 5, 3 }, ((Tensor<long>)Text.BertTokenizeFromFile("hello", file)![0]).ToArray());
        });
    }

    [Theory]
    [InlineData("relative")]
    [InlineData("dot")]
    [InlineData("parent")]
    public void FileAliasesShareTokenizerAndReloadIdentity(string spelling)
    {
        WithVocabDirectory((directory, file) =>
        {
            string absolute = Path.Combine(directory, file);
            WriteVocab(absolute);
            Directory.CreateDirectory(Path.Combine(directory, "child"));
            Directory.SetCurrentDirectory(directory);
            string alias = spelling == "relative" ? file : spelling == "dot" ? Path.Combine(".", file) : Path.Combine("child", "..", file);
            var original = Text.GetOrLoadBertFileTokenizer(absolute);
            Assert.Same(original, Text.GetOrLoadBertFileTokenizer(alias));
            WriteVocab(absolute, reversed: true);
            File.SetLastWriteTimeUtc(absolute, File.GetLastWriteTimeUtc(absolute).AddHours(1));
            var replacement = Text.GetOrLoadBertFileTokenizer(alias);
            Assert.NotSame(original, replacement);
            Assert.Same(replacement, Text.GetOrLoadBertFileTokenizer(absolute));
            Assert.Equal(new long[] { 2, 6, 3 }, ((Tensor<long>)Text.BertTokenizeFromFile("hello", alias)![0]).ToArray());
            File.Delete(absolute);
            Assert.Throws<FileNotFoundException>(() => Text.GetOrLoadBertFileTokenizer(alias));
        });
    }

    [Fact]
    public void FileCacheUsesPlatformPathCaseComparison()
    {
        WithVocabDirectory((directory, file) =>
        {
            string lower = Path.Combine(directory, file), upper = Path.Combine(directory, file.ToUpperInvariant());
            WriteVocab(lower);
            if (!OperatingSystem.IsWindows()) WriteVocab(upper, reversed: true);
            var original = Text.GetOrLoadBertFileTokenizer(lower);
            var other = Text.GetOrLoadBertFileTokenizer(upper);
            if (OperatingSystem.IsWindows()) Assert.Same(original, other);
            else
            {
                Assert.NotSame(original, other);
                Assert.Equal(new long[] { 2, 6, 3 }, ((Tensor<long>)Text.BertTokenizeFromFile("hello", upper)![0]).ToArray());
            }
        });
    }
}
