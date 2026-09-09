
namespace Lokad.Onnx.Backend.Tests;

[Collection("ProcessState")]
public class TokenizerLifetimeTests
{
    static string AssetPath() =>
        ModelFixture.RequireModelOrSkip("e5 tokenizer", "models", "multilingual-e5-small", "sentencepiece.bpe.model");

    static string SeedMe5s()
    {
        var target = Path.Combine(Runtime.AssemblyLocation, "me5s-sentencepiece.bpe.model");
        if (!File.Exists(target)) File.Copy(AssetPath(), target);
        return target;
    }

    static void Gc()
    {
        GC.Collect(2, GCCollectionMode.Forced, true, true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, true, true);
    }

    [SkippableFact]
    public void SecondSharedCall_AllocatesLittle()
    {
        string path = AssetPath();
        var texts = new[] { "Hello world", "query: What is the capital of France?", "a  b" };
        var first = Text.RobertaTokenizeFromFile(texts, path);
        Assert.NotNull(first);
        Gc();
        long before = GC.GetAllocatedBytesForCurrentThread();
        var second = Text.RobertaTokenizeFromFile(texts, path);
        long alloc = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.NotNull(second);
        Assert.Equal(
            ((Tensor<long>)first![0]).ToArray(),
            ((Tensor<long>)second![0]).ToArray());
        Assert.True(alloc < 10_000_000L, $"Second shared call allocated {alloc} bytes; expected a cache hit in the kilobyte range.");
    }

    [SkippableFact]
    public void EnsureMe5sTokenizer_SucceedsWhenSeeded()
    {
        Assert.True(File.Exists(SeedMe5s()));
        Assert.True(Text.EnsureMe5sTokenizer());
    }

    [SkippableFact]
    public void ConcurrentSharedBatch_MatchesSerial()
    {
        SeedMe5s();
        var texts = new[] { "Hello world", "query: What is the capital of France?" };
        var serial = Text.RobertaTokenize(texts, "me5s");
        Assert.NotNull(serial);
        var expected = ((Tensor<long>)serial![0]).ToArray();
        var tasks = new Task<long[]>[8];
        for (int i = 0; i < tasks.Length; i++)
            tasks[i] = Task.Run(() =>
            {
                var parts = Text.RobertaTokenize(texts, "me5s");
                Assert.NotNull(parts);
                return ((Tensor<long>)parts![0]).ToArray();
            });
        Task.WaitAll(tasks);
        foreach (var t in tasks) Assert.Equal(expected, t.Result);
    }

    static string TempCopyOfAsset()
    {
        var tmp = Path.Combine(Path.GetTempPath(), "lokad-tok-" + Guid.NewGuid().ToString("N") + ".bpe.model");
        File.Copy(AssetPath(), tmp);
        return tmp;
    }

    [SkippableFact]
    public void TouchedAsset_ReloadsWithStableResults()
    {
        string tmp = TempCopyOfAsset();
        try
        {
            var first = Text.GetOrLoadRobertaTokenizer(tmp);
            var baseline = Text.RobertaTokenizeFromFile("Hello world", tmp);
            Assert.NotNull(baseline);
            File.SetLastWriteTimeUtc(tmp, File.GetLastWriteTimeUtc(tmp).AddHours(1));
            var second = Text.GetOrLoadRobertaTokenizer(tmp);
            Assert.NotSame(first, second);
            var again = Text.RobertaTokenizeFromFile("Hello world", tmp);
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

    [SkippableFact]
    public void DeletedAsset_EvictsInsteadOfServingStale()
    {
        string tmp = TempCopyOfAsset();
        try
        {
            var first = Text.GetOrLoadRobertaTokenizer(tmp);
            File.Delete(tmp);
            Assert.Throws<FileNotFoundException>(() => Text.GetOrLoadRobertaTokenizer(tmp));
            File.Copy(AssetPath(), tmp);
            var second = Text.GetOrLoadRobertaTokenizer(tmp);
            Assert.NotSame(first, second);
        }
        finally
        {
            if (File.Exists(tmp)) File.Delete(tmp);
        }
    }

    [SkippableFact]
    public void GetOrLoad_IdentityAndMissingPath()
    {
        string path = AssetPath();
        var first = Text.GetOrLoadRobertaTokenizer(path);
        var second = Text.GetOrLoadRobertaTokenizer(path);
        Assert.Same(first, second);
        Assert.Throws<FileNotFoundException>(() => Text.GetOrLoadRobertaTokenizer("no-such-tokenizer.bpe.model"));
    }
}
