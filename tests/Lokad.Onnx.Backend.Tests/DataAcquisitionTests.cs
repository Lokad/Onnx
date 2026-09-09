using System.Collections.Generic;
using System.IO;

namespace Lokad.Onnx.Backend.Tests;

// D6: acquisition is explicit and local. The core exposes no downloader,
// inference never touches the network, and the documented tokenizer location
// resolves without a manual copy step.
public class DataAcquisitionTests
{
    static void WithTempDirectory(Action<string> exercise)
    {
        var directory = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        Directory.CreateDirectory(directory);
        try
        {
            exercise(directory);
        }
        finally
        {
            Directory.Delete(directory, true);
        }
    }

    [Fact]
    public void CoreExposesNoDownloader()
    {
        Assert.Null(typeof(Runtime).GetMethod("DownloadFile"));
    }

    [Fact]
    public void FindAssetUnderAncestors_SearchesUpward()
    {
        WithTempDirectory(root =>
        {
            var nested = Path.Combine(root, "a", "b");
            Directory.CreateDirectory(nested);
            Directory.CreateDirectory(Path.Combine(root, "models"));
            File.WriteAllText(Path.Combine(root, "models", "x.bin"), "x");
            Assert.Equal(
                Path.Combine(root, "models", "x.bin"),
                Text.FindAssetUnderAncestors(nested, "models", "x.bin"));
            Assert.Equal(
                Path.Combine(root, "models", "x.bin"),
                Text.FindAssetUnderAncestors(root, "models", "x.bin"));
            Assert.Null(Text.FindAssetUnderAncestors(nested, "models", "missing.bin"));
            Assert.Null(Text.FindAssetUnderAncestors(nested));
        });
    }

    [Fact]
    public void FindAssetUnderAncestors_IgnoresDirectories()
    {
        WithTempDirectory(root =>
        {
            Directory.CreateDirectory(Path.Combine(root, "models"));
            Assert.Null(Text.FindAssetUnderAncestors(root, "models"));
        });
    }

    static string? FindDocumentedTokenizer()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "models", "multilingual-e5-small", "sentencepiece.bpe.model");
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        return null;
    }

    [SkippableFact]
    public void EnsureMe5sTokenizer_UsesLocalAssets()
    {
        Skip.If(FindDocumentedTokenizer() is null, "e5 tokenizer asset not present; acquisition needs it locally.");
        Assert.True(Text.EnsureMe5sTokenizer());
        Assert.True(File.Exists(Text.Me5sTokenizerPath()));
    }

    [SkippableFact]
    public void Me5sTokenizerPath_PrefersAssemblyCopy()
    {
        var documented = FindDocumentedTokenizer();
        Skip.If(documented is null, "e5 tokenizer asset not present.");
        var target = Path.Combine(Runtime.AssemblyLocation, "me5s-sentencepiece.bpe.model");
        if (!File.Exists(target)) File.Copy(documented, target);
        Assert.Equal(target, Text.Me5sTokenizerPath());
    }
}
