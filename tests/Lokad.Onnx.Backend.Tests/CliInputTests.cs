namespace Lokad.Onnx.Backend.Tests;

using Lokad.Onnx.Tests.Support;

public class CliInputTests
{
    static void WithTempDirectory(Action<string> exercise)
    {
        var tempDirectory = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        Directory.CreateDirectory(tempDirectory);
        try
        {
            exercise(tempDirectory);
        }
        finally
        {
            Directory.Delete(tempDirectory, true);
        }
    }

    static string MnistPng() => TestSupport.CommittedImage("mnist2.png");

    static string SeedMe5sTokenizer()
    {
        ModelFixture.RequireModelOrSkip("e5 tokenizer", "models", "multilingual-e5-small", "sentencepiece.bpe.model");
        var target = Path.Combine(Runtime.AssemblyLocation, "me5s-sentencepiece.bpe.model");
        if (!File.Exists(target))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null)
            {
                var candidate = Path.Combine(dir.FullName, "models", "multilingual-e5-small", "sentencepiece.bpe.model");
                if (File.Exists(candidate))
                {
                    File.Copy(candidate, target);
                    break;
                }
                dir = dir.Parent;
            }
        }
        return target;
    }

    [Fact]
    public void UnknownExtension_MixedWithValid_ReturnsNull()
    {
        WithTempDirectory(directory =>
        {
            var img = Path.Combine(directory, "in.png");
            File.Copy(MnistPng(), img);
            File.WriteAllBytes(Path.Combine(directory, "bad.xyz"), new byte[] { 1, 2, 3 });
            Assert.Null(Data.GetInputTensorsFromFileArgs(new[] { img, Path.Combine(directory, "bad.xyz") }));
        });
    }

    [Fact]
    public void MissingTextFile_ReturnsNull()
    {
        WithTempDirectory(directory =>
        {
            Assert.Null(Data.GetInputTensorsFromFileArgs(new[] { Path.Combine(directory, "missing.txt") }));
        });
    }

    [Fact]
    public void BadTokenizerProps_ReturnsNull()
    {
        Assert.Null(Text.GetTextTensors("hi", "bogus"));
    }

    [SkippableFact]
    public void DefaultTextFile_TokenizesWithMe5s()
    {
        Assert.True(File.Exists(SeedMe5sTokenizer()));
        WithTempDirectory(directory =>
        {
            var txt = Path.Combine(directory, "hello.txt");
            File.WriteAllText(txt, "Hello world");
            var parts = Text.GetTextTensorsFromFileArg(txt, Array.Empty<string>());
            Assert.NotNull(parts);
            Assert.Equal(3, parts!.Length);
            Assert.Equal(new long[] { 0, 35378, 8999, 2 }, ((Tensor<long>)parts[0]).ToArray());
        });
    }

    [SkippableFact]
    public void ExplicitMe5sProps_Tokenize()
    {
        Assert.True(File.Exists(SeedMe5sTokenizer()));
        WithTempDirectory(directory =>
        {
            var txt = Path.Combine(directory, "hello.txt");
            File.WriteAllText(txt, "Hello world");
            var parts = Text.GetTextTensorsFromFileArg(txt, new string[] { "me5s" });
            Assert.NotNull(parts);
            Assert.Equal(new long[] { 0, 35378, 8999, 2 }, ((Tensor<long>)parts![0]).ToArray());
        });
    }

    [Fact]
    public void ImageSaveInput_WritesFileOnlyWhenAsked()
    {
        WithTempDirectory(directory =>
        {
            var img = Path.Combine(directory, "in.png");
            File.Copy(MnistPng(), img);
            var plain = Images.GetImageTensorFromFileArg(img, Array.Empty<string>(), 3, false);
            Assert.NotNull(plain);
            Assert.Empty(Directory.GetFiles(directory, "in_*.png"));
            var saved = Images.GetImageTensorFromFileArg(img, Array.Empty<string>(), 3, true);
            Assert.NotNull(saved);
            Assert.Single(Directory.GetFiles(directory, "in_*.png"));
        });
    }

    static string RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (int i = 0; i < 5; i++) dir = dir!.Parent!;
        return dir!.FullName;
    }

    static string CliDll()
    {
        var dll = Path.Combine(RepoRoot(), "src", "Lokad.Onnx.CLI", "bin", "Release", "net10.0", "Lokad.Onnx.CLI.dll");
        Assert.True(File.Exists(dll), "Build the Release CLI first (build.cmd): " + dll);
        return dll;
    }

    static int RunCli(params string[] args)
    {
        // No output redirection: the child inherits the console, which avoids
        // pipe-buffer deadlocks on the CLI verbose logging.
        var psi = new System.Diagnostics.ProcessStartInfo("dotnet");
        psi.ArgumentList.Add(CliDll());
        foreach (var a in args) psi.ArgumentList.Add(a);
        psi.UseShellExecute = false;
        psi.WorkingDirectory = AppContext.BaseDirectory;
        using var proc = System.Diagnostics.Process.Start(psi)!;
        if (!proc.WaitForExit(120000))
        {
            proc.Kill();
            Assert.Fail("CLI timed out: " + string.Join(" ", args));
        }
        return proc.ExitCode;
    }

    static string MnistModel() => TestSupport.CommittedModel("mnist-8.onnx");

    [Fact]
    public void Cli_Success_ZeroExit()
    {
        var code = RunCli("run", MnistModel(), MnistPng() + "::mnist");
        Assert.Equal(0, code);
    }

    [Fact]
    public void Cli_MissingNode_NotFoundExit()
    {
        var code = RunCli("run", MnistModel(), MnistPng() + "::mnist", "--node", "nope");
        Assert.Equal(4, code);
    }

    [Fact]
    public void Cli_UnknownExtension_InvalidInputExit()
    {
        WithTempDirectory(directory =>
        {
            var bad = Path.Combine(directory, "bad.xyz");
            File.WriteAllBytes(bad, new byte[] { 1, 2, 3 });
            Assert.Equal(5, RunCli("run", MnistModel(), bad));
        });
    }

    [Fact]
    public void Cli_SaveInput_WritesFile()
    {
        WithTempDirectory(directory =>
        {
            var img = Path.Combine(directory, "in.png");
            File.Copy(MnistPng(), img);
            var code = RunCli("run", MnistModel(), img + "::mnist", "--save-input");
            Assert.Equal(0, code);
            Assert.Single(Directory.GetFiles(directory, "in_*.png"));
        });
    }
}
