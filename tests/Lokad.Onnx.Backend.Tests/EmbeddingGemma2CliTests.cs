namespace Lokad.Onnx.Backend.Tests;

using System.Diagnostics;
using Lokad.Onnx.Tests.Support;

public class EmbeddingGemma2CliTests
{
    static (int Code, string Output, string Error) Run(params string[] arguments)
    {
        string dll = Path.Combine(TestSupport.RepoRoot(), "src/Lokad.Onnx.CLI/bin/Release/net10.0/Lokad.Onnx.CLI.dll");
        Assert.True(File.Exists(dll), "Build the Release CLI first.");
        var start = new ProcessStartInfo("dotnet") { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true };
        start.ArgumentList.Add(dll); foreach (string argument in arguments) start.ArgumentList.Add(argument);
        using var process = Process.Start(start) ?? throw new InvalidOperationException();
        var output = process.StandardOutput.ReadToEndAsync(); var error = process.StandardError.ReadToEndAsync();
        if (!process.WaitForExit(30000)) { process.Kill(true); process.WaitForExit(); Assert.Fail("CLI timeout."); }
        return (process.ExitCode, output.GetAwaiter().GetResult(), error.GetAwaiter().GetResult());
    }

    [Fact]
    public void HelpDescribesLocalAssetsPromptsDimensionsAndJsonLines()
    {
        var result = Run("embed", "--help");
        Assert.Equal(0, result.Code); Assert.Empty(result.Error);
        foreach (string text in new[] { "--query", "--documents", "FP32", "128", "768", "prefix", "JSONL", "stderr", "512" })
            Assert.Contains(text, result.Output);
    }

    [Theory]
    [InlineData("both")] [InlineData("neither")] [InlineData("duplicate")] [InlineData("dimensions")]
    [InlineData("tokens")] [InlineData("batch")] [InlineData("budget")] [InlineData("unknown")]
    [InlineData("missing-value")] [InlineData("extra-position")]
    public void InvalidOptionsWriteOnlyDiagnostics(string kind)
    {
        var args = new List<string> { "embed", "missing-embeddinggemma2-model-47329", "--query", "test" };
        if (kind == "both") args.AddRange(new[] { "--documents", "input.jsonl" });
        else if (kind == "neither") args.RemoveRange(2, 2);
        else if (kind == "duplicate") args.AddRange(new[] { "--query", "other" });
        else if (kind == "dimensions") args.Add("--dimensions=1024");
        else if (kind == "tokens") args.Add("--max-tokens=1");
        else if (kind == "batch") args.Add("--batch-size=0");
        else if (kind == "budget") args.Add("--padded-token-budget=511");
        else if (kind == "unknown") args.Add("--embedding-size=128");
        else if (kind == "missing-value") args.RemoveAt(args.Count - 1);
        else args.Add("extra-model");
        var result = Run(args.ToArray());
        Assert.Equal(2, result.Code); Assert.Empty(result.Output); Assert.NotEmpty(result.Error);
    }

    [Fact]
    public void EmptyQueryIsAccepted_AndMissingLocalAssetsReturnNotFound()
    {
        var result = Run("embed", "missing-embeddinggemma2-model-47329", "--query=");
        Assert.Equal(4, result.Code); Assert.Empty(result.Output); Assert.NotEmpty(result.Error);
    }
}
