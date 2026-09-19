namespace Lokad.Onnx.Backend.Tests;

using System.Diagnostics;
using Lokad.Onnx.Tests.Support;

public class ParakeetCliTests
{
    static (int Code, string Output, string Error) Run(params string[] arguments)
    {
        string dll = Path.Combine(TestSupport.RepoRoot(), "src", "Lokad.Onnx.CLI", "bin", "Release", "net10.0", "Lokad.Onnx.CLI.dll");
        Assert.True(File.Exists(dll), "Build the Release CLI first.");
        var start = new ProcessStartInfo("dotnet") { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true };
        start.ArgumentList.Add(dll);
        foreach (string argument in arguments) start.ArgumentList.Add(argument);
        using var process = Process.Start(start) ?? throw new InvalidOperationException("CLI failed to start");
        var output = process.StandardOutput.ReadToEndAsync(); var error = process.StandardError.ReadToEndAsync();
        if (!process.WaitForExit(30000)) { process.Kill(true); process.WaitForExit(); Assert.Fail("CLI timeout"); }
        return (process.ExitCode, output.GetAwaiter().GetResult(), error.GetAwaiter().GetResult());
    }

    [Theory]
    [InlineData("language")] [InlineData("empty-language")] [InlineData("unknown-model")]
    [InlineData("empty-model")] [InlineData("duplicate-model")] [InlineData("zero-tokens")]
    [InlineData("large-tokens")] [InlineData("bad-tokens")]
    public void InvalidModelPolicyProducesNoTranscript(string failure)
    {
        var options = new List<string> { "transcribe", "model", "input.wav", "--model-type", "parakeet", "--json" };
        switch (failure)
        {
            case "language": options.AddRange(["--language", "fr"]); break;
            case "empty-language": options.Add("--language="); break;
            case "unknown-model": options[4] = "rnnt"; break;
            case "empty-model": options[4] = ""; break;
            case "duplicate-model": options.AddRange(["--model-type", "whisper"]); break;
            case "zero-tokens": options.Add("--max-tokens=0"); break;
            case "large-tokens": options.Add("--max-tokens=4097"); break;
            default: options.Add("--max-tokens=1.5"); break;
        }
        var result = Run(options.ToArray());
        Assert.Equal(2, result.Code); Assert.Empty(result.Output); Assert.NotEmpty(result.Error);
    }

    [Theory]
    [InlineData(1)] [InlineData(445)] [InlineData(4096)]
    public void ValidParakeetPolicyReachesLocalAssetValidation(int tokens)
    {
        string absent = Path.Combine(Path.GetTempPath(), "lonnx-parakeet-absent-" + Guid.NewGuid().ToString("N"));
        var result = Run("transcribe", absent, "missing.wav", "--model-type=parakeet", "--max-tokens=" + tokens);
        Assert.Equal(4, result.Code); Assert.Empty(result.Output); Assert.Contains("Model directory not found", result.Error);
    }

    [Fact]
    public void HelpExplainsModelSelectionAndLimits()
    {
        var result = Run("transcribe", "--help");
        Assert.Equal(0, result.Code); Assert.Contains("--model-type", result.Output);
        Assert.Contains("4096", result.Output); Assert.Contains("444", result.Output); Assert.Empty(result.Error);
    }
}
