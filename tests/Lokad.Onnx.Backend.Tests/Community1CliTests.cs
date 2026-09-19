namespace Lokad.Onnx.Backend.Tests;

using System.Diagnostics;
using Lokad.Onnx.Tests.Support;

public class Community1CliTests
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
    public void HelpDescribesInputsAndTimelinePolicy()
    {
        var result = Run("diarize", "--help"); Assert.Equal(0, result.Code); Assert.Empty(result.Error);
        foreach (string text in new[] { "--segmentation", "--embedding", "--projection", "--plda", "--json", "ten minutes", "exclusive", "clipped" }) Assert.Contains(text, result.Output);
    }
    [Theory]
    [InlineData("missing-models")] [InlineData("duplicate")] [InlineData("empty")] [InlineData("unknown")] [InlineData("json-value")]
    public void InvalidOptionsProduceOnlyAnError(string kind)
    {
        var args = new List<string> { "diarize", "input.wav", "--segmentation=seg.onnx", "--embedding=encoder.onnx", "--projection=projection.onnx", "--plda=plda.json" };
        if (kind == "missing-models") args.RemoveAt(args.Count - 1);
        else if (kind == "duplicate") args.Add("--embedding=other.onnx");
        else if (kind == "empty") args[^1] = "--plda=";
        else args.Add(kind == "unknown" ? "--speakers=2" : "--json=maybe");
        var result = Run(args.ToArray()); Assert.Equal(2, result.Code); Assert.Empty(result.Output); Assert.NotEmpty(result.Error);
    }
    [Fact]
    public void MissingFilesFailBeforeAnyTimelineIsPrinted()
    {
        var result = Run("diarize", "missing-audio-83921.wav", "--segmentation=seg.onnx", "--embedding=encoder.onnx", "--projection=projection.onnx", "--plda=plda.json", "--json");
        Assert.Equal(4, result.Code); Assert.Empty(result.Output); Assert.NotEmpty(result.Error);
    }
}
