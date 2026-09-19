namespace Lokad.Onnx.Backend.Tests;

using System.Diagnostics;
using Lokad.Onnx.Tests.Support;

public class WhisperCliTests
{
    static (int Code, string Output, string Error) Run(params string[] arguments)
    {
        string dll = Path.Combine(TestSupport.RepoRoot(), "src", "Lokad.Onnx.CLI", "bin", "Release", "net10.0", "Lokad.Onnx.CLI.dll");
        Assert.True(File.Exists(dll), "Build the Release CLI first.");
        var start = new ProcessStartInfo("dotnet") { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true };
        start.ArgumentList.Add(dll);
        foreach (string argument in arguments) start.ArgumentList.Add(argument);
        using var process = Process.Start(start) ?? throw new InvalidOperationException("CLI did not start.");
        var output = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        if (!process.WaitForExit(30000))
        {
            process.Kill(true); process.WaitForExit();
            Assert.Fail("CLI timeout.");
        }
        return (process.ExitCode, output.GetAwaiter().GetResult(), error.GetAwaiter().GetResult());
    }

    [Fact]
    public void HelpDescribesActualInputPolicy()
    {
        var result = Run("transcribe", "--help");
        Assert.Equal(0, result.Code);
        Assert.Contains("--language", result.Output);
        Assert.Contains("30 seconds", result.Output);
        Assert.Contains("--json", result.Output);
        Assert.Contains("--recording", result.Output);
        Assert.Contains("--max-windows", result.Output);
        Assert.Empty(result.Error);
    }

    [Theory]
    [InlineData("missing-language")]
    [InlineData("empty-language")]
    [InlineData("missing-value")]
    [InlineData("too-many-paths")]
    [InlineData("duplicate-language")]
    [InlineData("duplicate-json")]
    [InlineData("unknown-option")]
    [InlineData("bad-json")]
    [InlineData("zero-tokens")]
    [InlineData("too-many-tokens")]
    [InlineData("noninteger-tokens")]
    [InlineData("recording-parakeet")]
    [InlineData("windows-without-recording")]
    [InlineData("windows-zero")]
    [InlineData("windows-too-many")]
    [InlineData("duplicate-recording")]
    [InlineData("bad-recording")]
    public void InvalidOptionsFailWithoutTranscriptOutput(string kind)
    {
        string[] arguments = kind switch
        {
            "missing-language" => ["transcribe", "model", "in.wav"],
            "empty-language" => ["transcribe", "model", "in.wav", "--language="],
            "missing-value" => ["transcribe", "model", "in.wav", "--language", "--json"],
            "too-many-paths" => ["transcribe", "model", "in.wav", "extra", "--language", "en"],
            "duplicate-language" => ["transcribe", "model", "in.wav", "--language=en", "--language=fr"],
            "duplicate-json" => ["transcribe", "model", "in.wav", "--language=en", "--json", "--json"],
            "unknown-option" => ["transcribe", "model", "in.wav", "--language=en", "--beam=5"],
            "bad-json" => ["transcribe", "model", "in.wav", "--language=en", "--json=maybe"],
            "zero-tokens" => ["transcribe", "model", "in.wav", "--language=en", "--max-tokens=0"],
            "too-many-tokens" => ["transcribe", "model", "in.wav", "--language=en", "--max-tokens=445"],
            "recording-parakeet" => ["transcribe", "model", "in.wav", "--model-type=parakeet", "--recording"],
            "windows-without-recording" => ["transcribe", "model", "in.wav", "--language=en", "--max-windows=1"],
            "windows-zero" => ["transcribe", "model", "in.wav", "--language=en", "--recording", "--max-windows=0"],
            "windows-too-many" => ["transcribe", "model", "in.wav", "--language=en", "--recording", "--max-windows=513"],
            "duplicate-recording" => ["transcribe", "model", "in.wav", "--language=en", "--recording", "--recording"],
            "bad-recording" => ["transcribe", "model", "in.wav", "--language=en", "--recording=maybe"],
            _ => ["transcribe", "model", "in.wav", "--language=en", "--max-tokens=bad"]
        };
        var result = Run(arguments);
        Assert.Equal(2, result.Code);
        Assert.Empty(result.Output);
        Assert.NotEmpty(result.Error);
    }

    [Theory]
    [InlineData("missing-file", 4)]
    [InlineData("missing-model", 4)]
    [InlineData("remote-model", 5)]
    [InlineData("bad-wave", 5)]
    [InlineData("overlong", 5)]
    [InlineData("invalid-language", 2)]
    [InlineData("bad-metadata", 5)]
    public void InvalidInputsFailBeforeLoadingModelWeights(string kind, int code)
    {
        string directory = Path.Combine(Path.GetTempPath(), "lonnx-whisper-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string wave = Path.Combine(directory, "input.wav");
            string model = directory;
            var data = kind == "overlong" ? new byte[16000 * 2 * 30 + 2] : new byte[2];
            File.WriteAllBytes(wave, WaveAudioTests.Wave(("fmt ", WaveAudioTests.Format(1, 1, 16000, 16, 16)), ("data", data)));
            File.WriteAllText(Path.Combine(directory, "generation_config.json"), "{\"lang_to_id\":{\"<|en|>\":50259}}");
            string language = "en";
            if (kind == "missing-file") wave += ".missing";
            if (kind == "missing-model") model = Path.Combine(directory, "absent");
            if (kind == "remote-model") model = "https://example.invalid/model";
            if (kind == "bad-wave") File.WriteAllBytes(wave, new byte[] { 1, 2, 3 });
            if (kind == "invalid-language") language = "unknown";
            if (kind == "bad-metadata") File.WriteAllText(Path.Combine(directory, "generation_config.json"), "{}");
            var result = Run("transcribe", model, wave, "--language", language, "--json");
            Assert.Equal(code, result.Code);
            Assert.Empty(result.Output);
            Assert.NotEmpty(result.Error);
        }
        finally { Directory.Delete(directory, true); }
    }
}
