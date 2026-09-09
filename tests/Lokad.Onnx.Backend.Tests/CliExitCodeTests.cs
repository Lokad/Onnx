namespace Lokad.Onnx.Backend.Tests;

using System.Diagnostics;
using Lokad.Onnx.Tests.Support;

/// <summary>
/// Exit codes are process properties: each case launches the built Release
/// CLI in a fresh process and asserts the observed exit class.
/// </summary>
public class CliExitCodeTests
{
    static string CliDll()
    {
        var dll = Path.Combine(TestSupport.RepoRoot(), "src", "Lokad.Onnx.CLI", "bin", "Release", "net10.0", "Lokad.Onnx.CLI.dll");
        Skip.IfNot(File.Exists(dll), "CLI Release build not present; build src/Lokad.Onnx.CLI in Release first.");
        return dll;
    }

    static int RunCli(params string[] args)
    {
        var psi = new ProcessStartInfo("dotnet", "")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        psi.ArgumentList.Add(CliDll());
        foreach (var a in args) psi.ArgumentList.Add(a);
        using var p = Process.Start(psi)!;
        if (!p.WaitForExit(180000))
        {
            try { p.Kill(true); } catch { }
            Assert.Fail("CLI timed out: " + string.Join(" ", args));
        }
        return p.ExitCode;
    }

    static string Mnist() => TestSupport.CommittedModel("mnist-8.onnx");

    [Fact]
    public void InvalidFilter_ExitsInvalidOptions()
    {
        Assert.Equal(2, RunCli("info", Mnist(), "--op-filter", "definitely-not-an-op"));
    }

    [Fact]
    public void ValidFilter_ExitsSuccess()
    {
        Assert.Equal(0, RunCli("info", Mnist(), "--op-filter", "MatMul"));
    }

    [Fact]
    public void MissingModel_ExitsNotFound()
    {
        Assert.Equal(4, RunCli("info", Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx")));
    }

    [Fact]
    public void MalformedModel_ExitsInvalidInput()
    {
        var garbage = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx");
        File.WriteAllBytes(garbage, new byte[] { 1, 2, 3, 4 });
        try
        {
            Assert.Equal(5, RunCli("info", garbage));
        }
        finally
        {
            File.Delete(garbage);
        }
    }

    [Fact]
    public void RunMissingModel_ExitsNotFound()
    {
        Assert.Equal(4, RunCli("run", Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx"), "x.png::mnist"));
    }

    [Fact]
    public void ValidMnistRun_ExitsSuccess()
    {
        Assert.Equal(0, RunCli("run", Mnist(), TestSupport.CommittedImage("mnist4.png") + "::mnist"));
    }
}
