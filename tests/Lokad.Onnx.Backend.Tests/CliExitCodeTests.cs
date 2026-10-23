namespace Lokad.Onnx.Backend.Tests;

using System.Diagnostics;
using Google.Protobuf;
using Lokad.Onnx.Tests.Support;
using global::Onnx;

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

    static (int Exit, string Output, string Error) RunCli(params string[] args)
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
        // Drain both pipes while the CLI runs; verbose output can exceed their
        // capacity before the process reaches its exit-code assertion.
        var stdout = p.StandardOutput.ReadToEndAsync();
        var stderr = p.StandardError.ReadToEndAsync();
        if (!p.WaitForExit(180000))
        {
            try { p.Kill(true); } catch { }
            Assert.Fail("CLI timed out: " + string.Join(" ", args));
        }
        Task.WhenAll(stdout, stderr).GetAwaiter().GetResult();
        return (p.ExitCode, stdout.Result, stderr.Result);
    }

    static string Mnist() => TestSupport.CommittedModel("mnist-8.onnx");

    [SkippableFact]
    public void InvalidFilter_ExitsInvalidOptions()
    {
        Assert.Equal(2, RunCli("info", Mnist(), "--op-filter", "definitely-not-an-op").Exit);
    }

    [SkippableFact]
    public void ValidFilter_ExitsSuccess()
    {
        Assert.Equal(0, RunCli("info", Mnist(), "--op-filter", "MatMul").Exit);
    }

    [SkippableFact]
    public void MissingModel_ExitsNotFound()
    {
        Assert.Equal(4, RunCli("info", Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx")).Exit);
    }

    [SkippableFact]
    public void MalformedModel_ExitsInvalidInput()
    {
        var garbage = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx");
        File.WriteAllBytes(garbage, new byte[] { 1, 2, 3, 4 });
        try
        {
            Assert.Equal(5, RunCli("info", garbage).Exit);
        }
        finally
        {
            File.Delete(garbage);
        }
    }

    [SkippableFact]
    public void RunMissingModel_ExitsNotFound()
    {
        Assert.Equal(4, RunCli("run", Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx"), "x.png::mnist").Exit);
    }

    [SkippableFact]
    public void ValidMnistRun_ExitsSuccess()
    {
        Assert.Equal(0, RunCli("run", Mnist(), TestSupport.CommittedImage("mnist4.png") + "::mnist").Exit);
    }

    static ValueInfoProto FloatInfo(string name) => new() { Name = name, Type = new TypeProto
        { TensorType = new TypeProto.Types.Tensor { ElemType = 1,
            Shape = new TensorShapeProto { Dim = { new TensorShapeProto.Types.Dimension { DimValue = 2 } } } } } };

    static GraphProto Branch(string name, bool nested)
    {
        var graph = new GraphProto { Name = name, Output = { FloatInfo("shared") } };
        if (nested) graph.Node.Add(IfNode("shared", false));
        else graph.Node.Add(new NodeProto { Name = "branch_private", OpType = "Constant", Output = { "shared" },
            Attribute = { new AttributeProto { Name = "value", Type = AttributeProto.Types.AttributeType.Tensor,
                T = new TensorProto { DataType = 1, Dims = { 2 }, FloatData = { 1, 2 } } } } });
        return graph;
    }

    static NodeProto IfNode(string output, bool nested) => new() { Name = "choose", OpType = "If", Input = { "condition" }, Output = { output },
        Attribute = { new AttributeProto { Name = "then_branch", Type = AttributeProto.Types.AttributeType.Graph, G = Branch("then", nested) },
            new AttributeProto { Name = "else_branch", Type = AttributeProto.Types.AttributeType.Graph, G = Branch("else", false) } } };

    static ModelProto InspectionModel(int kind)
    {
        var model = new ModelProto { IrVersion = 10, Graph = new GraphProto { Name = "inspection" },
            OpsetImport = { new OperatorSetIdProto { Version = 18 } } };
        if (kind < 3)
        {
            model.Graph.Output.Add(FloatInfo("x"));
            if (kind != 1) model.Graph.Input.Add(FloatInfo("x"));
            if (kind != 0) model.Graph.Initializer.Add(new TensorProto { Name = "x", DataType = 1, Dims = { 2 }, FloatData = { 1, 2 } });
            if (kind == 2)
            {
                model.Graph.Node.Add(new NodeProto { Name = "after", OpType = "Relu", Input = { "x" }, Output = { "y" } });
                model.Graph.Output.Add(FloatInfo("y"));
            }
        }
        else
        {
            model.Graph.Input.Add(new ValueInfoProto { Name = "condition", Type = new TypeProto
                { TensorType = new TypeProto.Types.Tensor { ElemType = 9, Shape = new TensorShapeProto() } } });
            model.Graph.Output.Add(FloatInfo("y"));
            model.Graph.Node.Add(IfNode("y", kind == 4));
        }
        return model;
    }

    [SkippableTheory]
    [InlineData(0, 0)]
    [InlineData(0, 1)]
    [InlineData(0, 2)]
    [InlineData(1, 0)]
    [InlineData(1, 1)]
    [InlineData(1, 2)]
    [InlineData(2, 0)]
    [InlineData(2, 1)]
    [InlineData(2, 2)]
    [InlineData(3, 0)]
    [InlineData(3, 1)]
    [InlineData(3, 2)]
    [InlineData(4, 0)]
    [InlineData(4, 1)]
    [InlineData(4, 2)]
    public void LegalOverlappingRolesAndGraphAttributesCanBeInspected(int kind, int mode)
    {
        string file = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx");
        try
        {
            File.WriteAllBytes(file, InspectionModel(kind).ToByteArray());
            var arguments = new List<string> { "info", file };
            if (mode != 0) arguments.Add(mode == 1 ? "--ops" : "--init");
            var result = RunCli(arguments.ToArray());
            Assert.True(result.Exit == 0, result.Output + result.Error);
            Assert.DoesNotContain("Unhandled", result.Output + result.Error);
            if (mode == 0 && kind == 2) Assert.Contains("x:float:2<initializer><input><output>", result.Output);
            if (mode == 0 && kind >= 3)
            {
                Assert.Contains("graph then (0 inputs, 1 outputs, 0 initializers, 1 nodes)", result.Output);
                Assert.Contains("graph else (0 inputs, 1 outputs, 0 initializers, 1 nodes)", result.Output);
                Assert.DoesNotContain("branch_private", result.Output);
            }
        }
        finally { if (File.Exists(file)) File.Delete(file); }
    }
}
