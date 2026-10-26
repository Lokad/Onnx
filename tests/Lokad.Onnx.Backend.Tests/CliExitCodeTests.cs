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
        => RunCliFrom(null, args);

    static (int Exit, string Output, string Error) RunCliFrom(string? workingDirectory, params string[] args)
    {
        var psi = new ProcessStartInfo("dotnet", "")
        {
            WorkingDirectory = workingDirectory ?? "",
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

    static void WithOperatorModel(string name, string domain, Action<string> inspect)
    {
        var model = new ModelProto { IrVersion = 10, Graph = new GraphProto { Name = "operator-name",
            Input = { FloatInfo("x") }, Output = { FloatInfo("y") },
            Node = { new NodeProto { Name = "op", OpType = name, Domain = domain, Input = { "x" }, Output = { "y" } } } },
            OpsetImport = { new OperatorSetIdProto { Domain = "", Version = 18 } } };
        if (domain.Length != 0) model.OpsetImport.Add(new OperatorSetIdProto { Domain = domain, Version = 1 });
        string path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".onnx");
        try
        {
            File.WriteAllBytes(path, model.ToByteArray());
            inspect(path);
        }
        finally { File.Delete(path); }
    }

    public static IEnumerable<object[]> InspectionNames()
    {
        foreach (var (name, domain, supported) in new[] {
            ("Relu", "", true), ("Abs", "", true), ("Relu", "com.example", false),
            ("ExternalAttention", "com.example", false), ("FancyNewOp", "", false),
            (((int)OpType.Relu).ToString(System.Globalization.CultureInfo.InvariantCulture), "", false),
            (" Relu ", "", false), ("Relu, Abs", "", false), ("relu", "", false), ("999999", "", false) })
            yield return new object[] { name, domain, supported };
    }

    [SkippableTheory]
    [MemberData(nameof(InspectionNames))]
    public void InspectionPreservesNameAndSupport(string name, string domain, bool supported)
    {
        WithOperatorModel(name, domain, path =>
        {
            var plain = RunCli("info", path);
            var listing = RunCli("info", path, "--ops");
            Assert.Equal(0, plain.Exit);
            Assert.Equal(0, listing.Exit);
            string identity = (domain.Length == 0 ? "" : domain + ":") + name;
            Assert.Contains(identity + (supported ? " [supported]" : " [unsupported]"), listing.Output);
            Assert.Contains("Node op has op type: " + name + ", inputs:", plain.Output);
        });
    }

    [SkippableTheory]
    [InlineData("FancyNewOp", "", "Abs", false)]
    [InlineData("ExternalAttention", "com.example", "Abs", false)]
    [InlineData("185", "", "Relu", false)]
    [InlineData(" Relu ", "", "Relu", false)]
    [InlineData("Relu, Abs", "", "Relu", false)]
    [InlineData("relu", "", "Relu", false)]
    [InlineData("Abs", "", "aBs", true)]
    [InlineData("Relu", "", "rElU", true)]
    [InlineData("Relu", "com.example", "Relu", true)]
    [InlineData("Relu", "", "Abs", false)]
    [InlineData("Abs", "", "Relu", false)]
    public void InspectionFiltersLiteralModelNames(string name, string domain, string filter, bool included)
    {
        WithOperatorModel(name, domain, path =>
        {
            var result = RunCli("info", path, "--op-filter", filter);
            Assert.Equal(0, result.Exit);
            if (included) Assert.Contains("Node op has op type: " + name + ", inputs:", result.Output);
            else Assert.DoesNotContain("Node op has op type:", result.Output);
        });
    }

    [SkippableTheory]
    [InlineData("185")]
    [InlineData("999999")]
    [InlineData(" Relu ")]
    [InlineData("Relu, Abs")]
    public void InvalidOperatorFilterSyntax_ExitsInvalidOptions(string filter)
    {
        Assert.Equal(2, RunCli("info", Mnist(), "--op-filter", filter).Exit);
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void RunInvalidImage_ExitsInvalidInput(bool truncated)
    {
        var path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".png");
        try
        {
            byte[] bytes = truncated
                ? File.ReadAllBytes(TestSupport.CommittedImage("mnist4.png"))[..16]
                : "This is not a PNG image"u8.ToArray();
            File.WriteAllBytes(path, bytes);
            var result = RunCli("run", Mnist(), path + "::mnist");
            Assert.Equal(5, result.Exit);
            Assert.Contains("Cannot prepare model inputs", result.Output + result.Error);
            Assert.DoesNotContain("Unhandled runtime error", result.Output + result.Error);
        }
        finally { File.Delete(path); }
    }

    [SkippableTheory]
    [InlineData(":224")]
    [InlineData("0:0")]
    [InlineData("224:")]
    [InlineData("nonsense")]
    public void RunMalformedResize_ExitsInvalidInput(string format)
    {
        var result = RunCli("run", Mnist(), TestSupport.CommittedImage("mnist4.png") + "::" + format);
        Assert.Equal(5, result.Exit);
        Assert.DoesNotContain("Unhandled runtime error", result.Output + result.Error);
    }

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

    [SkippableTheory]
    [InlineData(0, "missing")] [InlineData(1, "missing")] [InlineData(2, "missing")]
    [InlineData(0, "malformed")] [InlineData(1, "malformed")] [InlineData(2, "malformed")]
    [InlineData(0, "unshaped")] [InlineData(1, "unshaped")] [InlineData(2, "unshaped")]
    public void InspectionErrorsUseConsistentExitClasses(int mode, string failure)
    {
        string file = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx");
        try
        {
            if (failure == "malformed") File.WriteAllBytes(file, [1, 2, 3, 4]);
            else if (failure == "unshaped")
            {
                var model = new ModelProto { Graph = new GraphProto { Name = "unshaped", Output = {
                    new ValueInfoProto { Name = "y", Type = new TypeProto { TensorType = new TypeProto.Types.Tensor { ElemType = 1 } } } } } };
                File.WriteAllBytes(file, model.ToByteArray());
            }
            var args = new List<string> { "info", file };
            if (mode != 0) args.Add(mode == 1 ? "--ops" : "--init");
            var result = RunCli(args.ToArray());
            Assert.Equal(failure == "missing" ? 4 : 5, result.Exit);
            Assert.DoesNotContain("Unhandled runtime error", result.Output + result.Error);
        }
        finally { File.Delete(file); }
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

    [SkippableTheory]
    [InlineData("http-model.onnx")]
    [InlineData("https-model.onnx")]
    [InlineData("http-models/model.onnx")]
    [InlineData("HTTP-model.onnx")]
    [InlineData("model.onnx")]
    public void RunLocalRelativeModelPath_ProducesOutput(string relativePath)
    {
        string directory = Path.Combine(Path.GetTempPath(), "lonnx-cli-" + Guid.NewGuid().ToString("N"));
        string model = Path.Combine(directory, relativePath);
        string parent = Path.GetDirectoryName(model)!;
        Directory.CreateDirectory(parent);
        try
        {
            File.Copy(Mnist(), model);
            var result = RunCliFrom(directory, "run", relativePath, TestSupport.CommittedImage("mnist4.png") + "::mnist");
            Assert.Equal(0, result.Exit);
            Assert.Contains("Plus214_Output_0:float:1x10:", result.Output);
            Assert.DoesNotContain("Remote model locations", result.Output + result.Error);
        }
        finally
        {
            File.Delete(model);
            if (parent != directory) Directory.Delete(parent);
            Directory.Delete(directory);
        }
    }

    [SkippableTheory]
    [InlineData("http://example.invalid/model.onnx")]
    [InlineData("https://example.invalid/model.onnx")]
    [InlineData("HTTP://example.invalid/model.onnx")]
    [InlineData("HTTPS://example.invalid/model.onnx")]
    public void RunRemoteModelUrl_ExitsNotFoundWithoutLoading(string url)
    {
        var result = RunCli("run", url, TestSupport.CommittedImage("mnist4.png") + "::mnist");
        Assert.Equal(4, result.Exit);
        Assert.Contains("Remote model locations are not supported", result.Output);
    }

    [SkippableFact]
    public void SingleNodeProfile_ExitsSuccessAndPrintsOneNode()
    {
        var result = RunCli("run", Mnist(), TestSupport.CommittedImage("mnist4.png") + "::mnist",
            "--node", "Convolution28", "--profile", "--disable-simd");
        Assert.Equal(0, result.Exit);
        Assert.Contains("Total graph node count: 1", result.Output);
        Assert.Contains("Per-op stage breakdown", result.Output);
        Assert.DoesNotContain("Stack empty", result.Output + result.Error);
    }

    static ValueInfoProto FloatInfo(string name) => new() { Name = name, Type = new TypeProto
        { TensorType = new TypeProto.Types.Tensor { ElemType = 1,
            Shape = new TensorShapeProto { Dim = { new TensorShapeProto.Types.Dimension { DimValue = 2 } } } } } };

    [SkippableTheory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)]
    public void ExternalTensorAttributeCanBeInspected(int mode)
    {
        string file = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx");
        try
        {
            var model = new ModelProto { IrVersion = 8, Graph = new GraphProto { Name = "external-attribute", Output = { FloatInfo("y") } },
                OpsetImport = { new OperatorSetIdProto { Version = 14 } } };
            var tensor = new TensorProto { Name = "weights", DataType = 1, Dims = { 2 }, DataLocation = TensorProto.Types.DataLocation.External,
                ExternalData = { new StringStringEntryProto { Key = "location", Value = "missing-weights-" + Guid.NewGuid().ToString("N") + ".bin" } } };
            model.Graph.Node.Add(new NodeProto { Name = "constant", OpType = "Constant", Output = { "y" }, Attribute = {
                new AttributeProto { Name = "value", Type = AttributeProto.Types.AttributeType.Tensor, T = tensor } } });
            File.WriteAllBytes(file, model.ToByteArray());
            var args = new List<string> { "info", file };
            if (mode != 0) args.Add(mode == 1 ? "--ops" : "--init");
            var result = RunCli(args.ToArray());
            Assert.True(result.Exit == 0, result.Output + result.Error);
            if (mode == 0) Assert.Contains("tensor weights:float:2 (not materialized)", result.Output);
        }
        finally { File.Delete(file); }
    }

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
