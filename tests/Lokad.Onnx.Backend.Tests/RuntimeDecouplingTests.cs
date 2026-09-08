using System;
using System.Collections.Generic;
using Xunit;

namespace Lokad.Onnx.Backend.Tests;

public class RuntimeDecouplingTests
{
    [Fact]
    public void Runtime_HasNoStaticConstructor()
    {
        Assert.Null(typeof(Runtime).TypeInitializer);
    }

    [Theory]
    [InlineData(typeof(ComputationalGraph))]
    [InlineData(typeof(GraphExecution))]
    [InlineData(typeof(CPUExecutionProvider))]
    [InlineData(typeof(Model))]
    [InlineData(typeof(Data))]
    [InlineData(typeof(Images))]
    [InlineData(typeof(Text))]
    public void LibraryConcepts_DoNotInheritRuntime(Type t)
    {
        Assert.False(typeof(Runtime).IsAssignableFrom(t), t.FullName + " must not inherit Runtime.");
    }

    [Fact]
    public void Runtime_ExposesNoHostState()
    {
        foreach (var name in new[] { "Cts", "Ct", "InteractiveConsole", "Rng", "SessionId", "EntryAssembly", "IsUnitTestRun", "RuntimeInitialized", "DebugEnabled", "ToolName", "LogName", "PathSeparator", "RunFile" })
        {
            Assert.Null(typeof(Runtime).GetProperty(name));
        }
        foreach (var name in new[] { "Initialize", "RunCmd", "CopyDirectory", "ViewFilePath", "FailIfFileNotFound", "WarnIfFileExists", "GetProp" })
        {
            Assert.Null(typeof(Runtime).GetMethod(name));
        }
        foreach (var name in new[] { "Info", "Debug", "Error", "Warn", "Fatal", "Begin", "DownloadFile" })
        {
            Assert.NotEmpty(typeof(Runtime).GetMethods().Where(m => m.Name == name));
        }
        Assert.NotNull(typeof(Runtime).GetProperty("AssemblyLocation"));
        Assert.NotNull(typeof(Runtime).GetProperty("AssemblyVersion"));
    }

    [Fact]
    public void HandBuiltGraph_ExecutesWithoutHostInitialization()
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Inputs["x"] = DenseTensor<float>.OfShape(2);
        g.Outputs["y"] = DenseTensor<float>.OfShape(2);
        g.Nodes.Add(new Node { Name = "r", Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "y" } });
        g.RefreshLifetimeAnalysis();
        var user = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) } };
        Assert.True(g.Execute(user, false));
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
    }
}
