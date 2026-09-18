using Lokad.Onnx.Bench;

namespace Lokad.Onnx.Backend.Tests;

public class KBlockedRouteTests
{
    static TensorExecutionOptions KBlocked()
    {
        return TensorExecutionOptions.Auto with { UseKBlockedPanels = true };
    }

    static GemmMatrixResult RunOne(GemmShape s, TensorExecutionOptions o)
    {
        var results = GemmMatrix.Run(new GemmShape[] { s }, o);
        Assert.Single(results);
        return results[0];
    }

    [Fact]
    public void SwitchOff_PreservesLegacyRoute()
    {
        var r = RunOne(new GemmShape("kb-off", 16, 1024, 1024, true, "test"), TensorExecutionOptions.Auto);
        Assert.Equal("prep-grouped", r.Route);
        Assert.True(r.Pass, "maxScaled=" + r.MaxScaled);
    }

    [Fact]
    public void SmallN_PassthroughKeepsLegacyRoute()
    {
        var r = RunOne(new GemmShape("kb-small", 16, 64, 64, true, "test"), KBlocked());
        Assert.Equal("prep-grouped", r.Route);
        Assert.True(r.Pass, "maxScaled=" + r.MaxScaled);
    }

    [Fact]
    public void BlockedRoute_AgreesWithDouble()
    {
        var r = RunOne(new GemmShape("kb-proj", 16, 1024, 1024, true, "test"), KBlocked());
        Assert.Equal("prep-kblocked", r.Route);
        Assert.True(r.Pass, "maxScaled=" + r.MaxScaled);
    }

    [Fact]
    public void ReductionTail_AgreesWithDouble()
    {
        var r = RunOne(new GemmShape("kb-tail", 16, 300, 64, true, "test"), KBlocked());
        Assert.Equal("prep-kblocked", r.Route);
        Assert.True(r.Pass, "maxScaled=" + r.MaxScaled);
    }

    [Fact]
    public void DynamicPath_UntouchedBySwitch()
    {
        var r = RunOne(new GemmShape("kb-dyn", 16, 128, 128, false, "test"), KBlocked());
        Assert.Equal("tiled-2x4", r.Route);
        Assert.True(r.Pass, "maxScaled=" + r.MaxScaled);
    }


    [Fact]
    public void LateOptIn_PreparesBlockedClone()
    {
        var x = DenseTensor<float>.OfShape(new int[] { 16, 1024 });
        var w = DenseTensor<float>.OfShape(new int[] { 1024, 1024 });
        var span = w.Buffer.Span;
        for (int i = 0; i < span.Length; i++) span[i] = 0.01f;
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "kb-late";
        graph.Inputs["x"] = x;
        graph.Initializers["w"] = w;
        graph.Outputs["z"] = DenseTensor<float>.OfShape(new int[] { 16, 1024 });
        graph.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, Inputs = new[] { "x", "w" }, Outputs = new[] { "z" } });
        graph.RefreshLifetimeAnalysis();
        graph.Options = new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Auto with { UseKBlockedPanels = true });
        graph.RefreshLifetimeAnalysis();
        Assert.True(graph.Initializers.ContainsKey("kbpacked:w"), "late blocked clone missing");
    }
    [Fact]
    public void BatchedRank3_RoutesKBlockedAndAgrees()
    {
        var x = DenseTensor<float>.OfShape(new int[] { 1, 16, 1024 });
        var xs = x.Buffer.Span;
        for (int i = 0; i < xs.Length; i++) xs[i] = (float)((i % 97) - 48) * 0.01f;
        var w = DenseTensor<float>.OfShape(new int[] { 1024, 1024 });
        var ws = w.Buffer.Span;
        for (int i = 0; i < ws.Length; i++) ws[i] = (float)(((i + 31) % 97) - 48) * 0.01f;
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "kb-batch3";
        graph.Inputs["x"] = DenseTensor<float>.OfShape(new int[] { 1, 16, 1024 });
        graph.Initializers["w"] = w;
        graph.Outputs["z"] = DenseTensor<float>.OfShape(new int[] { 1, 16, 1024 });
        graph.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, Inputs = new[] { "x", "w" }, Outputs = new[] { "z" } });
        graph.Options = new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Auto with { UseKBlockedPanels = true });
        graph.RefreshLifetimeAnalysis();
        using var profilerScope = Profiler.BeginExecution(true);
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = x }, true), graph.LastErrorMessage);
        Assert.NotNull(graph.LastProfile);
        var node = Assert.Single(graph.LastProfile!);
        Assert.True(node.Detail.Contains(" route=prep-kblocked", StringComparison.Ordinal), "Detail: " + node.Detail);
        var xd = DenseTensor<double>.OfShape(new int[] { 1, 16, 1024 });
        var xds = xd.Buffer.Span;
        for (int i = 0; i < xds.Length; i++) xds[i] = (double)((i % 97) - 48) * 0.01;
        var wd = DenseTensor<double>.OfShape(new int[] { 1024, 1024 });
        var wds = wd.Buffer.Span;
        for (int i = 0; i < wds.Length; i++) wds[i] = (double)(((i + 31) % 97) - 48) * 0.01;
        Profiler.StartNodeProfile(2, OpType.MatMul);
        var zref = Tensor<double>.MatMul(xd, wd, TensorExecutionOptions.Scalar);
        Profiler.StopNodeProfile();
        var zf = ((Tensor<float>)graph.Outputs["z"]).ToArray();
        var zd = zref.ToArray();
        Assert.Equal(zf.Length, zd.Length);
        double worst = 0.0;
        for (int i = 0; i < zf.Length; i++) worst = Math.Max(worst, Math.Abs(zf[i] - zd[i]) / Math.Max(1.0, Math.Abs(zd[i])));
        Assert.True(worst <= 1e-4, "worst=" + worst);
    }
    [Fact]
    public void PackingReport_CountsBlockedClones()
    {
        var x = DenseTensor<float>.OfShape(new int[] { 16, 1024 });
        var w = DenseTensor<float>.OfShape(new int[] { 1024, 1024 });
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "kb-report";
        graph.Inputs["x"] = x;
        graph.Initializers["w"] = w;
        graph.Outputs["z"] = DenseTensor<float>.OfShape(new int[] { 16, 1024 });
        graph.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, Inputs = new[] { "x", "w" }, Outputs = new[] { "z" } });
        graph.RefreshLifetimeAnalysis();
        Assert.Equal(0, graph.PackingReport.KBlockedLive);
        Assert.Equal(0, graph.PackingReport.KBlockedRetainedBytes);
        Assert.Equal(1, graph.PackingReport.Live);
        graph.Options = new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Auto with { UseKBlockedPanels = true });
        graph.RefreshLifetimeAnalysis();
        Assert.Equal(1, graph.PackingReport.KBlockedLive);
        Assert.Equal(4194304, graph.PackingReport.KBlockedRetainedBytes);
        Assert.Equal(1, graph.PackingReport.Live);
    }
}
