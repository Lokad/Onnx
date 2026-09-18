using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

public class MatMulRouteTests
{
    static DenseTensor<float> Sequential(int rows, int cols)
    {
        var t = DenseTensor<float>.OfShape(new int[] { rows, cols });
        var span = t.Buffer.Span;
        for (int i = 0; i < span.Length; i++) span[i] = (float)((i % 97) + 1) * 0.01f;
        return t;
    }

    static string RunSingleOp(Func<Tensor<float>> op)
    {
        using var profilerScope = Profiler.BeginExecution(true);
        Profiler.StartNodeProfile(1, OpType.MatMul);
        var z = op();
        Profiler.StopNodeProfile();
        Assert.NotNull(z);
        var snap = Profiler.RouteCountsSnapshot();
        Assert.Single(snap);
        return snap.Keys.First();
    }

    [Fact]
    public void SmallTransientPack_ReportsTransRoute()
    {
        if (!Fma.IsSupported) return;
        var x = Sequential(64, 64);
        var y = Sequential(64, 64);
        var z = Tensor<float>.MatMul2D(x, y, TensorExecutionOptions.Auto);
        Assert.Equal(new int[] { 64, 64 }, z.Dimensions.ToArray());
        var route = RunSingleOp(() => Tensor<float>.MatMul2D(x, y, TensorExecutionOptions.Auto));
        Assert.True(route.StartsWith("trans-", StringComparison.Ordinal), "Expected trans- route, got: " + route);
    }

    [Fact]
    public void LargeTransientPack_ReportsTransTiled()
    {
        if (!Avx512F.IsSupported || !Fma.IsSupported) return;
        var route = RunSingleOp(() => Tensor<float>.MatMul2D(Sequential(80, 256), Sequential(256, 1152), TensorExecutionOptions.Auto));
        Assert.Equal("trans-tiled", route);
    }

    [Fact]
    public void SingleRow_ReportsIntrinsics()
    {
        if (!Fma.IsSupported) return;
        var route = RunSingleOp(() => Tensor<float>.MatMul2D(Sequential(1, 32), Sequential(32, 16), TensorExecutionOptions.Auto));
        Assert.Equal("intrinsics", route);
    }

    [Fact]
    public void SimdMode_ReportsSimd()
    {
        var route = RunSingleOp(() => Tensor<float>.MatMul2D(Sequential(4, 8), Sequential(8, 16), TensorExecutionOptions.Simd));
        Assert.Equal("simd", route);
    }

    [Fact]
    public void ScalarMode_ReportsScalar()
    {
        var route = RunSingleOp(() => Tensor<float>.MatMul2D(Sequential(4, 8), Sequential(8, 16), TensorExecutionOptions.Scalar));
        Assert.Equal("scalar", route);
    }

    [Fact]
    public void Routes_AccumulateAcrossOps()
    {
        using var profilerScope = Profiler.BeginExecution(true);
        Profiler.StartNodeProfile(1, OpType.MatMul);
        Tensor<float>.MatMul2D(Sequential(2, 2), Sequential(2, 2), TensorExecutionOptions.Scalar);
        Tensor<float>.MatMul2D(Sequential(2, 2), Sequential(2, 2), TensorExecutionOptions.Scalar);
        Profiler.StopNodeProfile();
        var snap = Profiler.RouteCountsSnapshot();
        Assert.Single(snap);
        Assert.True(snap.TryGetValue("scalar", out long hits) && hits == 2, "Expected scalar x2.");
    }

    [Fact]
    public void DisabledProfiler_ReportIsNoOp()
    {
        Profiler.ReportKernelRoute("probe");
        Assert.Empty(Profiler.RouteCountsSnapshot());
    }

    [Fact]
    public void GraphNode_DetailCarriesReportedRoute()
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "route";
        g.Inputs["x"] = DenseTensor<float>.OfShape(new int[] { 4, 8 });
        g.Inputs["y"] = DenseTensor<float>.OfShape(new int[] { 8, 16 });
        g.Outputs["z"] = DenseTensor<float>.OfShape(new int[] { 4, 16 });
        g.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, Inputs = new[] { "x", "y" }, Outputs = new[] { "z" } });
        g.RefreshLifetimeAnalysis();
        using var profilerScope = Profiler.BeginExecution(true);
        var inputs = new Dictionary<string, ITensor> { { "x", Sequential(4, 8) }, { "y", Sequential(8, 16) } };
        Assert.True(g.Execute(inputs, true));
        Assert.NotNull(g.LastProfile);
        var node = Assert.Single(g.LastProfile!);
        Assert.True(node.Detail.Contains(" route=", StringComparison.Ordinal), "Detail missing route: " + node.Detail);
    }
}
