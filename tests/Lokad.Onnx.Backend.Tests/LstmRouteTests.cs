using System.Runtime.Intrinsics.X86;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

public class LstmRouteTests
{
    static DenseTensor<float> FillRank3(int d0, int d1, int d2)
    {
        var t = DenseTensor<float>.OfShape(new[] { d0, d1, d2 });
        var span = t.Buffer.Span;
        for (int i = 0; i < span.Length; i++) span[i] = (float)((i % 97) + 1) * 0.01f;
        return t;
    }

    static System.Collections.Generic.IReadOnlyDictionary<string, long> RunLstmOp(System.Func<OpResult> op)
    {
        using var profilerScope = Profiler.BeginExecution(true);
        Profiler.StartNodeProfile(1, OpType.LSTM);
        var r = op();
        Profiler.StopNodeProfile();
        Assert.Equal(OpStatus.Success, r.Status);
        return Profiler.RouteCountsSnapshot();
    }

    [Fact]
    public void UnpreparedShared_ReportsSharedRowdotAndVectorGates()
    {
        if (!Fma.IsSupported) return;
        var x = FillRank3(32, 1, 4);
        var w = FillRank3(1, 32, 4);
        var r = FillRank3(1, 32, 8);
        var snap = RunLstmOp(() => CPU.Lstm(x, w, r, null, null, null, null, null, "forward", null, null, null, null, 8, false, 0, 1, null, null));
        Assert.True(snap.TryGetValue("lstm-shared-rowdot", out long d) && d == 1, "shared-rowdot x1");
        Assert.True(snap.TryGetValue("lstm-gates-vector", out long g) && g == 1, "gates-vector x1");
    }

    [Fact]
    public void ClippedGates_ReportsScalarGates()
    {
        if (!Fma.IsSupported) return;
        var x = FillRank3(4, 1, 4);
        var w = FillRank3(1, 32, 4);
        var r = FillRank3(1, 32, 8);
        var snap = RunLstmOp(() => CPU.Lstm(x, w, r, null, null, null, null, null, "forward", null, null, null, 0.1f, 8, false, 0, 1, null, null));
        Assert.True(snap.TryGetValue("lstm-scalar", out long d) && d == 1, "scalar x1");
        Assert.True(snap.TryGetValue("lstm-gates-scalar", out long g) && g == 1, "gates-scalar x1");
    }

    [Fact]
    public void UnpreparedShortSequence_ReportsScalar()
    {
        if (!Fma.IsSupported) return;
        var x = FillRank3(4, 1, 4);
        var w = FillRank3(1, 32, 4);
        var r = FillRank3(1, 32, 8);
        var snap = RunLstmOp(() => CPU.Lstm(x, w, r, null, null, null, null, null, "forward", null, null, null, null, 8, false, 0, 1, null, null));
        Assert.True(snap.TryGetValue("lstm-scalar", out long d) && d == 1, "scalar x1");
        Assert.True(snap.TryGetValue("lstm-gates-vector", out long g) && g == 1, "gates-vector x1");
    }

    static ComputationalGraph BuildLstmGraph(DenseTensor<float> x, DenseTensor<float> w, DenseTensor<float> r, int hidden)
    {
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "lstm-route-test";
        graph.Inputs["x"] = x;
        graph.Initializers["w"] = w;
        graph.Initializers["r"] = r;
        graph.Outputs["y"] = Tensor<float>.Zeros(x.Dimensions[0], 1, x.Dimensions[1], hidden).ToDenseTensor();
        graph.Nodes.Add(new Node { Name = "lstm", Op = OpType.LSTM, OpTypeName = "LSTM", Domain = "", Inputs = new[] { "x", "w", "r" }, Outputs = new[] { "y" }, Attributes = new Dictionary<string, object> { ["hidden_size"] = hidden, ["direction"] = "forward" } });
        return graph;
    }

    [Fact]
    public void PreparedGraph_ReportsPackedPanelInDetail()
    {
        if (!Fma.IsSupported) return;
        var x = FillRank3(8, 1, 8);
        var w = FillRank3(1, 64, 8);
        var r = FillRank3(1, 64, 16);
        var graph = BuildLstmGraph(x, w, r, 16);
        graph.RefreshLifetimeAnalysis();
        using var profilerScope = Profiler.BeginExecution(true);
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = x }, true), graph.LastErrorMessage);
        Assert.NotNull(graph.LastProfile);
        var node = Assert.Single(graph.LastProfile!);
        Assert.True(node.Detail.Contains(" route=lstm-packed-panel", StringComparison.Ordinal), "Detail missing packed-panel: " + node.Detail);
    }
}
