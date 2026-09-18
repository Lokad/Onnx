using System.Text.Json;
using Lokad.Onnx.Bench;

namespace Lokad.Onnx.Backend.Tests;

public class ProfileVoiceTests
{
    [Fact]
    public void Report_RoundTripsSyntheticGraph()
    {
        var x = DenseTensor<float>.OfShape(new int[] { 4, 8 });
        var w = DenseTensor<float>.OfShape(new int[] { 8, 16 });
        var xs = x.Buffer.Span;
        for (int i = 0; i < xs.Length; i++) xs[i] = 0.01f;
        var ws = w.Buffer.Span;
        for (int i = 0; i < ws.Length; i++) ws[i] = 0.02f;
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "report-test";
        graph.Inputs["x"] = DenseTensor<float>.OfShape(new int[] { 4, 8 });
        graph.Initializers["w"] = w;
        graph.Outputs["z"] = DenseTensor<float>.OfShape(new int[] { 4, 16 });
        graph.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, Inputs = new[] { "x", "w" }, Outputs = new[] { "z" } });
        graph.RefreshLifetimeAnalysis();
        string json = VoiceReport.Build(graph, new Dictionary<string, ITensor> { ["x"] = x }, ExecutionOptions.Default);
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        Assert.Equal(1, root.GetProperty("nodes").GetArrayLength());
        var node = root.GetProperty("nodes")[0];
        Assert.Equal("MatMul", node.GetProperty("op").GetString());
        Assert.Equal("prep-grouped", node.GetProperty("route").GetString());
        Assert.True(root.GetProperty("routes").TryGetProperty("prep-grouped", out _));
        Assert.True(root.TryGetProperty("topAlloc", out _));
        Assert.Equal(1, root.GetProperty("packingLegacy").GetInt32());
    }
}
