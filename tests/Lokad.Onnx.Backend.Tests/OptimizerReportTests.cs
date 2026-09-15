using System;
using System.Collections.Generic;
using System.Linq;
using Lokad.Onnx.Optimization;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// P02: the unified optimizer pipeline exposes per-pass rewrite counts and
/// execution time in its report, in deterministic pass order.
/// </summary>
public class OptimizerReportTests
{
    static OnnxModel StrayModel()
    {
        var mp = new OnnxModel { Name = "tiny-report" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "z", ElementType = TensorElementType.Float, Dims = new[] { 2 } });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "x", "x" }, Outputs = new[] { "z" }, Attributes = new Dictionary<string, object>() });
        var stray = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new[] { 2 }, Data = new[] { 1f, 2f } });
        mp.Nodes.Add(new OnnxNode { OpType = "Constant", Inputs = new string[0], Outputs = new[] { "stray" }, Attributes = new Dictionary<string, object> { ["value"] = stray } });
        return mp;
    }

    [Fact]
    public void Run_ReportsElapsedAlongsideCounts()
    {
        var graph = Model.Load(StrayModel(), runOptimizer: false)!;
        var report = GraphOptimizer.Run(graph);
        var fold = Assert.Single(report, c => c.Pass == "constfold");
        Assert.True(fold.Rewritten > 0);
        Assert.True(fold.Elapsed >= TimeSpan.Zero);
        Assert.All(report, c => Assert.True(c.Elapsed >= TimeSpan.Zero));
    }

    [Fact]
    public void Run_PassSequenceIsDeterministic()
    {
        var first = GraphOptimizer.Run(Model.Load(StrayModel(), runOptimizer: false)!);
        var second = GraphOptimizer.Run(Model.Load(StrayModel(), runOptimizer: false)!);
        Assert.Equal(first.Select(c => c.Pass), second.Select(c => c.Pass));
        Assert.Equal(first.Select(c => c.Rewritten), second.Select(c => c.Rewritten));
    }
}