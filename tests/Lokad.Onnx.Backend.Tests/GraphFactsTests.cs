using System.Collections.Generic;
using System.Linq;
using Lokad.Onnx.Optimization;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// G01-M1: the shared fact snapshot answers producer, consumer, output, dtype,
/// dimension and constant queries on a hand-built graph, and the pass runner with
/// zero registered passes changes nothing. No fusion reads the substrate yet.
/// </summary>
public class GraphFactsTests
{
    static OnnxModel TinyModel()
    {
        var mp = new OnnxModel { Name = "tiny-facts" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "z", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Initializers.Add(new OnnxTensor { Name = "w", ElementType = TensorElementType.Float, Dims = new[] { 4, 4 }, Data = new float[16] });
        var two = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new int[0], Data = new[] { 2f } });
        mp.Nodes.Add(new OnnxNode { OpType = "Constant", Inputs = new string[0], Outputs = new[] { "c1" }, Attributes = new Dictionary<string, object> { ["value"] = two } });
        mp.Nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { "x", "c1" }, Outputs = new[] { "m" }, Attributes = new Dictionary<string, object>() });
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "m", "w" }, Outputs = new[] { "y" }, Attributes = new Dictionary<string, object>() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "m", "x" }, Outputs = new[] { "z" }, Attributes = new Dictionary<string, object>() });
        return mp;
    }

    [Fact]
    public void FactsDescribeTinyGraph()
    {
        var graph = Model.Load(TinyModel())!;
        var facts = GraphFacts.Build(graph);
        Assert.Equal(0, facts.Producer["c1"]);
        Assert.Equal(1, facts.Producer["m"]);
        Assert.Equal(2, facts.Producer["y"]);
        Assert.Equal(3, facts.Producer["z"]);
        Assert.Equal(new[] { 1 }, facts.Consumers["c1"]);
        Assert.Equal(new[] { 1, 3 }, facts.Consumers["x"]);
        Assert.True(facts.IsGraphOutput("y"));
        Assert.True(facts.IsGraphOutput("z"));
        Assert.False(facts.IsGraphOutput("m"));
        Assert.True(facts.IsSingleConsumer("c1"));
        Assert.False(facts.IsSingleConsumer("x"));
        Assert.Equal(TensorElementType.Float, facts.Dtypes["x"]);
        Assert.Equal(TensorElementType.Float, facts.Dtypes["w"]);
        Assert.Equal(TensorElementType.Float, facts.Dtypes["c1"]);
        Assert.Equal(TensorElementType.Float, facts.Dtypes["m"]);
        Assert.Equal(TensorElementType.Float, facts.Dtypes["y"]);
        Assert.Equal(new[] { 2, 4 }, facts.KnownDims["x"]);
        Assert.Equal(new[] { 4, 4 }, facts.KnownDims["w"]);
        Assert.Empty(facts.KnownDims["c1"]);
        Assert.False(facts.KnownDims.ContainsKey("m"));
        Assert.False(facts.Constants.ContainsKey("w"));
        Assert.True(facts.Constants.ContainsKey("c1"));
        Assert.False(facts.Constants.ContainsKey("x"));
    }

    [Fact]
    public void EmptyPassListChangesNothing()
    {
        var graph = Model.Load(TinyModel())!;
        int before = graph.Nodes.Count;
        var report = GraphOptimizer.Run(graph);
        Assert.Empty(report);
        Assert.Equal(before, graph.Nodes.Count);
    }
}
