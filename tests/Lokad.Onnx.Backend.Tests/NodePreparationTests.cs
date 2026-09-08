extern alias OnnxSharp;

using OnnxSharp::Onnx;

namespace Lokad.Onnx.Backend.Tests;

public class NodePreparationTests
{
    [Fact]
    public void UnnamedNodes_GetDistinctSequentialIds()
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Inputs["x"] = DenseTensor<float>.OfShape(2);
        g.Outputs["y"] = DenseTensor<float>.OfShape(2);
        g.Nodes.Add(new Node { Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "t" } });
        g.Nodes.Add(new Node { Op = OpType.Relu, Inputs = new[] { "t" }, Outputs = new[] { "y" } });
        g.Nodes.Add(new Node { Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "u" } });
        g.IntermediateOutputs["t"] = null;
        g.RefreshLifetimeAnalysis();
        var ids = g.Nodes.Select(n => n.ID).ToArray();
        Assert.Equal(new long[] { 0, 1, 2 }, ids);
    }

    [Fact]
    public void LoadedGraphs_HaveSequentialIds()
    {
        var g1 = OnnxImport.Load(Path.Combine(AppContext.BaseDirectory, "models", "mnist-8.onnx"))!;
        var g2 = OnnxImport.Load(Path.Combine(AppContext.BaseDirectory, "models", "mnist-8.onnx"))!;
        var ids1 = g1.Nodes.Select(n => n.ID).ToArray();
        var ids2 = g2.Nodes.Select(n => n.ID).ToArray();
        Assert.Equal(ids1, ids2);
        for (int i = 0; i < ids1.Length; i++) Assert.Equal(i, ids1[i]);
    }

    [Fact]
    public void ImportedIntArrayAttributes_ArriveCanonicalized()
    {
        var attr = new AttributeProto { Name = "axes", Type = AttributeProto.Types.AttributeType.Ints };
        attr.Ints.Add(0L);
        attr.Ints.Add(1L);
        var node = new NodeProto { Name = "r", OpType = "ReduceMean" };
        node.Attribute.Add(attr);
        var dto = node.ToNodeDto(null);
        var axes = Assert.IsType<int[]>(dto.Attributes["axes"]);
        Assert.Equal(new int[] { 0, 1 }, axes);
    }
}
