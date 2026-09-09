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

    [Fact]
    public void Prepare_CanonicalizesLosslessLongArrayAttributes()
    {
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "test";
        graph.Inputs["x"] = DenseTensor<float>.OfShape(1, 1, 4, 4);
        graph.Outputs["y"] = DenseTensor<float>.OfShape(1, 1, 2, 2);
        graph.Nodes.Add(new Node
        {
            Name = "pool", Op = OpType.MaxPool, Inputs = new[] { "x" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>
            {
                ["kernel_shape"] = new long[] { 2, 2 },
                ["strides"] = new long[] { 2, 2 },
            },
        });
        graph.RefreshLifetimeAnalysis();
        Assert.Equal(new int[] { 2, 2 }, Assert.IsType<int[]>(graph.Nodes[0].Attributes!["kernel_shape"]));
        Assert.Equal(new int[] { 2, 2 }, Assert.IsType<int[]>(graph.Nodes[0].Attributes!["strides"]));
        var input = DenseTensor<float>.OfValues(new float[1, 1, 4, 4] { { { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f }, { 9f, 10f, 11f, 12f }, { 13f, 14f, 15f, 16f } } } });
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { { "x", input } }, true));
        Assert.Equal(new float[] { 6f, 8f, 14f, 16f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
    }

    [Fact]
    public void Prepare_LeavesOverflowingLongArrayAttributesUntouched()
    {
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "test";
        graph.Inputs["x"] = DenseTensor<float>.OfShape(1, 1, 4, 4);
        graph.Outputs["y"] = DenseTensor<float>.OfShape(1, 1, 2, 2);
        graph.Nodes.Add(new Node
        {
            Name = "pool", Op = OpType.MaxPool, Inputs = new[] { "x" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["kernel_shape"] = new long[] { 2, 4294967296L } },
        });
        graph.RefreshLifetimeAnalysis();
        Assert.IsType<long[]>(graph.Nodes[0].Attributes!["kernel_shape"]);
        Assert.Throws<OverflowException>(() => graph.Nodes[0].Ints("kernel_shape"));
        var input = DenseTensor<float>.OfValues(new float[1, 1, 4, 4] { { { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f }, { 9f, 10f, 11f, 12f }, { 13f, 14f, 15f, 16f } } } });
        Assert.False(graph.Execute(new Dictionary<string, ITensor> { { "x", input } }, true));
    }

    [Fact]
    public void GetInputTensors_Null_ThrowsArgumentNull()
    {
        var graph = new ComputationalGraph();
        Assert.Throws<ArgumentNullException>(() => graph.GetInputTensors(null!));
    }
}
