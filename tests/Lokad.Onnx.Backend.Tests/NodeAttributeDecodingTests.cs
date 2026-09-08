using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

public class NodeAttributeDecodingTests
{
    static ComputationalGraph Graph(int opset)
    {
        return new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = opset },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
    }

    [Fact]
    public void Conv_LongGroup_IsHonored()
    {
        var graph = Graph(13);
        var x = DenseTensor<float>.OfValues(new float[1, 2, 3, 3] { { { { 1f, 2f, 3f }, { 4f, 5f, 6f }, { 7f, 8f, 9f } }, { { 9f, 8f, 7f }, { 6f, 5f, 4f }, { 3f, 2f, 1f } } } });
        var w = DenseTensor<float>.OfValues(new float[2, 1, 2, 2] { { { { 1f, 0f }, { 0f, 1f } } }, { { { 1f, 1f }, { 1f, 1f } } } });
        graph.Inputs["x"] = x; graph.Inputs["w"] = w;
        var node = new Node
        {
            Name = "conv", Op = OpType.Conv, Inputs = new[] { "x", "w" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>
            {
                ["auto_pad"] = "NOTSET", ["pads"] = new long[] { 0, 0, 0, 0 },
                ["strides"] = new long[] { 1, 1 }, ["group"] = 2L,
            },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs[0];
        var expected = Tensor<float>.Conv2D(x, w, 2, MathOps.PadType.Value, padvalue: 0, kernelshape: null, strides: new[] { 1, 1 }, bias: null, dilations: null);
        Assert.Equal(expected.Dimensions.ToArray(), y.Dimensions.ToArray());
        var a = y.ToArray(); var e = expected.ToArray();
        for (int i = 0; i < e.Length; i++) Assert.Equal(e[i], a[i], 5);
    }

    [Fact]
    public void MaxPool_LongCeilAndStorageOrder_DoNotBreak()
    {
        var graph = Graph(13);
        var x = DenseTensor<float>.OfValues(new float[1, 1, 4, 4] { { { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f }, { 9f, 10f, 11f, 12f }, { 13f, 14f, 15f, 16f } } } });
        graph.Inputs["x"] = x;
        var node = new Node
        {
            Name = "pool", Op = OpType.MaxPool, Inputs = new[] { "x" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>
            {
                ["auto_pad"] = "VALID", ["kernel_shape"] = new long[] { 2, 2 }, ["strides"] = new long[] { 2, 2 },
                ["ceil_mode"] = 0L, ["storage_order"] = 0L,
            },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, r.Status);
    }

    [Fact]
    public void Reshape_LongAllowZero_OneKeepsZero()
    {
        var graph = Graph(13);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        graph.Inputs["s"] = DenseTensor<long>.OfValues(new long[] { 2, 0 });
        var node = new Node
        {
            Name = "reshape", Op = OpType.Reshape, Inputs = new[] { "x", "s" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["allowzero"] = 1L },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void Reshape_LongAllowZero_ZeroCopiesDim()
    {
        var graph = Graph(13);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        graph.Inputs["s"] = DenseTensor<long>.OfValues(new long[] { 2, 0 });
        var node = new Node
        {
            Name = "reshape", Op = OpType.Reshape, Inputs = new[] { "x", "s" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["allowzero"] = 0L },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new[] { 2, 3 }, ((Tensor<float>)r.Outputs[0]).Dimensions.ToArray());
    }

    [Fact]
    public void LayerNorm_MissingEpsilon_IsFiniteOnConstantInput()
    {
        var graph = Graph(13);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[,] { { 5f, 5f, 5f, 5f } });
        graph.Inputs["scale"] = DenseTensor<float>.OfValues(new float[] { 1f, 1f, 1f, 1f });
        graph.Inputs["bias"] = DenseTensor<float>.OfValues(new float[] { 0f, 0f, 0f, 0f });
        var node = new Node
        {
            Name = "ln", Op = OpType.LayerNormalization, Inputs = new[] { "x", "scale", "bias" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["axis"] = -1L },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, r.Status);
        foreach (var v in ((Tensor<float>)r.Outputs[0]).ToArray()) Assert.True(float.IsFinite(v));
    }

    [Fact]
    public void LayerNorm_WrongBiasDtype_FailsExplicitly()
    {
        var graph = Graph(13);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f } });
        graph.Inputs["scale"] = DenseTensor<float>.OfValues(new float[] { 1f, 1f, 1f, 1f });
        graph.Inputs["bias"] = DenseTensor<int>.OfValues(new int[] { 0, 0, 0, 0 });
        var node = new Node
        {
            Name = "ln", Op = OpType.LayerNormalization, Inputs = new[] { "x", "scale", "bias" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["axis"] = -1L },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void LayerNorm_StashType_FailsExplicitly()
    {
        var graph = Graph(13);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f } });
        graph.Inputs["scale"] = DenseTensor<float>.OfValues(new float[] { 1f, 1f, 1f, 1f });
        var node = new Node
        {
            Name = "ln", Op = OpType.LayerNormalization, Inputs = new[] { "x", "scale" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["axis"] = -1L, ["stash_type"] = 1L },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }
}

