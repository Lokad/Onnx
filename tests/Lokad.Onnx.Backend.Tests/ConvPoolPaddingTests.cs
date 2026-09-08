using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class ConvPoolPaddingTests
{
    static DenseTensor<float> F4(float[,,,] v) => DenseTensor<float>.OfValues(v);

    [Fact]
    public void Conv_OmittedPads_MeansZeros()
    {
        var x = F4(new float[1, 1, 3, 3] { { { { 1f, 2f, 3f }, { 4f, 5f, 6f }, { 7f, 8f, 9f } } } });
        var w = F4(new float[1, 1, 2, 2] { { { { 1f, 1f }, { 1f, 1f } } } });
        var r = CPUExecutionProvider.Conv(x, w, null, null, null, 1, null, null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs[0];
        Assert.Equal(new[] { 1, 1, 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 12f, 16f, 24f, 28f }, y.ToArray());
    }

    [Fact]
    public void Conv_OddSameUpperAndLower_DifferCorrectly()
    {
        var x = F4(new float[1, 1, 1, 5] { { { { 1f, 2f, 3f, 4f, 5f } } } });
        var w = F4(new float[1, 1, 1, 2] { { { { 1f, 1f } } } });
        var ru = CPUExecutionProvider.Conv(x, w, null, "SAME_UPPER", null, 1, null, null, new[] { 1, 2 }, null);
        var rl = CPUExecutionProvider.Conv(x, w, null, "SAME_LOWER", null, 1, null, null, new[] { 1, 2 }, null);
        Assert.Equal(OpStatus.Success, ru.Status);
        Assert.Equal(OpStatus.Success, rl.Status);
        Assert.Equal(new float[] { 3f, 7f, 5f }, ((Tensor<float>)ru.Outputs[0]).ToArray());
        Assert.Equal(new float[] { 1f, 5f, 9f }, ((Tensor<float>)rl.Outputs[0]).ToArray());
    }

    [Fact]
    public void Conv_Same_UnequalAxes_HandledPerAxis()
    {
        var x = DenseTensor<float>.OfShape(1, 1, 4, 6);
        for (int i = 0; i < x.Length; i++) x.SetValue(i, i);
        var w = DenseTensor<float>.OfShape(1, 1, 3, 3);
        w.Fill(1f);
        var ru = CPUExecutionProvider.Conv(x, w, null, "SAME_UPPER", null, 1, null, null, new[] { 2, 2 }, null);
        var rl = CPUExecutionProvider.Conv(x, w, null, "SAME_LOWER", null, 1, null, null, new[] { 2, 2 }, null);
        Assert.Equal(OpStatus.Success, ru.Status);
        Assert.Equal(OpStatus.Success, rl.Status);
        Assert.Equal(new[] { 1, 1, 2, 3 }, ((Tensor<float>)ru.Outputs[0]).Dimensions.ToArray());
        Assert.Equal(new[] { 1, 1, 2, 3 }, ((Tensor<float>)rl.Outputs[0]).Dimensions.ToArray());
        Assert.False(((Tensor<float>)ru.Outputs[0]).ToArray().SequenceEqual(((Tensor<float>)rl.Outputs[0]).ToArray()));
    }

    [Fact]
    public void Conv_AsymmetricExplicitPads_ShapeAndValues()
    {
        var x = F4(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });
        var w = F4(new float[1, 1, 2, 2] { { { { 1f, 1f }, { 1f, 1f } } } });
        var r = CPUExecutionProvider.Conv(x, w, null, "NOTSET", null, 1, null, new[] { 1, 0, 0, 0 }, new[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs[0];
        Assert.Equal(new[] { 1, 1, 2, 1 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 3f, 10f }, y.ToArray());
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Conv(x, w, null, "NOTSET", null, 1, null, new[] { 1, 0, 0 }, new[] { 1, 1 }, null).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Conv(x, w, null, "BOGUS", null, 1, null, null, null, null).Status);
    }

    [Fact]
    public void MaxPool_OmittedPads_MeansZeros()
    {
        var x = F4(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });
        var r = CPUExecutionProvider.MaxPool(x, null, 0, null, new[] { 2, 2 }, null, 0, new[] { 2, 2 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 4f }, ((Tensor<float>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void MaxPool_Dilation_ReadsDilatedTaps()
    {
        var x = F4(new float[1, 1, 1, 5] { { { { 1f, 2f, 3f, 4f, 5f } } } });
        var r = CPUExecutionProvider.MaxPool(x, "VALID", 0, new[] { 1, 2 }, new[] { 1, 2 }, null, 0, new[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs[0];
        Assert.Equal(new[] { 1, 1, 1, 3 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 3f, 4f, 5f }, y.ToArray());
    }

    [Fact]
    public void MaxPool_CeilMode_GrowsEdgeShape()
    {
        var x = F4(new float[1, 1, 1, 5] { { { { 1f, 2f, 3f, 4f, 5f } } } });
        var rf = CPUExecutionProvider.MaxPool(x, "VALID", 0, null, new[] { 1, 2 }, null, 0, new[] { 1, 2 }, null);
        var rc = CPUExecutionProvider.MaxPool(x, "VALID", 1, null, new[] { 1, 2 }, null, 0, new[] { 1, 2 }, null);
        Assert.Equal(OpStatus.Success, rf.Status);
        Assert.Equal(OpStatus.Success, rc.Status);
        Assert.Equal(new float[] { 2f, 4f }, ((Tensor<float>)rf.Outputs[0]).ToArray());
        Assert.Equal(new float[] { 2f, 4f, 5f }, ((Tensor<float>)rc.Outputs[0]).ToArray());
    }

    [Fact]
    public void MaxPool_UnsupportedVariants_FailExplicitly()
    {
        var x = F4(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.MaxPool(x, null, 0, null, new[] { 2, 2 }, null, 1, new[] { 2, 2 }, null).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.MaxPool(x, "BOGUS", 0, null, new[] { 2, 2 }, null, 0, new[] { 2, 2 }, null).Status);
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 11 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = x;
        var node = new Node
        {
            Name = "pool", Op = OpType.MaxPool, Inputs = new[] { "x" }, Outputs = new[] { "y", "i" },
            Attributes = new Dictionary<string, object>
            {
                ["auto_pad"] = "VALID", ["kernel_shape"] = new long[] { 2, 2 }, ["strides"] = new long[] { 2, 2 },
            },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Indices", r.Message ?? "");
    }
}
