
namespace Lokad.Onnx.Backend.Tests;

public class DomainTypesTests
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
    public void Cast_AcceptsTensorElementType()
    {
        var x = DenseTensor<float>.OfValues(new float[] { 1.9f, -1.9f, 2.5f, -2.5f, 0.5f });
        var r = CPUExecutionProvider.Cast(x, TensorElementType.Int32, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new int[] { 1, -1, 2, -2, 0 }, ((Tensor<int>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void Cast_InvalidCode_FailsWithAttributeNotSupported()
    {
        var graph = Graph(13);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[] { 1f });
        var node = new Node
        {
            Name = "cast", Op = OpType.Cast, Inputs = new[] { "x" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["to"] = 999L },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void Resize_NearestMatchesExpectedValues()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });
        var y = Tensor<float>.Resize(x, new[] { 1, 1, 4, 4 }, MathOps.ResizeMode.Nearest,
            MathOps.ResizeCoordinateTransformation.HalfPixel, MathOps.ResizeNearestMode.RoundPreferFloor, -0.75f, null);
        Assert.Equal(new float[] { 1f, 1f, 2f, 2f, 1f, 1f, 2f, 2f, 3f, 3f, 4f, 4f, 3f, 3f, 4f, 4f }, y.ToArray());
    }

    [Fact]
    public void Resize_InvalidMode_Fails()
    {
        var graph = Graph(13);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });
        graph.Initializers["s"] = DenseTensor<long>.OfValues(new long[] { 1, 1, 4, 4 });
        var node = new Node
        {
            Name = "resize", Op = OpType.Resize, Inputs = new[] { "x", "", "", "s" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["mode"] = "bogus" },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void Resize_LinearIgnoresNearestMode()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });
        var r = CPUExecutionProvider.Resize(x, null, null,
            DenseTensor<int>.OfValues(new int[] { 1, 1, 4, 4 }), "linear", "half_pixel", "garbage", -0.75f, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
    }

    [Fact]
    public void Conv_GarbagePad_Fails()
    {
        var graph = Graph(13);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[1, 1, 3, 3] { { { { 1f, 2f, 3f }, { 4f, 5f, 6f }, { 7f, 8f, 9f } } } });
        graph.Inputs["w"] = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 0f }, { 0f, 1f } } } });
        var node = new Node
        {
            Name = "conv", Op = OpType.Conv, Inputs = new[] { "x", "w" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["auto_pad"] = "BOGUS" },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void MaxPool_GarbagePad_Fails()
    {
        var graph = Graph(13);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[1, 1, 4, 4] { { { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f }, { 9f, 10f, 11f, 12f }, { 13f, 14f, 15f, 16f } } } });
        var node = new Node
        {
            Name = "pool", Op = OpType.MaxPool, Inputs = new[] { "x" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>
            {
                ["auto_pad"] = "BOGUS",
                ["kernel_shape"] = new long[] { 2, 2 },
            },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }
}
