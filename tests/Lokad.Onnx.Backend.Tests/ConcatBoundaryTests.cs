using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins Concat axis validation: out-of-range axes fail (ORT 1.29 refuses
/// them at load) while negative axes work, at provider and node level.
/// </summary>
public class ConcatBoundaryTests
{
    [Fact]
    public void OutOfRangeAxis_FailsCleanly()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var y = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 7f, 8f } });
        Assert.Throws<System.ArgumentException>(() => CPU.Concat(new ITensor[] { x, y }, 5, null));
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 13 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = x;
        graph.Inputs["y"] = y;
        var node = new Node
        {
            Name = "n", Op = OpType.Concat, OpTypeName = OpType.Concat.ToString(), Domain = "",
            OpsetVersion = 13, IsFused = false,
            Inputs = new[] { "x", "y" }, Outputs = new[] { "z" },
            Attributes = new Dictionary<string, object> { ["axis"] = 5L },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void NegativeAxis_Concatenates()
    {
        // ORT 1.29: [[1, 2, 3, 4]].
        var result = CPU.Concat(new ITensor[]
        {
            DenseTensor<float>.OfValues(new float[,] { { 1f, 2f } }),
            DenseTensor<float>.OfValues(new float[,] { { 3f, 4f } }),
        }, -1, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var z = (Tensor<float>)result.Outputs![0];
        Assert.Equal(new int[] { 1, 4 }, z.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f, 3f, 4f }, z.ToArray());
    }
}