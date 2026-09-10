using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// First Tile coverage anywhere: basic, 2-D, zero-repeat and empty-input
/// tiling against ORT 1.29 probe values, plus the negative-repeat clean
/// failure both sides enforce (ORT InvalidArgument, Lokad descriptive
/// Failure).
/// </summary>
public class TileBoundaryTests
{
    static float[] RunTile(float[,] x, long[] repeats)
    {
        var result = CPU.Tile(
            DenseTensor<float>.OfValues(x),
            DenseTensor<long>.OfValues(repeats), null);
        Assert.Equal(OpStatus.Success, result.Status);
        return ((Tensor<float>)result.Outputs![0]).ToArray();
    }

    static int[] ShapeOf(float[,] x, long[] repeats)
    {
        var result = CPU.Tile(
            DenseTensor<float>.OfValues(x),
            DenseTensor<long>.OfValues(repeats), null);
        Assert.Equal(OpStatus.Success, result.Status);
        return ((Tensor<float>)result.Outputs![0]).Dimensions.ToArray();
    }

    [Fact]
    public void Basic1D_MatchesOrt()
    {
        // ORT 1.29: [1, 2, 1, 2, 1, 2].
        var result = CPU.Tile(
            DenseTensor<float>.OfValues(new float[] { 1f, 2f }),
            DenseTensor<long>.OfValues(new long[] { 3L }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new float[] { 1f, 2f, 1f, 2f, 1f, 2f }, ((Tensor<float>)result.Outputs![0]).ToArray());
    }

    [Fact]
    public void Tiled2D_MatchesOrt()
    {
        // ORT 1.29: rows [1,2],[3,4],[1,2],[3,4].
        Assert.Equal(new int[] { 4, 2 }, ShapeOf(new float[,] { { 1f, 2f }, { 3f, 4f } }, new long[] { 2L, 1L }));
        Assert.Equal(
            new float[] { 1f, 2f, 3f, 4f, 1f, 2f, 3f, 4f },
            RunTile(new float[,] { { 1f, 2f }, { 3f, 4f } }, new long[] { 2L, 1L }));
    }

    [Fact]
    public void ZeroRepeats_ReturnEmpty()
    {
        // ORT 1.29: shape [0], no elements.
        var result = CPU.Tile(
            DenseTensor<float>.OfValues(new float[] { 1f, 2f }),
            DenseTensor<long>.OfValues(new long[] { 0L }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var output = (Tensor<float>)result.Outputs![0];
        Assert.Equal(new int[] { 0 }, output.Dimensions.ToArray());
        Assert.Empty(output.ToArray());
    }

    [Fact]
    public void EmptyInput_TilesToEmpty()
    {
        // ORT 1.29: [2,0] tiled [2,3] -> shape [4,0], no elements.
        var result = CPU.Tile(
            DenseTensor<float>.OfShape(2, 0),
            DenseTensor<long>.OfValues(new long[] { 2L, 3L }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var output = (Tensor<float>)result.Outputs![0];
        Assert.Equal(new int[] { 4, 0 }, output.Dimensions.ToArray());
        Assert.Empty(output.ToArray());
    }

    [Fact]
    public void NegativeRepeats_FailsCleanly()
    {
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Tile(
            DenseTensor<float>.OfValues(new float[] { 1f, 2f }), new int[] { -1 }));
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 13 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        graph.Inputs["r"] = DenseTensor<long>.OfValues(new long[] { -1L });
        var node = new Node
        {
            Name = "n", Op = OpType.Tile, OpTypeName = OpType.Tile.ToString(), Domain = "",
            OpsetVersion = 13, IsFused = false,
            Inputs = new[] { "x", "r" }, Outputs = new[] { "z" },
            Attributes = new Dictionary<string, object>(),
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Repeats", r.Message ?? "");
    }

    [Fact]
    public void RankMismatch_FailsCleanly()
    {
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Tile(
            DenseTensor<float>.OfValues(new float[,] { { 1f, 2f } }), new int[] { 2, 1, 1 }));
    }
}