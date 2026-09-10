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

    [Fact]
    public void PartialZeroRepeats_YieldsEmpty()
    {
        // ORT 1.29: [1,2] tiled [2,0] yields [2,0], empty.
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f } });
        var r = CPU.Tile(x, DenseTensor<long>.OfValues(new long[] { 2L, 0L }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new int[] { 2, 0 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
    }

    [Fact]
    public void EmptyRepeats_FailsCleanly()
    {
        // ORT 1.29 fails the run when repeats length != input rank (either
        // side); the kernel rejects the mismatch up front, as above.
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Tile(
            DenseTensor<float>.OfValues(new float[] { 1f, 2f }), new int[0]));
    }

    [Fact]
    public void Tile_UInt_MatchesOrt()
    {
        // ORT 1.29: repeats [2, 1] over [[1, 2], [3, max]] (u32) and [2] over [max, 7] (u64).
        var t32 = CPU.Tile(DenseTensor<uint>.OfValues(new uint[,] { { 1u, 2u }, { 3u, 4294967295u } }), DenseTensor<long>.OfValues(new long[] { 2L, 1L }), null);
        Assert.Equal(OpStatus.Success, t32.Status);
        var y32 = (Tensor<uint>)t32.Outputs[0];
        Assert.Equal(new int[] { 4, 2 }, y32.Dimensions.ToArray());
        Assert.Equal(new uint[] { 1u, 2u, 3u, 4294967295u, 1u, 2u, 3u, 4294967295u }, y32.ToArray());
        var t64 = CPU.Tile(DenseTensor<ulong>.OfValues(new ulong[] { 18446744073709551615ul, 7ul }), DenseTensor<long>.OfValues(new long[] { 2L }), null);
        Assert.Equal(OpStatus.Success, t64.Status);
        Assert.Equal(new ulong[] { 18446744073709551615ul, 7ul, 18446744073709551615ul, 7ul }, ((Tensor<ulong>)t64.Outputs[0]).ToArray());
    }

    [Fact]
    public void Tile_Bool_MatchesOrt()
    {
        // ORT 1.29: repeats [2, 1] over [[T, F]].
        var r = CPU.Tile(DenseTensor<bool>.OfValues(new bool[,] { { true, false } }), DenseTensor<long>.OfValues(new long[] { 2L, 1L }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<bool>)r.Outputs[0];
        Assert.Equal(new int[] { 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new bool[] { true, false, true, false }, y.ToArray());
    }

    [Fact]
    public void Tile_Sub32_MatchesOrt()
    {
        // ORT 1.29: repeats [2, 1] over [[v0, v1]] across int8/uint8/int16/uint16.
        var reps = DenseTensor<long>.OfValues(new long[] { 2L, 1L });
        var t8 = CPU.Tile(DenseTensor<sbyte>.OfValues(new sbyte[,] { { 1, -2 } }), reps, null);
        Assert.Equal(OpStatus.Success, t8.Status);
        Assert.Equal(new sbyte[] { 1, -2, 1, -2 }, ((Tensor<sbyte>)t8.Outputs[0]).ToArray());
        var tu8 = CPU.Tile(DenseTensor<byte>.OfValues(new byte[,] { { 1, 200 } }), reps, null);
        Assert.Equal(OpStatus.Success, tu8.Status);
        Assert.Equal(new byte[] { 1, 200, 1, 200 }, ((Tensor<byte>)tu8.Outputs[0]).ToArray());
        var t16 = CPU.Tile(DenseTensor<short>.OfValues(new short[,] { { 1, -2000 } }), reps, null);
        Assert.Equal(OpStatus.Success, t16.Status);
        Assert.Equal(new short[] { 1, -2000, 1, -2000 }, ((Tensor<short>)t16.Outputs[0]).ToArray());
        var tu16 = CPU.Tile(DenseTensor<ushort>.OfValues(new ushort[,] { { 1, 60000 } }), reps, null);
        Assert.Equal(OpStatus.Success, tu16.Status);
        Assert.Equal(new ushort[] { 1, 60000, 1, 60000 }, ((Tensor<ushort>)tu16.Outputs[0]).ToArray());
    }

    [Fact]
    public void Tile_Half_MatchesOrt()
    {
        // ORT 1.29 float16: repeats [2, 1] over [[1, 2]].
        var r = CPU.Tile(DenseTensor<Half>.OfValues(new Half[] { (Half)1f, (Half)2f }, new int[] { 1, 2 }), DenseTensor<long>.OfValues(new long[] { 2L, 1L }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<Half>)r.Outputs[0];
        Assert.Equal(new int[] { 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new Half[] { (Half)1f, (Half)2f, (Half)1f, (Half)2f }, y.ToArray());
    }

    [Fact]
    public void FloatRepeats_RejectedCleanly()
    {
        // ORT refuses non-int repeats at load; the provider threw a bare
        // ArgumentException from ToIntArray instead of a Failure.
        var x = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        var f = DenseTensor<float>.OfValues(new float[] { 2f });
        Assert.Equal(OpStatus.Failure, CPU.Tile(x, f, null).Status);
    }
    [Fact]
    public void HugeRepeats_FailsCleanly()
    {
        // ORT 1.29 fails the run (the output would exceed 4GB); unchecked
        // narrowing turned 2^32+1 repeats into 1 (silently succeeding).
        Assert.Throws<System.ArgumentException>(() => CPU.Tile(
            DenseTensor<float>.OfValues(new float[] { 7f }),
            DenseTensor<long>.OfValues(new long[] { 4294967297L }), null));
    }

    [Fact]
    public void HugeRepeatsOnEmptyDim_YieldsEmpty()
    {
        // Zero-size dimensions still tile to empty: no elements are needed.
        var r = CPU.Tile(
            DenseTensor<float>.OfShape(0),
            DenseTensor<long>.OfValues(new long[] { 1099511627776L }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new int[] { 0 }, ((Tensor<float>)r.Outputs![0]).Dimensions.ToArray());
    }

}
