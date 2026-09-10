using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

public class GemmTransposeBiasTests
{
    static DenseTensor<float> F(float[,] v) => DenseTensor<float>.OfValues(v);

    [Fact]
    public void TransposeCombinations_Nonsquare_AreCorrect()
    {
        var a = F(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        var b = F(new float[,] { { 7f, 8f }, { 9f, 10f }, { 11f, 12f } });
        var r00 = CPUExecutionProvider.Gemm(a, b, null, 1f, 0f, null, 0, 0);
        Assert.Equal(OpStatus.Success, r00.Status);
        var y00 = (Tensor<float>)r00.Outputs[0];
        Assert.Equal(new[] { 2, 2 }, y00.Dimensions.ToArray());
        Assert.Equal(58f, y00[0, 0], 4); Assert.Equal(64f, y00[0, 1], 4);
        Assert.Equal(139f, y00[1, 0], 4); Assert.Equal(154f, y00[1, 1], 4);

        var at = F(new float[,] { { 1f, 4f }, { 2f, 5f }, { 3f, 6f } });
        var r10 = CPUExecutionProvider.Gemm(at, b, null, 1f, 0f, null, 1, 0);
        Assert.Equal(OpStatus.Success, r10.Status);
        var y10 = (Tensor<float>)r10.Outputs[0];
        Assert.Equal(new[] { 2, 2 }, y10.Dimensions.ToArray());
        Assert.Equal(58f, y10[0, 0], 4); Assert.Equal(154f, y10[1, 1], 4);

        var bt = F(new float[,] { { 7f, 9f, 11f }, { 8f, 10f, 12f } });
        var r01 = CPUExecutionProvider.Gemm(a, bt, null, 1f, 0f, null, 0, 1);
        Assert.Equal(OpStatus.Success, r01.Status);
        Assert.Equal(58f, ((Tensor<float>)r01.Outputs[0])[0, 0], 4);

        var r11 = CPUExecutionProvider.Gemm(at, bt, null, 1f, 0f, null, 1, 1);
        Assert.Equal(OpStatus.Success, r11.Status);
        Assert.Equal(64f, ((Tensor<float>)r11.Outputs[0])[0, 1], 4);
    }

    [Fact]
    public void VectorInputs_Rejected()
    {
        // ORT 1.29 refuses 1-D Gemm inputs at load (rank 2 required);
        // Lokad returns a descriptive Failure instead of promoting.
        var a = F(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var b = F(new float[,] { { 5f, 6f }, { 7f, 8f } });
        var vec = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        var ra = CPUExecutionProvider.Gemm(vec, b, null, 1f, 0f, null, 0, 0);
        Assert.Equal(OpStatus.Failure, ra.Status);
        var rb = CPUExecutionProvider.Gemm(a, vec, null, 1f, 0f, null, 0, 0);
        Assert.Equal(OpStatus.Failure, rb.Status);
    }

    [Fact]
    public void SquareTransposed_IsNotSilentUntransposed()
    {
        var a = F(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var b = F(new float[,] { { 5f, 6f }, { 7f, 8f } });
        var r = CPUExecutionProvider.Gemm(a, b, null, 1f, 0f, null, 1, 0);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs[0];
        Assert.Equal(26f, y[0, 0], 4); Assert.Equal(30f, y[0, 1], 4);
        Assert.Equal(38f, y[1, 0], 4); Assert.Equal(44f, y[1, 1], 4);
    }

    [Fact]
    public void BiasForms_BroadcastCorrectly()
    {
        var a = F(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var b = F(new float[,] { { 5f, 6f }, { 7f, 8f } });
        var row = DenseTensor<float>.OfValues(new float[,] { { 100f, 200f } });
        var rr = CPUExecutionProvider.Gemm(a, b, row, 1f, 1f, null, 0, 0);
        Assert.Equal(OpStatus.Success, rr.Status);
        Assert.Equal(119f, ((Tensor<float>)rr.Outputs[0])[0, 0], 4);
        var col = DenseTensor<float>.OfValues(new float[,] { { 100f }, { 200f } });
        var rc = CPUExecutionProvider.Gemm(a, b, col, 1f, 1f, null, 0, 0);
        Assert.Equal(OpStatus.Success, rc.Status);
        Assert.Equal(119f, ((Tensor<float>)rc.Outputs[0])[0, 0], 4);
        Assert.Equal(243f, ((Tensor<float>)rc.Outputs[0])[1, 0], 4);
        var bad = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Gemm(a, b, bad, 1f, 1f, null, 0, 0).Status);
        var rank3 = DenseTensor<float>.OfShape(2, 2, 1); rank3.Fill(1f);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Gemm(a, b, rank3, 1f, 1f, null, 0, 0).Status);
    }

    [Fact]
    public void LongFlags_AsImported_AreHonored()
    {
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 13 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["a"] = F(new float[,] { { 1f, 4f }, { 2f, 5f }, { 3f, 6f } });
        graph.Inputs["b"] = F(new float[,] { { 7f, 8f }, { 9f, 10f }, { 11f, 12f } });
        var node = new Node
        {
            Name = "gemm", Op = OpType.Gemm, Inputs = new[] { "a", "b" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["transA"] = 1L, ["transB"] = 0L, ["alpha"] = 1f, ["beta"] = 0f },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(58f, ((Tensor<float>)r.Outputs[0])[0, 0], 4);
    }

    [Fact]
    public void AlphaBetaEdges_BehaveExplicitly()
    {
        var a = F(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var b = F(new float[,] { { 5f, 6f }, { 7f, 8f } });
        var bias = DenseTensor<float>.OfValues(new float[] { 100f, 200f });
        var r0 = CPUExecutionProvider.Gemm(a, b, bias, 0f, 1f, null, 0, 0);
        Assert.Equal(OpStatus.Success, r0.Status);
        Assert.Equal(100f, ((Tensor<float>)r0.Outputs[0])[0, 0], 4);
        var r1 = CPUExecutionProvider.Gemm(a, b, bias, 1f, 0f, null, 0, 0);
        Assert.Equal(OpStatus.Success, r1.Status);
        Assert.Equal(19f, ((Tensor<float>)r1.Outputs[0])[0, 0], 4);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Gemm(a, b, null, 1f, 1f, null, 2, 0).Status);
    }

    [Fact]
    public void ExceptionalAlphaBeta_MatchOrt()
    {
        // ORT 1.29 on A=[[0,0],[1,2]], B=I, C=ones: inf alpha/beta and NaN
        // inputs propagate through the scale/bias epilogue as below.
        var a = F(new float[,] { { 0f, 0f }, { 1f, 2f } });
        var b = F(new float[,] { { 1f, 0f }, { 0f, 1f } });
        var c = F(new float[,] { { 1f, 1f }, { 1f, 1f } });
        var ai = CPUExecutionProvider.Gemm(a, b, c, float.PositiveInfinity, 1f, null, 0, 0);
        Assert.Equal(OpStatus.Success, ai.Status);
        var ay = ((Tensor<float>)ai.Outputs[0]).ToArray();
        Assert.True(float.IsNaN(ay[0]));
        Assert.True(float.IsNaN(ay[1]));
        Assert.Equal(float.PositiveInfinity, ay[2]);
        Assert.Equal(float.PositiveInfinity, ay[3]);
        var bi = CPUExecutionProvider.Gemm(a, b, c, 1f, float.PositiveInfinity, null, 0, 0);
        Assert.Equal(OpStatus.Success, bi.Status);
        foreach (var v in ((Tensor<float>)bi.Outputs[0]).ToArray()) Assert.Equal(float.PositiveInfinity, v);
        var ni = CPUExecutionProvider.Gemm(DenseTensor<float>.OfValues(new float[,] { { float.NaN, 1f }, { 1f, 1f } }), b, c, 1f, 1f, null, 0, 0);
        Assert.Equal(OpStatus.Success, ni.Status);
        var ny = ((Tensor<float>)ni.Outputs[0]).ToArray();
        Assert.True(float.IsNaN(ny[0]));
        Assert.True(float.IsNaN(ny[1]));
        Assert.Equal(2f, ny[2]);
        Assert.Equal(2f, ny[3]);
    }

    [Fact]
    public void RowColumnBias_MatchesOrt()
    {
        // ORT 1.29 on A=[[1,2],[3,4]], B=[[5,6],[7,8]]: 2-D [1,2] row and
        // [2,1] column biases take the matrix-broadcast epilogue branch.
        var a = F(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var b = F(new float[,] { { 5f, 6f }, { 7f, 8f } });
        var row = CPUExecutionProvider.Gemm(a, b, F(new float[,] { { 10f, 20f } }), 1f, 1f, null, 0, 0);
        Assert.Equal(OpStatus.Success, row.Status);
        Assert.Equal(new float[] { 29f, 42f, 53f, 70f }, ((Tensor<float>)row.Outputs[0]).ToArray());
        var col = CPUExecutionProvider.Gemm(a, b, F(new float[,] { { 10f }, { 20f } }), 1f, 1f, null, 0, 0);
        Assert.Equal(OpStatus.Success, col.Status);
        Assert.Equal(new float[] { 29f, 32f, 63f, 70f }, ((Tensor<float>)col.Outputs[0]).ToArray());
    }

    [Fact]
    public void MixedDtypes_RejectedCleanly()
    {
        // ORT refuses mixed-dtype Gemm at load (MatMul A/B already pins
        // its guard); Gemm A/B and C must fail descriptively instead.
        var a = F(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var bInt = DenseTensor<int>.OfValues(new int[,] { { 1, 0 }, { 0, 1 } });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Gemm(a, bInt, null, 1f, 0f, null, 0, 0).Status);
        var b = F(new float[,] { { 5f, 6f }, { 7f, 8f } });
        var cInt = DenseTensor<int>.OfValues(new int[] { 1, 2 });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Gemm(a, b, cInt, 1f, 1f, null, 0, 0).Status);
    }
}
