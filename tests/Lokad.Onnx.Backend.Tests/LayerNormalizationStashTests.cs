using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

// R2: stash_type selects stage-one compute precision (1 means 32-bit float;
// it never selects outputs) and nodes declare one to three positional
// outputs: Y, then Mean, then InvStdDev, with stats always in float32.
public class LayerNormalizationStashTests
{
    static ComputationalGraph Graph(int opset)
    {
        return new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = opset },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
    }

    static Node LnNode(string[] inputs, string[] outputs, Dictionary<string, object>? attrs)
    {
        return new Node
        {
            Name = "ln", Op = OpType.LayerNormalization, Inputs = inputs, Outputs = outputs,
            Attributes = attrs ?? new Dictionary<string, object>(),
        };
    }

    static (float[] Mean, float[] Inv, int Outer) IndependentStats(float[,] x, double eps)
    {
        int rows = x.GetLength(0), cols = x.GetLength(1);
        var mean = new float[rows];
        var inv = new float[rows];
        for (int i = 0; i < rows; i++)
        {
            double m = 0;
            for (int j = 0; j < cols; j++) m += x[i, j];
            m /= cols;
            double v = 0;
            for (int j = 0; j < cols; j++) { double d = x[i, j] - m; v += d * d; }
            v /= cols;
            mean[i] = (float)m;
            inv[i] = (float)(1.0 / System.Math.Sqrt(v + eps));
        }
        return (mean, inv, rows);
    }

    [Fact]
    public void OmittedStash_MatchesExplicitDefault_Float()
    {
        var x = new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } };
        float[] Run(Dictionary<string, object>? attrs)
        {
            var graph = Graph(17);
            graph.Inputs["x"] = DenseTensor<float>.OfValues(x);
            graph.Inputs["scale"] = DenseTensor<float>.OfValues(new float[] { 1f, 2f, 1f, 0.5f });
            graph.Inputs["bias"] = DenseTensor<float>.OfValues(new float[] { 0.5f, -0.5f, 1f, 0f });
            var r = LnNode(new[] { "x", "scale", "bias" }, new[] { "y" }, attrs).Execute(graph, ExecutionProvider.CPU, null);
            Assert.Equal(OpStatus.Success, r.Status);
            Assert.Single(r.Outputs);
            return ((Tensor<float>)r.Outputs[0]).ToArray();
        }
        var omitted = Run(new Dictionary<string, object> { ["axis"] = -1L });
        var explicitDefault = Run(new Dictionary<string, object> { ["axis"] = -1L, ["stash_type"] = 1L });
        Assert.Equal(omitted, explicitDefault);
        var (mean, inv, _) = IndependentStats(x, 1e-5);
        float[] scale = { 1f, 2f, 1f, 0.5f }, bias = { 0.5f, -0.5f, 1f, 0f };
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++)
                Assert.Equal((x[i, j] - mean[i]) * inv[i] * scale[j] + bias[j], explicitDefault[i * 4 + j], 5);
    }

    [Fact]
    public void OmittedStash_MatchesExplicitDefault_Double()
    {
        var graph = Graph(17);
        graph.Inputs["x"] = DenseTensor<double>.OfValues(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        graph.Inputs["scale"] = DenseTensor<double>.OfValues(new double[] { 1.0, 1.0 });
        float[] Run(Dictionary<string, object>? attrs)
        {
            var r = LnNode(new[] { "x", "scale" }, new[] { "y" }, attrs).Execute(graph, ExecutionProvider.CPU, null);
            Assert.Equal(OpStatus.Success, r.Status);
            return ((Tensor<double>)r.Outputs[0]).ToArray().Select(v => (float)v).ToArray();
        }
        Assert.Equal(
            Run(new Dictionary<string, object> { ["axis"] = -1L }),
            Run(new Dictionary<string, object> { ["axis"] = -1L, ["stash_type"] = 1L }));
    }

    [Fact]
    public void ThreeOutputs_ReturnStandardFloatStats()
    {
        var x = new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } };
        var graph = Graph(17);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(x);
        graph.Inputs["scale"] = DenseTensor<float>.OfValues(new float[] { 1f, 1f, 1f, 1f });
        graph.Inputs["bias"] = DenseTensor<float>.OfValues(new float[] { 0f, 0f, 0f, 0f });
        var r = LnNode(new[] { "x", "scale", "bias" }, new[] { "y", "m", "v" },
            new Dictionary<string, object> { ["axis"] = -1L, ["stash_type"] = 1L }).Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(3, r.Outputs.Length);
        var y = (Tensor<float>)r.Outputs[0];
        var m = (Tensor<float>)r.Outputs[1];
        var v = (Tensor<float>)r.Outputs[2];
        Assert.Equal(new[] { 2, 4 }, y.Dimensions.ToArray());
        Assert.Equal(new[] { 2, 1 }, m.Dimensions.ToArray());
        Assert.Equal(new[] { 2, 1 }, v.Dimensions.ToArray());
        var (mean, inv, _) = IndependentStats(x, 1e-5);
        Assert.Equal(mean, m.ToArray());
        Assert.Equal(inv, v.ToArray());
        var ya = y.ToArray();
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++)
                Assert.Equal((x[i, j] - mean[i]) * inv[i], ya[i * 4 + j], 5);
    }

    [Fact]
    public void TwoOutputs_ReturnMeanOnly()
    {
        var graph = Graph(17);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 3f }, { 5f, 7f } });
        graph.Inputs["scale"] = DenseTensor<float>.OfValues(new float[] { 1f, 1f });
        var r = LnNode(new[] { "x", "scale" }, new[] { "y", "m" },
            new Dictionary<string, object> { ["axis"] = -1L }).Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(2, r.Outputs.Length);
        Assert.Equal(new[] { 2f, 6f }, ((Tensor<float>)r.Outputs[1]).ToArray());
    }

    [Fact]
    public void DoubleThreeOutputs_StatsStayFloat()
    {
        var graph = Graph(17);
        graph.Inputs["x"] = DenseTensor<double>.OfValues(new double[,] { { 1.0, 2.0, 3.0, 4.0 } });
        graph.Inputs["scale"] = DenseTensor<double>.OfValues(new double[] { 1.0, 1.0, 1.0, 1.0 });
        var r = LnNode(new[] { "x", "scale" }, new[] { "y", "m", "v" },
            new Dictionary<string, object> { ["axis"] = -1L, ["stash_type"] = 1L }).Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(TensorElementType.Double, r.Outputs[0].ElementType);
        Assert.Equal(TensorElementType.Float, r.Outputs[1].ElementType);
        Assert.Equal(TensorElementType.Float, r.Outputs[2].ElementType);
        Assert.Equal(new[] { 1, 1 }, ((Tensor<float>)r.Outputs[1]).Dimensions.ToArray());
        Assert.Equal(2.5f, ((Tensor<float>)r.Outputs[1]).ToArray()[0], 5);
    }

    [Theory]
    [InlineData(0L)]
    [InlineData(2L)]
    [InlineData(16L)]
    public void UnsupportedStash_FailsExplicitly(long stash)
    {
        var graph = Graph(17);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f } });
        graph.Inputs["scale"] = DenseTensor<float>.OfValues(new float[] { 1f, 1f, 1f, 1f });
        var r = LnNode(new[] { "x", "scale" }, new[] { "y" },
            new Dictionary<string, object> { ["axis"] = -1L, ["stash_type"] = stash }).Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("stash_type", r.Message ?? "");
    }

    [Fact]
    public void WrongTypedStash_FailsExplicitly()
    {
        var graph = Graph(17);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f } });
        graph.Inputs["scale"] = DenseTensor<float>.OfValues(new float[] { 1f, 1f, 1f, 1f });
        var r = LnNode(new[] { "x", "scale" }, new[] { "y" },
            new Dictionary<string, object> { ["axis"] = -1L, ["stash_type"] = "1" }).Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(4)]
    public void OutputCountOutsideOneToThree_Fails(int count)
    {
        var graph = Graph(17);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f } });
        graph.Inputs["scale"] = DenseTensor<float>.OfValues(new float[] { 1f, 1f, 1f, 1f });
        var outputs = Enumerable.Range(0, count).Select(i => "y" + i).ToArray();
        var r = LnNode(new[] { "x", "scale" }, outputs,
            new Dictionary<string, object> { ["axis"] = -1L }).Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void EmptyNormalizedAxis_FailsCleanly()
    {
        var x = DenseTensor<float>.OfShape(2, 0);
        var scale = DenseTensor<float>.OfShape(0);
        var ex = Assert.Throws<ArgumentException>(() => Tensor<float>.LayerNormalization(x, scale, null, -1, 1e-5f));
        Assert.Contains("positive extent", ex.Message);
        var xd = DenseTensor<double>.OfShape(2, 0);
        var sd = DenseTensor<double>.OfShape(0);
        ex = Assert.Throws<ArgumentException>(() => Tensor<double>.LayerNormalization(xd, sd, null, 0, 1e-5));
        Assert.Contains("positive extent", ex.Message);
        var graph = Graph(18);
        graph.Inputs["x"] = DenseTensor<float>.OfShape(2, 0);
        graph.Inputs["scale"] = DenseTensor<float>.OfShape(0);
        var r = LnNode(new[] { "x", "scale" }, new[] { "y" },
            new Dictionary<string, object> { ["axis"] = -1L }).Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("positive extent", r.Message ?? "");
    }

    [Fact]
    public void ScalarRank_FailsCleanly()
    {
        var x = new DenseTensor<float>(new float[] { 1f }, Array.Empty<int>());
        var scale = new DenseTensor<float>(new float[] { 1f }, Array.Empty<int>());
        Assert.Throws<ArgumentException>(() => Tensor<float>.LayerNormalization(x, scale, null, 0, 1e-5f));
    }

    [Fact]
    public void EmptyOuterAxis_SucceedsEmpty()
    {
        var x = DenseTensor<float>.OfShape(0, 4);
        var scale = DenseTensor<float>.OfValues(new float[] { 1f, 1f, 1f, 1f });
        var y = Tensor<float>.LayerNormalization(x, scale, null, 1, 1e-5f);
        Assert.Equal(new int[] { 0, 4 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
    }

    [Fact]
    public void StridedInput_MatchesDense()
    {
        var full = DenseTensor<float>.OfValues(new float[2, 4]
        {
            { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f },
        });
        Tensor<float> slice = full[.., 1..3];
        var scale = DenseTensor<float>.OfValues(new float[] { 1f, 1f });
        var y = Tensor<float>.LayerNormalization(slice, scale, null, -1, 1e-5f);
        Assert.Equal(new int[] { 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { -1f, 1f, -1f, 1f }, y.ToArray().Select(v => (float)System.Math.Round(v, 4)).ToArray());
    }
}
