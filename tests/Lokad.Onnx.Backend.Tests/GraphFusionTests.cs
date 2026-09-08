using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

public class GraphFusionTests
{
    static OnnxValueInfo NamedIO(string name, int d0, int d1) =>
        new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = new[] { d0, d1 } };

    static OnnxTensor FloatInit(string name, int[] dims, float[] values) =>
        new OnnxTensor { Name = name, ElementType = TensorElementType.Float, Dims = dims, Data = values };

    static Dictionary<string, object> Attrs(params (string Name, object Value)[] attrs)
    {
        var d = new Dictionary<string, object>();
        foreach (var a in attrs) d[a.Name] = a.Value;
        return d;
    }

    static OnnxNode Nod(string op, string[] inputs, string[] outputs, Dictionary<string, object>? attrs) =>
        new OnnxNode { OpType = op, Inputs = inputs, Outputs = outputs, Attributes = attrs ?? new Dictionary<string, object>() };

    static OnnxModel TinyLayerNormModel()
    {
        var mp = new OnnxModel { Name = "tiny-ln" };
        mp.Opset[""] = 11;
        mp.Inputs.Add(NamedIO("x", 2, 4));
        mp.Outputs.Add(NamedIO("z", 2, 4));
        mp.Initializers.Add(FloatInit("gamma", new[] { 4 }, new[] { 1f, 2f, 3f, 4f }));
        mp.Initializers.Add(FloatInit("beta", new[] { 4 }, new[] { 0.5f, -0.5f, 1f, 0f }));
        mp.Initializers.Add(FloatInit("two", new int[0], new[] { 2f }));
        var eps = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new int[0], Data = new[] { 1e-5f } });
        mp.Nodes.Add(Nod("ReduceMean", new[] { "x" }, new[] { "m" }, Attrs(("axes", new long[] { -1 }), ("keepdims", 1L))));
        mp.Nodes.Add(Nod("Sub", new[] { "x", "m" }, new[] { "d" }, null));
        mp.Nodes.Add(Nod("Pow", new[] { "d", "two" }, new[] { "s" }, null));
        mp.Nodes.Add(Nod("ReduceMean", new[] { "s" }, new[] { "v" }, Attrs(("axes", new long[] { -1 }), ("keepdims", 1L))));
        mp.Nodes.Add(Nod("Constant", new string[0], new[] { "e" }, Attrs(("value", eps))));
        mp.Nodes.Add(Nod("Add", new[] { "v", "e" }, new[] { "ve" }, null));
        mp.Nodes.Add(Nod("Sqrt", new[] { "ve" }, new[] { "sd" }, null));
        mp.Nodes.Add(Nod("Div", new[] { "d", "sd" }, new[] { "n" }, null));
        mp.Nodes.Add(Nod("Mul", new[] { "n", "gamma" }, new[] { "g" }, null));
        mp.Nodes.Add(Nod("Add", new[] { "g", "beta" }, new[] { "z" }, null));
        return mp;
    }

    static float[] IndependentLayerNorm(float[,] x, float[] gamma, float[] beta, double eps)
    {
        int rows = x.GetLength(0), cols = x.GetLength(1);
        var y = new float[rows * cols];
        for (int i = 0; i < rows; i++)
        {
            double mean = 0;
            for (int j = 0; j < cols; j++) mean += x[i, j];
            mean /= cols;
            double variance = 0;
            for (int j = 0; j < cols; j++) variance += (x[i, j] - mean) * (x[i, j] - mean);
            variance /= cols;
            double denom = System.Math.Sqrt(variance + eps);
            for (int j = 0; j < cols; j++) y[i * cols + j] = (float)((x[i, j] - mean) / denom * gamma[j] + beta[j]);
        }
        return y;
    }

    [Fact]
    public void DecomposedLayerNorm_FusesToNativeNode()
    {
        var graph = Model.Load(TinyLayerNormModel())!;
        Assert.Single(graph.Nodes);
        Assert.Equal(OpType.LayerNormalization, graph.Nodes[0].Op);
        Assert.Equal(new[] { "z" }, graph.Nodes[0].Outputs);

        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } });
        var inputs = new Dictionary<string, ITensor> { { "x", x } };
        Assert.True(graph.Execute(inputs, true));
        var actual = ((Tensor<float>)graph.Outputs["z"]).ToArray();
        var expected = IndependentLayerNorm(
            new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } },
            new float[] { 1f, 2f, 3f, 4f }, new float[] { 0.5f, -0.5f, 1f, 0f }, 1e-5);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 5);
    }

    [Fact]
    public void PlainDiv_SurvivesFusion()
    {
        var mp = new OnnxModel { Name = "plain-div" };
        mp.Opset[""] = 11;
        mp.Inputs.Add(NamedIO("x", 1, 2));
        mp.Outputs.Add(NamedIO("z", 1, 2));
        mp.Initializers.Add(FloatInit("two", new[] { 2 }, new[] { 2f, 2f }));
        mp.Nodes.Add(Nod("Div", new[] { "x", "two" }, new[] { "z" }, null));
        var graph = Model.Load(mp)!;
        Assert.Single(graph.Nodes);
        Assert.Equal(OpType.Div, graph.Nodes[0].Op);
    }
}
