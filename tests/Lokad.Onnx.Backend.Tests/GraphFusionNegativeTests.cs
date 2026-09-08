using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class GraphFusionNegativeTests
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

    static OnnxModel ValidModel()
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

    static OnnxModel SwapInputs(OnnxModel mp, string op, int a, int b)
    {
        var n = mp.Nodes.First(n => n.OpType == op);
        var tmp = n.Inputs[a]; n.Inputs[a] = n.Inputs[b]; n.Inputs[b] = tmp;
        return mp;
    }

    static void AssertNoFusion(OnnxModel mp)
    {
        var graph = Model.Load(mp)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.LayerNormalization);
    }

    [Fact]
    public void ValidPattern_Fuses()
    {
        var graph = Model.Load(ValidModel())!;
        Assert.Single(graph.Nodes);
        Assert.Equal(OpType.LayerNormalization, graph.Nodes[0].Op);
    }

    [Fact]
    public void ReversedSub_DoesNotFuse()
    {
        var mp = ValidModel();
        SwapInputs(mp, "Sub", 0, 1);
        AssertNoFusion(mp);
    }

    [Fact]
    public void ReversedDiv_DoesNotFuse()
    {
        var mp = ValidModel();
        SwapInputs(mp, "Div", 0, 1);
        AssertNoFusion(mp);
    }

    [Fact]
    public void ReversedPow_DoesNotFuse()
    {
        var mp = ValidModel();
        SwapInputs(mp, "Pow", 0, 1);
        AssertNoFusion(mp);
    }

    [Fact]
    public void MismatchedReduceMeanSource_DoesNotFuse()
    {
        var mp = ValidModel();
        mp.Inputs.Add(NamedIO("y", 2, 4));
        mp.Nodes.First(n => n.OpType == "ReduceMean" && n.Outputs[0] == "m").Inputs[0] = "y";
        AssertNoFusion(mp);
    }

    [Fact]
    public void ExposedCenteredIntermediate_BlocksFusionAndStaysCorrect()
    {
        var mp = ValidModel();
        mp.Outputs.Add(NamedIO("d", 2, 4));
        var graph = Model.Load(mp)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.LayerNormalization);
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } });
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { { "x", x } }, true));
        var d = ((Tensor<float>)graph.Outputs["d"]).ToArray();
        Assert.Equal(8, d.Length); Assert.Equal(-1.5f, d[0], 4); Assert.Equal(1.5f, d[3], 4);
    }

    [Fact]
    public void ExposedEpsilonConstant_SurvivesWithCorrectValue()
    {
        var mp = ValidModel();
        mp.Outputs.Add(new OnnxValueInfo { Name = "e", ElementType = TensorElementType.Float, Dims = new int[0] });
        var graph = Model.Load(mp)!;
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } });
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { { "x", x } }, true));
        var e = ((Tensor<float>)graph.Outputs["e"]).ToArray();
        Assert.Single(e);
        Assert.Equal(1e-5f, e[0], 7);
    }

    [Fact]
    public void FusedMatchesUnfusedNumerics()
    {
        var fused = Model.Load(ValidModel())!;
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } });
        Assert.True(fused.Execute(new Dictionary<string, ITensor> { { "x", x } }, true));
        var actual = ((Tensor<float>)fused.Outputs["z"]).ToArray();
        double eps = 1e-5;
        float[] gamma = new[] { 1f, 2f, 3f, 4f };
        float[] beta = new[] { 0.5f, -0.5f, 1f, 0f };
        float[,] input = new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } };
        for (int i = 0; i < 2; i++)
        {
            double mean = 0;
            for (int j = 0; j < 4; j++) mean += input[i, j];
            mean /= 4;
            double variance = 0;
            for (int j = 0; j < 4; j++) variance += (input[i, j] - mean) * (input[i, j] - mean);
            variance /= 4;
            double denom = System.Math.Sqrt(variance + eps);
            for (int j = 0; j < 4; j++)
            {
                float expected = (float)((input[i, j] - mean) / denom * gamma[j] + beta[j]);
                Assert.Equal(expected, actual[i * 4 + j], 5);
            }
        }
    }

    [Fact]
    public void AllGraphOutputsKeepProducers()
    {
        var mp = ValidModel();
        mp.Outputs.Add(NamedIO("d", 2, 4));
        var graph = Model.Load(mp)!;
        var produced = new HashSet<string>(graph.Nodes.SelectMany(n => n.Outputs).Where(s => !string.IsNullOrEmpty(s)));
        foreach (var o in graph.Outputs.Keys)
        {
            Assert.True(graph.Inputs.ContainsKey(o) || graph.Initializers.ContainsKey(o) || produced.Contains(o), "dangling output " + o);
        }
    }
}


