extern alias OnnxSharp;

using System.Collections.Generic;
using System.Linq;
using OnnxSharp::Onnx;

namespace Lokad.Onnx.Backend.Tests;

public class GraphFusionTests
{
    static ValueInfoProto NamedIO(string name, int d0, int d1)
    {
        var shape = new TensorShapeProto();
        shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = d0 });
        shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = d1 });
        return new ValueInfoProto { Name = name, Type = new TypeProto { TensorType = new TypeProto.Types.Tensor { ElemType = 1, Shape = shape } } };
    }

    static TensorProto FloatInit(string name, int[] dims, float[] values)
    {
        var tp = new TensorProto { Name = name, DataType = (int)TensorElementType.Float };
        foreach (var d in dims) tp.Dims.Add(d);
        foreach (var v in values) tp.FloatData.Add(v);
        return tp;
    }

    static AttributeProto IntsAttr(string name, params long[] values)
    {
        var ap = new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.Ints };
        foreach (var v in values) ap.Ints.Add(v);
        return ap;
    }

    static NodeProto Nod(string op, string[] inputs, string[] outputs, params AttributeProto[] attrs)
    {
        var n = new NodeProto { OpType = op };
        foreach (var i in inputs) n.Input.Add(i);
        foreach (var o in outputs) n.Output.Add(o);
        foreach (var a in attrs) n.Attribute.Add(a);
        return n;
    }

    static ModelProto TinyLayerNormModel()
    {
        var mp = new ModelProto();
        mp.OpsetImport.Add(new OperatorSetIdProto { Domain = "", Version = 11 });
        mp.Graph = new GraphProto { Name = "tiny-ln" };
        mp.Graph.Input.Add(NamedIO("x", 2, 4));
        mp.Graph.Output.Add(NamedIO("z", 2, 4));
        mp.Graph.Initializer.Add(FloatInit("gamma", new[] { 4 }, new[] { 1f, 2f, 3f, 4f }));
        mp.Graph.Initializer.Add(FloatInit("beta", new[] { 4 }, new[] { 0.5f, -0.5f, 1f, 0f }));
        mp.Graph.Initializer.Add(FloatInit("two", new int[0], new[] { 2f }));
        var eps = new TensorProto { DataType = (int)TensorElementType.Float };
        eps.FloatData.Add(1e-5f);
        var kd = new AttributeProto { Name = "keepdims", Type = AttributeProto.Types.AttributeType.Int, I = 1 };
        mp.Graph.Node.Add(Nod("ReduceMean", new[] { "x" }, new[] { "m" }, IntsAttr("axes", -1), kd));
        mp.Graph.Node.Add(Nod("Sub", new[] { "x", "m" }, new[] { "d" }));
        mp.Graph.Node.Add(Nod("Pow", new[] { "d", "two" }, new[] { "s" }));
        mp.Graph.Node.Add(Nod("ReduceMean", new[] { "s" }, new[] { "v" }, IntsAttr("axes", -1), kd));
        mp.Graph.Node.Add(Nod("Add", new[] { "v", "e" }, new[] { "ve" }));
        mp.Graph.Node.Add(Nod("Constant", new string[0], new[] { "e" },
            new AttributeProto { Name = "value", Type = AttributeProto.Types.AttributeType.Tensor, T = eps }));
        mp.Graph.Node.Add(Nod("Sqrt", new[] { "ve" }, new[] { "sd" }));
        mp.Graph.Node.Add(Nod("Div", new[] { "d", "sd" }, new[] { "n" }));
        mp.Graph.Node.Add(Nod("Mul", new[] { "n", "gamma" }, new[] { "g" }));
        mp.Graph.Node.Add(Nod("Add", new[] { "g", "beta" }, new[] { "z" }));
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
        var mp = new ModelProto();
        mp.OpsetImport.Add(new OperatorSetIdProto { Domain = "", Version = 11 });
        mp.Graph = new GraphProto { Name = "plain-div" };
        mp.Graph.Input.Add(NamedIO("x", 1, 2));
        mp.Graph.Output.Add(NamedIO("z", 1, 2));
        mp.Graph.Initializer.Add(FloatInit("two", new[] { 2 }, new[] { 2f, 2f }));
        mp.Graph.Node.Add(Nod("Div", new[] { "x", "two" }, new[] { "z" }));
        var graph = Model.Load(mp)!;
        Assert.Single(graph.Nodes);
        Assert.Equal(OpType.Div, graph.Nodes[0].Op);
    }
}
