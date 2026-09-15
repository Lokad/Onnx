using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Covers arithmetic constant folding: elementwise, unary, and structural extension
/// ops fold constant-only chains to initializers with exact values, while data
/// inputs, graph outputs, and over-budget values keep their nodes.
/// </summary>
public class GraphConstantsArithmeticTests
{
    static OnnxValueInfo NamedIO(string name, int[] dims) =>
        new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = dims };

    static OnnxTensor FloatInit(string name, int[] dims, float[] values) =>
        new OnnxTensor { Name = name, ElementType = TensorElementType.Float, Dims = dims, Data = values };

    static OnnxNode Nod(string op, string[] inputs, string[] outputs) =>
        new OnnxNode { OpType = op, Inputs = inputs, Outputs = outputs, Attributes = new Dictionary<string, object>() };

    static float[] RunZ(OnnxModel mp, string outputName)
    {
        var graph = Model.Load(mp)!;
        Assert.True(graph.Execute(new Dictionary<string, ITensor>(), true));
        return ((Tensor<float>)graph.Outputs[outputName]).ToArray();
    }

    [Fact]
    public void MulAddChain_FoldsToInitializer()
    {
        var mp = new OnnxModel { Name = "mularith" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("x", new[] { 2 }));
        mp.Outputs.Add(NamedIO("z", new[] { 2 }));
        mp.Initializers.Add(FloatInit("A", new[] { 2 }, new[] { 1f, 2f }));
        mp.Initializers.Add(FloatInit("B", new[] { 2 }, new[] { 3f, 4f }));
        mp.Initializers.Add(FloatInit("C", new[] { 2 }, new[] { 10f, 20f }));
        mp.Nodes.Add(Nod("Mul", new[] { "A", "B" }, new[] { "m" }));
        mp.Nodes.Add(Nod("Add", new[] { "m", "C" }, new[] { "a" }));
        mp.Nodes.Add(Nod("Mul", new[] { "a", "x" }, new[] { "z" }));
        var graph = Model.Load(mp)!;
        Assert.Single(graph.Nodes);
        Assert.True(graph.Initializers.ContainsKey("a"));
        var x = new DenseTensor<float>(new[] { 2f, 0.5f }, new[] { 2 });
        var graph2 = Model.Load(mp)!;
        Assert.True(graph2.Execute(new Dictionary<string, ITensor> { { "x", x } }, true));
        Assert.Equal(new float[] { 26f, 14f }, ((Tensor<float>)graph2.Outputs["z"]).ToArray());
    }

    [Fact]
    public void SinAbsSubDiv_FoldExactly()
    {
        var mp = new OnnxModel { Name = "unaryarith" };
        mp.Opset[""] = 14;
        mp.Outputs.Add(NamedIO("z", new[] { 2 }));
        mp.Initializers.Add(FloatInit("A", new[] { 2 }, new[] { 0f, 1f }));
        mp.Initializers.Add(FloatInit("B", new[] { 2 }, new[] { -3f, 4f }));
        mp.Nodes.Add(Nod("Sin", new[] { "A" }, new[] { "s" }));
        mp.Nodes.Add(Nod("Abs", new[] { "B" }, new[] { "b" }));
        mp.Nodes.Add(Nod("Sub", new[] { "b", "s" }, new[] { "d" }));
        mp.Nodes.Add(Nod("Div", new[] { "d", "b" }, new[] { "a" }));
        mp.Nodes.Add(Nod("Identity", new[] { "a" }, new[] { "z" }));
        var graph = Model.Load(mp)!;
        Assert.Single(graph.Nodes);
        Assert.True(graph.Initializers.ContainsKey("a"));
        var actual = RunZ(mp, "z");
        var want = new float[] { 1f, (4f - MathF.Sin(1f)) / 4f };
        Assert.Equal(want.Length, actual.Length);
        for (int i = 0; i < want.Length; i++) Assert.True(Math.Abs(want[i] - actual[i]) < 1e-6f, "index=" + i);
    }

    [Fact]
    public void MatMulConstant_FoldsWithCorrectValues()
    {
        var mp = new OnnxModel { Name = "mmatfold" };
        mp.Opset[""] = 14;
        mp.Outputs.Add(NamedIO("z", new[] { 2, 2 }));
        mp.Initializers.Add(FloatInit("X", new[] { 2, 3 }, new[] { 1f, 2f, 3f, 4f, 5f, 6f }));
        mp.Initializers.Add(FloatInit("Y", new[] { 3, 2 }, new[] { 1f, 0f, 0f, 1f, 1f, 1f }));
        mp.Nodes.Add(Nod("MatMul", new[] { "X", "Y" }, new[] { "m" }));
        mp.Nodes.Add(Nod("Identity", new[] { "m" }, new[] { "z" }));
        var graph = Model.Load(mp)!;
        Assert.Single(graph.Nodes);
        Assert.True(graph.Initializers.ContainsKey("m"));
        Assert.Equal(new float[] { 4f, 5f, 10f, 11f }, RunZ(mp, "z"));
    }

    [Fact]
    public void DataDependentLeg_Declines()
    {
        var mp = new OnnxModel { Name = "datadep" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("x", new[] { 2 }));
        mp.Outputs.Add(NamedIO("z", new[] { 2 }));
        mp.Initializers.Add(FloatInit("A", new[] { 2 }, new[] { 1f, 2f }));
        mp.Nodes.Add(Nod("Mul", new[] { "A", "x" }, new[] { "z" }));
        var graph = Model.Load(mp)!;
        Assert.Single(graph.Nodes);
        Assert.False(graph.Initializers.ContainsKey("z"));
    }

    [Fact]
    public void GraphOutput_Declines()
    {
        var mp = new OnnxModel { Name = "graphout" };
        mp.Opset[""] = 14;
        mp.Outputs.Add(NamedIO("a", new[] { 2 }));
        mp.Outputs.Add(NamedIO("z", new[] { 2 }));
        mp.Initializers.Add(FloatInit("A", new[] { 2 }, new[] { 1f, 2f }));
        mp.Initializers.Add(FloatInit("B", new[] { 2 }, new[] { 3f, 4f }));
        mp.Nodes.Add(Nod("Add", new[] { "A", "B" }, new[] { "a" }));
        mp.Nodes.Add(Nod("Identity", new[] { "a" }, new[] { "z" }));
        var graph = Model.Load(mp)!;
        Assert.Equal(2, graph.Nodes.Count);
    }

    [Fact]
    public void OverBudgetConcat_Declines()
    {
        var mp = new OnnxModel { Name = "bigconcat" };
        mp.Opset[""] = 14;
        mp.Outputs.Add(NamedIO("z", new[] { 18000000 }));
        var left = new float[9000000];
        var right = new float[9000000];
        for (int i = 0; i < left.Length; i++) { left[i] = 1f; right[i] = 2f; }
        mp.Initializers.Add(FloatInit("L", new[] { 9000000 }, left));
        mp.Initializers.Add(FloatInit("R", new[] { 9000000 }, right));
        mp.Nodes.Add(Nod("Concat", new[] { "L", "R" }, new[] { "c" }));
        mp.Nodes.Add(Nod("Identity", new[] { "c" }, new[] { "z" }));
        var graph = Model.Load(mp)!;
        Assert.Equal(2, graph.Nodes.Count);
        Assert.False(graph.Initializers.ContainsKey("c"));
    }

    [Fact]
    public void Segmentation_SincChainsFold()
    {
        var graph = ModelFixture.LoadRequiredModel("PyannoteSegmentation", "models", "speaker-diarization-community-1", "onnx", "segmentation", "model.onnx");
        Assert.True(graph.Initializers.ContainsKey("/sincnet/conv1d.0/Concat_2_output_0"));
        foreach (var n in graph.Nodes)
        {
            Assert.False(n.Op == OpType.Concat && n.Name.StartsWith("/sincnet/conv1d.0/Concat"), "Sinc concat survived: " + n.Name);
            Assert.False(n.Op == OpType.Sin && n.Name.StartsWith("/sincnet/conv1d.0/Sin"), "Sinc sin survived: " + n.Name);
        }
    }
}
