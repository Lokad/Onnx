using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

// Covers the MatMul-times-scalar fusion (M3): MatMul feeding only a Mul by a
// single-use scalar Constant becomes one Gemm with alpha set to the scale.
// The Gemm scale pass is bit-identical to the removed Mul (same single
// rounding per element, including signed zero), so agreement is exact and
// the Mul dispatch plus its M-by-N intermediate disappear per site.
public class GraphFusionMatMulScaleTests
{
    static OnnxValueInfo IO(string name, int d0, int d1) =>
        new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = new[] { d0, d1 } };

    static OnnxValueInfo IOd(string name, int d0, int d1) =>
        new OnnxValueInfo { Name = name, ElementType = TensorElementType.Double, Dims = new[] { d0, d1 } };

    static OnnxNode Nod(string op, string[] inputs, string[] outputs, Dictionary<string, object>? attrs) =>
        new OnnxNode { OpType = op, Domain = "", Inputs = inputs, Outputs = outputs, Attributes = attrs ?? new Dictionary<string, object>() };

    static OnnxNode ConstFloat(string output, float value) =>
        Nod("Constant", new string[0], new[] { output }, new Dictionary<string, object>
        {
            { "value", Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new int[0], Data = new[] { value } }) },
        });

    static OnnxModel Case(float scale, bool scaleFirst, string extraOutput)
    {
        var mp = new OnnxModel { Name = "mm-scale" };
        mp.Opset[""] = 17;
        mp.Inputs.Add(IO("a", 2, 3));
        mp.Inputs.Add(IO("b", 3, 2));
        mp.Outputs.Add(IO("y", 2, 2));
        if (extraOutput == "mm")
            mp.Outputs.Add(IO("m", 2, 2));
        mp.Nodes.Add(Nod("MatMul", new[] { "a", "b" }, new[] { "m" }, null));
        mp.Nodes.Add(ConstFloat("s", scale));
        mp.Nodes.Add(scaleFirst
            ? Nod("Mul", new[] { "s", "m" }, new[] { "y" }, null)
            : Nod("Mul", new[] { "m", "s" }, new[] { "y" }, null));
        return mp;
    }

    static Dictionary<string, ITensor> Feeds() => new Dictionary<string, ITensor>
    {
        ["a"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } }),
        ["b"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 0f }, { 0f, 1f }, { 1f, 1f } }),
    };

    static int Bits(float f) => BitConverter.SingleToInt32Bits(f);

    [Fact]
    public void MatMulScalarMul_FusesToGemm()
    {
        var graph = Model.Load(Case(0.5f, false, ""))!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Mul);
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Constant);
        var mm = Assert.Single(graph.Nodes, n => n.Op == OpType.MatMul);
        Assert.Equal(0.5f, Assert.IsType<float>(mm.Attributes!["fuse_scale"]));
        Assert.True(graph.Execute(Feeds(), true), graph.LastErrorMessage);
        var got = ((Tensor<float>)graph.Outputs["y"]).ToArray();
        var want = new float[] { 2f, 2.5f, 5f, 5.5f };
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++)
            Assert.Equal(Bits(want[i]), Bits(got[i]));
    }

    [Fact]
    public void ScaleFirstLeg_FusesToGemm()
    {
        var graph = Model.Load(Case(0.25f, true, ""))!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Mul);
        var mm = Assert.Single(graph.Nodes, n => n.Op == OpType.MatMul);
        Assert.Equal(0.25f, Assert.IsType<float>(mm.Attributes!["fuse_scale"]));
        Assert.True(graph.Execute(Feeds(), true), graph.LastErrorMessage);
        var got = ((Tensor<float>)graph.Outputs["y"]).ToArray();
        var want = new float[] { 1f, 1.25f, 2.5f, 2.75f };
        Assert.Equal(want, got);
    }

    [Fact]
    public void MatMulProduct_NeverNegativeZero()
    {
        // Load-bearing lemma for the fusion above: every MatMul accumulation
        // lane starts at +0, and +0 plus any product is +0, so a product can
        // never come out -0. The removed Mul therefore never observes -0,
        // which is what makes the Gemm alpha pass (with its +0 term)
        // bit-identical on all reachable inputs. If a future kernel changes
        // accumulation, this fails loudly and the fusion must be re-examined.
        var mp = new OnnxModel { Name = "mm-never-negzero" };
        mp.Opset[""] = 17;
        mp.Inputs.Add(IO("a", 1, 2));
        mp.Inputs.Add(IO("b", 2, 1));
        mp.Outputs.Add(IO("y", 1, 1));
        mp.Nodes.Add(Nod("MatMul", new[] { "a", "b" }, new[] { "y" }, null));
        var graph = Model.Load(mp)!;
        var feeds = new Dictionary<string, ITensor>
        {
            ["a"] = DenseTensor<float>.OfValues(new float[,] { { -0f, 0f } }),
            ["b"] = DenseTensor<float>.OfValues(new float[,] { { 1f }, { 0f } }),
        };
        Assert.True(graph.Execute(feeds, true), graph.LastErrorMessage);
        var got = ((Tensor<float>)graph.Outputs["y"]).ToArray();
        Assert.Equal(Bits(0f), Bits(got[0]));
    }


    [Fact]
    public void MultiUseScale_KeepsMul()
    {
        var mp = new OnnxModel { Name = "mm-shared-scale" };
        mp.Opset[""] = 17;
        mp.Inputs.Add(IO("a", 2, 2));
        mp.Inputs.Add(IO("b", 2, 2));
        mp.Outputs.Add(IO("y1", 2, 2));
        mp.Outputs.Add(IO("y2", 2, 2));
        mp.Nodes.Add(Nod("MatMul", new[] { "a", "b" }, new[] { "m1" }, null));
        mp.Nodes.Add(Nod("MatMul", new[] { "b", "a" }, new[] { "m2" }, null));
        mp.Nodes.Add(ConstFloat("s", 0.5f));
        mp.Nodes.Add(Nod("Mul", new[] { "m1", "s" }, new[] { "y1" }, null));
        mp.Nodes.Add(Nod("Mul", new[] { "m2", "s" }, new[] { "y2" }, null));
        var graph = Model.Load(mp)!;
        Assert.Equal(2, graph.Nodes.Count(n => n.Op == OpType.Mul));
        Assert.DoesNotContain(graph.Nodes, n => n.Attributes is not null && n.Attributes.ContainsKey("fuse_scale"));
    }

    [Fact]
    public void MultiUseMatMul_KeepsMul()
    {
        var mp = new OnnxModel { Name = "mm-shared-mm" };
        mp.Opset[""] = 17;
        mp.Inputs.Add(IO("a", 2, 2));
        mp.Inputs.Add(IO("b", 2, 2));
        mp.Outputs.Add(IO("y1", 2, 2));
        mp.Outputs.Add(IO("y2", 2, 2));
        mp.Nodes.Add(Nod("MatMul", new[] { "a", "b" }, new[] { "m" }, null));
        mp.Nodes.Add(ConstFloat("s", 0.5f));
        mp.Nodes.Add(Nod("Mul", new[] { "m", "s" }, new[] { "y1" }, null));
        mp.Nodes.Add(Nod("Add", new[] { "m", "m" }, new[] { "y2" }, null));
        var graph = Model.Load(mp)!;
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Mul);
        Assert.DoesNotContain(graph.Nodes, n => n.Attributes is not null && n.Attributes.ContainsKey("fuse_scale"));
    }

    [Fact]
    public void GraphOutputMatMul_KeepsMul()
    {
        var graph = Model.Load(Case(0.5f, false, "mm"))!;
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Mul);
        Assert.DoesNotContain(graph.Nodes, n => n.Attributes is not null && n.Attributes.ContainsKey("fuse_scale"));
    }

    [Fact]
    public void CustomDomainMul_BlocksFusion()
    {
        var mp = Case(0.5f, false, "");
        mp.Nodes.First(n => n.OpType == "Mul").Domain = "custom";
        mp.Opset["custom"] = 1;
        var graph = Model.Load(mp)!;
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Mul);
        Assert.DoesNotContain(graph.Nodes, n => n.Attributes is not null && n.Attributes.ContainsKey("fuse_scale"));
    }

    [Fact]
    public void DoubleMatMul_KeepsMul()
    {
        var mp = new OnnxModel { Name = "mm-double" };
        mp.Opset[""] = 17;
        mp.Inputs.Add(IOd("a", 2, 2));
        mp.Inputs.Add(IOd("b", 2, 2));
        mp.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Double, Dims = new[] { 2, 2 } });
        mp.Nodes.Add(Nod("MatMul", new[] { "a", "b" }, new[] { "m" }, null));
        mp.Nodes.Add(Nod("Constant", new string[0], new[] { "s" }, new Dictionary<string, object>
        {
            { "value", Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Double, Dims = new int[0], Data = new[] { 0.5 } }) },
        }));
        mp.Nodes.Add(Nod("Mul", new[] { "m", "s" }, new[] { "y" }, null));
        var graph = Model.Load(mp)!;
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Mul);
        Assert.DoesNotContain(graph.Nodes, n => n.Attributes is not null && n.Attributes.ContainsKey("fuse_scale"));
    }

    [SkippableFact]
    public void Encoder_Fuses48ScaleSites()
    {
        var graph = ModelFixture.LoadRequiredModel("ParakeetEncoder", "models", "parakeet-tdt-0.6b-v3", "onnx", "encoder-model.onnx");
        int scaled = graph.Nodes.Count(n => n.Op == OpType.MatMul && n.Attributes is not null
            && n.Attributes.TryGetValue("fuse_scale", out var a) && a is float f && f == 0.5f);
        Assert.Equal(48, scaled);
        foreach (var n in graph.Nodes)
        {
            if (n.Op != OpType.Mul || n.Inputs.Length != 2 || n.IsFused) continue;
            bool matMulLeg = graph.Nodes.Any(m => m.Op == OpType.MatMul && m.Outputs.Contains(n.Inputs[0]))
                || graph.Nodes.Any(m => m.Op == OpType.MatMul && m.Outputs.Contains(n.Inputs[1]));
            Assert.False(matMulLeg, "unfused MatMul-scale Mul remains: " + n.Name);
        }
    }
}

