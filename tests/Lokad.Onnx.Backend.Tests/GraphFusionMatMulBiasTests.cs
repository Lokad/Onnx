using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

// Covers the MatMul-plus-bias fusion (M3): MatMul feeding only an Add of a
// scalar or row-vector float bias becomes one node with a fuse_bias
// epilogue. The epilogue keeps the product-then-add order, so agreement is
// exact and the Add dispatch plus its M-by-N intermediate disappear.
public class GraphFusionMatMulBiasTests
{
    static OnnxValueInfo IO(string name, int d0, int d1) =>
        new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = new[] { d0, d1 } };

    static OnnxNode Nod(string op, string[] inputs, string[] outputs, Dictionary<string, object>? attrs) =>
        new OnnxNode { OpType = op, Domain = "", Inputs = inputs, Outputs = outputs, Attributes = attrs ?? new Dictionary<string, object>() };

    static OnnxNode ConstTensor(string output, float[] values, int[] dims) =>
        Nod("Constant", new string[0], new[] { output }, new Dictionary<string, object>
        {
            { "value", Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = dims, Data = values }) },
        });

    static OnnxModel Case(string biasKind, bool biasFirst)
    {
        var mp = new OnnxModel { Name = "mm-bias" };
        mp.Opset[""] = 17;
        mp.Inputs.Add(IO("a", 2, 3));
        mp.Inputs.Add(IO("b", 3, 2));
        mp.Outputs.Add(IO("y", 2, 2));
        mp.Nodes.Add(Nod("MatMul", new[] { "a", "b" }, new[] { "m" }, null));
        if (biasKind == "const")
            mp.Nodes.Add(ConstTensor("s", new float[] { 10f, 20f }, new[] { 2 }));
        else if (biasKind == "scalar")
            mp.Nodes.Add(ConstTensor("s", new float[] { 5f }, new int[0]));
        else
            mp.Initializers.Add(new OnnxTensor { Name = "s", ElementType = TensorElementType.Float, Dims = new[] { 2 }, Data = new float[] { 10f, 20f } });
        mp.Nodes.Add(biasFirst
            ? Nod("Add", new[] { "s", "m" }, new[] { "y" }, null)
            : Nod("Add", new[] { "m", "s" }, new[] { "y" }, null));
        return mp;
    }

    static Dictionary<string, ITensor> Feeds() => new Dictionary<string, ITensor>
    {
        ["a"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } }),
        ["b"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 0f }, { 0f, 1f }, { 1f, 1f } }),
    };

    static int Bits(float f) => BitConverter.SingleToInt32Bits(f);

    static void CheckFused(string biasKind, bool biasFirst, float[] want)
    {
        var graph = Model.Load(Case(biasKind, biasFirst))!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Add);
        var mm = Assert.Single(graph.Nodes, n => n.Op == OpType.MatMul);
        Assert.Equal("s", Assert.IsType<string>(mm.Attributes!["fuse_bias"]));
        Assert.True(graph.Execute(Feeds(), true), graph.LastErrorMessage);
        var got = ((Tensor<float>)graph.Outputs["y"]).ToArray();
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++)
            Assert.Equal(Bits(want[i]), Bits(got[i]));
    }

    [Fact]
    public void RowBiasConstant_FusesBitwise()
    {
        CheckFused("const", false, new float[] { 14f, 25f, 20f, 31f });
    }

    [Fact]
    public void BiasFirstLeg_FusesBitwise()
    {
        CheckFused("const", true, new float[] { 14f, 25f, 20f, 31f });
    }

    [Fact]
    public void ScalarBias_FusesBitwise()
    {
        CheckFused("scalar", false, new float[] { 9f, 10f, 15f, 16f });
    }

    [Fact]
    public void InitializerBias_FusesBitwise()
    {
        CheckFused("init", false, new float[] { 14f, 25f, 20f, 31f });
    }

    [Fact]
    public void GraphOutputAdd_FusesBitwise()
    {
        var mp = Case("const", false);
        var graph = Model.Load(mp)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Add);
        var mm = Assert.Single(graph.Nodes, n => n.Op == OpType.MatMul);
        Assert.Equal(new[] { "y" }, mm.Outputs);
    }

    [Fact]
    public void MultiUseProduct_Declines()
    {
        var mp = Case("const", false);
        mp.Outputs.Add(IO("m2", 2, 2));
        mp.Nodes.Add(Nod("Identity", new[] { "m" }, new[] { "m2" }, null));
        var graph = Model.Load(mp)!;
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Add);
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.MatMul && n.Attributes is not null && n.Attributes.ContainsKey("fuse_bias"));
    }

    [Fact]
    public void FedBiasName_Declines()
    {
        var mp = Case("init", false);
        mp.Inputs.Add(IO("s", 2, 1));
        var graph = Model.Load(mp)!;
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Add);
    }

    [Fact]
    public void BadLengthBias_Declines()
    {
        var mp = new OnnxModel { Name = "mm-bias-bad" };
        mp.Opset[""] = 17;
        mp.Inputs.Add(IO("a", 2, 3));
        mp.Inputs.Add(IO("b", 3, 2));
        mp.Outputs.Add(IO("y", 2, 2));
        mp.Nodes.Add(Nod("MatMul", new[] { "a", "b" }, new[] { "m" }, null));
        mp.Initializers.Add(new OnnxTensor { Name = "s", ElementType = TensorElementType.Float, Dims = new[] { 3 }, Data = new float[] { 1f, 2f, 3f } });
        mp.Nodes.Add(Nod("Add", new[] { "m", "s" }, new[] { "y" }, null));
        var graph = Model.Load(mp)!;
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Add);
    }

    static List<string> FusedSites(ComputationalGraph graph)
    {
        var found = new List<string>();
        foreach (var n in graph.Nodes)
        {
            if (n.Op == OpType.MatMul && n.Attributes is not null && n.Attributes.ContainsKey("fuse_bias"))
                found.Add(n.Name + "<-" + n.Attributes["fuse_bias"]);
        }
        found.Sort();
        return found;
    }

    [SkippableFact]
    public void Decoder_FusesThreeBiasSites()
    {
        var graph = ModelFixture.LoadRequiredModel("ParakeetDecoder", "models", "parakeet-tdt-0.6b-v3", "onnx", "decoder_joint-model.onnx");
        var found = FusedSites(graph);
        Assert.Equal(3, found.Count);
        Assert.Contains("/joint/enc/MatMul<-joint.enc.bias", found);
        Assert.Contains("/joint/pred/MatMul<-joint.pred.bias", found);
        Assert.Contains("/joint/joint_net/joint_net.2/MatMul<-joint.joint_net.2.bias", found);
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Add && (n.Name == "/joint/enc/Add" || n.Name == "/joint/pred/Add" || n.Name == "/joint/joint_net/joint_net.2/Add"));
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Add && n.Name == "/joint/Add");
    }

    [SkippableFact]
    public void Encoder_FusesPreEncodeBias()
    {
        var graph = ModelFixture.LoadRequiredModel("ParakeetEncoder", "models", "parakeet-tdt-0.6b-v3", "onnx", "encoder-model.onnx");
        var found = FusedSites(graph);
        Assert.Equal(1, found.Count);
        Assert.Contains("/pre_encode/out/MatMul<-pre_encode.out.bias", found);
    }

    [SkippableFact]
    public void SegmentationBias_BlockedAtControlFlow()
    {
        // The three linear biases bottom out at data-dependent If/Shape
        // control flow the dtype prover correctly refuses: recorded here so
        // a future prover change shows up as a count change, not silence.
        var graph = ModelFixture.LoadRequiredModel("PyannoteSegmentation", "models", "speaker-diarization-community-1", "onnx", "segmentation", "model.onnx");
        Assert.Equal(0, FusedSites(graph).Count);
    }
}
