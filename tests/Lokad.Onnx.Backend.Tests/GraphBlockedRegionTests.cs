using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

// Covers blocked residual regions (M5): recognition fuses Conv/Relu/Add
// chains with single-use internal links into one region head; execution
// converts once, runs prepared-filter blocked steps with in-domain
// epilogues, converts once. Agreement with unfused execution at 1e-4.
public class GraphBlockedRegionTests
{
    static OnnxValueInfo IO(string name, int c, int h, int w) =>
        new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = new[] { -1, c, -1, -1 } };

    static OnnxTensor FloatInit(string name, int[] dims, float[] values) =>
        new OnnxTensor { Name = name, ElementType = TensorElementType.Float, Dims = dims, Data = values };

    static OnnxNode Nod(string op, string[] inputs, string[] outputs, Dictionary<string, object>? attrs) =>
        new OnnxNode { OpType = op, Domain = "", Inputs = inputs, Outputs = outputs, Attributes = attrs ?? new Dictionary<string, object>() };

    static Dictionary<string, object> ConvAttrs() => new Dictionary<string, object>
    {
        { "kernel_shape", new long[] { 3L, 3L } },
        { "strides", new long[] { 1L, 1L } },
        { "pads", new long[] { 1L, 1L, 1L, 1L } },
    };

    static float[] Rand(int n, int seed)
    {
        var rnd = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rnd.NextDouble() * 2 - 1);
        return a;
    }

    static OnnxModel ResidualBlockModel(bool constBias)
    {
        var mp = new OnnxModel { Name = "resblock" };
        mp.Opset[""] = 17;
        mp.Inputs.Add(IO("x", 16, 10, 10));
        mp.Outputs.Add(IO("y", 16, 10, 10));
        mp.Initializers.Add(FloatInit("w1", new[] { 16, 16, 3, 3 }, Rand(16 * 16 * 9, 11)));
        mp.Initializers.Add(FloatInit("w2", new[] { 16, 16, 3, 3 }, Rand(16 * 16 * 9, 12)));
        if (!constBias)
        {
            mp.Initializers.Add(FloatInit("b1", new[] { 16 }, Rand(16, 13)));
            mp.Initializers.Add(FloatInit("b2", new[] { 16 }, Rand(16, 14)));
        }
        mp.Nodes.Add(Nod("Conv", new[] { "x", "w1", "b1" }, new[] { "c1" }, ConvAttrs()));
        mp.Nodes.Add(Nod("Relu", new[] { "c1" }, new[] { "r1" }, null));
        mp.Nodes.Add(Nod("Conv", new[] { "r1", "w2", "b2" }, new[] { "c2" }, ConvAttrs()));
        mp.Nodes.Add(Nod("Add", new[] { "c2", "x" }, new[] { "a" }, null));
        mp.Nodes.Add(Nod("Relu", new[] { "a" }, new[] { "y" }, null));
        if (constBias)
        {
            var b1 = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new[] { 16 }, Data = Rand(16, 13) });
            var b2 = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new[] { 16 }, Data = Rand(16, 14) });
            mp.Nodes.Add(Nod("Constant", new string[0], new[] { "b1" }, new Dictionary<string, object> { { "value", b1 } }));
            mp.Nodes.Add(Nod("Constant", new string[0], new[] { "b2" }, new Dictionary<string, object> { { "value", b2 } }));
        }
        return mp;
    }

    static ComputationalGraph UnfusedResidualBlock(bool constBias)
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "resblock-unfused";
        g.Opset[""] = 17;
        g.Inputs["x"] = DenseTensor<float>.OfShape(1, 16, 10, 10);
        g.InputDescs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { -1, 16, -1, -1 } });
        g.Outputs["y"] = DenseTensor<float>.OfShape(1, 16, 10, 10);
        var mp = ResidualBlockModel(constBias);
        foreach (var init in mp.Initializers)
            g.Initializers[init.Name] = Model.ToTensor(init);
        foreach (var np in mp.Nodes)
        {
            g.Nodes.Add(new Node
            {
                Name = np.Name,
                Op = Enum.Parse<OpType>(np.OpType, false),
                OpTypeName = np.OpType ?? "",
                Domain = np.Domain ?? "",
                Inputs = np.Inputs,
                Outputs = np.Outputs,
                Attributes = new Dictionary<string, object>(np.Attributes),
            });
            foreach (var o in np.Outputs)
                if (!g.Outputs.ContainsKey(o) && !g.IntermediateOutputs.ContainsKey(o))
                    g.IntermediateOutputs.Add(o, null);
        }
        g.RefreshLifetimeAnalysis();
        return g;
    }

    static Dictionary<string, ITensor> Feeds(int n, int c, int h, int w, int seed)
    {
        return new Dictionary<string, ITensor>
        {
            ["x"] = new DenseTensor<float>(new Memory<float>(Rand(n * c * h * w, seed)), new[] { n, c, h, w }),
        };
    }

    static double WorstScaled(float[] actual, float[] expected)
    {
        double worst = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double e = System.Math.Abs((double)actual[i] - expected[i]) / (1.0 + System.Math.Abs((double)expected[i]));
            if (e > worst) worst = e;
        }
        return worst;
    }

    static void AssertAgree(ComputationalGraph fused, ComputationalGraph plain, Dictionary<string, ITensor> feeds, string what)
    {
        Assert.True(fused.Execute(new Dictionary<string, ITensor>(feeds), true), fused.LastErrorMessage);
        Assert.True(plain.Execute(new Dictionary<string, ITensor>(feeds), true), plain.LastErrorMessage);
        var a = ((Tensor<float>)fused.Outputs["y"]).ToArray();
        var b = ((Tensor<float>)plain.Outputs["y"]).ToArray();
        Assert.Equal(a.Length, b.Length);
        double worst = WorstScaled(a, b);
        Assert.True(worst <= 1e-4, what + " worst=" + worst.ToString("E2"));
    }

    [Fact]
    public void ResidualBlock_FusesAndAgrees()
    {
        var graph = Model.Load(ResidualBlockModel(false))!;
        Assert.Single(graph.Nodes);
        var head = graph.Nodes[0];
        Assert.Equal(OpType.Conv, head.Op);
        Assert.True(head.Attributes is not null && head.Attributes.ContainsKey("fuse_region"), "region marker is missing.");
        var plain = UnfusedResidualBlock(false);
        AssertAgree(graph, plain, Feeds(1, 16, 10, 10, 21), "n1");
        AssertAgree(graph, plain, Feeds(2, 16, 10, 10, 22), "n2");
        AssertAgree(graph, plain, Feeds(1, 16, 6, 5, 23), "odd");
    }

    [Fact]
    public void ConstantBias_Fuses()
    {
        var graph = Model.Load(ResidualBlockModel(true))!;
        Assert.Single(graph.Nodes);
        var plain = UnfusedResidualBlock(true);
        AssertAgree(graph, plain, Feeds(1, 16, 10, 10, 24), "const-bias");
    }

    [Fact]
    public void FanOut_StopsRegion()
    {
        var mp = new OnnxModel { Name = "fanout" };
        mp.Opset[""] = 17;
        mp.Inputs.Add(IO("x", 16, 8, 8));
        mp.Outputs.Add(IO("y1", 16, 8, 8));
        mp.Outputs.Add(IO("y2", 16, 8, 8));
        mp.Initializers.Add(FloatInit("w1", new[] { 16, 16, 3, 3 }, Rand(16 * 16 * 9, 31)));
        mp.Initializers.Add(FloatInit("b1", new[] { 16 }, Rand(16, 32)));
        mp.Nodes.Add(Nod("Conv", new[] { "x", "w1", "b1" }, new[] { "c1" }, ConvAttrs()));
        mp.Nodes.Add(Nod("Relu", new[] { "c1" }, new[] { "r1" }, null));
        mp.Nodes.Add(Nod("Add", new[] { "r1", "r1" }, new[] { "y1" }, null));
        mp.Nodes.Add(Nod("Mul", new[] { "r1", "r1" }, new[] { "y2" }, null));
        var graph = Model.Load(mp)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Attributes is not null && n.Attributes.ContainsKey("fuse_region"));
    }

    [Fact]
    public void SingleConv_NoRegion()
    {
        var mp = new OnnxModel { Name = "single" };
        mp.Opset[""] = 17;
        mp.Inputs.Add(IO("x", 16, 8, 8));
        mp.Outputs.Add(IO("y", 16, 8, 8));
        mp.Initializers.Add(FloatInit("w1", new[] { 16, 16, 3, 3 }, Rand(16 * 16 * 9, 41)));
        mp.Initializers.Add(FloatInit("b1", new[] { 16 }, Rand(16, 42)));
        mp.Nodes.Add(Nod("Conv", new[] { "x", "w1", "b1" }, new[] { "c1" }, ConvAttrs()));
        mp.Nodes.Add(Nod("Relu", new[] { "c1" }, new[] { "y" }, null));
        var graph = Model.Load(mp)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Attributes is not null && n.Attributes.ContainsKey("fuse_region"));
    }

    [Fact]
    public void WeightReplace_Repacks()
    {
        var graph = Model.Load(ResidualBlockModel(false))!;
        var plain = UnfusedResidualBlock(false);
        var w1 = FloatInit("w1", new[] { 16, 16, 3, 3 }, Rand(16 * 16 * 9, 51));
        graph.Initializers["w1"] = Model.ToTensor(w1);
        plain.Initializers["w1"] = Model.ToTensor(w1);
        graph.RefreshLifetimeAnalysis();
        plain.RefreshLifetimeAnalysis();
        AssertAgree(graph, plain, Feeds(1, 16, 10, 10, 52), "repacked");
    }

    [Fact]
    public void ScalarOptions_Executes()
    {
        var graph = Model.Load(ResidualBlockModel(false))!;
        var plain = UnfusedResidualBlock(false);
        var opts = new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Scalar);
        var gctx = graph.CreateExecution(opts);
        var pctx = plain.CreateExecution(opts);
        var feeds = Feeds(1, 16, 10, 10, 61);
        Assert.True(gctx.Execute(new Dictionary<string, ITensor>(feeds), true), gctx.LastErrorMessage);
        Assert.True(pctx.Execute(new Dictionary<string, ITensor>(feeds), true), pctx.LastErrorMessage);
        double worst = WorstScaled(((Tensor<float>)gctx.Outputs["y"]).ToArray(), ((Tensor<float>)pctx.Outputs["y"]).ToArray());
        Assert.True(worst <= 1e-4, "scalar-region worst=" + worst.ToString("E2"));
    }

    [SkippableFact]
    public void Embedding_FindsResidualRegions()
    {
        var graph = ModelFixture.LoadRequiredModel("VoiceEmbedding", "models", "speaker-diarization-community-1", "onnx", "embedding", "embedding_encoder.onnx");
        int regions = graph.Nodes.Count(n => n.Op == OpType.Conv && n.Attributes is not null && n.Attributes.ContainsKey("fuse_region"));
        Assert.Equal(13, regions);
    }
}

