using System;
using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

// Sigmoid feeding a single-consumer Mul must fuse into the multiply node
// with a fuse_sigmoid epilogue (Swish x*sigmoid(x) and gated a*sigmoid(b));
// exposed intermediates and shared sigmoids must block fusion while leaving
// unfused execution intact. Fused and unfused agree within float rounding.
public class GraphFusionSwishTests
{
    static OnnxValueInfo NamedIO(string name, params int[] dims) =>
        new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = dims };

    static OnnxNode Nod(string op, string[] inputs, string[] outputs) =>
        new OnnxNode { OpType = op, Domain = "", Inputs = inputs, Outputs = outputs, Attributes = new Dictionary<string, object>() };

    static OnnxModel TinySwishModel()
    {
        var mp = new OnnxModel { Name = "tiny-swish" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("x", 2, 5));
        mp.Outputs.Add(NamedIO("z", 2, 5));
        mp.Nodes.Add(Nod("Sigmoid", new[] { "x" }, new[] { "t" }));
        mp.Nodes.Add(Nod("Mul", new[] { "x", "t" }, new[] { "z" }));
        return mp;
    }

    static OnnxModel TinyGluModel(bool sigmoidFirst)
    {
        var mp = new OnnxModel { Name = "tiny-glu" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("a", 2, 5));
        mp.Inputs.Add(NamedIO("b", 2, 5));
        mp.Outputs.Add(NamedIO("z", 2, 5));
        mp.Nodes.Add(Nod("Sigmoid", new[] { "b" }, new[] { "s" }));
        mp.Nodes.Add(sigmoidFirst
            ? Nod("Mul", new[] { "s", "a" }, new[] { "z" })
            : Nod("Mul", new[] { "a", "s" }, new[] { "z" }));
        return mp;
    }

    static DenseTensor<float> XInput()
    {
        // Straddles the sigmoid knees plus saturation on both sides.
        return DenseTensor<float>.OfValues(new float[,] { { -9f, -3f, -0.5f, 0f, 0.5f }, { 1f, 2f, 4f, 8f, 12f } });
    }

    static void AssertScaledNear(float[] expected, float[] actual, double tol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        int at = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            double err = Math.Abs((double)actual[i] - expected[i]) / (1.0 + Math.Abs((double)expected[i]));
            if (err > worst) { worst = err; at = i; }
        }
        Assert.True(worst <= tol, what + " worst=" + worst.ToString("E2") + " at " + at);
    }

    static float[] RunTwin(OnnxModel mp, Dictionary<string, ITensor> inputs)
    {
        var graph = Model.Load(mp);
        Assert.True(graph is not null);
        Assert.True(graph.Execute(inputs, true));
        return ((Tensor<float>)graph.Outputs["z"]).ToArray();
    }

    [Fact]
    public void Swish_FusesToSingleMulEpilogue()
    {
        var graph = Model.Load(TinySwishModel());
        Assert.True(graph is not null);
        Assert.Single(graph.Nodes);
        var fused = graph.Nodes[0];
        Assert.Equal(OpType.Mul, fused.Op);
        Assert.True(fused.IsFused);
        Assert.True(fused.Attributes is not null && fused.Attributes.TryGetValue("fuse_sigmoid", out var av) && (int)av == 1);
        Assert.Equal(new[] { "x", "x" }, fused.Inputs);
        Assert.True(CPUExecutionProvider.SupportsNode(fused));
    }

    [Fact]
    public void Swish_FusedMatchesUnfusedTwin()
    {
        var fusedOut = RunTwin(TinySwishModel(), new Dictionary<string, ITensor> { { "x", XInput() } });
        var mp2 = TinySwishModel();
        mp2.Outputs.Add(NamedIO("t", 2, 5));
        var blocked = Model.Load(mp2);
        Assert.True(blocked is not null);
        Assert.Equal(2, blocked.Nodes.Count);
        var plainOut = RunTwin(mp2, new Dictionary<string, ITensor> { { "x", XInput() } });
        AssertScaledNear(plainOut, fusedOut, 1e-6, "swish-twin");
    }

    [Fact]
    public void Glu_FusesEitherLeg()
    {
        var second = Model.Load(TinyGluModel(false));
        Assert.True(second is not null);
        Assert.Single(second.Nodes);
        var secondAttrs = second.Nodes[0].Attributes;
        Assert.True(secondAttrs is not null && secondAttrs.TryGetValue("fuse_sigmoid", out var b) && (int)b == 1);
        var first = Model.Load(TinyGluModel(true));
        Assert.True(first is not null);
        Assert.Single(first.Nodes);
        var firstAttrs = first.Nodes[0].Attributes;
        Assert.True(firstAttrs is not null && firstAttrs.TryGetValue("fuse_sigmoid", out var a) && (int)a == 0);
        var inputs = new Dictionary<string, ITensor> { { "a", XInput() }, { "b", XInput() } };
        var mp2 = TinyGluModel(false);
        mp2.Outputs.Add(NamedIO("s", 2, 5));
        AssertScaledNear(RunTwin(mp2, inputs), RunTwin(TinyGluModel(false), inputs), 1e-6, "glu-twin");
        AssertScaledNear(RunTwin(mp2, inputs), RunTwin(TinyGluModel(true), inputs), 1e-6, "glu-leg-swap");
    }

    [Fact]
    public void SharedSigmoid_BlocksFusion()
    {
        var mp = new OnnxModel { Name = "shared-sig" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("x", 2, 5));
        mp.Outputs.Add(NamedIO("z1", 2, 5));
        mp.Outputs.Add(NamedIO("z2", 2, 5));
        mp.Nodes.Add(Nod("Sigmoid", new[] { "x" }, new[] { "t" }));
        mp.Nodes.Add(Nod("Mul", new[] { "x", "t" }, new[] { "z1" }));
        mp.Nodes.Add(Nod("Mul", new[] { "t", "x" }, new[] { "z2" }));
        var graph = Model.Load(mp);
        Assert.True(graph is not null);
        Assert.Equal(3, graph.Nodes.Count);
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true));
    }

    [Fact]
    public void MulSigmoid_ProviderMatchesUnfusedAcrossShapes()
    {
        var rnd = new Random(91);
        int[][] shapes = new int[][] { new[] { 1 }, new[] { 7 }, new[] { 8 }, new[] { 1000 }, new[] { 4, 64 }, new[] { 2, 3 } };
        foreach (var dims in shapes)
        {
            int n = 1;
            foreach (var d in dims) n *= d;
            var av = new float[n];
            var bv = new float[n];
            for (int i = 0; i < n; i++)
            {
                av[i] = (float)(rnd.NextDouble() * 20 - 10);
                bv[i] = (float)(rnd.NextDouble() * 20 - 10);
            }
            var a = new DenseTensor<float>(av, dims);
            var b = new DenseTensor<float>(bv, dims);
            foreach (int leg in new[] { 0, 1 })
            {
                var sig = leg == 1 ? b : a;
                var other = leg == 1 ? a : b;
                var s = CPUExecutionProvider.Sigmoid(sig, null);
                Assert.Equal(OpStatus.Success, s.Status);
                var m = CPUExecutionProvider.Mul(other, (Tensor<float>)s.Outputs[0], null, null);
                Assert.Equal(OpStatus.Success, m.Status);
                var f = CPUExecutionProvider.MulSigmoid(a, b, leg, null, null);
                Assert.Equal(OpStatus.Success, f.Status);
                AssertScaledNear(((Tensor<float>)m.Outputs[0]).ToArray(), ((Tensor<float>)f.Outputs[0]).ToArray(), 1e-6,
                    "mulsig leg=" + leg + " dims=" + string.Join("x", dims));
                var fs = CPUExecutionProvider.MulSigmoid(a, b, leg, ExecutionOptions.Scalar, null);
                Assert.Equal(OpStatus.Success, fs.Status);
                AssertScaledNear(((Tensor<float>)m.Outputs[0]).ToArray(), ((Tensor<float>)fs.Outputs[0]).ToArray(), 1e-6,
                    "mulsig-scalar leg=" + leg + " dims=" + string.Join("x", dims));
            }
        }
    }

    [SkippableFact]
    public void Encoder_FusesNinetySixSigmoidMulPairs()
    {
        var graph = ModelFixture.LoadRequiredModel("ParakeetEncoder", "models", "parakeet-tdt-0.6b-v3", "onnx", "encoder-model.onnx");
        int swish = 0, glu = 0;
        foreach (var node in graph.Nodes)
        {
            if (node.Op != OpType.Mul || !node.IsFused) continue;
            var attrs = node.Attributes;
            if (attrs is null || !attrs.TryGetValue("fuse_sigmoid", out var av)) continue;
            if (node.Inputs.Length == 2 && node.Inputs[0] == node.Inputs[1]) swish++;
            else glu++;
        }
        Assert.Equal(72, swish);
        Assert.Equal(24, glu);
    }

    [Fact]
    public void MulSigmoid_MismatchedBroadcastShapes()
    {
        // Same lengths but different shapes broadcast to [2,2]; the fused
        // kernel must follow the broadcast, not the flat pairing.
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f }, { -2f } });
        var b = DenseTensor<float>.OfValues(new float[] { 0.5f, -1.5f });
        foreach (int leg in new[] { 0, 1 })
        {
            var sig = leg == 1 ? (Tensor<float>)b : (Tensor<float>)a;
            var other = leg == 1 ? (Tensor<float>)a : (Tensor<float>)b;
            var s = CPUExecutionProvider.Sigmoid(sig, null);
            Assert.Equal(OpStatus.Success, s.Status);
            var m = CPUExecutionProvider.Mul(other, (Tensor<float>)s.Outputs[0], null, null);
            Assert.Equal(OpStatus.Success, m.Status);
            var expected = ((Tensor<float>)m.Outputs[0]).ToArray();
            Assert.Equal(new[] { 2, 2 }, ((Tensor<float>)m.Outputs[0]).Dimensions.ToArray());
            var f = CPUExecutionProvider.MulSigmoid(a, b, leg, null, null);
            Assert.Equal(OpStatus.Success, f.Status);
            var actual = (Tensor<float>)f.Outputs[0];
            Assert.Equal(new[] { 2, 2 }, actual.Dimensions.ToArray());
            AssertScaledNear(expected, actual.ToArray(), 1e-6, "mulsig-mismatch leg=" + leg);
        }
    }

    [Fact]
    public void MulSigmoid_TransposedInputMatchesUnfused()
    {
        var rnd = new Random(93);
        var av = new float[12];
        var bv = new float[12];
        for (int i = 0; i < 12; i++)
        {
            av[i] = (float)(rnd.NextDouble() * 8 - 4);
            bv[i] = (float)(rnd.NextDouble() * 8 - 4);
        }
        var a = new DenseTensor<float>(av, new[] { 3, 4 });
        var bt = Tensor<float>.Transpose(new DenseTensor<float>(bv, new[] { 4, 3 }), new[] { 1, 0 });
        foreach (int leg in new[] { 0, 1 })
        {
            var sig = leg == 1 ? bt : (Tensor<float>)a;
            var other = leg == 1 ? (Tensor<float>)a : bt;
            var s = CPUExecutionProvider.Sigmoid(sig, null);
            Assert.Equal(OpStatus.Success, s.Status);
            var m = CPUExecutionProvider.Mul(other, (Tensor<float>)s.Outputs[0], null, null);
            Assert.Equal(OpStatus.Success, m.Status);
            var f = CPUExecutionProvider.MulSigmoid(a, bt, leg, null, null);
            Assert.Equal(OpStatus.Success, f.Status);
            AssertScaledNear(((Tensor<float>)m.Outputs[0]).ToArray(), ((Tensor<float>)f.Outputs[0]).ToArray(), 1e-6,
                "mulsig-transposed leg=" + leg);
        }
    }

    [Fact]
    public void MulSigmoid_ProviderBroadcastFallback()
    {
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, -2f, 3f }, { -4f, 5f, -6f } });
        var b = DenseTensor<float>.OfValues(new float[] { 0.5f, -0.5f, 1.5f });
        var s = CPUExecutionProvider.Sigmoid(b, null);
        Assert.Equal(OpStatus.Success, s.Status);
        var m = CPUExecutionProvider.Mul(a, (Tensor<float>)s.Outputs[0], null, null);
        Assert.Equal(OpStatus.Success, m.Status);
        var f = CPUExecutionProvider.MulSigmoid(a, b, 1, null, null);
        Assert.Equal(OpStatus.Success, f.Status);
        AssertScaledNear(((Tensor<float>)m.Outputs[0]).ToArray(), ((Tensor<float>)f.Outputs[0]).ToArray(), 1e-6, "mulsig-broadcast");
    }
}
