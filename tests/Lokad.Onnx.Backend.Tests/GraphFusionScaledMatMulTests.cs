using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

// P07 scale fusion: Mul(data, scalar-scale) feeding MatMul rewrites to the
// ScaledMatMul fused op. Composite-first: the fused kernel sequences the exact
// legacy Mul then MatMul, so these gates prove plumbing with identical values
// before any fused kernel arrives.
public class GraphFusionScaledMatMulTests
{
    static DenseTensor<float> Rand(int[] dims, int seed)
    {
        var rnd = new System.Random(seed);
        var n = 1;
        foreach (var d in dims) n *= d;
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)rnd.NextDouble() * 2f - 1f;
        return new DenseTensor<float>(a.AsMemory(), dims);
    }

    static float[] Out(ITensor t) => ((Tensor<float>)t).ToDenseTensor().Buffer.ToArray();

    [Fact]
    public void ScaledMatMul_CompositeMatchesLegacyBitwise()
    {
        var a = Rand(new[] { 201, 384 }, 7);
        var b = Rand(new[] { 384, 384 }, 11);
        var s = DenseTensor<float>.Scalar(0.125f);
        var legacy = CPUExecutionProvider.Mul(a, s, null, null);
        Assert.Equal(OpStatus.Success, legacy.Status);
        var mm = CPUExecutionProvider.MatMul(legacy.Outputs![0]!, b, null, null);
        Assert.Equal(OpStatus.Success, mm.Status);
        var fused = CPUExecutionProvider.ScaledMatMul(a, b, s, null, null);
        Assert.Equal(OpStatus.Success, fused.Status);
        Assert.Equal(Out(mm.Outputs![0]!), Out(fused.Outputs![0]!));
    }

    static OnnxModel ScaleFusionModel()
    {
        var mp = new OnnxModel { Name = "tiny-scalematmul" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "r", ElementType = TensorElementType.Float, Dims = new[] { 4 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "t", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y1", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y2", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y3", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y4", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Initializers.Add(new OnnxTensor { Name = "s", ElementType = TensorElementType.Float, Dims = new int[0], Data = new float[] { 0.125f } });
        mp.Initializers.Add(new OnnxTensor { Name = "w", ElementType = TensorElementType.Float, Dims = new[] { 4, 3 }, Data = new float[] { 1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 1f, 1f, 1f } });
        Dictionary<string, object> NoAttrs() => new Dictionary<string, object>();
        mp.Nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { "x", "s" }, Outputs = new[] { "m1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "m1", "w" }, Outputs = new[] { "y1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { "s", "x" }, Outputs = new[] { "m2" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "m2", "w" }, Outputs = new[] { "y2" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { "x", "r" }, Outputs = new[] { "m3" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "m3", "w" }, Outputs = new[] { "y3" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { "x", "t" }, Outputs = new[] { "m4" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "m4", "w" }, Outputs = new[] { "y4" }, Attributes = NoAttrs() });
        return mp;
    }

    [Fact]
    public void ScaleRegionsFuseDecoysStay()
    {
        var graph = Model.Load(ScaleFusionModel(), runOptimizer: false)!;
        var report = Optimization.GraphOptimizer.Run(graph);
        Assert.Contains(report, c => c.Pass == "scalematmul");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.ScaledMatMul && n.Outputs.Length == 1 && n.Outputs[0] == "y1");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.ScaledMatMul && n.Outputs.Length == 1 && n.Outputs[0] == "y2");
        Assert.DoesNotContain(graph.Nodes, n => n.Outputs.Length == 1 && (n.Outputs[0] == "m1" || n.Outputs[0] == "m2"));
        // Runtime (non-singleton) scale keeps the two-node form.
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Mul && n.Outputs.Length == 1 && n.Outputs[0] == "m3");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.MatMul && n.Outputs.Length == 1 && n.Outputs[0] == "y3");
        // Two runtime operands keep the two-node form.
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Mul && n.Outputs.Length == 1 && n.Outputs[0] == "m4");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.MatMul && n.Outputs.Length == 1 && n.Outputs[0] == "y4");
    }

    static System.Collections.Generic.Dictionary<string, ITensor> ScaleFeed(int seed)
    {
        var rnd = new System.Random(seed);
        var x = new float[8];
        var r = new float[4];
        var t = new float[8];
        for (int i = 0; i < 8; i++) { x[i] = (float)rnd.NextDouble() * 8f - 4f; t[i] = (float)rnd.NextDouble() * 8f - 4f; }
        for (int i = 0; i < 4; i++) r[i] = (float)rnd.NextDouble() * 8f - 4f;
        x[0] = System.BitConverter.Int32BitsToSingle(0x7FC00001);
        x[1] = float.PositiveInfinity;
        x[2] = -0f;
        r[0] = System.BitConverter.Int32BitsToSingle(0x7FC00002);
        t[0] = float.NegativeInfinity;
        return new System.Collections.Generic.Dictionary<string, ITensor>
        {
            ["x"] = new DenseTensor<float>(new System.Memory<float>(x), new int[] { 2, 4 }),
            ["r"] = new DenseTensor<float>(new System.Memory<float>(r), new int[] { 4 }),
            ["t"] = new DenseTensor<float>(new System.Memory<float>(t), new int[] { 2, 4 }),
        };
    }

    [Fact]
    public void ScaleFusion_BitwiseTwinWithExceptionalValues()
    {
        var fused = Model.Load(ScaleFusionModel())!;
        var plain = Model.Load(ScaleFusionModel(), runOptimizer: false)!;
        var feed = ScaleFeed(99);
        Assert.True(fused.Execute(feed, true), fused.LastErrorMessage);
        Assert.True(plain.Execute(feed, true), plain.LastErrorMessage);
        foreach (var name in new[] { "y1", "y2", "y3", "y4" })
        {
            var f = ((Tensor<float>)fused.Outputs[name]!).ToDenseTensor().Buffer.ToArray();
            var p = ((Tensor<float>)plain.Outputs[name]!).ToDenseTensor().Buffer.ToArray();
            Assert.Equal(p.Length, f.Length);
            Assert.True(p.AsSpan().SequenceEqual(f.AsSpan()), name + " diverged");
        }
    }

    static OnnxModel SequenceChainModel()
    {
        // Mirrors the DINOv3 Q path: SplitToSequence -> SequenceAt + RoPE-less
        // concat -> scalar Mul -> MatMul. The Mul dtype must prove through
        // the sequence chain or the fusion starves (AddRelu lesson).
        var mp = new OnnxModel { Name = "tiny-seqchain" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2, 8 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Initializers.Add(new OnnxTensor { Name = "s", ElementType = TensorElementType.Float, Dims = new int[0], Data = new float[] { 0.5f } });
        mp.Initializers.Add(new OnnxTensor { Name = "w", ElementType = TensorElementType.Float, Dims = new[] { 8, 3 }, Data = new float[24] });
        mp.Initializers.Add(new OnnxTensor { Name = "idx", ElementType = TensorElementType.Int64, Dims = new[] { 1 }, Data = new long[] { 0 } });
        Dictionary<string, object> NoAttrs() => new Dictionary<string, object>();
        mp.Nodes.Add(new OnnxNode { OpType = "SplitToSequence", Inputs = new[] { "x" }, Outputs = new[] { "seq" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "SequenceAt", Inputs = new[] { "seq", "idx" }, Outputs = new[] { "piece" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { "piece", "s" }, Outputs = new[] { "m" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "m", "w" }, Outputs = new[] { "y" }, Attributes = NoAttrs() });
        return mp;
    }

    [Fact]
    public void SequenceChainDtypeProvesAndFuses()
    {
        var graph = Model.Load(SequenceChainModel(), runOptimizer: false)!;
        var report = Optimization.GraphOptimizer.Run(graph);
        Assert.Contains(report, c => c.Pass == "scalematmul");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.ScaledMatMul && n.Outputs.Length == 1 && n.Outputs[0] == "y");
    }

    static void AssertFusedMatchesLegacy(string name, int[] adims, int[] bdims, float s)
    {
        var a = Rand(adims, 21);
        var b = Rand(bdims, 22);
        var scale = DenseTensor<float>.Scalar(s);
        var legacy = CPUExecutionProvider.Mul(a, scale, null, null);
        Assert.Equal(OpStatus.Success, legacy.Status);
        var mm = CPUExecutionProvider.MatMul(legacy.Outputs![0]!, b, null, null);
        Assert.Equal(OpStatus.Success, mm.Status);
        var fused = CPUExecutionProvider.ScaledMatMul(a, b, scale, null, null);
        Assert.Equal(OpStatus.Success, fused.Status);
        var f = Out(fused.Outputs![0]!);
        var e = Out(mm.Outputs![0]!);
        Assert.Equal(e.Length, f.Length);
        Assert.True(e.AsSpan().SequenceEqual(f.AsSpan()), name + " diverged");
    }

    [Fact]
    public void ScaledMatMul_MatchesLegacyBitwise()
    {
        // 3x4-packed territory (DINO scores geometry) plus even-m packed,
        // rank-4 batched, tails and exceptional values via seeded data.
        AssertFusedMatchesLegacy("dino-3x4", new[] { 201, 64 }, new[] { 64, 201 }, 0.125f);
        AssertFusedMatchesLegacy("even-2x4", new[] { 200, 64 }, new[] { 64, 200 }, 0.125f);
        AssertFusedMatchesLegacy("batch4d", new[] { 1, 6, 12, 32 }, new[] { 1, 6, 32, 40 }, -2.5f);
        AssertFusedMatchesLegacy("tail-k", new[] { 66, 33 }, new[] { 33, 45 }, 1e-10f);
    }

    [Fact]
    public void ScaledMatMul_MissingScaleIsNotSuccess()
    {
        var a = Rand(new[] { 4, 8 }, 7);
        var b = Rand(new[] { 8, 4 }, 11);
        var r = CPUExecutionProvider.ScaledMatMul(a, b, null, null, null);
        Assert.NotEqual(OpStatus.Success, r.Status);
    }

    [Fact]
    public void TrailingDivCompositeMatchesLegacyBitwise()
    {
        // Odd-m shapes decline the alpha twin and run the identical
        // composite (same ops, same order): must agree bit-for-bit.
        foreach (var d in new[] { 2.0f, 0.1f, 5.656854f })
        {
            var a = Rand(new[] { 31, 32 }, 31);
            var b = Rand(new[] { 32, 27 }, 32);
            var ab = a.Buffer.ToArray();
            ab[0] = System.BitConverter.Int32BitsToSingle(0x7FC00001);
            ab[1] = float.PositiveInfinity;
            ab[2] = -0f;
            var ax = new DenseTensor<float>(ab.AsMemory(), new[] { 31, 32 });
            var div = DenseTensor<float>.Scalar(d);
            var mm = CPUExecutionProvider.MatMul(ax, b, null, null);
            Assert.Equal(OpStatus.Success, mm.Status);
            var legacy = CPUExecutionProvider.Div(mm.Outputs![0]!, div, null, null);
            Assert.Equal(OpStatus.Success, legacy.Status);
            var fused = CPUExecutionProvider.ScaledMatMulTrailing(ax, b, div, null, null);
            Assert.Equal(OpStatus.Success, fused.Status);
            var f = Out(fused.Outputs![0]!);
            var e = Out(legacy.Outputs![0]!);
            Assert.Equal(e.Length, f.Length);
            Assert.True(e.AsSpan().SequenceEqual(f.AsSpan()), "d=" + d + " diverged");
        }
    }

    static OnnxModel TrailingDivModel()
    {
        // Mirrors the E5 attention tail: MatMul -> Div(const) -> mask-Add.
        var mp = new OnnxModel { Name = "tiny-trailingdiv" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 4, 6 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "b", ElementType = TensorElementType.Float, Dims = new[] { 4, 5 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "r", ElementType = TensorElementType.Float, Dims = new[] { 1 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y1", ElementType = TensorElementType.Float, Dims = new[] { 4, 5 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y2", ElementType = TensorElementType.Float, Dims = new[] { 4, 5 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y3", ElementType = TensorElementType.Float, Dims = new[] { 4, 5 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y4", ElementType = TensorElementType.Float, Dims = new[] { 4, 5 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y5", ElementType = TensorElementType.Float, Dims = new[] { 4, 5 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y6", ElementType = TensorElementType.Float, Dims = new[] { 4, 5 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "t3", ElementType = TensorElementType.Float, Dims = new[] { 4, 5 } });
        mp.Initializers.Add(new OnnxTensor { Name = "s", ElementType = TensorElementType.Float, Dims = new int[0], Data = new float[] { 2.0f } });
        mp.Initializers.Add(new OnnxTensor { Name = "w", ElementType = TensorElementType.Float, Dims = new[] { 6, 5 }, Data = new float[30] });
        Dictionary<string, object> NoAttrs() => new Dictionary<string, object>();
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "x", "w" }, Outputs = new[] { "t1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Div", Inputs = new[] { "t1", "s" }, Outputs = new[] { "d1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "d1", "b" }, Outputs = new[] { "y1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "x", "w" }, Outputs = new[] { "t2" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Div", Inputs = new[] { "t2", "s" }, Outputs = new[] { "d2" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "d2", "b" }, Outputs = new[] { "y2" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "t2", "b" }, Outputs = new[] { "y3" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "x", "w" }, Outputs = new[] { "t3" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Div", Inputs = new[] { "t3", "s" }, Outputs = new[] { "d3" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "d3", "b" }, Outputs = new[] { "y4" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "x", "w" }, Outputs = new[] { "t5" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Div", Inputs = new[] { "t5", "r" }, Outputs = new[] { "d5" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "d5", "b" }, Outputs = new[] { "y5" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "x", "w" }, Outputs = new[] { "t6" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Div", Inputs = new[] { "s", "t6" }, Outputs = new[] { "d6" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "d6", "b" }, Outputs = new[] { "y6" }, Attributes = NoAttrs() });
        return mp;
    }

    [Fact]
    public void TrailingDivRegionsFuseDecoysStay()
    {
        var graph = Model.Load(TrailingDivModel(), runOptimizer: false)!;
        var report = Optimization.GraphOptimizer.Run(graph);
        Assert.Contains(report, c => c.Pass == "scalematmul");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.ScaledMatMul && n.Outputs.Length == 1 && n.Outputs[0] == "d1");
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Div && n.Outputs.Length == 1 && n.Outputs[0] == "d1");
        // Second consumer on the MatMul->Div link keeps the two-node form.
        Assert.Contains(graph.Nodes, n => n.Op == OpType.MatMul && n.Outputs.Length == 1 && n.Outputs[0] == "t2");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Div && n.Outputs.Length == 1 && n.Outputs[0] == "d2");
        // Graph-output exposure on the link keeps the two-node form.
        Assert.Contains(graph.Nodes, n => n.Op == OpType.MatMul && n.Outputs.Length == 1 && n.Outputs[0] == "t3");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Div && n.Outputs.Length == 1 && n.Outputs[0] == "d3");
        // Runtime (non-singleton) divisor keeps the two-node form.
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Div && n.Outputs.Length == 1 && n.Outputs[0] == "d5");
        // Divisor at index 0 is not the ORT scale shape.
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Div && n.Outputs.Length == 1 && n.Outputs[0] == "d6");
    }

    static double MaxScaledDiff(float[] got, float[] want)
    {
        // NaN positions must be preserved exactly (payloads may differ:
        // alpha*NaN vs NaN/d take different propagation paths); bitwise-equal
        // elements (signed zeros, same-signed infinities, lucky finites)
        // contribute zero; everything else prices against 1+|want|.
        double worst = 0;
        for (int i = 0; i < got.Length; i++)
        {
            if (float.IsNaN(want[i]))
            {
                if (!float.IsNaN(got[i])) return double.NaN;
                continue;
            }
            if (got[i] == want[i]) continue;
            double denom = 1.0 + System.Math.Abs((double)want[i]);
            double dd = System.Math.Abs((double)got[i] - want[i]) / denom;
            if (double.IsNaN(dd) || !(dd <= worst)) { if (double.IsNaN(dd)) return double.NaN; worst = dd; }
        }
        return worst;
    }

    static float[] TrailingLegacy(Tensor<float> a, Tensor<float> b, float d)
    {
        var mm = CPUExecutionProvider.MatMul(a, b, null, null);
        Assert.Equal(OpStatus.Success, mm.Status);
        var div = CPUExecutionProvider.Div(mm.Outputs![0]!, DenseTensor<float>.Scalar(d), null, null);
        Assert.Equal(OpStatus.Success, div.Status);
        return Out(div.Outputs![0]!);
    }

    static float[] TrailingFused(Tensor<float> a, Tensor<float> b, float d)
    {
        var fused = CPUExecutionProvider.ScaledMatMulTrailing(a, b, DenseTensor<float>.Scalar(d), null, null);
        Assert.Equal(OpStatus.Success, fused.Status);
        return Out(fused.Outputs![0]!);
    }

    [Fact]
    public void TrailingAlpha_EnvelopeWithinGate()
    {
        // E5-1 M2 envelope: alpha-epilogue vs Div-then-MatMul over QK shapes
        // and divisor systematics. Inexact divisors must differ (proves the
        // twin ran, not the composite) while staying deep inside 1e-4.
        var cases = new (int[] ad, int[] bd, float d)[]
        {
            (new[] { 30, 32 }, new[] { 32, 30 }, 5.656854f),
            (new[] { 8, 32 }, new[] { 32, 8 }, 0.1f),
            (new[] { 30, 32 }, new[] { 32, 30 }, 1e-10f),
            (new[] { 30, 32 }, new[] { 32, 30 }, 1e10f),
            (new[] { 8, 32 }, new[] { 32, 8 }, -2.5f),
            (new[] { 30, 35 }, new[] { 35, 27 }, 3.1415927f),
        };
        int seed = 101;
        foreach (var (ad, bd, d) in cases)
        {
            var a = Rand(ad, seed++);
            var b = Rand(bd, seed++);
            var ab = a.Buffer.ToArray();
            ab[0] = System.BitConverter.Int32BitsToSingle(0x7FC00001);
            ab[1] = float.PositiveInfinity;
            var ax = new DenseTensor<float>(ab.AsMemory(), ad);
            var e = TrailingLegacy(ax, b, d);
            var f = TrailingFused(ax, b, d);
            Assert.Equal(e.Length, f.Length);
            double worst = MaxScaledDiff(f, e);
            Assert.True(worst <= 1e-4, "d=" + d + " envelope " + worst);
            Assert.True(worst > 0, "d=" + d + " suspiciously bitwise (twin bypassed?)");
        }
    }

    [Fact]
    public void TrailingAlpha_UnitDivisorSkipsDivBitwise()
    {
        var a = Rand(new[] { 30, 32 }, 111);
        var b = Rand(new[] { 32, 30 }, 112);
        var mm = CPUExecutionProvider.MatMul(a, b, null, null);
        Assert.Equal(OpStatus.Success, mm.Status);
        var f = TrailingFused(a, b, 1f);
        var e = Out(mm.Outputs![0]!);
        Assert.True(e.AsSpan().SequenceEqual(f.AsSpan()), "div-by-1 must skip the Div bitwise");
    }

    [Fact]
    public void TrailingAlpha_DegenerateDivisorsStayExact()
    {
        // Zero/NaN/Inf divisors route the identical composite: IEEE edge
        // behavior (Inf, NaN payloads, signed zeros) preserved exactly.
        foreach (var d in new[] { 0f, -0f, float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        {
            var a = Rand(new[] { 8, 32 }, 121);
            var b = Rand(new[] { 32, 8 }, 122);
            var e = TrailingLegacy(a, b, d);
            var f = TrailingFused(a, b, d);
            Assert.True(e.AsSpan().SequenceEqual(f.AsSpan()), "d=" + d + " must stay bitwise via composite");
        }
    }

    [Fact]
    public void TrailingDiv_BitwiseTwinWithExceptionalValues()
    {
        var fused = Model.Load(TrailingDivModel())!;
        var plain = Model.Load(TrailingDivModel(), runOptimizer: false)!;
        var rnd = new System.Random(77);
        var x = new float[24];
        var b = new float[20];
        for (int i = 0; i < 24; i++) x[i] = (float)rnd.NextDouble() * 8f - 4f;
        for (int i = 0; i < 20; i++) b[i] = (float)rnd.NextDouble() * 8f - 4f;
        x[0] = System.BitConverter.Int32BitsToSingle(0x7FC00001);
        x[1] = float.PositiveInfinity;
        x[2] = -0f;
        var feed = new System.Collections.Generic.Dictionary<string, ITensor>
        {
            ["x"] = new DenseTensor<float>(x.AsMemory(), new int[] { 4, 6 }),
            ["b"] = new DenseTensor<float>(b.AsMemory(), new int[] { 4, 5 }),
            ["r"] = new DenseTensor<float>(new float[] { 2.0f }.AsMemory(), new int[] { 1 }),
        };
        Assert.True(fused.Execute(feed, true), fused.LastErrorMessage);
        Assert.True(plain.Execute(feed, true), plain.LastErrorMessage);
        foreach (var name in new[] { "y1", "y2", "y3", "y4", "y5", "y6", "t3" })
        {
            var f = ((Tensor<float>)fused.Outputs[name]!).ToDenseTensor().Buffer.ToArray();
            var p = ((Tensor<float>)plain.Outputs[name]!).ToDenseTensor().Buffer.ToArray();
            Assert.Equal(p.Length, f.Length);
            Assert.True(p.AsSpan().SequenceEqual(f.AsSpan()), name + " diverged");
        }
    }
}
