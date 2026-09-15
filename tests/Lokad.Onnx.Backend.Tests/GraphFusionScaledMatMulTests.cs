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

    [Fact]
    public void ScaledMatMul_MissingScaleIsNotSuccess()
    {
        var a = Rand(new[] { 4, 8 }, 7);
        var b = Rand(new[] { 8, 4 }, 11);
        var r = CPUExecutionProvider.ScaledMatMul(a, b, null, null, null);
        Assert.NotEqual(OpStatus.Success, r.Status);
    }
}
