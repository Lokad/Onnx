using System.Linq;
using System.Runtime.InteropServices;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

public class GraphBufferReuseTests
{
    static float[] ToArray(ITensor tensor) => ((Tensor<float>)tensor).ToArray();

    static void AssertBitwise(float[] expected, ITensor actual)
    {
        Assert.Equal(expected, ToArray(actual));
    }

    [Fact]
    public void Pool_RentsNew_ReusesReturned_AndCapsBuffered()
    {
        var pool = new TensorBufferPool();
        var a = pool.Rent<float>(4);
        Assert.Equal(1, pool.AllocatedNew);
        Assert.Equal(0, pool.Reused);
        pool.Return(a);
        Assert.Equal(1, pool.Returned);
        var b = pool.Rent<float>(4);
        Assert.Equal(1, pool.Reused);
        Assert.Same(a, b);
        pool.Rent<float>(5);
        Assert.Equal(2, pool.AllocatedNew);
        for (int i = 0; i < 40; i++) pool.Return(new float[8]);
        Assert.Equal(33, pool.Returned);
        Assert.Equal(8, pool.Dropped);
    }

    [Fact]
    public void PoolPeakOutstanding_TracksHighWater()
    {
        var pool = new TensorBufferPool();
        Assert.Equal(0, pool.PeakOutstandingBytes);
        var a = pool.Rent<byte>(100);
        var b = pool.Rent<byte>(200);
        Assert.Equal(300, pool.PeakOutstandingBytes);
        pool.Return(a);
        var c = pool.Rent<byte>(50);
        Assert.Equal(300, pool.PeakOutstandingBytes);
        pool.Return(b);
        pool.Return(c);
        var d = pool.Rent<byte>(100);
        Assert.Same(a, d);
        Assert.Equal(300, pool.PeakOutstandingBytes);
        pool.Return(new byte[1000]);
        Assert.Equal(300, pool.PeakOutstandingBytes);
        pool.Return(d);
    }
    [Fact]
    public void PooledAddMulDiv_MatchAllocating_Bitwise()
    {
        var options = ExecutionOptions.Default;
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var b = DenseTensor<float>.OfValues(new float[] { 10f, 20f });
        foreach (var op in new[] { OpType.Add, OpType.Mul, OpType.Div })
        {
            var pool = new TensorBufferPool();
            OpResult legacy = op switch
            {
                OpType.Add => CPU.Add(a, b, options, null),
                OpType.Mul => CPU.Mul(a, b, options, null),
                _ => CPU.Div(a, b, options, null),
            };
            OpResult pooled = op switch
            {
                OpType.Add => CPU.Add(a, b, options, pool),
                OpType.Mul => CPU.Mul(a, b, options, pool),
                _ => CPU.Div(a, b, options, pool),
            };
            Assert.Equal(OpStatus.Success, legacy.Status);
            Assert.Equal(OpStatus.Success, pooled.Status);
            AssertBitwise(ToArray(legacy.Outputs[0]), pooled.Outputs[0]);
            Assert.Equal(1, pool.AllocatedNew);
        }
    }

    [Fact]
    public void PooledMatMul_MatchesAllocating_Bitwise()
    {
        var options = ExecutionOptions.Default;
        var twoD_a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f }, { 7f, 8f, 9f }, { 10f, 11f, 12f }, { 13f, 14f, 15f } });
        var twoD_b = DenseTensor<float>.OfValues(new float[,] { { 1f, 0f }, { 0f, 1f }, { 1f, 1f } });
        var batched_a = DenseTensor<float>.OfShape(2, 3, 4);
        batched_a.Fill(0.5f);
        var batched_b = DenseTensor<float>.OfShape(2, 4, 5);
        batched_b.Fill(-0.25f);
        var square_a = DenseTensor<float>.OfShape(3, 3);
        square_a.Fill(0.5f);
        var mixed_b = DenseTensor<float>.OfShape(3);
        mixed_b.Fill(2f);
        var vector_a = DenseTensor<float>.OfValues(new float[] { 7f });
        var vector_b = DenseTensor<float>.OfValues(new float[] { 3f });
        var cases = new[]
        {
            (twoD_a, twoD_b),
            (batched_a, batched_b),
            (square_a, mixed_b),
            (vector_a, vector_b),
        };
        foreach (var (x, y) in cases)
        {
            var pool = new TensorBufferPool();
            var legacy = CPU.MatMul(x, y, options, null);
            var pooled = CPU.MatMul(x, y, options, pool);
            Assert.Equal(OpStatus.Success, legacy.Status);
            Assert.Equal(OpStatus.Success, pooled.Status);
            AssertBitwise(ToArray(legacy.Outputs[0]), pooled.Outputs[0]);
            Assert.Equal(1, pool.AllocatedNew);
        }
    }

    [Fact]
    public void PooledMatMul_OnDirtyBuffer_MatchesAllocating()
    {
        var options = ExecutionOptions.Default;
        var pool = new TensorBufferPool();
        var a = DenseTensor<float>.OfShape(4, 32);
        a.Fill(0.5f);
        var b = DenseTensor<float>.OfShape(32, 32);
        b.Fill(-0.25f);
        var legacy = CPU.MatMul(a, b, options, null);
        Assert.Equal(OpStatus.Success, legacy.Status);
        var dirtied = 0;
        for (int i = 0; i < 4; i++)
        {
            var r = CPU.MatMul(a, b, options, pool);
            Assert.Equal(OpStatus.Success, r.Status);
            AssertBitwise(ToArray(legacy.Outputs[0]), r.Outputs[0]);
            var buffer = ((DenseTensor<float>)r.Outputs[0]).Buffer;
            Assert.True(MemoryMarshal.TryGetArray<float>(buffer, out var segment) && segment.Array is not null);
            for (int k = 0; k < segment.Array.Length; k++) segment.Array[k] = 1234.5f;
            dirtied++;
            pool.Return(segment.Array!);
        }
        Assert.Equal(4, dirtied);
        Assert.True(pool.Reused >= 3);
    }

    [Fact]
    public void PooledSoftmaxErfTransposeLayerNorm_MatchAllocating_Bitwise()
    {
        var options = ExecutionOptions.Default;
        var pool = new TensorBufferPool();
        var logits = DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f }, { 1f, 1f, 1f } });
        var legacySm = CPU.Softmax(logits, 1, null, null, 13);
        var pooledSm = CPU.Softmax(logits, 1, options, pool, 13);
        Assert.Equal(OpStatus.Success, legacySm.Status);
        Assert.Equal(OpStatus.Success, pooledSm.Status);
        AssertBitwise(ToArray(legacySm.Outputs[0]), pooledSm.Outputs[0]);
        var erfIn = DenseTensor<float>.OfValues(new float[] { 0f, 0.5f, 1f });
        var legacyErf = CPU.Erf(erfIn, options, null);
        var pooledErf = CPU.Erf(erfIn, options, pool);
        Assert.Equal(OpStatus.Success, legacyErf.Status);
        Assert.Equal(OpStatus.Success, pooledErf.Status);
        AssertBitwise(ToArray(legacyErf.Outputs[0]), pooledErf.Outputs[0]);
        var square = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var legacyTr = CPU.Transpose(square, new[] { 1, 0 }, options, null);
        var pooledTr = CPU.Transpose(square, new[] { 1, 0 }, options, pool);
        Assert.Equal(OpStatus.Success, legacyTr.Status);
        Assert.Equal(OpStatus.Success, pooledTr.Status);
        AssertBitwise(ToArray(legacyTr.Outputs[0]), pooledTr.Outputs[0]);
        var rect = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        var legacyRect = CPU.Transpose(rect, new[] { 1, 0 }, options, null);
        var pooledRect = CPU.Transpose(rect, new[] { 1, 0 }, options, pool);
        Assert.Equal(OpStatus.Success, legacyRect.Status);
        Assert.Equal(OpStatus.Success, pooledRect.Status);
        AssertBitwise(ToArray(legacyRect.Outputs[0]), pooledRect.Outputs[0]);
        Assert.Equal(new[] { 3, 2 }, ((Tensor<float>)pooledRect.Outputs[0]).Dimensions.ToArray());
        var lnIn = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } });
        var gamma = DenseTensor<float>.OfValues(new float[] { 1f, 1f, 1f, 1f });
        var beta = DenseTensor<float>.OfValues(new float[] { 0f, 0f, 0f, 0f });
        var legacyLn = CPU.LayerNormalization(lnIn, gamma, beta, -1, 1e-5f, null, 1, options, null);
        var pooledLn = CPU.LayerNormalization(lnIn, gamma, beta, -1, 1e-5f, null, 1, options, pool);
        Assert.Equal(OpStatus.Success, legacyLn.Status);
        Assert.Equal(OpStatus.Success, pooledLn.Status);
        AssertBitwise(ToArray(legacyLn.Outputs[0]), pooledLn.Outputs[0]);
        Assert.True(pool.AllocatedNew >= 4);
    }

    [Fact]
    public void VectorizedErf_MatchesScalarReference()
    {
        var data = new float[1024];
        for (int i = 0; i < data.Length; i++) data[i] = -6f + 12f * i / (data.Length - 1);
        var x = DenseTensor<float>.OfValues(data);
        var expected = x.Apply(MathOps.Erf);
        foreach (var options in new[] { TensorExecutionOptions.Scalar, TensorExecutionOptions.Simd, TensorExecutionOptions.Intrinsics })
        {
            var actual = Tensor<float>.Erf(x, options);
            float worst = 0f;
            for (int i = 0; i < data.Length; i++) worst = System.Math.Max(worst, System.Math.Abs(expected[i] - actual[i]));
            Assert.True(worst <= 1e-6f, "erf worst drift " + worst + " in " + options);
        }
    }

    [Fact]
    public void VectorizedGelu_MatchesScalarReference_AndPools()
    {
        var data = new float[1024];
        for (int i = 0; i < data.Length; i++) data[i] = -6f + 12f * i / (data.Length - 1);
        var x = DenseTensor<float>.OfValues(data);
        var expected = x.Apply((float v) => 0.5f * v * (1f + MathOps.Erf(v * 0.7071067811865476f)));
        foreach (var options in new[] { TensorExecutionOptions.Scalar, TensorExecutionOptions.Simd, TensorExecutionOptions.Intrinsics })
        {
            var actual = Tensor<float>.Gelu(x, options);
            float worst = 0f;
            for (int i = 0; i < data.Length; i++) worst = System.Math.Max(worst, System.Math.Abs(expected[i] - actual[i]));
            Assert.True(worst <= 1e-6f, "gelu worst drift " + worst + " in " + options);
        }
        var pool = new TensorBufferPool();
        var execOptions = new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Simd);
        var pooled = CPU.Gelu(x, null, execOptions, pool);
        Assert.Equal(OpStatus.Success, pooled.Status);
        float pooledWorst = 0f;
        var pooledArr = (Tensor<float>)pooled.Outputs[0];
        for (int i = 0; i < data.Length; i++) pooledWorst = System.Math.Max(pooledWorst, System.Math.Abs(expected[i] - pooledArr[i]));
        Assert.True(pooledWorst <= 1e-6f, "pooled gelu worst drift " + pooledWorst);
        Assert.Equal(1, pool.AllocatedNew);
    }

    [Fact]
    public void PooledOutput_ReturnedToPool_IsReused()
    {
        var options = ExecutionOptions.Default;
        var pool = new TensorBufferPool();
        var a = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        var b = DenseTensor<float>.OfValues(new float[] { 3f, 4f });
        var first = CPU.Add(a, b, options, pool);
        Assert.Equal(OpStatus.Success, first.Status);
        Assert.True(MemoryMarshal.TryGetArray<float>(((DenseTensor<float>)first.Outputs[0]).Buffer, out var segment) && segment.Array is not null);
        pool.Return(segment.Array!);
        var second = CPU.Add(a, b, options, pool);
        Assert.Equal(OpStatus.Success, second.Status);
        Assert.Equal(1, pool.Reused);
        AssertBitwise(new float[] { 4f, 6f }, second.Outputs[0]);
    }

    [Fact]
    public void ExpandView_PinsBaseBuffer_AcrossReuse()
    {
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "test";
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[] { 1f, 2f, 3f, 4f });
        graph.Inputs["w"] = DenseTensor<float>.OfValues(new float[] { 10f, 20f, 30f, 40f });
        graph.Inputs["u"] = DenseTensor<float>.OfValues(new float[] { 2f, 2f, 2f, 2f });
        graph.Inputs["u2"] = DenseTensor<float>.OfValues(new float[] { 3f, 3f, 3f, 3f });
        graph.Initializers["eshape"] = DenseTensor<long>.OfValues(new long[] { 3, 4 });
        graph.Outputs["z"] = DenseTensor<float>.OfShape(3, 4);
        graph.Nodes.Add(new Node { Name = "add", Op = OpType.Add, Inputs = new[] { "x", "w" }, Outputs = new[] { "t" } });
        graph.Nodes.Add(new Node { Name = "expand", Op = OpType.Expand, Inputs = new[] { "t", "eshape" }, Outputs = new[] { "v" } });
        graph.Nodes.Add(new Node { Name = "mul", Op = OpType.Mul, Inputs = new[] { "u", "u2" }, Outputs = new[] { "s" } });
        graph.Nodes.Add(new Node { Name = "add2", Op = OpType.Add, Inputs = new[] { "v", "s" }, Outputs = new[] { "z" } });
        graph.IntermediateOutputs["t"] = null;
        graph.IntermediateOutputs["v"] = null;
        graph.IntermediateOutputs["s"] = null;
        graph.RefreshLifetimeAnalysis();
        var inputs = new System.Collections.Generic.Dictionary<string, ITensor>
        {
            { "x", graph.Inputs["x"] },
            { "w", graph.Inputs["w"] },
            { "u", graph.Inputs["u"] },
            { "u2", graph.Inputs["u2"] },
        };
        Assert.True(graph.Execute(inputs, true));
        var row = new float[] { 17f, 28f, 39f, 50f };
        AssertBitwise(row.Concat(row).Concat(row).ToArray(), graph.Outputs["z"]);
        AssertBitwise(new float[] { 11f, 22f, 33f, 44f }, graph.IntermediateOutputs["t"]!);
    }

    [Fact]
    public void ContextReuse_CallerHeldOutputStaysValid()
    {
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "test";
        graph.Inputs["x"] = DenseTensor<float>.OfShape(2);
        graph.Outputs["y"] = DenseTensor<float>.OfShape(2);
        graph.Nodes.Add(new Node { Name = "r", Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "y" } });
        graph.RefreshLifetimeAnalysis();
        var ctx = graph.CreateExecution(null);
        var first = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) } };
        Assert.True(ctx.Execute(first, true));
        var held = ((Tensor<float>)ctx.Outputs["y"]).ToArray();
        var second = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 3f, -4f }) } };
        Assert.True(ctx.Execute(second, true));
        Assert.Equal(new float[] { 0f, 2f }, held);
        Assert.Equal(new float[] { 3f, 0f }, ((Tensor<float>)ctx.Outputs["y"]).ToArray());
    }

    [Fact]
    public void ContextReuse_AlternatingShapes()
    {
        var m = new OnnxModel { Name = "m" };
        m.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { -1 }, DimParams = new string?[] { null } });
        m.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { -1 }, DimParams = new string?[] { null } });
        m.Nodes.Add(new OnnxNode { Name = "r", OpType = "Relu", Inputs = new[] { "x" }, Outputs = new[] { "y" } });
        var graph = Model.Load(m);
        var ctx = graph.CreateExecution(null);
        foreach (int n in new int[] { 2, 5, 2 })
        {
            var data = new float[n];
            for (int i = 0; i < n; i++) data[i] = (float)i - 1f;
            var user = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(data) } };
            Assert.True(ctx.Execute(user, true));
            var y = ((Tensor<float>)ctx.Outputs["y"]).ToArray();
            Assert.Equal(n, y.Length);
            for (int i = 0; i < n; i++) Assert.Equal(Math.Max(0f, data[i]), y[i], 5);
        }
    }

    [Fact]
    public async Task ContextReuse_ConcurrentContexts()
    {
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "test";
        graph.Inputs["x"] = DenseTensor<float>.OfShape(2);
        graph.Outputs["y"] = DenseTensor<float>.OfShape(2);
        graph.Nodes.Add(new Node { Name = "r", Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "y" } });
        graph.RefreshLifetimeAnalysis();
        var execA = graph.CreateExecution(null);
        var execB = graph.CreateExecution(null);
        using var barrier = new Barrier(2);
        System.Func<Dictionary<string, ITensor>, GraphExecution, float[]> run = (user, exec) =>
        {
            barrier.SignalAndWait();
            if (!exec.Execute(user, true)) return Array.Empty<float>();
            return ((Tensor<float>)exec.Outputs["y"]).ToArray();
        };
        var userA = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) } };
        var userB = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 3f, -4f }) } };
        var tA = Task.Run(() => run(userA, execA));
        var tB = Task.Run(() => run(userB, execB));
        await Task.WhenAll(tA, tB);
        Assert.Equal(new float[] { 0f, 2f }, await tA);
        Assert.Equal(new float[] { 3f, 0f }, await tB);
    }

    [Fact]
    public void DoubleExecute_WithPooling_IsStable()
    {
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "test";
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        graph.Inputs["y"] = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 7f, 8f } });
        graph.Outputs["z"] = DenseTensor<float>.OfShape(2, 2);
        graph.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, Inputs = new[] { "x", "y" }, Outputs = new[] { "m" } });
        graph.Nodes.Add(new Node { Name = "add", Op = OpType.Add, Inputs = new[] { "m", "x" }, Outputs = new[] { "z" } });
        graph.RefreshLifetimeAnalysis();
        var inputs = new System.Collections.Generic.Dictionary<string, ITensor>
        {
            { "x", graph.Inputs["x"] },
            { "y", graph.Inputs["y"] },
        };
        Assert.True(graph.Execute(inputs, true));
        var first = ToArray(graph.Outputs["z"]);
        Assert.True(graph.Execute(inputs, true));
        AssertBitwise(first, graph.Outputs["z"]);
    }
}
