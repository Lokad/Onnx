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
        Assert.Equal(41, pool.Returned);
        Assert.Equal(0, pool.Dropped);
        for (int i = 0; i < 480; i++) pool.Return(new float[8]);
        Assert.Equal(513, pool.Returned);
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
    static ComputationalGraph MatMulAddGraph()
    {
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "pool-retain";
        graph.Inputs["a"] = DenseTensor<float>.OfShape(64, 64);
        graph.Inputs["b"] = DenseTensor<float>.OfShape(64, 64);
        graph.Inputs["c"] = DenseTensor<float>.OfShape(64, 64);
        graph.Outputs["y"] = DenseTensor<float>.OfShape(64, 64);
        graph.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, Inputs = new[] { "a", "b" }, Outputs = new[] { "t" } });
        graph.Nodes.Add(new Node { Name = "add", Op = OpType.Add, Inputs = new[] { "t", "c" }, Outputs = new[] { "y" } });
        graph.IntermediateOutputs["t"] = null;
        graph.RefreshLifetimeAnalysis();
        return graph;
    }

    static Dictionary<string, ITensor> MatMulAddFeeds(float a, float b, float c)
    {
        var fa = new float[64 * 64];
        var fb = new float[64 * 64];
        var fc = new float[64 * 64];
        for (int i = 0; i < fa.Length; i++) { fa[i] = a; fb[i] = b; fc[i] = c; }
        return new Dictionary<string, ITensor>
        {
            ["a"] = new DenseTensor<float>(new Memory<float>(fa), new[] { 64, 64 }),
            ["b"] = new DenseTensor<float>(new Memory<float>(fb), new[] { 64, 64 }),
            ["c"] = new DenseTensor<float>(new Memory<float>(fc), new[] { 64, 64 }),
        };
    }

    [Fact]
    public void PoolRetainedAcrossRuns_ReusesStorage()
    {
        var graph = MatMulAddGraph();
        var ctx = graph.CreateExecution(null);
        Assert.True(ctx.Execute(MatMulAddFeeds(1f, 2f, 3f), true));
        Assert.Equal(131f, ToArray(ctx.Outputs["y"])[0]);
        long firstNew = ctx.LastPoolAllocatedNewBytes;
        Assert.True(firstNew >= 2L * 64 * 64 * 4, "first run must allocate t and y, saw " + firstNew);
        Assert.True(ctx.Execute(MatMulAddFeeds(1f, 2f, 3f), true));
        Assert.Equal(131f, ToArray(ctx.Outputs["y"])[0]);
        Assert.True(ctx.LastPoolReused > 0, "second run must reuse retained storage.");
        Assert.True(ctx.LastPoolAllocatedNewBytes < firstNew, "second run must allocate less: " + ctx.LastPoolAllocatedNewBytes + " vs " + firstNew);
    }

    [Fact]
    public void RetainedOutput_HeldAcrossRuns_StaysValid()
    {
        var graph = MatMulAddGraph();
        var ctx = graph.CreateExecution(null);
        Assert.True(ctx.Execute(MatMulAddFeeds(1f, 2f, 3f), true));
        var held = (Tensor<float>)ctx.Outputs["y"];
        Assert.Equal(131f, held.ToArray()[0]);
        Assert.True(ctx.Execute(MatMulAddFeeds(2f, 1f, 0f), true));
        Assert.Equal(131f, held.ToArray()[0]);
        Assert.Equal(128f, ToArray(ctx.Outputs["y"])[0]);
    }

    [Fact]
    public void PoolRetention_SurvivesReset()
    {
        var graph = MatMulAddGraph();
        var ctx = graph.CreateExecution(null);
        Assert.True(ctx.Execute(MatMulAddFeeds(1f, 2f, 3f), true));
        ctx.Reset();
        Assert.True(ctx.Execute(MatMulAddFeeds(1f, 2f, 3f), true));
        Assert.Equal(131f, ToArray(ctx.Outputs["y"])[0]);
        Assert.True(ctx.LastPoolReused > 0, "retained pool must survive Reset.");
    }


    [Fact]
    public void PublicExecute_ReusesSharedPoolAcrossRuns()
    {
        var graph = MatMulAddGraph();
        var first = MatMulAddFeeds(1f, 2f, 3f);
        Assert.True(graph.Execute(first, true));
        Assert.Equal(131f, ToArray(graph.Outputs["y"])[0]);
        long firstNew = graph.LastPoolAllocatedNewBytes;
        Assert.True(firstNew >= 2L * 64 * 64 * 4, "first run must allocate t and y, saw " + firstNew);
        var second = MatMulAddFeeds(1f, 2f, 3f);
        Assert.True(graph.Execute(second, true));
        Assert.Equal(131f, ToArray(graph.Outputs["y"])[0]);
        Assert.True(graph.LastPoolReused > 0, "second public run must reuse shared storage.");
        Assert.True(graph.LastPoolAllocatedNewBytes < firstNew, "second run must allocate less: " + graph.LastPoolAllocatedNewBytes + " vs " + firstNew);
    }

    [Fact]
    public void ConcurrentContexts_ExecuteCorrectly()
    {
        var graph = MatMulAddGraph();
        var exceptions = new System.Collections.Concurrent.ConcurrentQueue<Exception>();
        System.Threading.Tasks.Parallel.For(0, 8, i =>
        {
            try
            {
                var ctx = graph.CreateExecution(null);
                float v = 1f + (i % 3);
                var feeds = MatMulAddFeeds(v, 2f, 3f);
                if (!ctx.Execute(feeds, true)) throw new InvalidOperationException(ctx.LastErrorMessage ?? "execute failed");
                float want = 64f * v * 2f + 3f;
                float got = ToArray(ctx.Outputs["y"])[0];
                if (System.Math.Abs(got - want) > 1e-3f) throw new InvalidOperationException("got " + got + " want " + want);
            }
            catch (Exception ex)
            {
                exceptions.Enqueue(ex);
            }
        });
        Assert.True(exceptions.IsEmpty, "concurrent runs failed: " + string.Join("; ", exceptions.Select(e => e.Message)));
    }


    [Fact]
    public void ResetRecyclesRunEndIntermediates()
    {
        // Final-node side outputs nobody consumes stay checked out without
        // end-of-run recycling: every run rents them fresh. After Reset they
        // must come back from the pool instead (no new misses on the steady run).
        const int H = 4, S = 2, K = 3;
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "recycle-test";
        var xr = new float[S * K];
        var wr = new float[4 * H * K];
        var rr = new float[4 * H * H];
        for (int i = 0; i < xr.Length; i++) xr[i] = (i - 2) * 0.25f;
        for (int i = 0; i < wr.Length; i++) wr[i] = ((i * 37) % 97 - 48) * 0.01f;
        for (int i = 0; i < rr.Length; i++) rr[i] = ((i * 53) % 89 - 44) * 0.01f;
        graph.Inputs["x"] = new DenseTensor<float>(xr, new[] { S, 1, K });
        graph.Initializers["w"] = new DenseTensor<float>(wr, new[] { 1, 4 * H, K });
        graph.Initializers["r"] = new DenseTensor<float>(rr, new[] { 1, 4 * H, H });
        graph.Outputs["y"] = DenseTensor<float>.OfShape(S, 1, 1, H);
        graph.IntermediateOutputs["yh"] = null;
        graph.IntermediateOutputs["yc"] = null;
        graph.Nodes.Add(new Node
        {
            Name = "lstm",
            Op = OpType.LSTM,
            OpTypeName = "LSTM",
            Domain = "",
            Inputs = new[] { "x", "w", "r" },
            Outputs = new[] { "y", "yh", "yc" },
            Attributes = new Dictionary<string, object> { ["hidden_size"] = H },
        });
        graph.RefreshLifetimeAnalysis();
        var feeds = new Dictionary<string, ITensor> { ["x"] = graph.Inputs["x"] };
        Assert.True(graph.Execute(feeds, true));
        var first = ((Tensor<float>)graph.Outputs["y"]).ToArray();
        graph.Reset();
        long m1 = MissedFor(graph, H);
        Assert.True(graph.Execute(feeds, true));
        Assert.Equal(first, ((Tensor<float>)graph.Outputs["y"]).ToArray());
        graph.Reset();
        Assert.Equal(m1, MissedFor(graph, H));
    }

    static long MissedFor(ComputationalGraph graph, int len)
    {
        long m = 0;
        foreach (var s in graph.PoolDemandSnapshot()) if (s.Type == "Single" && s.Length == len) m += s.Missed;
        return m;
    }

    [Fact]
    public void ResetPinAccounting_NamesAliasPinnedRemainder()
    {
        // The run-end sweep census attributes each owned run-end binding by
        // decision: storage pinned by a live view counts Pinned, unaliased
        // dead storage counts ReturnedAtReset. The dead final Add output is
        // owned and unaliased (no last-use entry, so no pass releases it).
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "pin-test";
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[] { 1f, 2f, 3f, 4f });
        graph.Inputs["w"] = DenseTensor<float>.OfValues(new float[] { 10f, 20f, 30f, 40f });
        graph.Inputs["u"] = DenseTensor<float>.OfValues(new float[] { 2f, 2f, 2f, 2f });
        graph.Inputs["u2"] = DenseTensor<float>.OfValues(new float[] { 3f, 3f, 3f, 3f });
        graph.Outputs["z"] = DenseTensor<float>.OfShape(4);
        graph.Nodes.Add(new Node { Name = "mul", Op = OpType.Mul, Inputs = new[] { "u", "u2" }, Outputs = new[] { "s" } });
        graph.Nodes.Add(new Node { Name = "add2", Op = OpType.Add, Inputs = new[] { "s", "w" }, Outputs = new[] { "z" } });
        graph.Nodes.Add(new Node { Name = "add3", Op = OpType.Add, Inputs = new[] { "x", "u" }, Outputs = new[] { "d2" } });
        graph.IntermediateOutputs["s"] = null;
        graph.IntermediateOutputs["d2"] = null;
        graph.RefreshLifetimeAnalysis();
        var inputs = new System.Collections.Generic.Dictionary<string, ITensor>
        {
            { "x", graph.Inputs["x"] },
            { "w", graph.Inputs["w"] },
            { "u", graph.Inputs["u"] },
            { "u2", graph.Inputs["u2"] },
        };
        Assert.True(graph.Execute(inputs, true));
        AssertBitwise(new float[] { 16f, 26f, 36f, 46f }, graph.Outputs["z"]);
        graph.Reset();
        var pin = graph.PoolPinSnapshot().Single(s => s.Type == "Single" && s.Length == 4);
        Assert.Equal(0, pin.Pinned);
        Assert.Equal(1, pin.ReturnedAtReset);
    }

    [Fact]
    public void ResetPinAccounting_IntermediateViewDoesNotProtectBase()
    {
        // Two-phase contract (W2): liveness among intermediates never
        // protects storage -- only escaping roots do. The hand-placed view
        // is itself reset-owned, so the base returns instead of pinning.
        // The view leaves no record; the base counts ReturnedAtReset.
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "pin-view-test";
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
        AssertBitwise(new float[] { 11f, 22f, 33f, 44f }, graph.IntermediateOutputs["t"]);
        var baseBuffer = ((DenseTensor<float>)graph.IntermediateOutputs["t"]).Buffer;
        Assert.True(MemoryMarshal.TryGetArray(baseBuffer, out ArraySegment<float> window) && window.Array is not null);
        graph.IntermediateOutputs["w"] = new DenseTensor<float>(new Memory<float>(window.Array, 1, 3), new[] { 3 });
        graph.Reset();
        var pin = graph.PoolPinSnapshot().Single(s => s.Type == "Single" && s.Length == 4);
        Assert.Equal(0, pin.Pinned);
        Assert.Equal(1, pin.ReturnedAtReset);
    }

    [Fact]
    public void PinCensus_CountsPinnedVsReturned()
    {
        // Pool-level census math behind PoolPinSnapshot, no engine involved.
        // Storage ids are stable per array and distinct across arrays.
        var pool = new TensorBufferPool();
        float[] a1 = pool.Rent<float>(4);
        float[] a2 = pool.Rent<float>(4);
        Assert.Equal(pool.StorageId(a1), pool.StorageId(a1));
        Assert.NotEqual(pool.StorageId(a1), pool.StorageId(a2));
        pool.BumpPin(typeof(float), 4, true, pool.StorageId(a1), "Intermediate");
        pool.BumpPin(typeof(float), 4, true, pool.StorageId(a1), "Intermediate");
        pool.BumpPin(typeof(float), 4, false, pool.StorageId(a2), "Returned");
        pool.BumpPin(typeof(byte), 8, false, 99, "Returned");
        var snap = pool.SnapshotPins();
        var f = snap.Single(s => s.Type == "Single" && s.Length == 4);
        Assert.Equal(2, f.Pinned);
        Assert.Equal(1, f.ReturnedAtReset);
        var b = snap.Single(s => s.Type == "Byte" && s.Length == 8);
        Assert.Equal(0, b.Pinned);
        Assert.Equal(1, b.ReturnedAtReset);
        var details = pool.SnapshotPinDetails();
        Assert.Equal(4, details.Length);
        Assert.Equal(2, details.Count(r => r.Pinned && r.RootReason == "Intermediate"));
        Assert.True(details.Where(r => r.Pinned).Select(r => r.StorageId).Distinct().Count() == 1);
        Assert.Contains(details, r => !r.Pinned && r.RootReason == "Returned");
    }

    [Fact]
    public void ResetPinDetails_ContextPathMatchesPublicPath()
    {
        // Two-phase agreement on both execution paths with a real engine
        // view: the dead trailing Expand view survives to Reset (final-node
        // outputs escape the release passes) but, being reset-owned itself,
        // does not protect its base -- the base returns on both paths with
        // identical decisions. Consumed views die guard-free mid-run.
        foreach (bool useContext in new[] { false, true })
        {
            var graph = new ComputationalGraph();
            graph.Metadata["Name"] = "pin-ctx-test";
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
            graph.Nodes.Add(new Node { Name = "expandLast", Op = OpType.Expand, Inputs = new[] { "t", "eshape" }, Outputs = new[] { "vd" } });
            graph.IntermediateOutputs["t"] = null;
            graph.IntermediateOutputs["v"] = null;
            graph.IntermediateOutputs["vd"] = null;
            graph.IntermediateOutputs["s"] = null;
            graph.RefreshLifetimeAnalysis();
            var inputs = new System.Collections.Generic.Dictionary<string, ITensor>
            {
                { "x", graph.Inputs["x"] },
                { "w", graph.Inputs["w"] },
                { "u", graph.Inputs["u"] },
                { "u2", graph.Inputs["u2"] },
            };
            var row = new float[] { 17f, 28f, 39f, 50f };
            ComputationalGraph runner = graph;
            if (useContext)
            {
                runner = graph.CreateExecution(null);
            }
            Assert.True(runner.Execute(inputs, true));
            AssertBitwise(row.Concat(row).Concat(row).ToArray(), runner.Outputs["z"]);
            runner.Reset();
            var details = runner.PoolPinDetailSnapshot();
            Assert.Empty(details.Where(r => r.Pinned));
            var ret = details.Where(r => !r.Pinned).ToArray();
            Assert.Single(ret);
            Assert.Equal("Returned", ret[0].RootReason);
            var totals = runner.PoolPinSnapshot().Single(s => s.Type == "Single" && s.Length == 4);
            Assert.Equal(0, totals.Pinned);
            Assert.Equal(1, totals.ReturnedAtReset);
        }
    }

    [Fact]
    public void ResetOutputViewProtectsItsBase()
    {
        // A graph output that views rented storage pins its base with root
        // reason Output: outputs are never recycled, so their storage must
        // never re-enter circulation while they reference it. The output
        // stays valid across Reset and reruns; pins accumulate deterministi-
        // cally one per run.
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "pin-output-test";
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[] { 1f, 2f, 3f, 4f });
        graph.Inputs["w"] = DenseTensor<float>.OfValues(new float[] { 10f, 20f, 30f, 40f });
        graph.Inputs["u"] = DenseTensor<float>.OfValues(new float[] { 2f, 2f, 2f, 2f });
        graph.Inputs["u2"] = DenseTensor<float>.OfValues(new float[] { 3f, 3f, 3f, 3f });
        graph.Initializers["eshape"] = DenseTensor<long>.OfValues(new long[] { 3, 4 });
        graph.Outputs["z"] = DenseTensor<float>.OfShape(4);
        graph.Outputs["vout"] = DenseTensor<float>.OfShape(3, 4);
        graph.Nodes.Add(new Node { Name = "add", Op = OpType.Add, Inputs = new[] { "x", "w" }, Outputs = new[] { "t" } });
        graph.Nodes.Add(new Node { Name = "expand", Op = OpType.Expand, Inputs = new[] { "t", "eshape" }, Outputs = new[] { "vout" } });
        graph.Nodes.Add(new Node { Name = "mul", Op = OpType.Mul, Inputs = new[] { "u", "u2" }, Outputs = new[] { "s" } });
        graph.Nodes.Add(new Node { Name = "add2", Op = OpType.Add, Inputs = new[] { "s", "w" }, Outputs = new[] { "z" } });
        graph.IntermediateOutputs["t"] = null;
        graph.IntermediateOutputs["s"] = null;
        graph.RefreshLifetimeAnalysis();
        var inputs = new System.Collections.Generic.Dictionary<string, ITensor>
        {
            { "x", graph.Inputs["x"] },
            { "w", graph.Inputs["w"] },
            { "u", graph.Inputs["u"] },
            { "u2", graph.Inputs["u2"] },
        };
        var row = new float[] { 11f, 22f, 33f, 44f };
        Assert.True(graph.Execute(inputs, true));
        AssertBitwise(new float[] { 16f, 26f, 36f, 46f }, graph.Outputs["z"]);
        AssertBitwise(row.Concat(row).Concat(row).ToArray(), graph.Outputs["vout"]);
        var held = (Tensor<float>)graph.Outputs["vout"];
        graph.Reset();
        var pinned = graph.PoolPinDetailSnapshot().Where(r => r.Pinned).ToArray();
        Assert.Single(pinned);
        Assert.Equal("Output", pinned[0].RootReason);
        AssertBitwise(row.Concat(row).Concat(row).ToArray(), held);
        Assert.True(graph.Execute(inputs, true));
        AssertBitwise(row.Concat(row).Concat(row).ToArray(), graph.Outputs["vout"]);
        graph.Reset();
        Assert.Equal(2, graph.PoolPinSnapshot().Single(s => s.Type == "Single" && s.Length == 4).Pinned);
    }

    [Fact]
    public void ResetDuplicateBindingsReturnOnceRegardlessOfOrder()
    {
        // The same rented array under two intermediate names returns exactly
        // once per sweep (the pool would throw on a double return), and the
        // per-decision records match under both binding orders. Reruns reuse
        // the storage and stay correct.
        foreach (bool reversed in new[] { false, true })
        {
            var graph = new ComputationalGraph();
            graph.Metadata["Name"] = "pin-dup-test";
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
            var row = new float[] { 17f, 28f, 39f, 50f };
            Assert.True(graph.Execute(inputs, true));
            AssertBitwise(row.Concat(row).Concat(row).ToArray(), graph.Outputs["z"]);
            var t = graph.IntermediateOutputs["t"];
            Assert.NotNull(t);
            if (reversed)
            {
                graph.IntermediateOutputs.Remove("t");
                graph.IntermediateOutputs["dup"] = t;
                graph.IntermediateOutputs["t"] = t;
            }
            else
            {
                graph.IntermediateOutputs["dup"] = t;
            }
            graph.Reset();
            var details = graph.PoolPinDetailSnapshot();
            var rets = details.Where(r => !r.Pinned).ToArray();
            Assert.Equal(2, rets.Length);
            Assert.Equal(rets[0].StorageId, rets[1].StorageId);
            Assert.Empty(details.Where(r => r.Pinned));
            Assert.True(graph.Execute(inputs, true));
            AssertBitwise(row.Concat(row).Concat(row).ToArray(), graph.Outputs["z"]);
        }
    }

    [Fact]
    public void ResetAfterFailedExecuteRecoversCleanly()
    {
        // A failed run must not corrupt pool circulation: Reset after the
        // failure is safe and the next good run is correct and reuses.
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "pin-fail-test";
        graph.Inputs["x"] = DenseTensor<float>.OfShape(2);
        graph.Inputs["w"] = DenseTensor<float>.OfValues(new float[] { 10f, 20f });
        graph.Outputs["y"] = DenseTensor<float>.OfShape(2);
        graph.Nodes.Add(new Node { Name = "add", Op = OpType.Add, Inputs = new[] { "x", "w" }, Outputs = new[] { "t" } });
        graph.Nodes.Add(new Node { Name = "relu", Op = OpType.Relu, Inputs = new[] { "t" }, Outputs = new[] { "y" } });
        graph.RefreshLifetimeAnalysis();
        var bad = new System.Collections.Generic.Dictionary<string, ITensor>
        {
            { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f, 3f }) },
            { "w", graph.Inputs["w"] },
        };
        Assert.False(graph.Execute(bad, true));
        graph.Reset();
        var good = new System.Collections.Generic.Dictionary<string, ITensor>
        {
            { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) },
            { "w", graph.Inputs["w"] },
        };
        Assert.True(graph.Execute(good, true));
        Assert.Equal(new float[] { 9f, 22f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
        graph.Reset();
        Assert.True(graph.Execute(good, true));
        Assert.Equal(new float[] { 9f, 22f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
    }

    [Fact]
    public void ConcurrentContextsDoNotRecycleEachOthersLiveBuffers()
    {
        // Two contexts share one pool but own their bindings: resetting one
        // must leave the other's held outputs valid and its reruns correct.
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "pin-ctx-test";
        graph.Inputs["x"] = DenseTensor<float>.OfShape(2);
        graph.Initializers["w"] = DenseTensor<float>.OfValues(new float[] { 10f, 20f });
        graph.Initializers["u"] = DenseTensor<float>.OfValues(new float[] { 2f, 3f });
        graph.Outputs["y"] = DenseTensor<float>.OfShape(2);
        graph.Nodes.Add(new Node { Name = "add", Op = OpType.Add, Inputs = new[] { "x", "w" }, Outputs = new[] { "t" } });
        graph.Nodes.Add(new Node { Name = "mul", Op = OpType.Mul, Inputs = new[] { "t", "u" }, Outputs = new[] { "y" } });
        graph.IntermediateOutputs["t"] = null;
        graph.RefreshLifetimeAnalysis();
        var ctx1 = graph.CreateExecution(null);
        var ctx2 = graph.CreateExecution(null);
        var first = new System.Collections.Generic.Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) } };
        Assert.True(ctx1.Execute(first, true));
        var held = ((Tensor<float>)ctx1.Outputs["y"]).ToArray();
        Assert.Equal(new float[] { 22f, 66f }, held);
        var second = new System.Collections.Generic.Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 3f, 4f }) } };
        Assert.True(ctx2.Execute(second, true));
        ctx2.Reset();
        Assert.True(ctx2.Execute(second, true));
        ctx2.Reset();
        Assert.Equal(new float[] { 22f, 66f }, held);
        Assert.Equal(new float[] { 22f, 66f }, ((Tensor<float>)ctx1.Outputs["y"]).ToArray());
        Assert.True(ctx1.Execute(second, true));
        Assert.Equal(new float[] { 26f, 72f }, ((Tensor<float>)ctx1.Outputs["y"]).ToArray());
    }

[Fact]
    public void ResetPreservesCallerHeldOutputs()
    {
        // Graph outputs keep their storage across Reset plus rerun: the
        // end-of-run sweep must never recycle caller-visible storage.
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "held-output-test";
        graph.Inputs["x"] = DenseTensor<float>.OfShape(2);
        graph.Outputs["y"] = DenseTensor<float>.OfShape(2);
        graph.Nodes.Add(new Node { Name = "r", Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "y" } });
        graph.RefreshLifetimeAnalysis();
        var first = new Dictionary<string, ITensor> { ["x"] = DenseTensor<float>.OfValues(new float[] { -1f, 2f }) };
        Assert.True(graph.Execute(first, true));
        var held = (Tensor<float>)graph.Outputs["y"];
        var second = new Dictionary<string, ITensor> { ["x"] = DenseTensor<float>.OfValues(new float[] { 3f, -4f }) };
        graph.Reset();
        Assert.True(graph.Execute(second, true));
        Assert.Equal(new float[] { 0f, 2f }, held.ToArray());
        Assert.Equal(new float[] { 3f, 0f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
    }
}
