namespace Lokad.Onnx.Backend.Tests;

public class ReleasedBufferCacheTests
{
    [Fact]
    public void TransfersOnlyReleasedRentsAndClearsAccumulationDestinations()
    {
        var cache = new ReleasedBufferCache(1024, 8);
        var first = new TensorBufferPool(cache);
        var released = first.Rent<float>(8);
        Array.Fill(released, 123f);
        var live = first.Rent<float>(8);
        Array.Fill(live, 456f);
        first.Return(released);
        first.Return(new float[3]); // Foreign, never rented: do not carry it over.
        first.RetainReleasedBuffers();
        first.RetainReleasedBuffers(); // Drained; cannot transfer the same array twice.
        Assert.Equal(1, cache.Count);
        Assert.Equal(32, cache.Bytes);
        var second = new TensorBufferPool(cache);
        var reused = second.RentCleared<float>(8);
        Assert.Same(released, reused);
        Assert.NotSame(live, reused);
        Assert.Equal(new float[8], reused);
        Assert.Equal(Enumerable.Repeat(456f, 8), live);
        Assert.True(second.IsOwned(reused));
        Assert.Equal(1, second.Reused);
        Assert.Equal(32, second.ReusedBytes);
        Assert.Equal(0, second.AllocatedNew);
        Assert.Equal(0, cache.Count);
        Assert.Equal(0, cache.Shapes);
        Assert.Equal(0, cache.Bytes);
    }

    [Fact]
    public void ManyShapesAndOversizedBuffersStayWithinBothBounds()
    {
        var cache = new ReleasedBufferCache(64, 3);
        for (int length = 1; length <= 100; length++)
        {
            var pool = new TensorBufferPool(cache);
            var array = pool.Rent<float>(length);
            pool.Return(array);
            pool.RetainReleasedBuffers();
            Assert.InRange(cache.Bytes, 0, 64);
            Assert.InRange(cache.Count, 0, 3);
            Assert.InRange(cache.Shapes, 0, 3);
        }
        Assert.Null(cache.Take<float>(100));
        Assert.Equal(3, cache.Count);
        cache.Clear();
        Assert.Equal(0, cache.Count);
        Assert.Equal(0, cache.Bytes);
        Assert.Equal(0, cache.Shapes);
        var large = new TensorBufferPool(cache);
        large.Return(large.Rent<float>(17));
        large.RetainReleasedBuffers();
        Assert.Equal(0, cache.Count);
    }

    [Fact]
    public void ZeroLengthAndPerShapeCountsAreBounded()
    {
        var zeroCache = new ReleasedBufferCache(0, 3);
        var pool = new TensorBufferPool(zeroCache);
        var zeros = Enumerable.Range(0, 10).Select(_ => pool.Rent<float>(0)).ToArray();
        foreach (var array in zeros) pool.Return(array);
        pool.RetainReleasedBuffers();
        Assert.Equal(3, zeroCache.Count);
        Assert.Equal(0, zeroCache.Bytes);
        var shapes = new ReleasedBufferCache(4096, 256);
        for (int run = 0; run < 2; run++)
        {
            var p = new TensorBufferPool(shapes);
            var arrays = Enumerable.Range(0, 64).Select(_ => p.Rent<float>(1)).ToArray();
            foreach (var array in arrays) p.Return(array);
            p.RetainReleasedBuffers();
            Assert.Equal(32, shapes.Count);
            Assert.Equal(128, shapes.Bytes);
        }
    }

    [Fact]
    public void CacheAccountsManagedBoolHalfAndStructPayloadsExactly()
    {
        var cache = new ReleasedBufferCache(100, 10);
        var pool = new TensorBufferPool(cache);
        pool.Return(pool.Rent<bool>(3));
        pool.Return(pool.Rent<Half>(3));
        pool.Return(pool.Rent<decimal>(2));
        pool.Return(pool.Rent<System.Numerics.Complex>(2));
        pool.RetainReleasedBuffers();
        Assert.Equal(73, cache.Bytes);
        Assert.Equal(4, cache.Count);
        Assert.NotNull(cache.Take<bool>(3));
        Assert.Equal(70, cache.Bytes);
        Assert.NotNull(cache.Take<System.Numerics.Complex>(2));
        Assert.Equal(38, cache.Bytes);
        Assert.NotNull(cache.Take<Half>(3));
        Assert.Equal(32, cache.Bytes);
        Assert.NotNull(cache.Take<decimal>(2));
        Assert.Equal(0, cache.Bytes);
        Assert.Equal(0, cache.Shapes);
    }

    static ComputationalGraph Chain(bool reuse, bool keepAlias)
    {
        var graph = new ComputationalGraph { ReuseReleasedBuffers = reuse };
        graph.Metadata["Name"] = "released-storage-only";
        graph.Inputs["x"] = DenseTensor<float>.OfShape(8);
        graph.InputDescs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { -1 } });
        graph.Outputs["y"] = DenseTensor<float>.OfShape(8);
        graph.OutputDescs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { -1 } });
        graph.IntermediateOutputs["first"] = null;
        graph.IntermediateOutputs["second"] = null;
        graph.Nodes.Add(new Node { Name = "first", Op = OpType.Add, Inputs = new[] { "x", "x" }, Outputs = new[] { "first" } });
        if (keepAlias)
        {
            graph.Outputs["saved"] = DenseTensor<float>.OfShape(8);
            graph.OutputDescs.Add(new OnnxValueInfo { Name = "saved", ElementType = TensorElementType.Float, Dims = new[] { -1 } });
            graph.Nodes.Add(new Node { Name = "alias", Op = OpType.Identity, Inputs = new[] { "first" }, Outputs = new[] { "saved" } });
        }
        graph.Nodes.Add(new Node { Name = "second", Op = OpType.Add, Inputs = new[] { "first", "x" }, Outputs = new[] { "second" } });
        graph.Nodes.Add(new Node { Name = "last", Op = OpType.Add, Inputs = new[] { "second", "x" }, Outputs = new[] { "y" } });
        graph.RefreshLifetimeAnalysis();
        return graph;
    }

    static Dictionary<string, ITensor> Feed(int length, float start) => new()
    {
        ["x"] = DenseTensor<float>.OfValues(Enumerable.Range(0, length).Select(i => start + i).ToArray())
    };

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void OldOutputsAndLiveIntermediateAliasesSurviveShapesAndFailures(bool context, bool keepAlias)
    {
        var plan = Chain(true, keepAlias);
        var graph = context ? plan.CreateExecution(null) : plan;
        var old = new List<(Tensor<float> Tensor, float[] Values)>();
        foreach (int length in new[] { 8, 8, 13, 0, 8, 13, 8 })
        {
            graph.Reset();
            var feed = Feed(length, old.Count + 1);
            var input = ((Tensor<float>)feed["x"]).ToArray();
            Assert.True(graph.Execute(feed, false), graph.LastErrorMessage);
            Assert.Equal(input.Select(x => 4 * x), ((Tensor<float>)graph.Outputs["y"]).ToArray());
            Assert.Equal(input, ((Tensor<float>)feed["x"]).ToArray());
            if (keepAlias)
            {
                Assert.Equal(input.Select(x => 2 * x), ((Tensor<float>)graph.Outputs["saved"]).ToArray());
                Assert.NotNull(graph.IntermediateOutputs["first"]);
            }
            foreach (var tensor in graph.Outputs.Values.Concat(graph.IntermediateOutputs.Values).OfType<Tensor<float>>())
                old.Add((tensor, tensor.ToArray()));
            graph.Reset();
            Assert.False(graph.Execute(new Dictionary<string, ITensor>(), false));
            foreach (var held in old) Assert.Equal(held.Values, held.Tensor.ToArray());
            Assert.InRange(graph.ReleasedBuffers!.Bytes, 0, ReleasedBufferCache.DefaultByteLimit);
            Assert.InRange(graph.ReleasedBuffers.Count, 0, ReleasedBufferCache.DefaultCountLimit);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RepeatedCallsReduceFreshRentsAndKeepPerRunAccounting(bool context)
    {
        var plan = Chain(true, false);
        var graph = context ? plan.CreateExecution(null) : plan;
        Assert.True(graph.Execute(Feed(8, 1), false));
        int cold = graph.LastPoolAllocatedNew;
        long bytes = graph.LastPoolAllocatedNewBytes;
        Assert.Equal(1, graph.ReleasedBuffers!.Count);
        Assert.Equal(32, graph.ReleasedBuffers.Bytes);
        graph.Reset();
        Assert.True(graph.Execute(Feed(8, 2), false));
        Assert.Equal(cold - 1, graph.LastPoolAllocatedNew);
        Assert.Equal(bytes - 32, graph.LastPoolAllocatedNewBytes);
        Assert.Equal(2, graph.LastPoolReused);
        Assert.Equal(64, graph.LastPoolReusedBytes);
        var disabled = Chain(false, false);
        Assert.True(disabled.Execute(Feed(8, 1), false));
        disabled.Reset();
        Assert.True(disabled.Execute(Feed(8, 2), false));
        Assert.Equal(cold, disabled.LastPoolAllocatedNew);
        Assert.Null(disabled.ReleasedBuffers);
        Assert.Equal(((Tensor<float>)disabled.Outputs["y"]).ToArray(), ((Tensor<float>)graph.Outputs["y"]).ToArray());
    }

    [Fact]
    public void InvalidationAndPoolDisableReleaseRetainedStorage()
    {
        var graph = Chain(true, false);
        Assert.True(graph.Execute(Feed(8, 1), false));
        Assert.True(graph.ReleasedBuffers!.Bytes > 0);
        graph.InvalidatePreparation();
        Assert.Equal(0, graph.ReleasedBuffers.Bytes);
        graph.Reset();
        Assert.True(graph.Execute(Feed(8, 2), false));
        Assert.True(graph.ReleasedBuffers.Bytes > 0);
        graph.RefreshLifetimeAnalysis();
        Assert.Equal(0, graph.ReleasedBuffers.Bytes);
        Assert.True(graph.Execute(Feed(8, 3), false));
        var noPool = ExecutionOptions.Default with { Tensor = ExecutionOptions.Default.Tensor with { DisableBufferPool = true } };
        graph.Reset();
        Assert.True(graph.Execute(Feed(8, 4), false, ExecutionProvider.CPU, noPool));
        Assert.Equal(0, graph.ReleasedBuffers.Bytes);
        Assert.Equal(0, graph.LastPoolReused);
    }

    [Fact]
    public async Task IndependentContextsNeverShareFreeStorage()
    {
        var plan = Chain(true, false);
        Assert.True(plan.Execute(Feed(8, 1), false));
        var one = plan.CreateExecution(null);
        var two = plan.CreateExecution(null);
        async Task Exercise(GraphExecution graph, float start)
        {
            await Task.Yield();
            var held = new List<(Tensor<float> Tensor, float[] Values)>();
            for (int run = 0; run < 20; run++)
            {
                graph.Reset();
                var feed = Feed(8, start + run);
                Assert.True(graph.Execute(feed, false));
                var output = (Tensor<float>)graph.Outputs["y"];
                Assert.Equal(((Tensor<float>)feed["x"]).ToArray().Select(x => x * 4), output.ToArray());
                held.Add((output, output.ToArray()));
                foreach (var old in held) Assert.Equal(old.Values, old.Tensor.ToArray());
            }
        }
        await Task.WhenAll(Task.Run(() => Exercise(one, 1)), Task.Run(() => Exercise(two, 100)));
        Assert.NotSame(plan.ReleasedBuffers, one.ReleasedBuffers);
        Assert.NotSame(plan.ReleasedBuffers, two.ReleasedBuffers);
        Assert.NotSame(one.ReleasedBuffers, two.ReleasedBuffers);
        Assert.Equal(32, plan.ReleasedBuffers!.Bytes);
        Assert.Equal(32, one.ReleasedBuffers!.Bytes);
        Assert.Equal(32, two.ReleasedBuffers!.Bytes);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ExecuteNodeKeepsItsNonPooledOutputValidDuringLaterCacheReuse(bool context)
    {
        var plan = Chain(true, false);
        var graph = context ? plan.CreateExecution(null) : plan;
        Assert.True(graph.Execute(Feed(8, 1), false));
        graph.Reset();
        Assert.True(graph.ExecuteNode(Feed(8, 20), "first", false), graph.LastErrorMessage);
        var held = (Tensor<float>)graph.Outputs["first"];
        var values = held.ToArray();
        Assert.Equal(Enumerable.Range(20, 8).Select(x => 2f * x), values);
        Assert.Equal(1, graph.ReleasedBuffers!.Count); // ExecuteNode retains its existing no-pool route.
        Assert.Equal(0, graph.LastPoolReused);
        graph.Reset();
        Assert.True(graph.Execute(Feed(8, 100), false));
        Assert.Equal(values, held.ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PartialFailureDoesNotAdoptCheckedOutOutputs(bool context)
    {
        var plan = Chain(true, true);
        var graph = context ? plan.CreateExecution(null) : plan;
        Assert.True(graph.Execute(Feed(8, 1), false));
        var held = graph.Outputs.Values.Concat(graph.IntermediateOutputs.Values).OfType<Tensor<float>>()
            .Select(t => (Tensor: t, Values: t.ToArray())).ToArray();
        graph.Nodes.Add(new Node { Name = "fail", Op = OpType.Unknown, OpTypeName = "UnsupportedForTest", Inputs = new[] { "y" }, Outputs = new[] { "bad" } });
        graph.IntermediateOutputs["bad"] = null;
        graph.Reset();
        Assert.False(graph.Execute(Feed(8, 100), false));
        Assert.Equal("fail", graph.LastFailedNodeName);
        foreach (var old in held) Assert.Equal(old.Values, old.Tensor.ToArray());
        graph.Nodes.RemoveAt(graph.Nodes.Count - 1);
        graph.IntermediateOutputs.Remove("bad");
        Assert.True(graph.Execute(Feed(8, 200), false));
        foreach (var old in held) Assert.Equal(old.Values, old.Tensor.ToArray());
    }
}
