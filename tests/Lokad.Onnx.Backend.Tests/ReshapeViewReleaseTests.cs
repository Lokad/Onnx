namespace Lokad.Onnx.Backend.Tests;

public class ReshapeViewReleaseTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PublicMemoryPolicy_IsContextLocalAndPreservesHeldValues(bool savedOutput)
    {
        var plan = Chain(false, false, savedOutput);
        var memory = plan.CreateExecution(ExecutionOptions.Memory);
        var speed = plan.CreateExecution(ExecutionOptions.Default);
        var held = new List<(Tensor<float> Value, float[] Bits)>();
        foreach (int size in new[] { 8, 10, 0, 16, 8 })
        {
            var input = Feed(size, held.Count + 1);
            var before = ((Tensor<float>)input["x"]).ToArray();
            foreach (var context in new[] { memory, speed })
            {
                context.Reset();
                Assert.True(context.Execute(input, true), context.LastErrorMessage);
                Assert.Equal(before.Select(v => 4 * v), ((Tensor<float>)context.Outputs["y"]).ToArray());
                foreach (var tensor in context.Outputs.Values.Concat(context.IntermediateOutputs.Values).OfType<Tensor<float>>())
                    held.Add((tensor, tensor.ToArray()));
            }
            Assert.NotNull(speed.IntermediateOutputs["owner"]);
            if (!savedOutput)
            {
                Assert.Null(memory.IntermediateOutputs["owner"]);
                Assert.Null(memory.IntermediateOutputs["view"]);
                Assert.Null(memory.IntermediateOutputs["view2"]);
                Assert.True(memory.LastPoolAllocatedNew < speed.LastPoolAllocatedNew);
            }
            Assert.Equal(before, ((Tensor<float>)input["x"]).ToArray());
            foreach (var prior in held) Assert.Equal(prior.Bits, prior.Value.ToArray());
            memory.Reset();
            Assert.False(memory.Execute(new Dictionary<string, ITensor>(), true));
            foreach (var prior in held) Assert.Equal(prior.Bits, prior.Value.ToArray());
        }
        Assert.Equal(OptimizationMode.Speed, plan.Options.Optimization);
        Assert.Equal(OptimizationMode.Speed, speed.Options.Optimization);
        Assert.Equal(OptimizationMode.Memory, memory.Options.Optimization);
    }

    [Fact]
    public void ExplicitOptions_CanSelectMemoryPolicyPerCall()
    {
        var graph = Chain(false, false, false);
        var input = Feed(8, 1);
        var held = new List<(Tensor<float> Value, float[] Bits)>();
        foreach (var options in new[] { ExecutionOptions.Default, ExecutionOptions.Memory, ExecutionOptions.Default })
        {
            graph.Reset();
            Assert.True(graph.Execute(input, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
            Assert.Equal(options.Optimization == OptimizationMode.Memory, graph.IntermediateOutputs["owner"] is null);
            foreach (var tensor in graph.Outputs.Values.Concat(graph.IntermediateOutputs.Values).OfType<Tensor<float>>())
                held.Add((tensor, tensor.ToArray()));
            foreach (var prior in held) Assert.Equal(prior.Bits, prior.Value.ToArray());
        }
    }

    static ComputationalGraph Chain(bool enabled, bool reuse, bool saved)
    {
        var graph = new ComputationalGraph { ReleaseReshapeViews = enabled, ReuseReleasedBuffers = reuse };
        graph.Metadata["Name"] = "dead-reshape-alias-chain";
        graph.Inputs["x"] = DenseTensor<float>.OfShape(8);
        graph.InputDescs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { -1 } });
        graph.Initializers["matrix"] = DenseTensor<long>.OfValues(new long[] { 2, -1 });
        graph.Initializers["flat"] = DenseTensor<long>.OfValues(new long[] { -1 });
        graph.Outputs["y"] = DenseTensor<float>.OfShape(8);
        foreach (string name in new[] { "owner", "view", "view2", "sum" }) graph.IntermediateOutputs[name] = null;
        graph.Nodes.Add(new Node { Name = "owner", Op = OpType.Add, Inputs = new[] { "x", "x" }, Outputs = new[] { "owner" } });
        graph.Nodes.Add(new Node { Name = "view", Op = OpType.Reshape, Inputs = new[] { "owner", "matrix" }, Outputs = new[] { "view" } });
        graph.Nodes.Add(new Node { Name = "view2", Op = OpType.Reshape, Inputs = new[] { "view", "flat" }, Outputs = new[] { "view2" } });
        graph.Nodes.Add(new Node { Name = "sum", Op = OpType.Add, Inputs = new[] { "view2", "x" }, Outputs = new[] { "sum" } });
        if (saved)
        {
            graph.Outputs["saved"] = DenseTensor<float>.OfShape(8);
            graph.Nodes.Add(new Node { Name = "save", Op = OpType.Identity, Inputs = new[] { "view2" }, Outputs = new[] { "saved" } });
        }
        graph.Nodes.Add(new Node { Name = "last", Op = OpType.Add, Inputs = new[] { "sum", "x" }, Outputs = new[] { "y" } });
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
    public void DeadViewChainReleasesOwnerAndReducesFreshRents(bool context, bool reuse)
    {
        var reference = Chain(false, reuse, false);
        var candidate = Chain(true, reuse, false);
        if (context) { reference = reference.CreateExecution(null); candidate = candidate.CreateExecution(null); }
        var held = new List<(Tensor<float> Value, float[] Bits)>();
        foreach (int length in new[] { 8, 8, 10, 0, 16, 8 })
        {
            var feed = Feed(length, held.Count + 1);
            var x = ((Tensor<float>)feed["x"]).ToArray();
            foreach (var graph in new[] { reference, candidate })
            {
                graph.Reset();
                Assert.True(graph.Execute(feed, true), graph.LastErrorMessage);
                Assert.Equal(x.Select(v => 4 * v), ((Tensor<float>)graph.Outputs["y"]).ToArray());
                foreach (var value in graph.Outputs.Values.Concat(graph.IntermediateOutputs.Values).OfType<Tensor<float>>())
                    held.Add((value, value.ToArray()));
            }
            Assert.NotNull(reference.IntermediateOutputs["owner"]);
            foreach (string name in new[] { "owner", "view", "view2" }) Assert.Null(candidate.IntermediateOutputs[name]);
            Assert.True(candidate.LastPoolAllocatedNew < reference.LastPoolAllocatedNew);
            Assert.Equal(x, ((Tensor<float>)feed["x"]).ToArray());
            foreach (var previous in held) Assert.Equal(previous.Bits, previous.Value.ToArray());
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void OutputAliasesAndPriorValuesSurviveResetAndLateFailure(bool context, bool enabled)
    {
        var graph = Chain(enabled, true, true);
        if (context) graph = graph.CreateExecution(null);
        var held = new List<(Tensor<float> Value, float[] Bits)>();
        foreach (int length in new[] { 8, 10, 0, 16, 8 })
        {
            graph.Reset();
            var feed = Feed(length, held.Count + 1);
            var x = ((Tensor<float>)feed["x"]).ToArray();
            Assert.True(graph.Execute(feed, true), graph.LastErrorMessage);
            Assert.NotNull(graph.IntermediateOutputs["owner"]);
            Assert.Equal(x.Select(v => 2 * v), ((Tensor<float>)graph.Outputs["saved"]).ToArray());
            Assert.Equal(x.Select(v => 4 * v), ((Tensor<float>)graph.Outputs["y"]).ToArray());
            foreach (var value in graph.Outputs.Values.Concat(graph.IntermediateOutputs.Values).OfType<Tensor<float>>())
                held.Add((value, value.ToArray()));
            graph.Reset();
            Assert.False(graph.Execute(new Dictionary<string, ITensor>(), true));
            foreach (var previous in held) Assert.Equal(previous.Bits, previous.Value.ToArray());
        }
        graph.Initializers["invalidShape"] = DenseTensor<long>.OfValues(new long[] { 3 });
        graph.Outputs["bad"] = DenseTensor<float>.OfShape(3);
        graph.Nodes.Add(new Node { Name = "late-failure", Op = OpType.Reshape, Inputs = new[] { "y", "invalidShape" }, Outputs = new[] { "bad" } });
        graph.Reset();
        Assert.False(graph.Execute(Feed(8, 100), true));
        Assert.Equal("late-failure", graph.LastFailedNodeName);
        foreach (var previous in held) Assert.Equal(previous.Bits, previous.Value.ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DuplicateProducerStillRefusedBeforeReleasingAnything(bool enabled)
    {
        var graph = Chain(enabled, false, false);
        var reshape = graph.Nodes[1]; reshape.Outputs = new[] { "owner" }; graph.Nodes[1] = reshape;
        reshape = graph.Nodes[2]; reshape.Inputs = new[] { "owner", "flat" }; graph.Nodes[2] = reshape;
        graph.IntermediateOutputs.Remove("view");
        var feed = Feed(8, 1);
        Assert.False(graph.Execute(feed, true));
        Assert.Contains("duplicate producers", graph.LastErrorMessage);
        Assert.Equal(0, graph.LastPoolReturned);
        Assert.Equal(Enumerable.Range(1, 8).Select(v => (float)v), ((Tensor<float>)feed["x"]).ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void UnknownStaticStorageKeepsOriginalOwnerProtected(bool enabled)
    {
        var graph = Chain(enabled, true, false);
        graph.Initializers["opaque"] = new OpaqueTensor(new[] { 7f, 8f });
        Assert.True(graph.Execute(Feed(8, 1), true), graph.LastErrorMessage);
        Assert.NotNull(graph.IntermediateOutputs["owner"]);
        Assert.Equal(0, graph.LastPoolReturned);
        var held = (Tensor<float>)graph.IntermediateOutputs["owner"]!;
        var expected = held.ToArray();
        if (enabled)
        {
            Assert.Null(graph.IntermediateOutputs["view"]);
            Assert.Null(graph.IntermediateOutputs["view2"]);
        }
        graph.Reset();
        Assert.True(graph.Execute(Feed(8, 100), true), graph.LastErrorMessage);
        Assert.Equal(expected, held.ToArray());
        Assert.Equal(new[] { 7f, 8f }, ((Tensor<float>)graph.Initializers["opaque"]).ToArray());
    }

    sealed class OpaqueTensor : Tensor<float>
    {
        readonly float[] values;
        public OpaqueTensor(float[] values) : base(values.Length) { this.values = values; }
        public override float GetValue(int index) => values[index];
        public override void SetValue(int index, float value) => values[index] = value;
        public override Tensor<float> Clone() => new OpaqueTensor((float[])values.Clone());
        public override Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions) => new DenseTensor<TResult>(dimensions);
        public override Tensor<float> Reshape(ReadOnlySpan<int> dimensions) => throw new NotSupportedException();
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ExplicitContextsRetainIndependentOutputViews(bool enabled)
    {
        var plan = Chain(enabled, true, true);
        async Task Run(float start)
        {
            await Task.Yield();
            var graph = plan.CreateExecution(null);
            var held = new List<(Tensor<float> Value, float[] Bits)>();
            for (int i = 0; i < 12; i++)
            {
                graph.Reset();
                var feed = Feed(i % 2 == 0 ? 8 : 10, start + i);
                Assert.True(graph.Execute(feed, true), graph.LastErrorMessage);
                var output = (Tensor<float>)graph.Outputs["saved"];
                Assert.Equal(((Tensor<float>)feed["x"]).ToArray().Select(x => 2 * x), output.ToArray());
                held.Add((output, output.ToArray()));
                foreach (var previous in held) Assert.Equal(previous.Bits, previous.Value.ToArray());
            }
        }
        await Task.WhenAll(Task.Run(() => Run(1)), Task.Run(() => Run(100)));
    }
}
