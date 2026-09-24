using System.Numerics;

namespace Lokad.Onnx.Backend.Tests;

public class PreparedLstmWeightsTests
{
    const int H = 640, Count = 4 * H * H;
    const long Bytes = (long)Count * sizeof(float), PairBytes = 2 * Bytes;

    static float[] Values(int count, int seed)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = ((i * 17 + seed) % 97 - 48) / 2048f;
        return values;
    }

    static ComputationalGraph Graph(long budget = PairBytes)
    {
        var graph = new ComputationalGraph(budget); graph.Metadata["Name"] = "prepared-decoder-lstm"; graph.Opset[""] = 17;
        graph.Inputs["x"] = new DenseTensor<float>(Values(H, 1), new[] { 1, 1, H });
        graph.Inputs["h"] = new DenseTensor<float>(Values(H, 3), new[] { 1, 1, H });
        graph.Inputs["c"] = new DenseTensor<float>(Values(H, 5), new[] { 1, 1, H });
        graph.Initializers["w"] = new DenseTensor<float>(Values(Count, 7), new[] { 1, 4 * H, H });
        graph.Initializers["r"] = new DenseTensor<float>(Values(Count, 11), new[] { 1, 4 * H, H });
        graph.Initializers["b"] = new DenseTensor<float>(Values(8 * H, 13), new[] { 1, 8 * H });
        graph.Outputs["y"] = Tensor<float>.Zeros(1, 1, 1, H);
        graph.Outputs["yh"] = Tensor<float>.Zeros(1, 1, H); graph.Outputs["yc"] = Tensor<float>.Zeros(1, 1, H);
        graph.Nodes.Add(new Node { Name = "lstm", Op = OpType.LSTM, OpTypeName = "LSTM", Domain = "", OpsetVersion = 17,
            Inputs = new[] { "x", "w", "r", "b", "", "h", "c" }, Outputs = new[] { "y", "yh", "yc" },
            Attributes = new() { ["hidden_size"] = H } });
        return graph;
    }

    static DenseTensor<float> Weight(ComputationalGraph graph, string name) => Assert.IsType<DenseTensor<float>>(graph.Initializers[name]);
    static int[] Bits(ITensor tensor) => ((Tensor<float>)tensor).ToArray().Select(BitConverter.SingleToInt32Bits).ToArray();
    static ITensor[] Run(ComputationalGraph graph, ExecutionOptions? options = null)
    {
        var context = graph.CreateExecution(options ?? ExecutionOptions.Memory);
        Assert.Same(graph.PackedLstmWeights, context.PackedLstmWeights);
        var feeds = graph.Inputs.ToDictionary(p => p.Key, p => p.Value!);
        Assert.True(context.Execute(feeds, true, ExecutionProvider.CPU, options ?? ExecutionOptions.Memory), context.LastErrorMessage);
        var outputs = new[] { "y", "yh", "yc" }.Select(n => context.Outputs[n]!).ToArray();context.Reset();return outputs;
    }

    static ITensor[] Direct(ComputationalGraph graph, ExecutionOptions options)
    {
        var result = CPUExecutionProvider.Lstm(graph.Inputs["x"], graph.Initializers["w"], graph.Initializers["r"], graph.Initializers["b"],
            null, graph.Inputs["h"], graph.Inputs["c"], null, "forward", null, null, null, null, H, false, 0, 3, options, null);
        Assert.Equal(OpStatus.Success, result.Status);return result.Outputs;
    }

    static void Equal(ITensor[] expected, ITensor[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) { Assert.Equal(expected[i].Dims, actual[i].Dims);Assert.Equal(Bits(expected[i]), Bits(actual[i])); }
    }

    [Theory]
    [InlineData(0, 0)] [InlineData(Bytes, 0)] [InlineData(PairBytes - 1, 0)]
    [InlineData(PairBytes, 2)] [InlineData(PairBytes + 1, 2)] [InlineData(long.MaxValue, 2)]
    public void CompletePairAdmissionIsAtomicAndRepeatPreparationRetainsOwnedArrays(long budget, int count)
    {
        var graph = Graph(budget);var sources = graph.Initializers.ToDictionary(p => p.Key, p => Bits(p.Value));
        var expected = Direct(graph, ExecutionOptions.Scalar);graph.Prepare();
        Assert.Equal(count, graph.PackedLstmWeights.Count);Assert.Equal(count * Bytes, graph.RetainedPackedWeightBytes);
        var original = graph.PackedLstmWeights.Values.ToArray();graph.RefreshLifetimeAnalysis();
        Assert.Equal(original, graph.PackedLstmWeights.Values.ToArray());
        foreach (var record in original)
        {
            Assert.NotSame(record.SourceArray, record.Values);Assert.Equal(Count, record.Values.Length);
            Assert.DoesNotContain(graph.Initializers.Values, t => t is DenseTensor<float> d && d.Buffer.Span.Overlaps(record.Values));
        }
        Equal(expected, Run(graph));Equal(expected, Run(graph, ExecutionOptions.Scalar));
        foreach (var source in sources) Assert.Equal(source.Value, Bits(graph.Initializers[source.Key]));
        Assert.Equal(sources.Keys.Order(), graph.Initializers.Keys.Order());
    }

    [Fact]
    public void FreshContextsActuallyUsePreparedValuesWhileScalarAndDirectCallsKeepOriginalSources()
    {
        var graph = Graph();var expected = Direct(graph, ExecutionOptions.Scalar);var first = Run(graph);
        var held = first.Select(Bits).ToArray();Equal(expected, first);Equal(expected, Run(graph));
        foreach (var record in graph.PackedLstmWeights.Values) Array.Fill(record.Values, float.NaN);
        Equal(expected, Run(graph, ExecutionOptions.Scalar));Equal(expected, Direct(graph, ExecutionOptions.Default));
        if (Vector.IsHardwareAccelerated)
            Assert.All(Run(graph).SelectMany(t => ((Tensor<float>)t).ToArray()), x => Assert.True(float.IsNaN(x)));
        graph.InvalidatePreparation();Assert.Empty(graph.PackedLstmWeights);Assert.Equal(0, graph.RetainedPackedWeightBytes);
        Equal(expected, Run(graph));
        for (int i = 0; i < held.Length; i++) Assert.Equal(held[i], Bits(first[i]));
    }

    [Fact]
    public void ReplacementShapeChangesAndInvalidationRefuseStaleSources()
    {
        var graph = Graph();graph.Prepare();var original = graph.PackedLstmWeights.Values.ToArray();var w = Weight(graph, "w");
        var replacement = new DenseTensor<float>(w.ToArray(), new[] { 1, 4 * H, H });
        Assert.Null(GraphLstmPacking.Resolve(graph.PackedLstmWeights, replacement));
        graph.Initializers["w"] = replacement;graph.RefreshLifetimeAnalysis();
        Assert.Equal(2, graph.PackedLstmWeights.Count);Assert.Equal(PairBytes, graph.RetainedPackedWeightBytes);
        Assert.DoesNotContain(graph.PackedLstmWeights.Values, r => ReferenceEquals(r, original.Single(p => ReferenceEquals(p.Source, w))));
        var changed = new DenseTensor<float>(replacement.Buffer, new[] { 1, H, 4 * H });
        Assert.Null(GraphLstmPacking.Resolve(graph.PackedLstmWeights, changed));
        graph.Initializers["w"] = changed;graph.RefreshLifetimeAnalysis();Assert.Empty(graph.PackedLstmWeights);Assert.Equal(0, graph.RetainedPackedWeightBytes);
        graph.Initializers["w"] = replacement;graph.Prepare();graph.RefreshLifetimeAnalysis();
        replacement.Buffer.Span[19] += .125f;graph.InvalidatePreparation();Equal(Direct(graph, ExecutionOptions.Scalar), Run(graph));
    }

    [Theory]
    [InlineData("input")] [InlineData("output")] [InlineData("consumer")] [InlineData("offset")]
    [InlineData("shape")] [InlineData("direction")] [InlineData("domain")] [InlineData("hidden")] [InlineData("reversed")]
    public void IncompatibleWeightsAndConsumersRemoveTheWholePair(string kind)
    {
        var graph = Graph();graph.Prepare();Assert.Equal(2, graph.PackedLstmWeights.Count);
        switch (kind)
        {
            case "input": graph.Inputs["w"] = graph.Initializers["w"];break;
            case "output": graph.Outputs["w"] = graph.Initializers["w"];break;
            case "consumer": graph.Nodes.Add(new Node { Name = "read-w", Op = OpType.Identity, Inputs = new[] { "w" }, Outputs = new[] { "unused" } });break;
            case "offset": graph.Initializers["w"] = new DenseTensor<float>(new float[Count + 1].AsMemory(1), new[] { 1, 4 * H, H });break;
            case "shape": graph.Initializers["w"] = new DenseTensor<float>(new float[Count], new[] { 1, H, 4 * H });break;
            case "reversed": graph.Initializers["w"] = new DenseTensor<float>(new float[Count], new[] { 1, 4 * H, H }, true);break;
            case "direction": graph.Nodes[0].Attributes!["direction"] = "reverse";break;
            case "domain": var n = graph.Nodes[0];n.Domain = "custom";graph.Nodes[0] = n;break;
            case "hidden": graph.Nodes[0].Attributes!["hidden_size"] = H / 2;break;
        }
        graph.RefreshLifetimeAnalysis();Assert.Empty(graph.PackedLstmWeights);Assert.Equal(0, graph.RetainedPackedWeightBytes);
    }

    [Fact]
    public void SharedTensorUsesOneCloneAndDifferentWrappersCannotOverwriteItsMapping()
    {
        var graph = Graph(Bytes);graph.Initializers["r"] = graph.Initializers["w"];graph.Prepare();
        Assert.Single(graph.PackedLstmWeights);Assert.Equal(Bytes, graph.RetainedPackedWeightBytes);
        Equal(Direct(graph, ExecutionOptions.Scalar), Run(graph));
        var w = Weight(graph, "w");graph.Initializers["r"] = new DenseTensor<float>(w.Buffer, w.Dimensions.ToArray());
        graph.RefreshLifetimeAnalysis();Assert.Empty(graph.PackedLstmWeights);Assert.Equal(0, graph.RetainedPackedWeightBytes);
        Equal(Direct(graph, ExecutionOptions.Scalar), Run(graph));
    }

    [Theory]
    [InlineData(2, 1)] [InlineData(1, 2)]
    public void UnsupportedSequenceOrBatchKeepsTheOriginalRoute(int sequence, int batch)
    {
        var graph = Graph();graph.Prepare();
        foreach (var record in graph.PackedLstmWeights.Values) Array.Fill(record.Values, float.NaN);
        graph.Inputs["x"] = new DenseTensor<float>(Values(sequence * batch * H, 1), new[] { sequence, batch, H });
        graph.Inputs["h"] = new DenseTensor<float>(Values(batch * H, 3), new[] { 1, batch, H });
        graph.Inputs["c"] = new DenseTensor<float>(Values(batch * H, 5), new[] { 1, batch, H });
        Equal(Direct(graph, ExecutionOptions.Scalar), Run(graph));
    }

    [Theory]
    [InlineData(-1, 0)] [InlineData(0, 2)]
    public void MatrixConvolutionAndRecurrentMapsShareOneBudgetAcrossRefresh(long adjustment, int recurrentCount)
    {
        long convBytes = GraphConvPacking.Lanes == 0 ? 0 : 32 * 16 * 9 * 4;
        var graph = Graph(PairBytes + 16 + convBytes + adjustment);
        graph.Inputs["mx"] = Tensor<float>.Zeros(1, 2);graph.Initializers["mw"] = Tensor<float>.Zeros(2, 2);
        graph.Outputs["my"] = Tensor<float>.Zeros(1, 2);
        graph.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, Inputs = new[] { "mx", "mw" }, Outputs = new[] { "my" } });
        graph.Inputs["cx"] = Tensor<float>.Zeros(1, 16, 5, 5);graph.Initializers["cw"] = Tensor<float>.Zeros(32, 16, 3, 3);
        graph.Outputs["cy"] = Tensor<float>.Zeros(1, 32, 2, 2);
        graph.Nodes.Add(new Node { Name = "conv", Op = OpType.Conv, Inputs = new[] { "cx", "cw" }, Outputs = new[] { "cy" },
            Attributes = new() { ["strides"] = new[] { 2, 2 }, ["group"] = 1 } });
        for (int repeat = 0; repeat < 3; repeat++)
        {
            graph.RefreshLifetimeAnalysis();Assert.Single(graph.PackedWeights);
            Assert.Equal(recurrentCount, graph.PackedLstmWeights.Count);
            long actual = graph.PackedWeights.Values.Sum(v => v.Packed.Length * sizeof(float))
                + graph.PackedConvWeights.Values.Sum(v => (v.Values.LongLength + (v.WinogradValues?.LongLength ?? 0)) * sizeof(float))
                + graph.PackedLstmWeights.Values.Sum(v => v.Values.LongLength * sizeof(float));
            Assert.Equal(16 + convBytes + recurrentCount * Bytes, actual);
            Assert.Equal(actual, graph.RetainedPackedWeightBytes);Assert.True(actual <= graph.MaximumPackedWeightBytes);
        }
        graph.Nodes.RemoveAt(0);graph.RefreshLifetimeAnalysis();Assert.Empty(graph.PackedLstmWeights);
        Assert.Equal(16 + convBytes, graph.RetainedPackedWeightBytes);
        graph.InvalidatePreparation();Assert.Equal(0, graph.RetainedPackedWeightBytes);
        Assert.Empty(graph.PackedWeights);Assert.Empty(graph.PackedConvWeights);Assert.Empty(graph.PackedLstmWeights);
    }
}
