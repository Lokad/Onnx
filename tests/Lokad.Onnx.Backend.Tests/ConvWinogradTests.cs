namespace Lokad.Onnx.Backend.Tests;

public partial class ConvBlockedSpatialTests
{
    static void Close(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(expected[i]) && float.IsFinite(actual[i]), $"Nonfinite value at {i}");
            double error = Math.Abs((double)actual[i] - expected[i]) / Math.Max(1, Math.Abs((double)expected[i]));
            Assert.True(error <= 1e-4, $"Scaled error {error:R} at {i}");
        }
    }

    static long WinogradBytes(int c, int m) => GraphConvPacking.Lanes == 0 ? 0 : 16L * c * m * 4;
    static long WinogradScratch(int c, int m, int h, int w) => 4L * (16 * c * 8 + 16 * m * 8 + m * h * w);

    [Theory]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(1)]
    public void OptionalWinogradRespectsExactSharedBudget(int delta)
    {
        long direct = Bytes(), optional = WinogradBytes(16, 32);
        var graph = Graph(Math.Max(0, direct + optional + delta));
        graph.Prepare();
        if (GraphConvPacking.Lanes == 0) { Assert.Empty(graph.PackedConvWeights); return; }
        var record = Assert.Single(graph.PackedConvWeights).Value;
        Assert.Equal(delta >= 0, record.WinogradValues is not null);
        Assert.Equal(direct + (delta >= 0 ? optional : 0), graph.RetainedPackedWeightBytes);
        var directCopy = record.Values.ToArray(); var optionalCopy = record.WinogradValues?.ToArray();
        graph.Prepare(); Assert.Same(record, Assert.Single(graph.PackedConvWeights).Value);
        var expected = Reference(graph); var output = Run(graph); var held = output.ToArray();
        if (delta >= 0) Close(expected, held); else Equal(expected, held);
        Equal(held, Run(graph).ToArray()); Equal(held, output.ToArray()); Equal(directCopy, record.Values);
        if (optionalCopy is not null) Equal(optionalCopy, record.WinogradValues!);
        graph.InvalidatePreparation(); Assert.Empty(graph.PackedConvWeights); Assert.Equal(0, graph.RetainedPackedWeightBytes);
    }

    [Fact]
    public void AllDirectCandidatesPrecedeOptionalWinogradAllocation()
    {
        long direct = Bytes(), optional = WinogradBytes(16, 32);
        var graph = Graph(2 * direct + optional);
        graph.Initializers["second"] = new DenseTensor<float>(Values(32 * 16 * 9, 7), new[] { 32, 16, 3, 3 });
        var node = graph.Nodes[0]; node.Name = "second"; node.Inputs = new[] { "x", "second", "b" }; node.Outputs = new[] { "second-y" };
        graph.Nodes.Add(node); graph.Prepare();
        if (GraphConvPacking.Lanes == 0) { Assert.Empty(graph.PackedConvWeights); return; }
        Assert.Equal(2, graph.PackedConvWeights.Count);
        Assert.Single(graph.PackedConvWeights.Values.Where(r => r.WinogradValues is not null));
        Assert.Equal(2 * direct + optional, graph.RetainedPackedWeightBytes);
        var records = graph.PackedConvWeights.Values.ToArray(); graph.Prepare();
        Assert.Equal(records, graph.PackedConvWeights.Values.ToArray());
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(1)]
    public void MatrixDirectAndOptionalWeightsShareTheBudget(int delta)
    {
        long direct = Bytes(), optional = WinogradBytes(16, 32);
        var graph = Graph(Math.Max(16, direct + optional + 16 + delta));
        graph.Inputs["mx"] = DenseTensor<float>.OfValues(new float[,] { { 1, 2 } });
        graph.Initializers["mw"] = DenseTensor<float>.OfValues(new float[,] { { 1, 2 }, { 3, 4 } });
        graph.Outputs["my"] = Lokad.Onnx.Tensor<float>.Zeros(1, 2);
        graph.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, OpTypeName = "MatMul", Domain = "", OpsetVersion = 18,
            Inputs = new[] { "mx", "mw" }, Outputs = new[] { "my" } });
        graph.Prepare();
        long expected = 16 + direct + (delta >= 0 ? optional : 0);
        Assert.Equal(expected, graph.RetainedPackedWeightBytes);
        graph.Prepare(); Assert.Equal(expected, graph.RetainedPackedWeightBytes);
        Assert.True(expected <= graph.MaximumPackedWeightBytes);
    }

    [Fact]
    public void StrideTwoKeepsOnlyTheDirectRepresentation()
    {
        var graph = Graph(c: 32, stride: 2); var expected = Reference(graph); Equal(expected, Run(graph).ToArray());
        if (GraphConvPacking.Lanes != 0)
        {
            Assert.Null(Assert.Single(graph.PackedConvWeights).Value.WinogradValues);
            Assert.Equal(Bytes(32, 32), graph.RetainedPackedWeightBytes);
        }
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(5, 33)]
    [InlineData(6, 34)]
    [InlineData(5, 65)]
    [InlineData(6, 66)]
    public void WinogradGraphMatchesIndependentDoubleConvolution(int h, int w)
    {
        const int c = 16, m = 32;
        var graph = Graph(c, m, h, w, 1, long.MaxValue, false);
        var x = Operand(graph, "x").ToArray(); var weights = Operand(graph, "w").ToArray(); var bias = Operand(graph, "b").ToArray();
        var reference = new float[m * h * w];
        for (int oc = 0; oc < m; oc++)
        for (int y = 0; y < h; y++)
        for (int z = 0; z < w; z++)
        {
            double sum = 0;
            for (int ic = 0; ic < c; ic++)
            for (int ky = 0; ky < 3; ky++)
            for (int kx = 0; kx < 3; kx++)
            {
                int iy = y + ky - 1, ix = z + kx - 1;
                if ((uint)iy < (uint)h && (uint)ix < (uint)w)
                    sum += (double)x[(ic * h + iy) * w + ix] * weights[((oc * c + ic) * 3 + ky) * 3 + kx];
            }
            reference[(oc * h + y) * w + z] = (float)(sum + bias[oc]);
        }
        var first = Run(graph); var held = first.ToArray(); Close(reference, held);
        Equal(held, Run(graph).ToArray()); Equal(held, first.ToArray());
        Equal(x, Operand(graph, "x").ToArray()); Equal(weights, Operand(graph, "w").ToArray());
        if (GraphConvPacking.Lanes != 0)
        {
            Assert.NotNull(Assert.Single(graph.PackedConvWeights).Value.WinogradValues);
            Assert.Equal(WinogradScratch(c, m, h, w), graph.LastScratchBytes);
        }
    }

    [Fact]
    public void FiniteWeightTransformOverflowRetainsDirectFallback()
    {
        var graph = Graph(); Operand(graph, "w").Buffer.Span.Fill(float.MaxValue);
        var expected = Reference(graph); Equal(expected, Run(graph).ToArray());
        if (GraphConvPacking.Lanes != 0)
        {
            Assert.Null(Assert.Single(graph.PackedConvWeights).Value.WinogradValues);
            Assert.Equal(Bytes(), graph.RetainedPackedWeightBytes);
        }
    }

    [Theory]
    [InlineData("x")]
    [InlineData("b")]
    public void RuntimeWinogradRefusalKeepsDirectResultAndRecovers(string operand)
    {
        var graph = Graph(); var held = Run(graph); var snapshot = held.ToArray();
        var target = Operand(graph, operand); var original = target.ToArray();
        target.Buffer.Span.Fill(float.MaxValue);
        var expected = Reference(graph); Equal(expected, Run(graph).ToArray()); Equal(snapshot, held.ToArray());
        original.AsSpan().CopyTo(target.Buffer.Span);
        Equal(snapshot, Run(graph).ToArray()); Equal(snapshot, held.ToArray());
    }

    [Fact]
    public void OptionalPreparationAndScratchBoundsRejectOversizeWithoutAllocation()
    {
        Assert.True(GraphConvPacking.WinogradShape(16, 32, 16 * 32 * 9, out int elements)); Assert.Equal(16 * 16 * 32, elements);
        Assert.False(GraphConvPacking.WinogradShape(int.MaxValue, int.MaxValue, long.MaxValue, out _));
        Assert.False(GraphConvPacking.WinogradShape(4096, 4096, 9L * 4096 * 4096, out _));
        Assert.False(GraphConvPacking.WinogradShape(16, 32, GraphPacking.MaxPackedBytes / 4, out _));
        Assert.True(ConvBlockedSpatial.PlanWinograd(32, 32, 80, 998, out int a, out int b, out int d));
        Assert.Equal(WinogradScratch(32, 32, 80, 998), (long)(a + b + d) * 4);
        Assert.False(ConvBlockedSpatial.PlanWinograd(256, 256, 1024, 1024, out _, out _, out _));
        Assert.False(ConvBlockedSpatial.PlanWinograd(int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, out _, out _, out _));
    }
}
