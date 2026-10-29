namespace Lokad.Onnx.Backend.Tests;

public partial class ConvBlockedSpatialTests
{
    [Theory]
    [InlineData(16, 1, 1)]
    [InlineData(16, 1, 32)]
    [InlineData(32, 3, 35)]
    [InlineData(64, 7, 36)]
    [InlineData(16, 9, 64)]
    [InlineData(16, 5, 33)]
    [InlineData(32, 6, 34)]
    [InlineData(64, 7, 65)]
    [InlineData(32, 13, 135)]
    [InlineData(16, 7, 32)]
    [InlineData(64, 5, 33)]
    [InlineData(128, 7, 80)]
    [InlineData(256, 5, 36)]
    public void LargerOptInTilesMatchDoubleConvolution(int c, int h, int w)
        => CheckLargerTiles(c, 32, h, w, false);

    [Theory]
    [InlineData(16, 48)]
    [InlineData(128, 48)]
    [InlineData(256, 48)]
    [InlineData(128, 128)]
    [InlineData(256, 256)]
    public void OptInShapesMatchDoubleConvolutionOnNormalData(int c, int m)
        => CheckLargerTiles(c, m, 7, 33, true);

    static void CheckLargerTiles(int c, int m, int h, int w, bool normal)
    {
        var graph = Graph(c, m, h, w, 1, long.MaxValue, false);
        var random = new Random(1234);
        float Value(double scale) => (float)(normal
            ? Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble()) * scale
            : (random.NextDouble() * 2 - 1) * scale);
        var x = Operand(graph, "x"); var weights = Operand(graph, "w");
        for (int i = 0; i < x.Length; i++) x.Buffer.Span[i] = Value(2);
        for (int i = 0; i < weights.Length; i++) weights.Buffer.Span[i] = Value(.1);
        var inputCopy = x.ToArray(); var weightCopy = weights.ToArray(); var bias = Operand(graph, "b").ToArray();
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
                    sum += (double)inputCopy[(ic * h + iy) * w + ix] * weightCopy[((oc * c + ic) * 3 + ky) * 3 + kx];
            }
            reference[(oc * h + y) * w + z] = (float)(sum + bias[oc]);
        }
        graph.PrepareWinograd4Weights();
        var record = graph.PackedConvWeights.Values.SingleOrDefault();
        if (record is not null)
        {
            int tile = c <= 64 || GraphConvPacking.Lanes == 16 ? 4 : 2;
            Assert.Equal(tile, record.OutputTileSize);
            int weightChannels = tile == 4 && c > 64 ? c + 1 : c;
            Assert.Equal((tile + 2) * (tile + 2) * weightChannels * m, record.WinogradValues!.Length);
        }
        var held = Run(graph); var snapshot = held.ToArray(); Close(reference, snapshot);
        Equal(snapshot, Run(graph).ToArray()); Equal(snapshot, held.ToArray());
        Equal(inputCopy, x.ToArray()); Equal(weightCopy, weights.ToArray());
        graph.PrepareWinograd4Weights();
        if (record is not null) Assert.Same(record, Assert.Single(graph.PackedConvWeights).Value);
        graph.Initializers["w"] = new DenseTensor<float>(new float[c * m * 9], new[] { m, c, 3, 3 });
        var changed = Run(graph);
        for (int oc = 0; oc < m; oc++)
        for (int i = 0; i < h * w; i++) Assert.Equal(bias[oc], changed.GetValue(oc * h * w + i));
        Equal(snapshot, held.ToArray());
        graph.Initializers["w"] = weights;
        graph.InvalidatePreparation();
        Equal(snapshot, Run(graph).ToArray());
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(1)]
    public void LargerTilesRespectExactBudget(int delta)
    {
        long direct = Bytes(), optional = GraphConvPacking.Lanes == 0 ? 0 : 36L * 16 * 32 * 4;
        var graph = Graph(Math.Max(0, direct + optional + delta));
        graph.PrepareWinograd4Weights();
        if (GraphConvPacking.Lanes == 0) { Assert.Empty(graph.PackedConvWeights); return; }
        var record = Assert.Single(graph.PackedConvWeights).Value;
        Assert.Equal(delta >= 0, record.WinogradValues is not null);
        Assert.Equal(direct + (delta >= 0 ? optional : 0), graph.RetainedPackedWeightBytes);
        graph.Prepare(); Assert.Same(record, Assert.Single(graph.PackedConvWeights).Value);
        Close(Reference(graph), Run(graph).ToArray());
    }

    [Fact]
    public void LargerReductionKeepsTheSmallerTransform()
    {
        var graph = Graph(384, 32, 5, 9, 1, long.MaxValue, false);
        graph.PrepareWinograd4Weights();
        if (GraphConvPacking.Lanes != 0)
        {
            var record = Assert.Single(graph.PackedConvWeights).Value;
            Assert.Equal(2, record.OutputTileSize);
            Assert.Equal(16 * 384 * 32, record.WinogradValues!.Length);
        }
        Close(Reference(graph), Run(graph).ToArray());
    }

    [Theory]
    [InlineData(0)] [InlineData(31)] [InlineData(32)]
    [InlineData(511)] [InlineData(512)] [InlineData(18431)]
    public void CorruptedPreparedComponentsFallBackWithoutChangingHeldOutputs(int index)
    {
        var graph = Graph(); graph.PrepareWinograd4Weights();
        if (GraphConvPacking.Lanes == 0) return;
        var prepared = Assert.Single(graph.PackedConvWeights).Value.WinogradValues!;
        var held = Run(graph); var snapshot = held.ToArray(); float original = prepared[index];
        try
        {
            foreach (float value in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity })
            {
                prepared[index] = value;
                Equal(Reference(graph), Run(graph).ToArray()); Equal(snapshot, held.ToArray());
            }
        }
        finally { prepared[index] = original; }
        Equal(snapshot, Run(graph).ToArray()); Equal(snapshot, held.ToArray());
    }

    [Theory]
    [InlineData("x")]
    [InlineData("b")]
    public void LargerTilesKeepNonfiniteAndOverflowFallback(string name)
    {
        var graph = Graph(); graph.PrepareWinograd4Weights();
        var held = Run(graph); var snapshot = held.ToArray();
        var target = Operand(graph, name); var original = target.ToArray();
        foreach (float value in new[] { float.MaxValue, float.PositiveInfinity, float.NaN })
        {
            target.Buffer.Span.Fill(value);
            Equal(Reference(graph), Run(graph).ToArray()); Equal(snapshot, held.ToArray());
            original.AsSpan().CopyTo(target.Buffer.Span);
            Equal(snapshot, Run(graph).ToArray());
        }
    }
}
