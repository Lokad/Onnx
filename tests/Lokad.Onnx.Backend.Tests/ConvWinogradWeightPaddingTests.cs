namespace Lokad.Onnx.Backend.Tests;

public partial class ConvBlockedSpatialTests
{
    [Theory]
    [InlineData(128)]
    [InlineData(256)]
    public void LargerWeightShapeHonorsThePerWeightCapIncludingPadding(int channels)
    {
        const int outputs = 48;
        int expected = 36 * (channels + 1) * outputs;
        Assert.True(GraphConvPacking.WinogradShape(channels, outputs, 64L * channels * 9, out int elements, 4));
        Assert.Equal(expected, elements);
        long directAtCap = GraphPacking.MaxPackedBytes / 4 - expected;
        Assert.True(GraphConvPacking.WinogradShape(channels, outputs, directAtCap, out elements, 4));
        Assert.Equal(expected, elements);
        Assert.False(GraphConvPacking.WinogradShape(channels, outputs, directAtCap + 1, out _, 4));
    }

    [Theory]
    [InlineData(128)]
    [InlineData(256)]
    public void LargerWeightPanelsInitializePaddingAndKeepRefusal(int channels)
    {
        if (GraphConvPacking.Lanes != 16) return;
        const int outputs = 48, height = 7, width = 33;
        var input = Enumerable.Repeat(1f, channels * height * width).ToArray();
        var weights = Enumerable.Repeat(1f / channels, outputs * channels * 9).ToArray();
        var prepared = ConvBlockedSpatial.PrepareWinograd4(weights, channels, outputs, 16)!;
        Assert.Equal(36 * (channels + 1) * outputs, prepared.Length);
        for (int component = 0; component < 36; component++)
        for (int panel = 0; panel < outputs / 16; panel++)
        for (int lane = 0; lane < 16; lane++)
            Assert.Equal(0, BitConverter.SingleToInt32Bits(prepared[(component * outputs / 16 + panel) * (channels + 1) * 16 + channels * 16 + lane]));
        Assert.True(ConvBlockedSpatial.PlanWinograd(channels, outputs, height, width, out int ni, out int np, out int no, 4));
        var transformed = new float[ni]; var products = new float[np]; var blocked = new float[no];
        var destination = Enumerable.Repeat(192837f, no).ToArray();
        Assert.True(ConvBlockedSpatial.ExecuteWinograd(input, prepared, default, default, destination,
            transformed, products, blocked, channels, outputs, height, width, 16, false, 4));
        for (int oc = 0; oc < outputs; oc++)
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            float expected = (y == 0 || y == height - 1 ? 2 : 3) * (x == 0 || x == width - 1 ? 2 : 3);
            Assert.True(Math.Abs(destination[(oc * height + y) * width + x] - expected) <= 1e-4f * expected);
        }
        Array.Fill(destination, 192837f);
        prepared[channels * 16] = float.NaN;
        Assert.False(ConvBlockedSpatial.ExecuteWinograd(input, prepared, default, default, destination,
            transformed, products, blocked, channels, outputs, height, width, 16, false, 4));
        Assert.All(destination, value => Assert.Equal(192837f, value));
    }

    [Theory]
    [InlineData(128, -1)] [InlineData(128, 0)] [InlineData(128, 1)]
    [InlineData(256, -1)] [InlineData(256, 0)] [InlineData(256, 1)]
    public void LargerWeightPaddingCountsAgainstTheExactGraphBudget(int channels, int delta)
    {
        if (GraphConvPacking.Lanes != 16) return;
        const int outputs = 48;
        long direct = Bytes(channels, outputs), optional = 36L * (channels + 1) * outputs * 4;
        var graph = Graph(channels, outputs, 1, 1, 1, direct + optional + delta, false);
        graph.PrepareWinograd4Weights();
        var record = Assert.Single(graph.PackedConvWeights).Value;
        Assert.Equal(delta >= 0, record.WinogradValues is not null);
        Assert.Equal(direct + (delta >= 0 ? optional : 0), graph.RetainedPackedWeightBytes);
        Close(Reference(graph), Run(graph).ToArray());
    }
}
