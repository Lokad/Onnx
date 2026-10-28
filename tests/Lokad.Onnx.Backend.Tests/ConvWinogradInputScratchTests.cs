namespace Lokad.Onnx.Backend.Tests;

public class ConvWinogradInputScratchTests
{
    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)]
    public void PoisonedScratchCannotAffectLargerTransformOutputs(int channels)
    {
        int lanes = GraphConvPacking.Lanes;
        if (lanes == 0 || channels > 64 && lanes != 16) return;
        const int outputs = 32, height = 7, width = 33;
        var input = Enumerable.Repeat(1f, channels * height * width).ToArray();
        var weights = Enumerable.Repeat(1f / channels, outputs * channels * 9).ToArray();
        var prepared = ConvBlockedSpatial.PrepareWinograd4(weights, channels, outputs, lanes)!;
        Assert.True(ConvBlockedSpatial.PlanWinograd(channels, outputs, height, width,
            out int ni, out int np, out int no, 4));
        var transformed = new float[ni]; var products = new float[np];
        var blocked = new float[no]; var destination = new float[no];
        for (int repeat = 0; repeat < 2; repeat++)
        {
            Array.Fill(transformed, float.NaN); Array.Fill(products, float.NaN);
            Array.Fill(blocked, float.NaN); Array.Fill(destination, float.NaN);
            Assert.True(ConvBlockedSpatial.ExecuteWinograd(input, prepared, default, default,
                destination, transformed, products, blocked, channels, outputs, height, width, lanes, false, 4));
            for (int oc = 0; oc < outputs; oc++)
            for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                float expected = (y == 0 || y == height - 1 ? 2 : 3) * (x == 0 || x == width - 1 ? 2 : 3);
                float actual = destination[(oc * height + y) * width + x];
                Assert.True(float.IsFinite(actual) && Math.Abs(actual - expected) <= 1e-4f * expected);
            }
        }
    }
}
