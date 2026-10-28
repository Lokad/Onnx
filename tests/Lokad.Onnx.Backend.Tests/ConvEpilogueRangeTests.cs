namespace Lokad.Onnx.Backend.Tests;

public class ConvEpilogueRangeTests
{
    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(7)] [InlineData(8)]
    [InlineData(15)] [InlineData(16)] [InlineData(17)] [InlineData(33)]
    public void CombinedBoundMatchesFiniteMagnitudeOracleWithoutChangingInputs(int length)
    {
        const float limit = float.MaxValue / 4;
        float[] special = { 0f, -0f, float.Epsilon, -float.Epsilon, limit, -limit,
            MathF.BitIncrement(limit), -MathF.BitIncrement(limit), float.MaxValue, -float.MaxValue,
            float.PositiveInfinity, float.NegativeInfinity,
            BitConverter.Int32BitsToSingle(0x7fc12345),
            BitConverter.Int32BitsToSingle(unchecked((int)0xffa54321)) };
        var values = Enumerable.Repeat(.25f, length).ToArray();
        Assert.True(ConvBlockedSpatial.EpilogueRange(values));
        foreach (float value in special)
        foreach (int index in new[] { 0, length / 2, Math.Max(0, length - 1) }.Distinct())
        {
            if (length == 0) continue;
            values[index] = value;
            var before = values.Select(BitConverter.SingleToInt32Bits).ToArray();
            bool expected = values.All(x => float.IsFinite(x) && MathF.Abs(x) <= limit);
            Assert.Equal(expected, ConvBlockedSpatial.EpilogueRange(values));
            Assert.Equal(before, values.Select(BitConverter.SingleToInt32Bits));
            values[index] = .25f;
        }
    }
}
