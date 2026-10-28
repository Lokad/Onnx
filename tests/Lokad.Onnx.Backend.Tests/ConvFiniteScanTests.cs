namespace Lokad.Onnx.Backend.Tests;

public class ConvFiniteScanTests
{
    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(7)] [InlineData(8)]
    [InlineData(15)] [InlineData(16)] [InlineData(17)] [InlineData(31)]
    [InlineData(32)] [InlineData(33)] [InlineData(63)] [InlineData(64)]
    [InlineData(65)] [InlineData(129)] [InlineData(257)]
    public void ScanMatchesIndependentOracleAtEveryPositionWithoutChangingInputs(int length)
    {
        float[] special = { 0f, -0f, float.Epsilon, -float.Epsilon,
            float.MaxValue, -float.MaxValue, float.PositiveInfinity, float.NegativeInfinity,
            BitConverter.Int32BitsToSingle(0x7fc12345),
            BitConverter.Int32BitsToSingle(unchecked((int)0xffa54321)) };
        var values = Enumerable.Repeat(.25f, length).ToArray();
        Assert.True(ConvBlockedSpatial.Finite(values));
        foreach (float value in special)
        for (int index = 0; index < length; index++)
        {
            values[index] = value;
            var before = values.Select(BitConverter.SingleToInt32Bits).ToArray();
            Assert.Equal(values.All(float.IsFinite), ConvBlockedSpatial.Finite(values));
            Assert.Equal(before, values.Select(BitConverter.SingleToInt32Bits));
            values[index] = .25f;
        }
    }
}
