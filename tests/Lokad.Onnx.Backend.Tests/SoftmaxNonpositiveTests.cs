using System.Numerics;

namespace Lokad.Onnx.Backend.Tests;

public class SoftmaxNonpositiveTests
{
    static void CheckDomain(Vector<float> input)
    {
        var reference = MathOps.ExpVectorEstrinReference(input);
        var pruned = MathOps.ExpVectorEstrinPrunedInline(input);
        var actual = MathOps.ExpVectorNonpositive(input);
        for (int lane = 0; lane < Vector<float>.Count; lane++)
        {
            Assert.Equal(BitConverter.SingleToInt32Bits(reference[lane]), BitConverter.SingleToInt32Bits(actual[lane]));
            Assert.Equal(BitConverter.SingleToInt32Bits(pruned[lane]), BitConverter.SingleToInt32Bits(actual[lane]));
        }
    }

    [Fact]
    public void NonpositiveRandomBitsAndActiveRangeMatchBothReferences()
    {
        var random = new Random(20260920);
        var values = new float[Vector<float>.Count];
        for (int sample = 0; sample < 262144; sample++)
        {
            for (int lane = 0; lane < values.Length; lane++)
                values[lane] = sample % 2 == 0 ? (float)(-random.NextDouble() * 90)
                    : BitConverter.Int32BitsToSingle(random.Next() | int.MinValue);
            CheckDomain(new Vector<float>(values));
        }
    }

    [Fact]
    public void RoundingCutoffAndExceptionalNeighborsMatchBothReferences()
    {
        foreach (float edge in new[] { 0f, -0f, -float.Epsilon, float.NaN,
            BitConverter.Int32BitsToSingle(0x7FA12345), BitConverter.Int32BitsToSingle(unchecked((int)0xFFA12345)),
            float.NegativeInfinity, -float.MaxValue })
            CheckDomain(new Vector<float>(edge));

        // Half-integer range reductions and both underflow/lower-clamp boundaries.
        for (int n = 0; n < 130; n++)
        {
            float center = -(n + 0.5f) / 1.44269504088896341f;
            int bits = BitConverter.SingleToInt32Bits(center);
            for (int delta = -32; delta <= 32; delta++)
                CheckDomain(new Vector<float>(BitConverter.Int32BitsToSingle(bits + delta)));
        }
        foreach (float center in new[] { -88.722839f, -87.33654475f })
        {
            int bits = BitConverter.SingleToInt32Bits(center);
            for (int delta = -4096; delta <= 4096; delta++)
                CheckDomain(new Vector<float>(BitConverter.Int32BitsToSingle(bits + delta)));
        }
    }

    [Fact]
    public void GeneralExponentialRetainsPositiveInputContract()
    {
        foreach (float input in new[] { float.Epsilon, 0.3f, 1f, 20f, 87f, 88.722839f,
            MathF.BitIncrement(88.722839f), float.MaxValue, float.PositiveInfinity })
        {
            var vector = new Vector<float>(input);
            var expected = MathOps.ExpVectorEstrinReference(vector);
            var actual = MathOps.ExpVectorEstrin(vector);
            for (int lane = 0; lane < Vector<float>.Count; lane++)
                Assert.Equal(BitConverter.SingleToInt32Bits(expected[lane]), BitConverter.SingleToInt32Bits(actual[lane]));
        }
    }
}
