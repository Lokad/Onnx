using System.Numerics;

namespace Lokad.Onnx.Backend.Tests;

public class LstmVectorActivationTests
{
    [Fact]
    public void TanhKeepsScalarAccuracyAcrossItsRangeAndFloatBitPatterns()
    {
        var random = new Random(12894); int width = Vector<float>.Count;
        var values = new float[width]; var actual = new float[width];
        for (int sample = 0; sample < 10000; sample++)
        {
            for (int i = 0; i < width; i++) values[i] = sample % 2 == 0
                ? (sample * width + i - 25000) * .0005f
                : BitConverter.Int32BitsToSingle((int)random.NextInt64(int.MinValue, (long)int.MaxValue + 1));
            CPUExecutionProvider.LstmVectorTanh(new Vector<float>(values)).CopyTo(actual);
            for (int i = 0; i < width; i++)
            {
                float expected = MathF.Tanh(values[i]);
                if (float.IsNaN(expected)) Assert.True(float.IsNaN(actual[i]));
                else Assert.True(float.IsFinite(actual[i]) && Math.Abs((double)expected - actual[i]) <= 5e-7,
                    $"tanh({values[i]:R}): expected {expected:R}, got {actual[i]:R}");
            }
        }
    }

    [Fact]
    public void TanhRetainsInfinityZerosAndNan()
    {
        foreach (float value in new[] { float.PositiveInfinity, float.NegativeInfinity, 0f, -0f, float.NaN })
        {
            var actual = CPUExecutionProvider.LstmVectorTanh(new Vector<float>(value));
            for (int i = 0; i < Vector<float>.Count; i++)
                if (float.IsNaN(value)) Assert.True(float.IsNaN(actual[i]));
                else Assert.Equal(BitConverter.SingleToInt32Bits(MathF.Tanh(value)), BitConverter.SingleToInt32Bits(actual[i]));
        }
    }
}
