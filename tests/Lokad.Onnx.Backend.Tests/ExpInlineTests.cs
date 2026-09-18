using System.Numerics;
using System.Runtime.CompilerServices;

namespace Lokad.Onnx.Backend.Tests;

public class ExpInlineTests
{
    // Keep a reduction live across each exponential, as the softmax caller does.
    [MethodImpl(MethodImplOptions.NoInlining)]
    static Vector<float> InlineWithLiveSum(Vector<float> input, Vector<float> sum) =>
        sum + MathOps.ExpVectorEstrinPrunedInline(input);

    [Fact]
    public void RandomBitsActiveRangeAndExceptionalLanesAreBitIdentical()
    {
        var random = new Random(20260919);
        var values = new float[Vector<float>.Count];
        float[] edges = { -0f, 0f, float.Epsilon, -float.Epsilon, float.NaN,
            BitConverter.Int32BitsToSingle(0x7FA12345), float.PositiveInfinity, float.NegativeInfinity,
            -float.MaxValue, float.MaxValue, -88.722839f, MathF.BitIncrement(-88.722839f),
            MathF.BitDecrement(-88.722839f), 88.722839f, MathF.BitIncrement(88.722839f), -87f, -1f };
        for (int sample = 0; sample < 262144; sample++)
        {
            for (int lane = 0; lane < values.Length; lane++)
                values[lane] = sample < edges.Length ? edges[(sample + lane) % edges.Length]
                    : sample % 2 == 0 ? (float)(-random.NextDouble() * 90)
                    : BitConverter.Int32BitsToSingle((int)random.NextInt64(int.MinValue, (long)int.MaxValue + 1));
            var input = new Vector<float>(values);
            var expected = MathOps.ExpVectorEstrinPruned(input);
            var actual = MathOps.ExpVectorEstrinPrunedInline(input);
            var accumulated = InlineWithLiveSum(input, new Vector<float>(0.75f));
            var expectedSum = expected + new Vector<float>(0.75f);
            for (int lane = 0; lane < values.Length; lane++)
            {
                Assert.Equal(BitConverter.SingleToInt32Bits(expected[lane]), BitConverter.SingleToInt32Bits(actual[lane]));
                Assert.Equal(BitConverter.SingleToInt32Bits(expectedSum[lane]), BitConverter.SingleToInt32Bits(accumulated[lane]));
            }
        }
    }

    [Fact]
    public void RangeReductionAndUnderflowBoundaryNeighborsStayExact()
    {
        var values = new float[Vector<float>.Count];
        foreach (float center in new[] { -88.722839f, 88.722839f, -87.33654475f, -0.3465735903f, 0.3465735903f })
        {
            int bits = BitConverter.SingleToInt32Bits(center);
            for (int delta = -4096; delta <= 4096; delta++)
            {
                for (int lane = 0; lane < values.Length; lane++) values[lane] = BitConverter.Int32BitsToSingle(bits + delta + lane);
                var input = new Vector<float>(values);
                var expected = MathOps.ExpVectorEstrinPruned(input);
                var actual = MathOps.ExpVectorEstrinPrunedInline(input);
                for (int lane = 0; lane < values.Length; lane++)
                    Assert.Equal(BitConverter.SingleToInt32Bits(expected[lane]), BitConverter.SingleToInt32Bits(actual[lane]));
            }
        }
    }
}
