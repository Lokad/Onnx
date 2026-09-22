namespace Lokad.Onnx.Backend.Tests;

using System.Reflection;

public class SparseMelTests
{
    delegate DenseTensor<float> Frontend(ReadOnlySpan<float> input, int rate, CancellationToken cancellation);
    static readonly Frontend Dense = DenseWeSpeakerReference.LogMelFilterbank;

    public static IEnumerable<object[]> Cases()
    {
        foreach (int length in new[] { 400, 559, 560, 561, 16000, 160000, 480000 })
        foreach (int signal in Enumerable.Range(0, 8))
            yield return new object[] { length, signal };
    }

    static float[] Signal(int length, int kind)
    {
        var result = new float[length]; uint random = 0x82731645;
        for (int i = 0; i < length; i++)
        {
            random ^= random << 13; random ^= random >> 17; random ^= random << 5;
            result[i] = kind switch
            {
                0 => 0f, 1 => -1f, 2 => 1f, 3 => i % 2 == 0 ? -1f : 1f,
                4 => (random % 65537 - 32768f) / 32768f,
                5 => i % 2 == 0 ? -float.Epsilon : float.Epsilon,
                6 => (i % 37 - 18) * 1e-15f,
                _ => i % 2 == 0 ? -0f : 0f
            };
        }
        return result;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void CompleteFeaturesPreserveDenseBitsForBoundariesAndSignals(int length, int signal)
    {
        var input = Signal(length, signal);
        var inputBits = input.Select(BitConverter.SingleToInt32Bits).ToArray();
        var expected = Dense(input, 16000, CancellationToken.None);
        var actual = WeSpeakerAudio.LogMelFilterbank(input, 16000, CancellationToken.None);
        Assert.Equal(expected.Dimensions.ToArray(), actual.Dimensions.ToArray());
        Assert.Equal(expected.ToArray().Select(BitConverter.SingleToInt32Bits), actual.ToArray().Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(inputBits, input.Select(BitConverter.SingleToInt32Bits));
        Assert.All(actual.ToArray(), value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void ActualCoefficientSupportOmitsOnlyZerosAndPreservesAllNonzeroOrder()
    {
        var flags = BindingFlags.Static | BindingFlags.NonPublic;
        var weights = (float[])typeof(WeSpeakerAudio).GetField("MelWeights", flags)!.GetValue(null)!;
        var support = ((int Start, int End)[])typeof(WeSpeakerAudio).GetField("MelSupport", flags)!.GetValue(null)!;
        Assert.Equal(80 * 256, weights.Length); Assert.Equal(80, support.Length);
        int retained = 0;
        for (int band = 0; band < 80; band++)
        {
            var (first, last) = support[band];
            Assert.InRange(first, 0, 256); Assert.InRange(last, first, 256);
            for (int bin = 0; bin < 256; bin++)
            {
                float weight = weights[band * 256 + bin];
                Assert.True(float.IsFinite(weight)); Assert.InRange(weight, 0f, 1f);
                if (bin < first || bin >= last) Assert.Equal(0f, weight);
                else retained++;
            }
            if (first < last)
            {
                Assert.True(weights[band * 256 + first] > 0);
                Assert.True(weights[band * 256 + last - 1] > 0);
            }
        }
        Assert.InRange(retained, 1, 20479);
    }

    [Fact]
    public async Task ConcurrentFeaturesOwnOutputsAndPreserveHeldResults()
    {
        var input = Signal(16000, 4);
        var first = WeSpeakerAudio.LogMelFilterbank(input, 16000, CancellationToken.None);
        var held = first.ToArray();
        var expected = Dense(input, 16000, CancellationToken.None).ToArray();
        var results = await Task.WhenAll(Enumerable.Range(0, 3).Select(_ => Task.Run(
            () => WeSpeakerAudio.LogMelFilterbank(input, 16000, CancellationToken.None))));
        foreach (var output in results)
            Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), output.ToArray().Select(BitConverter.SingleToInt32Bits));
        results[0].Buffer.Span[0] = 12345;
        Assert.Equal(held.Select(BitConverter.SingleToInt32Bits), first.ToArray().Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), results[1].ToArray().Select(BitConverter.SingleToInt32Bits));
    }
}
