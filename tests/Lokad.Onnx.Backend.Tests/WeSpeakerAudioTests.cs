namespace Lokad.Onnx.Backend.Tests;

using System.Text.Json;

public class WeSpeakerAudioTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void CompleteFeaturesAgreeWithIndependentDirectFourierReference(int index)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "wespeaker-frontend.json")));
        var item = document.RootElement.GetProperty("cases")[index];
        float[] samples = item.GetProperty("input").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        var bits = samples.Select(BitConverter.SingleToInt32Bits).ToArray();
        var output = WeSpeakerAudio.LogMelFilterbank(samples, 16000, CancellationToken.None);
        Assert.Equal(item.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()), output.Dimensions.ToArray());
        var expected = item.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        var actual = output.ToArray(); Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < actual.Length; i++)
            Assert.True(float.IsFinite(actual[i]) && Math.Abs((double)actual[i] - expected[i]) / Math.Max(1, Math.Abs((double)expected[i])) <= 1e-4,
                $"{item.GetProperty("kind").GetString()} [{i}]: {actual[i]:R} versus {expected[i]:R}");
        Assert.Equal(bits, samples.Select(BitConverter.SingleToInt32Bits));
    }

    [Theory]
    [InlineData(400, 1)]
    [InlineData(559, 1)]
    [InlineData(560, 2)]
    [InlineData(561, 2)]
    [InlineData(480000, 2998)]
    public void UsesOnlyCompleteFrames(int length, int frames)
    {
        var output = WeSpeakerAudio.LogMelFilterbank(new float[length], 16000, CancellationToken.None);
        Assert.Equal(new[] { 1, frames, 80 }, output.Dimensions.ToArray());
        Assert.All(output.ToArray(), value => Assert.Equal(0f, value));
    }

    [Fact]
    public void UnusedTailDoesNotChangeFeaturesButIsStillValidated()
    {
        var samples = Enumerable.Range(0, 719).Select(i => (i % 19 - 9) / 16f).ToArray();
        var expected = WeSpeakerAudio.LogMelFilterbank(samples.AsSpan(0, 560), 16000, CancellationToken.None).ToArray();
        Array.Fill(samples, .75f, 560, 159);
        Assert.Equal(expected, WeSpeakerAudio.LogMelFilterbank(samples, 16000, CancellationToken.None).ToArray());
        samples[^1] = float.NaN;
        Assert.Throws<ArgumentException>(() => WeSpeakerAudio.LogMelFilterbank(samples, 16000, CancellationToken.None));
    }

    [Theory]
    [InlineData(-1f)]
    [InlineData(.25f)]
    [InlineData(1f)]
    public void ConstantFramesHaveZeroCenteredFeatures(float value)
    {
        var output = WeSpeakerAudio.LogMelFilterbank(Enumerable.Repeat(value, 16000).ToArray(), 16000, CancellationToken.None);
        Assert.All(output.ToArray(), item => Assert.Equal(0f, item));
    }

    [Fact]
    public void SingleFrameCenteringIsZeroForAnySignal()
    {
        var samples = Enumerable.Range(0, 400).Select(i => (i % 17 - 8) / 16f).ToArray();
        Assert.All(WeSpeakerAudio.LogMelFilterbank(samples, 16000, CancellationToken.None).ToArray(), value => Assert.Equal(0f, value));
    }

    [Theory]
    [InlineData(560)]
    [InlineData(16000)]
    public void FrameMeanRemovalPreservesSmallSignalUnderExactDcOffset(int length)
    {
        // Every value and its DC-shifted counterpart are exactly representable
        // in float. Frame centering should remove the constant before the FFT.
        uint state = 20260920;
        var samples = new float[length];
        for (int i = 0; i < length; i++)
        {
            state = unchecked(state * 1664525 + 1013904223);
            samples[i] = ((int)(state >> 27) - 16) / 1048576f;
        }
        var shifted = samples.Select(value => value + .5f).ToArray();
        Assert.Equal(samples, shifted.Select(value => value - .5f));
        var expected = WeSpeakerAudio.LogMelFilterbank(samples, 16000, CancellationToken.None).ToArray();
        var actual = WeSpeakerAudio.LogMelFilterbank(shifted, 16000, CancellationToken.None).ToArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.True(double.IsFinite(actual[i]) && Math.Abs((double)actual[i] - expected[i]) / Math.Max(1, Math.Abs((double)expected[i])) <= 1e-4,
                $"DC invariance at feature {i}: {actual[i]:R} versus {expected[i]:R}");
    }

    [Fact]
    public async Task ResultsAndConcurrentScratchAreIndependent()
    {
        var samples = Enumerable.Range(0, 16000).Select(i => (i % 97 - 48) / 64f).ToArray();
        var before = samples.ToArray();
        var first = WeSpeakerAudio.LogMelFilterbank(samples, 16000, CancellationToken.None); var held = first.ToArray();
        var tasks = Enumerable.Range(0, 3).Select(_ => Task.Run(() => WeSpeakerAudio.LogMelFilterbank(samples, 16000, CancellationToken.None))).ToArray();
        var results = await Task.WhenAll(tasks);
        foreach (var result in results) Assert.Equal(held, result.ToArray());
        Assert.Equal(before, samples);
        results[0].Buffer.Span[0] = 999; samples[0] = 1;
        Assert.Equal(held, first.ToArray()); Assert.Equal(held, results[1].ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(399)]
    [InlineData(480001)]
    public void RefusesUnsupportedLengths(int length) => Assert.Throws<ArgumentOutOfRangeException>(
        () => WeSpeakerAudio.LogMelFilterbank(new float[length], 16000, CancellationToken.None));

    [Theory]
    [InlineData(0)]
    [InlineData(8000)]
    [InlineData(44100)]
    public void RefusesUnsupportedRates(int rate) => Assert.Throws<ArgumentOutOfRangeException>(
        () => WeSpeakerAudio.LogMelFilterbank(new float[400], rate, CancellationToken.None));

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    [InlineData(1.001f)]
    [InlineData(-1.001f)]
    public void RefusesNonfiniteOrUnnormalizedSamples(float value)
    {
        var samples = new float[400]; samples[^1] = value;
        Assert.Throws<ArgumentException>(() => WeSpeakerAudio.LogMelFilterbank(samples, 16000, CancellationToken.None));
        Assert.Equal(value, samples[^1]);
    }

    [Fact]
    public void CancellationAndInvalidRequestsDoNotAffectLaterCalls()
    {
        var samples = new float[560]; samples[10] = .5f;
        var expected = WeSpeakerAudio.LogMelFilterbank(samples, 16000, CancellationToken.None).ToArray();
        using var cancellation = new CancellationTokenSource(); cancellation.Cancel();
        Assert.Throws<OperationCanceledException>(() => WeSpeakerAudio.LogMelFilterbank(samples, 16000, cancellation.Token));
        Assert.Throws<ArgumentOutOfRangeException>(() => WeSpeakerAudio.LogMelFilterbank(samples, 8000, CancellationToken.None));
        Assert.Equal(expected, WeSpeakerAudio.LogMelFilterbank(samples, 16000, CancellationToken.None).ToArray());
    }
}
