namespace Lokad.Onnx.Backend.Tests;

using System.Text.Json;

public class WhisperAudioTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(16000)]
    [InlineData(480013)]
    public void SilenceHasExactFeatures(int length)
    {
        var output = WhisperAudio.LogMelSpectrogram(new float[length], 16000);
        Assert.Equal(new[] { 1, 128, 3000 }, output.Dimensions.ToArray());
        Assert.All(output.ToArray(), value => Assert.Equal(-1.5f, value));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    public void FeaturesAgreeWithIndependentNumpyReference(int index)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "whisper-frontend.json")));
        var row = doc.RootElement.GetProperty("cases")[index];
        float[] samples = Samples(row.GetProperty("kind").GetString()!, row.GetProperty("length").GetInt32());
        var original = samples.Select(BitConverter.SingleToInt32Bits).ToArray();
        var actual = WhisperAudio.LogMelSpectrogram(samples, 16000);
        Assert.Equal(original, samples.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(new[] { 1, 128, 3000 }, actual.Dimensions.ToArray());
        var output = actual.ToArray();
        Assert.All(output, value => Assert.True(float.IsFinite(value)));
        int[] frames = row.GetProperty("frames").EnumerateArray().Select(f => f.GetInt32()).ToArray();
        double tolerance = doc.RootElement.GetProperty("absolute_tolerance").GetDouble();
        // NumPy advanced indexing stores the selected rows in [frame,mel] order.
        for (int f = 0; f < frames.Length; f++)
            for (int mel = 0; mel < 128; mel++)
            {
                float want = row.GetProperty("values")[f][mel].GetSingle();
                float got = output[mel * 3000 + frames[f]];
                Assert.True(Math.Abs((double)got - want) <= tolerance,
                    $"{row.GetProperty("name").GetString()} frame {frames[f]} mel {mel}: {got:R} != {want:R}");
            }
    }

    [Fact]
    public void TruncationMatchesExplicitThirtySecondClip()
    {
        float[] samples = Samples("noise", 480013);
        Assert.Equal(WhisperAudio.LogMelSpectrogram(samples.AsSpan(0, 480000), 16000).ToArray(),
            WhisperAudio.LogMelSpectrogram(samples, 16000).ToArray());
    }

    [Fact]
    public void PaddingMatchesExplicitSilenceAndResultsStayIndependent()
    {
        float[] samples = Samples("noise", 481);
        float[] padded = new float[480000];
        samples.CopyTo(padded, 0);
        var first = WhisperAudio.LogMelSpectrogram(samples, 16000);
        var held = first.ToArray();
        var second = WhisperAudio.LogMelSpectrogram(padded, 16000);
        Assert.Equal(held, second.ToArray());
        second.Buffer.Span[0] = 123;
        samples[0] = 123;
        Assert.Equal(held, first.ToArray());
    }

    [Fact]
    public async Task ConcurrentCallsOwnSeparateScratch()
    {
        float[] input = Samples("noise", 481);
        var expected = WhisperAudio.LogMelSpectrogram(input, 16000).ToArray();
        var results = await Task.WhenAll(Enumerable.Range(0, 3).Select(_ => Task.Run(() => WhisperAudio.LogMelSpectrogram(input, 16000).ToArray())));
        foreach (var result in results) Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(8000)]
    [InlineData(44100)]
    public void RefusesUnsupportedSampleRates(int rate) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => WhisperAudio.LogMelSpectrogram(new float[1], rate));

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void RefusesNonfiniteSamplesIncludingTruncatedTail(float value)
    {
        Assert.Throws<ArgumentException>(() => WhisperAudio.LogMelSpectrogram(new[] { value }, 16000));
        var longInput = new float[480001];
        longInput[^1] = value;
        Assert.Throws<ArgumentException>(() => WhisperAudio.LogMelSpectrogram(longInput, 16000));
    }

    static float[] Samples(string kind, int length)
    {
        var values = new float[length];
        switch (kind)
        {
            case "impulse": values[0] = .5f; break;
            case "edges":
                values[0] = .5f; values[199] = -.25f; values[200] = .75f; values[479800] = -.5f; values[479999] = .25f;
                break;
            case "noise":
            case "quiet":
                uint state = 20260918;
                for (int i = 0; i < length; i++)
                {
                    state = unchecked(1664525 * state + 1013904223);
                    values[i] = ((int)((state >> 8) & 65535) - 32768) / 131072f;
                    if (kind == "quiet") values[i] *= 1e-6f;
                }
                break;
            case "tones":
                for (int i = 0; i < length; i++)
                {
                    double time = i / 16000.0;
                    values[i] = (float)(.2 * Math.Sin(2 * Math.PI * 440 * time) + .1 * Math.Sin(2 * Math.PI * 1000 * time) + .05 * Math.Sin(2 * Math.PI * 3100 * time));
                }
                break;
            case "dc": Array.Fill(values, .25f); break;
            default: throw new ArgumentException(kind);
        }
        return values;
    }
}
