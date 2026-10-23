namespace Lokad.Onnx.Backend.Tests;

using System.Text.Json;

public class AudioResamplerTests
{
    [Fact]
    public void IndependentShortCoprimeRequestsMatchFullScipyArrays()
    {
        using var reference = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "fixtures", "audio-resampling-short.json")));
        foreach (var item in reference.RootElement.GetProperty("cases").EnumerateArray())
        {
            var input = item.GetProperty("input").EnumerateArray().Select(v => v.GetSingle()).ToArray();
            var saved = input.Select(BitConverter.SingleToInt32Bits).ToArray();
            var expected = item.GetProperty("output").EnumerateArray().Select(v => v.GetSingle()).ToArray();
            var actual = AudioResampler.Resample(input, item.GetProperty("source_rate").GetInt32(), item.GetProperty("destination_rate").GetInt32());
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < actual.Length; i++) Assert.InRange(Math.Abs(actual[i] - expected[i]), 0, 1e-6f);
            Assert.Equal(saved, input.Select(BitConverter.SingleToInt32Bits));
        }
    }

    [Theory]
    [InlineData(48000)]
    [InlineData(48001)]
    [InlineData(191999)]
    public void OneSampleRequestsAllocateTheirOutputWithoutLargeSetupStorage(int sourceRate)
    {
        var input = new[] { .5f };
        AudioResampler.Resample(input, 48000, 16000);
        long start = GC.GetAllocatedBytesForCurrentThread();
        var output = AudioResampler.Resample(input, sourceRate, 16000);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - start;
        Assert.Single(output);
        Assert.InRange(allocated, 0, 1024);
        Assert.Equal(.5f, input[0]);
        Assert.True(float.IsFinite(output[0]));
    }

    [Fact]
    public void IndependentScipyFullArrays_AgreeAndKeepInput()
    {
        using var reference = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "fixtures", "audio-resampling.json")));
        foreach (var item in reference.RootElement.GetProperty("cases").EnumerateArray())
        {
            var input = item.GetProperty("input").EnumerateArray().Select(v => v.GetSingle()).ToArray();
            var saved = input.ToArray();
            var expected = item.GetProperty("output").EnumerateArray().Select(v => v.GetSingle()).ToArray();
            var actual = AudioResampler.Resample(input, item.GetProperty("source_rate").GetInt32(), item.GetProperty("destination_rate").GetInt32());
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < actual.Length; i++) Assert.InRange(Math.Abs(actual[i] - expected[i]), 0, 1e-6f);
            Assert.Equal(saved, input);
        }
    }

    [Theory]
    [InlineData(8000, 16000)]
    [InlineData(44100, 16000)]
    [InlineData(48000, 16000)]
    [InlineData(16000, 48000)]
    public void PassbandToneHasExpectedFrequencyAndAmplitude(int sourceRate, int destinationRate)
    {
        var input = Enumerable.Range(0, sourceRate).Select(i => (float)(0.5 * Math.Sin(2 * Math.PI * 1000 * i / sourceRate))).ToArray();
        var actual = AudioResampler.Resample(input, sourceRate, destinationRate);
        Assert.Equal(destinationRate, actual.Length);
        for (int i = 200; i < actual.Length - 200; i++)
            Assert.InRange(Math.Abs(actual[i] - 0.5 * Math.Sin(2 * Math.PI * 1000 * i / destinationRate)), 0, 1e-4);
    }

    [Fact]
    public void DownsamplingRejectsAboveNyquistTone()
    {
        var input = Enumerable.Range(0, 48000).Select(i => (float)Math.Sin(2 * Math.PI * 12000 * i / 48000)).ToArray();
        var actual = AudioResampler.Resample(input, 48000, 16000);
        double rms = Math.Sqrt(actual.Skip(200).Take(actual.Length - 400).Average(x => (double)x * x));
        Assert.InRange(rms, 0, 1e-3);
    }

    [Fact]
    public void EqualRatePreservesBitsButOwnsItsOutput()
    {
        var input = new[] { -0f, float.Epsilon, -float.Epsilon, 1.5f, -1.5f };
        var output = AudioResampler.Resample(input, 16000, 16000);
        Assert.NotSame(input, output);
        Assert.Equal(input.Select(BitConverter.SingleToInt32Bits), output.Select(BitConverter.SingleToInt32Bits));
        output[1] = 9;
        Assert.Equal(float.Epsilon, input[1]);
    }

    [Theory]
    [InlineData(8000, 16000, 2)]
    [InlineData(48000, 16000, 1)]
    [InlineData(44100, 16000, 1)]
    public void TinyAndEmptyClipsHaveCeilingDuration(int source, int target, int expected)
    {
        Assert.Empty(AudioResampler.Resample(Array.Empty<float>(), source, target));
        Assert.Equal(expected, AudioResampler.Resample(new[] { 0.5f }, source, target).Length);
        Assert.All(AudioResampler.Resample(new float[37], source, target), x => Assert.Equal(0, x));
    }

    [Fact]
    public void CenteredImpulseHasNoExtraDelay()
    {
        var input = new float[64];
        input[12] = 1;
        var output = AudioResampler.Resample(input, 16000, 48000);
        Assert.Equal(36, Array.IndexOf(output, output.Max()));
    }

    [Theory]
    [InlineData(0, 16000)]
    [InlineData(16000, 0)]
    [InlineData(7999, 16000)]
    [InlineData(16000, 192001)]
    public void InvalidRatesRefused(int source, int target) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => AudioResampler.Resample(Array.Empty<float>(), source, target));

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void NonfiniteInputRefusedEvenAtEqualRate(float value) =>
        Assert.Throws<ArgumentException>(() => AudioResampler.Resample(new[] { value }, 16000, 16000));
}
