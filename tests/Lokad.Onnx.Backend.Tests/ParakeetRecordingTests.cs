namespace Lokad.Onnx.Backend.Tests;

using System.Reflection;

public class ParakeetRecordingTests
{
    const int Rate = 16000;
    delegate ParakeetRecording Policy(float[] pcm, int limit, Func<int, int, ParakeetTranscription> decode, CancellationToken cancellation);
    static readonly Policy Run = typeof(ParakeetTranscriber).Assembly.GetType("Lokad.Onnx.ParakeetRecordingPolicy", true)!
        .GetMethod("Run", BindingFlags.NonPublic | BindingFlags.Static)!.CreateDelegate<Policy>();
    static ParakeetTranscription Result(string text = "speech", ParakeetStopReason stop = ParakeetStopReason.EndOfAudio)
        => new(text, Array.AsReadOnly(new[] { 1 }), Array.AsReadOnly(new[] { 2 }), Array.AsReadOnly(new[] { 3 }), stop, 4, 5);
    static float[] Loud(int count) => Enumerable.Repeat(.1f, count).ToArray();

    [Fact]
    public void LongestQuietRunWinsWithLaterTieAndNoSkippedSamples()
    {
        var pcm = Loud(61 * Rate);
        Array.Fill(pcm, .002f, 25 * Rate, Rate / 2);
        Array.Fill(pcm, .002f, 28 * Rate, Rate / 2);
        var calls = new List<(int, int)>();
        var result = Run(pcm, 256, (start, length) => { calls.Add((start, length)); return Result(); }, CancellationToken.None);
        Assert.Equal(new[] { (0, 452000), (452000, 480000), (932000, 44000) }, calls);
        Assert.Equal(new[] { ParakeetWindowBoundary.Quiet, ParakeetWindowBoundary.HardLimit, ParakeetWindowBoundary.EndOfRecording }, result.Windows.Select(w => w.Boundary));
        Assert.Equal(61, result.ProcessedSeconds);
        Assert.Equal("speech speech speech", result.Text);
        Assert.Equal(ParakeetRecordingStopReason.Completed, result.StopReason);
        Assert.Throws<NotSupportedException>(() => ((IList<ParakeetWindow>)result.Windows).Clear());
    }

    [Theory]
    [InlineData(19, .002f, 30)]
    [InlineData(20, .002f, 28.1)]
    [InlineData(20, .004f, 30)]
    public void QuietRequiresMinimumDurationAndAmplitude(int blocks, float amplitude, double firstLength)
    {
        var pcm = Loud(31 * Rate);
        Array.Fill(pcm, amplitude, 28 * Rate, blocks * 160);
        var result = Run(pcm, 256, (_, _) => Result(), CancellationToken.None);
        Assert.Equal(firstLength, result.Windows[0].AudioSeconds);
        Assert.Equal(31, result.ProcessedSeconds);
    }

    [Theory]
    [InlineData(1)] [InlineData(256)] [InlineData(257)]
    public void TinyTailRetainsExactRecordingDuration(int extra)
    {
        var result = Run(Loud(30 * Rate + extra), 256, (_, _) => Result(), CancellationToken.None);
        Assert.Equal(2, result.Windows.Count);
        Assert.Equal(extra / (double)Rate, result.Windows[1].AudioSeconds);
        Assert.Equal(30 + extra / (double)Rate, result.ProcessedSeconds);
    }

    [Fact]
    public void TokenLimitRetainsRawPartialWindowAndCommitsOnlyEarlierWindows()
    {
        int calls = 0;
        var result = Run(Loud(65 * Rate), 256, (_, _) => ++calls == 1 ? Result("complete") : Result("partial", ParakeetStopReason.TokenLimit), CancellationToken.None);
        Assert.Equal(2, calls);
        Assert.Equal(ParakeetRecordingStopReason.TokenLimit, result.StopReason);
        Assert.Equal("complete", result.Text);
        Assert.Equal("partial", result.Windows[1].Decoding.Text);
        Assert.Equal(30, result.ProcessedSeconds);
    }

    [Fact]
    public void WindowLimitDoesNotDecodeOrClaimRemainingAudio()
    {
        var result = Run(Loud(31 * Rate), 1, (_, _) => Result(), CancellationToken.None);
        Assert.Single(result.Windows);
        Assert.Equal(30, result.ProcessedSeconds);
        Assert.Equal(ParakeetRecordingStopReason.WindowLimit, result.StopReason);
        var exact = Run(Loud(30 * Rate), 1, (_, _) => Result(), CancellationToken.None);
        Assert.Equal(ParakeetRecordingStopReason.Completed, exact.StopReason);
    }

    [Fact]
    public void EmptyAndSilentRecordingsHaveNoTextButRetainProgress()
    {
        var empty = Run(Array.Empty<float>(), 1, (_, _) => throw new Exception("No empty inference"), CancellationToken.None);
        Assert.Empty(empty.Windows); Assert.Empty(empty.Text);
        var silent = Run(new float[31 * Rate], 256, (_, _) => Result("", ParakeetStopReason.SilentInput), CancellationToken.None);
        Assert.Equal(27.5, silent.Windows[0].AudioSeconds);
        Assert.Empty(silent.Text); Assert.Equal(31, silent.ProcessedSeconds);
    }

    [Fact]
    public void CancellationBeforeAndDuringWindowDoesNotReturnACompletedResult()
    {
        using var cancellation = new CancellationTokenSource();
        Assert.Throws<OperationCanceledException>(() => Run(Loud(31 * Rate), 256, (_, _) =>
        { cancellation.Cancel(); return Result(); }, cancellation.Token));
        Assert.Throws<OperationCanceledException>(() => Run(Array.Empty<float>(), 1, (_, _) => Result(), cancellation.Token));
    }
}
