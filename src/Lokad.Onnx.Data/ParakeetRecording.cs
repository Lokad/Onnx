namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

/// <summary>Independent Parakeet windows with explicit per-window decoding and recording work limits.</summary>
/// <remarks>MaxTokens applies to each window. A token limit stops the recording without committing
/// that window to Text or ProcessedSeconds; its partial text remains in Windows.</remarks>
public sealed record ParakeetRecordingOptions(ParakeetTranscriptionOptions Decoding, int MaxWindows)
{
    public static ParakeetRecordingOptions Default => new(ParakeetTranscriptionOptions.Default, 256);
}

/// <summary>Why processing of a Parakeet recording ended.</summary>
public enum ParakeetRecordingStopReason { Completed, TokenLimit, WindowLimit }

/// <summary>How the audio window ends. Quiet is an amplitude heuristic, not a detected word boundary.</summary>
public enum ParakeetWindowBoundary { EndOfRecording, Quiet, HardLimit }

/// <summary>One independent window, with recording-relative audio position and local decoder decisions.</summary>
/// <remarks>Decoding.FrameIndices are local to the window, nominally 80 ms per frame, not word alignment.
/// A nonsilent final tail shorter than 257 samples is zero-padded for inference only.</remarks>
public sealed record ParakeetWindow(double StartSeconds, double AudioSeconds,
    ParakeetWindowBoundary Boundary, ParakeetTranscription Decoding);

/// <summary>Owned recording transcript and immutable windows, including any uncommitted partial window.</summary>
/// <remarks>Text joins completed nonempty window texts with one space. ProcessedSeconds excludes a
/// token-limited window. Independent windows can lose or duplicate words at a hard cut; inspect Boundaries.</remarks>
public sealed record ParakeetRecording(string Text, IReadOnlyList<ParakeetWindow> Windows,
    ParakeetRecordingStopReason StopReason, double DurationSeconds, double ProcessedSeconds);

internal static class ParakeetRecordingPolicy
{
    const int Rate = ParakeetTranscriber.SampleRate;
    const int Block = Rate / 100;
    const int SearchStart = Rate * 25;
    const int MinimumQuietBlocks = 20;

    internal static ParakeetRecording Run(float[] samples, int maxWindows,
        Func<int, int, ParakeetTranscription> decode, CancellationToken cancellation)
    {
        var windows = new List<ParakeetWindow>();
        var completed = new List<string>();
        int position = 0;
        var stop = ParakeetRecordingStopReason.Completed;
        while (position < samples.Length)
        {
            cancellation.ThrowIfCancellationRequested();
            if (windows.Count == maxWindows) { stop = ParakeetRecordingStopReason.WindowLimit; break; }
            int length = Math.Min(ParakeetTranscriber.MaximumSamples, samples.Length - position);
            var boundary = ParakeetWindowBoundary.EndOfRecording;
            if (length < samples.Length - position)
            {
                int cut = QuietCut(samples.AsSpan(position, length));
                boundary = cut == 0 ? ParakeetWindowBoundary.HardLimit : ParakeetWindowBoundary.Quiet;
                if (cut != 0) length = cut;
            }
            var result = decode(position, length);
            cancellation.ThrowIfCancellationRequested();
            windows.Add(new ParakeetWindow(Seconds(position), Seconds(length), boundary, result));
            if (result.StopReason == ParakeetStopReason.TokenLimit)
            { stop = ParakeetRecordingStopReason.TokenLimit; break; }
            position += length;
            if (result.Text.Length != 0) completed.Add(result.Text);
        }
        cancellation.ThrowIfCancellationRequested();
        return new ParakeetRecording(string.Join(" ", completed), Array.AsReadOnly(windows.ToArray()),
            stop, Seconds(samples.Length), Seconds(position));
    }

    static int QuietCut(ReadOnlySpan<float> samples)
    {
        int runStart = SearchStart, runBlocks = 0, bestStart = 0, bestBlocks = 0;
        for (int offset = SearchStart; offset < samples.Length; offset += Block)
        {
            double sum = 0;
            for (int i = offset; i < offset + Block; i++) sum += (double)samples[i] * samples[i];
            if (sum <= Block * 0.000009)
            {
                if (runBlocks == 0) runStart = offset;
                runBlocks++;
                if (runBlocks >= MinimumQuietBlocks && runBlocks >= bestBlocks)
                { bestStart = runStart; bestBlocks = runBlocks; }
            }
            else runBlocks = 0;
        }
        return bestBlocks == 0 ? 0 : bestStart + bestBlocks * Block / 2;
    }

    static double Seconds(int samples) => (double)samples / Rate;
}
