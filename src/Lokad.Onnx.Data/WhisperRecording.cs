namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;

/// <summary>Greedy segment-timestamp decoding with independent windows and explicit work limits.</summary>
/// <remarks>Each window extracts its own features and starts with a fresh language/task prefix.
/// Previous text, temperature fallback and word alignment are not used. MaxNewTokens applies
/// to each window, including timestamps and EOS; exhausting it stops the recording.</remarks>
public sealed record WhisperRecordingOptions(WhisperTranscriptionOptions Decoding, int MaxWindows)
{
    public static WhisperRecordingOptions ForLanguage(string language) => new(WhisperTranscriptionOptions.ForLanguage(language), 256);
}

/// <summary>Why processing of a bounded recording ended.</summary>
public enum WhisperRecordingStopReason { Completed, TokenLimit, WindowLimit, NoProgress }

/// <summary>One immutable text segment. Times are model estimates, clipped to the recording, not word boundaries.</summary>
/// <remarks>TokenIds contains only this segment's text tokens. Raw timestamps are retained on WhisperWindow.Decoding.</remarks>
public sealed record WhisperSegment(double StartSeconds, double EndSeconds, string Text, IReadOnlyList<int> TokenIds);

/// <summary>The observed model window and its committed advance; unfinished text remains in Decoding.</summary>
public sealed record WhisperWindow(double StartSeconds, double AudioSeconds, double AdvancedSeconds, WhisperTranscription Decoding);

/// <summary>Owned transcript, timed segments and all window decisions for a recording.</summary>
/// <remarks>ProcessedSeconds is the committed audio position, which may precede the last observed
/// model window when a limit stops processing. Text includes only committed segments.</remarks>
public sealed record WhisperRecording(string Text, IReadOnlyList<WhisperSegment> Segments,
    IReadOnlyList<WhisperWindow> Windows, WhisperRecordingStopReason StopReason, double DurationSeconds, double ProcessedSeconds);

internal static class WhisperRecordingPolicy
{
    const int End = 50257;
    const int Begin = WhisperTimestampRules.Begin;
    const int TimestampSamples = WhisperTimestampRules.SamplesPerTimestamp;

    internal static WhisperRecording Run(int sampleCount, int maxWindows,
        Func<int, int, WhisperTranscription> decode, Func<IEnumerable<int>, string> text, CancellationToken cancellation)
    {
        var segments = new List<WhisperSegment>();
        var windows = new List<WhisperWindow>();
        int position = 0;
        var stop = WhisperRecordingStopReason.Completed;
        while (position < sampleCount)
        {
            cancellation.ThrowIfCancellationRequested();
            if (windows.Count == maxWindows) { stop = WhisperRecordingStopReason.WindowLimit; break; }
            int length = Math.Min(WhisperAudio.SampleCount, sampleCount - position);
            WhisperTranscription result = decode(position, length);
            cancellation.ThrowIfCancellationRequested();
            int advance = Collect(position, length, result, text, segments);
            windows.Add(new WhisperWindow(Seconds(position), Seconds(length), Seconds(advance), result));
            position += advance;
            if (result.StopReason == WhisperStopReason.TokenLimit) { stop = WhisperRecordingStopReason.TokenLimit; break; }
            if (advance == 0) { stop = WhisperRecordingStopReason.NoProgress; break; }
        }
        cancellation.ThrowIfCancellationRequested();
        return new WhisperRecording(text(segments.SelectMany(s => s.TokenIds)), Array.AsReadOnly(segments.ToArray()),
            Array.AsReadOnly(windows.ToArray()), stop, Seconds(sampleCount), Seconds(position));
    }

    static int Collect(int offset, int length, WhisperTranscription result, Func<IEnumerable<int>, string> text, List<WhisperSegment> segments)
    {
        if (result.SkippedAsNoSpeech || result.StopReason == WhisperStopReason.SilentInput)
            return result.StopReason == WhisperStopReason.TokenLimit ? 0 : length;
        int[] tokens = result.TokenIds.ToArray();
        if (tokens.Any(id => id < 0 || id >= WhisperGeneration.VocabularySize || id > End && id < Begin)
            || tokens.Take(Math.Max(0, tokens.Length - 1)).Contains(End))
            throw new InvalidDataException("Unexpected Whisper recording tokens.");
        if (tokens.Length > 0 && tokens[tokens.Length - 1] == End) Array.Resize(ref tokens, tokens.Length - 1);
        bool terminal = tokens.Length >= 2 && tokens[tokens.Length - 2] < Begin && tokens[tokens.Length - 1] >= Begin;
        var cuts = new List<int>();
        for (int i = 1; i < tokens.Length; i++) if (tokens[i - 1] >= Begin && tokens[i] >= Begin) cuts.Add(i);
        bool limited = result.StopReason == WhisperStopReason.TokenLimit;
        if (cuts.Count > 0 || limited)
        {
            if (terminal) cuts.Add(tokens.Length);
            int first = 0, lastEnd = 0;
            foreach (int cut in cuts)
            {
                if (tokens[first] < Begin || tokens[cut - 1] < Begin)
                    throw new InvalidDataException("Whisper segment is missing a timestamp boundary.");
                int start = (tokens[first] - Begin) * TimestampSamples;
                int end = (tokens[cut - 1] - Begin) * TimestampSamples;
                if (start < lastEnd || end < start) throw new InvalidDataException("Whisper timestamps moved backwards.");
                Add(start, end, tokens.Skip(first).Take(cut - first));
                lastEnd = end; first = cut;
            }
            // Keep an unfinished suffix for inspection, then decode its audio again in
            // the next window. A token limit returns only these completed segments.
            if (!limited && terminal) return length;
            return Math.Min(length, lastEnd);
        }
        int lastTimestamp = tokens.LastOrDefault(id => id >= Begin, Begin);
        int duration = lastTimestamp > Begin ? (lastTimestamp - Begin) * TimestampSamples : length;
        Add(0, duration, tokens);
        return length;

        void Add(int start, int end, IEnumerable<int> source)
        {
            start = Math.Min(start, length); end = Math.Min(end, length);
            int[] ids = source.Where(id => id < End).ToArray();
            string decoded = text(ids);
            if (end <= start || string.IsNullOrWhiteSpace(decoded)) return;
            segments.Add(new WhisperSegment(Seconds(offset + start), Seconds(offset + end), decoded, Array.AsReadOnly(ids)));
        }
    }

    static double Seconds(int samples) => (double)samples / WhisperAudio.SampleRate;
}
