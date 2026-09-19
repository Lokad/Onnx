namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

// Window, count and reconstruction rules follow pyannote.audio 4.0.0.
// See NOTICE.txt. Time coordinates are SincNet receptive fields, not 10 / 589.
internal sealed record Community1Interval(double Start, double End, int Speaker);
internal sealed record Community1Frames(bool[] Values, int Frames, int Speakers);

internal static class Community1Timeline
{
    internal const int WindowSamples = 160000, StepSamples = 16000, LocalFrames = 589;
    internal const int MaximumSamples = 16000 * 600;
    internal const double FrameDuration = 991.0 / 16000, FrameStep = 270.0 / 16000;
    internal static int ChunkCount(int samples)
    {
        if (samples < 1 || samples > MaximumSamples) throw new ArgumentOutOfRangeException(nameof(samples));
        if (samples < WindowSamples) return 1;
        int remainder = samples - WindowSamples;
        return 1 + remainder / StepSamples + (remainder % StepSamples == 0 ? 0 : 1);
    }

    internal static float[] Window(float[] samples, int chunk)
    {
        int count = ChunkCount(samples.Length);
        if (chunk < 0 || chunk >= count) throw new ArgumentOutOfRangeException(nameof(chunk));
        var output = new float[WindowSamples]; int start = chunk * StepSamples;
        Array.Copy(samples, start, output, 0, Math.Min(WindowSamples, samples.Length - start));
        return output;
    }

    internal static bool[] Powerset(float[] scores)
    {
        if (scores.Length != LocalFrames * 7) throw new ArgumentException("Expected 589 by 7 segmentation scores.");
        ReadOnlySpan<int> bits = new[] { 0, 1, 2, 4, 3, 5, 6 };
        var result = new bool[LocalFrames * 3];
        for (int t = 0; t < LocalFrames; t++)
        {
            int best = 0;
            for (int c = 0; c < 7; c++)
            {
                float value = scores[t * 7 + c];
                if (!float.IsFinite(value)) throw new ArgumentException("Nonfinite segmentation score.");
                if (value > scores[t * 7 + best]) best = c;
            }
            for (int s = 0; s < 3; s++) result[t * 3 + s] = (bits[best] & (1 << s)) != 0;
        }
        return result;
    }

    internal static float[][] EmbeddingMasks(bool[] activity)
    {
        if (activity.Length != LocalFrames * 3) throw new ArgumentException("Expected one segmentation chunk.");
        var masks = new float[3][];
        for (int s = 0; s < 3; s++)
        {
            var full = new float[LocalFrames]; var clean = new float[LocalFrames]; int selected = 0;
            for (int t = 0; t < LocalFrames; t++)
            {
                if (!activity[t * 3 + s]) continue;
                full[t] = 1; int active = 0;
                for (int k = 0; k < 3; k++) if (activity[t * 3 + k]) active++;
                if (active < 2) { clean[t] = 1; selected++; }
            }
            // Native min_num_samples finds the 400-sample fbank boundary by exceptions.
            // ceil(589 * 400 / 160000) == 2; the clean-mask test is strictly greater.
            masks[s] = selected > 2 ? clean : full;
        }
        return masks;
    }

    static int Validate(bool[] activity, int chunks)
    {
        if (chunks < 1 || chunks > ChunkCount(MaximumSamples) || activity.Length != chunks * LocalFrames * 3)
            throw new ArgumentException("Invalid segmentation chunk count or shape.");
        return checked((int)Math.Round((10 + chunks - 1 + .5 * FrameDuration - .5 * FrameDuration) / FrameStep) + 1);
    }
    static int Offset(int chunk) => (int)Math.Round((chunk + .5 * FrameDuration - .5 * FrameDuration) / FrameStep);

    internal static int[] Count(bool[] activity, int chunks, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested(); int frames = Validate(activity, chunks);
        var sums = new float[frames]; var contributions = new int[frames];
        for (int c = 0; c < chunks; c++)
        {
            cancellation.ThrowIfCancellationRequested(); int offset = Offset(c);
            for (int t = 0; t < LocalFrames; t++)
            {
                int n = 0; for (int s = 0; s < 3; s++) if (activity[(c * LocalFrames + t) * 3 + s]) n++;
                sums[offset + t] += n; contributions[offset + t]++;
            }
        }
        var result = new int[frames];
        for (int t = 0; t < frames; t++) if (contributions[t] != 0) result[t] = (int)Math.Round(sums[t] / contributions[t]);
        return result;
    }

    internal static Community1Frames Reconstruct(bool[] activity, int chunks, int[] labels,
        int[] count, bool exclusive, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested(); int frames = Validate(activity, chunks);
        if (labels.Length != chunks * 3 || labels.Any(v => v < -2 || v == -1 || v >= Community1Clusterer.MaximumTrainingEmbeddings)
            || count.Length != frames || count.Any(v => v < 0 || v > 3)) throw new ArgumentException("Invalid reconstruction labels/counts.");
        int clusters = labels.Max() + 1;
        if (clusters <= 0) return new Community1Frames(Array.Empty<bool>(), frames, 0);
        int speakers = Math.Max(clusters, exclusive ? Math.Min(1, count.Max()) : count.Max());
        var votes = new float[checked(frames * speakers)]; var binary = new bool[votes.Length];
        for (int c = 0; c < chunks; c++)
        {
            cancellation.ThrowIfCancellationRequested(); int offset = Offset(c);
            for (int t = 0; t < LocalFrames; t++)
            {
                for (int s = 0; s < 3; s++)
                {
                    int label = labels[c * 3 + s]; if (label < 0 || !activity[(c * LocalFrames + t) * 3 + s]) continue;
                    bool duplicate = false;
                    for (int k = 0; k < s; k++) if (labels[c * 3 + k] == label && activity[(c * LocalFrames + t) * 3 + k]) duplicate = true;
                    if (!duplicate) votes[(offset + t) * speakers + label]++;
                }
            }
        }
        var order = new int[speakers];
        for (int t = 0; t < frames; t++)
        {
            cancellation.ThrowIfCancellationRequested(); int offset = t * speakers;
            for (int s = 0; s < speakers; s++) order[s] = s;
            Array.Sort(order, (a, b) => { int byVote = votes[offset + b].CompareTo(votes[offset + a]); return byVote != 0 ? byVote : a.CompareTo(b); });
            int selected = exclusive ? Math.Min(count[t], 1) : count[t];
            for (int s = 0; s < selected; s++) binary[offset + order[s]] = true;
        }
        return new Community1Frames(binary, frames, speakers);
    }

    internal static Community1Interval[] Intervals(Community1Frames frames, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();
        if (frames.Frames < 1 || frames.Speakers < 0 || frames.Values.Length != checked(frames.Frames * frames.Speakers))
            throw new ArgumentException("Invalid diarization frame shape.");
        double Center(int t) { double start = t * FrameStep; return .5 * (start + (start + FrameDuration)); }
        var result = new List<Community1Interval>();
        for (int s = 0; s < frames.Speakers; s++)
        {
            cancellation.ThrowIfCancellationRequested(); int start = -1;
            for (int t = 0; t < frames.Frames; t++)
            {
                bool active = frames.Values[t * frames.Speakers + s];
                if (active && start < 0) start = t;
                if (!active && start >= 0) { result.Add(new Community1Interval(Center(start), Center(t), s)); start = -1; }
            }
            if (start >= 0 && frames.Frames - 1 > start) result.Add(new Community1Interval(Center(start), Center(frames.Frames - 1), s));
        }
        return result.OrderBy(v => v.Start).ThenBy(v => v.End).ThenBy(v => v.Speaker).ToArray();
    }
}
