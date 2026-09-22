namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;

/// <summary>A speaker interval in seconds from the beginning of the supplied PCM.</summary>
public sealed record DiarizationInterval(double Start, double End, int Speaker);

/// <summary>An owned speaker centroid. Count-only overlap speakers have a zero centroid and HasEmbedding=false.</summary>
public sealed record DiarizationSpeaker(int Speaker, IReadOnlyList<double> Centroid, bool HasEmbedding);

/// <summary>Whether the automatic pipeline found speech and usable training embeddings.</summary>
public enum Community1DiarizationStatus { Completed, NoSpeech, NoUsableEmbeddings }

/// <summary>Owned ordinary/overlap and exclusive speaker timelines, with matching speaker centroids.</summary>
public sealed record Community1Diarization(IReadOnlyList<DiarizationInterval> Intervals,
    IReadOnlyList<DiarizationInterval> ExclusiveIntervals, IReadOnlyList<DiarizationSpeaker> Speakers,
    Community1DiarizationStatus Status, double AudioDuration, int Windows);

/// <summary>Managed automatic diarization using local Community-1 segmentation, WeSpeaker and PLDA models.</summary>
/// <remarks>One instance serializes requests. Uses ten-second windows advancing one second,
/// native mask/count/reconstruction rules, and automatic VBx clustering. Equal vote scores select
/// the lowest canonical speaker label, independent of native NumPy sort dispatch. Intervals are
/// clipped to the supplied recording. Forced speaker counts are not supported.</remarks>
public sealed class Community1Diarizer
{
    public const int SampleRate = 16000;
    public const int MaximumSamples = Community1Timeline.MaximumSamples;
    readonly ComputationalGraph segmentation;
    readonly WeSpeakerEmbedder embedder;
    readonly Community1Clusterer clusterer;
    readonly object gate = new object();

    /// <summary>Loads local FP32 segmentation, split embedding encoder, prepared projection and PLDA assets.</summary>
    public Community1Diarizer(string segmentationPath, string embeddingEncoderPath, string projectionPath, string pldaPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(segmentationPath);
        segmentation = OnnxImport.Load(segmentationPath, 32L * 1024 * 1024)
            ?? throw new InvalidDataException("Could not load Community-1 segmentation: " + OnnxImport.LastErrorMessage, OnnxImport.LastErrorCause);
        if (segmentation.Inputs.Count != 1 || !segmentation.Inputs.ContainsKey("waveform")
            || segmentation.Outputs.Count != 1 || !segmentation.Outputs.ContainsKey("scores"))
            throw new NotSupportedException("Expected the Community-1 segmentation export with waveform input and scores output.");
        embedder = new WeSpeakerEmbedder(embeddingEncoderPath, projectionPath);
        clusterer = new Community1Clusterer(pldaPath);
    }

    /// <summary>Diarizes finite normalized mono 16 kHz PCM of at most ten minutes. Empty input returns NoSpeech.</summary>
    /// <remarks>Requests preserve inputs and own all outputs. Cancellation is checked between graph calls;
    /// an in-flight graph call completes first. Digital silence is processed by the segmentation model.
    /// Speakers inferred only from overlap counts carry HasEmbedding=false. Completed does not assert
    /// human-labeled speaker accuracy; model numerical qualification is documented separately.</remarks>
    public Community1Diarization Diarize(ReadOnlySpan<float> samples, int sampleRate, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();
        if (sampleRate != SampleRate) throw new ArgumentOutOfRangeException(nameof(sampleRate), "Diarization requires mono 16000 Hz PCM.");
        if (samples.Length > MaximumSamples) throw new ArgumentOutOfRangeException(nameof(samples), "At most ten minutes are supported per request.");
        foreach (float value in samples) if (!float.IsFinite(value) || value < -1 || value > 1)
            throw new ArgumentException("PCM samples must be finite and in [-1,1].", nameof(samples));
        double duration = (double)samples.Length / SampleRate;
        if (samples.IsEmpty) return Empty(Community1DiarizationStatus.NoSpeech, duration, 0);
        lock (gate)
        {
            cancellation.ThrowIfCancellationRequested(); var pcm = samples.ToArray();
            int chunks = Community1Timeline.ChunkCount(pcm.Length);
            var activity = new bool[chunks * Community1Timeline.LocalFrames * 3];
            var execution = segmentation.CreateExecution(ExecutionOptions.Memory);
            try
            {
                for (int c = 0; c < chunks; c++)
                {
                    cancellation.ThrowIfCancellationRequested(); execution.Reset();
                    var input = new DenseTensor<float>(Community1Timeline.Window(pcm, c), new[] { 1, 1, Community1Timeline.WindowSamples });
                    if (!execution.Execute(new Dictionary<string, ITensor> { ["waveform"] = input }, true, ExecutionProvider.CPU, ExecutionOptions.Memory))
                        throw new InvalidDataException(execution.LastErrorMessage, execution.LastErrorCause);
                    if (!execution.Outputs.TryGetValue("scores", out var output) || output is not Tensor<float> scores
                        || !scores.Dimensions.SequenceEqual(new[] { 1, Community1Timeline.LocalFrames, 7 }))
                        throw new InvalidDataException("Unexpected Community-1 segmentation shape.");
                    var binary = Community1Timeline.Powerset(scores.ToArray()); Array.Copy(binary, 0, activity, c * binary.Length, binary.Length);
                }
            }
            finally { execution.Reset(); }
            var count = Community1Timeline.Count(activity, chunks, cancellation);
            if (count.All(v => v == 0)) return Empty(Community1DiarizationStatus.NoSpeech, duration, chunks);
            var embeddings = new WeSpeakerEmbedding[chunks * 3];
            using (var request = embedder.CreatePipelineRequest())
            {
                for (int c = 0; c < chunks; c++)
                {
                    cancellation.ThrowIfCancellationRequested(); var binary = new bool[Community1Timeline.LocalFrames * 3];
                    Array.Copy(activity, c * binary.Length, binary, 0, binary.Length);
                    var result = request.ExtractPipeline(Community1Timeline.Window(pcm, c), Community1Timeline.EmbeddingMasks(binary), cancellation);
                    Array.Copy(result, 0, embeddings, c * 3, 3);
                }
            }
            var clustered = clusterer.Cluster(embeddings, new DenseTensor<bool>(activity, new[] { chunks, Community1Timeline.LocalFrames, 3 }), cancellation);
            if (clustered.Centroids.Count == 0) return Empty(Community1DiarizationStatus.NoUsableEmbeddings, duration, chunks);
            var labels = clustered.Labels.ToArray();
            var ordinary = Community1Timeline.Intervals(Community1Timeline.Reconstruct(activity, chunks, labels, count, false, cancellation), cancellation);
            var exclusive = Community1Timeline.Intervals(Community1Timeline.Reconstruct(activity, chunks, labels, count, true, cancellation), cancellation);
            return Assemble(ordinary, exclusive, clustered.Centroids, duration, chunks, cancellation);
        }
    }

    internal static Community1Diarization Assemble(Community1Interval[] ordinary, Community1Interval[] exclusive,
        IReadOnlyList<IReadOnlyList<double>> centroids, double duration, int chunks, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();
        Community1Interval[] Clip(Community1Interval[] input) => input.Select(v => v with { Start = Math.Max(0, v.Start), End = Math.Min(duration, v.End) }).Where(v => v.End > v.Start).ToArray();
        ordinary = Clip(ordinary); exclusive = Clip(exclusive);
        var visible = ordinary.Select(v => v.Speaker).Distinct().OrderBy(v => v).ToArray();
        var mapping = visible.Select((label, index) => (label, index)).ToDictionary(v => v.label, v => v.index);
        var speakers = visible.Select(label => new DiarizationSpeaker(mapping[label],
            Array.AsReadOnly(label < centroids.Count ? centroids[label].ToArray() : new double[256]), label < centroids.Count)).ToArray();
        DiarizationInterval[] Convert(Community1Interval[] input) => input.Select(v => new DiarizationInterval(v.Start, v.End, mapping[v.Speaker])).ToArray();
        return new Community1Diarization(Array.AsReadOnly(Convert(ordinary)), Array.AsReadOnly(Convert(exclusive)), Array.AsReadOnly(speakers),
            visible.Length == 0 ? Community1DiarizationStatus.NoSpeech : Community1DiarizationStatus.Completed, duration, chunks);
    }

    static Community1Diarization Empty(Community1DiarizationStatus status, double duration, int chunks) => new(
        Array.AsReadOnly(Array.Empty<DiarizationInterval>()), Array.AsReadOnly(Array.Empty<DiarizationInterval>()),
        Array.AsReadOnly(Array.Empty<DiarizationSpeaker>()), status, duration, chunks);
}
