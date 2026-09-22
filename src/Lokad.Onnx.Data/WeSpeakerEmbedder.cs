namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;

/// <summary>Whether sufficient frame data existed to calculate an embedding.</summary>
public enum WeSpeakerEmbeddingStatus { Completed, InsufficientFrames }

/// <summary>An owned, unnormalized speaker representation and its frame coverage.</summary>
/// <remarks>Completed is a numerical result, not a speech, speaker identity or reliability decision.</remarks>
public sealed record WeSpeakerEmbedding(IReadOnlyList<float> Values, WeSpeakerEmbeddingStatus Status,
    int EncoderFrames, int PositiveFrames);

/// <summary>Managed short-recording embeddings using the pyannote Community-1 WeSpeaker split export.</summary>
/// <remarks>Loads local ONNX assets only. One instance serializes requests; each request owns its
/// execution contexts and result. Frontend numerical qualification is documented in tests/pyannote/frontend.</remarks>
public sealed class WeSpeakerEmbedder
{
    const string EncodedName = "/resnet/pool/Reshape_output_0";
    readonly ComputationalGraph encoder, projection;
    readonly object gate = new object();

    /// <summary>Loads embedding_encoder.onnx and the prepared projection.onnx without downloads or native ORT.</summary>
    public WeSpeakerEmbedder(string encoderPath, string projectionPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(encoderPath);
        ArgumentException.ThrowIfNullOrEmpty(projectionPath);
        encoder = Load(encoderPath, 64L * 1024 * 1024);
        projection = Load(projectionPath, 8L * 1024 * 1024);
        RequireNames(encoder, "fbank_features", EncodedName);
        RequireNames(projection, "pooled", "embedding");
    }

    /// <summary>Extracts 256 values from 400 through 480000 normalized mono 16 kHz samples.</summary>
    /// <remarks>An empty weights span selects unweighted statistics. Otherwise weights must be finite
    /// in [0,1], at most 480000 values, and cover the complete recording uniformly. They are resized
    /// to encoder frames using nearest-neighbor selection floor(frame * weights.Length / EncoderFrames).
    /// Fewer than two positive resized weights returns InsufficientFrames with no vector. This check
    /// does not detect silence, exclude overlapping speech or establish speaker reliability.
    /// Cancellation is checked between model calls; an in-flight call completes first.</remarks>
    public WeSpeakerEmbedding Extract(ReadOnlySpan<float> samples, int sampleRate,
        ReadOnlySpan<float> weights, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();
        if (weights.Length > WeSpeakerAudio.MaximumSamples)
            throw new ArgumentOutOfRangeException(nameof(weights), "At most 480000 frame weights are supported.");
        foreach (float weight in weights)
            if (!float.IsFinite(weight) || weight < 0 || weight > 1)
                throw new ArgumentException("Frame weights must be finite and in [0,1].", nameof(weights));
        lock (gate)
        {
            cancellation.ThrowIfCancellationRequested();
            var features = WeSpeakerAudio.LogMelFilterbank(samples, sampleRate, cancellation);
            int frames = (features.Dimensions[1] + 7) / 8, positive = 0;
            var resized = new float[frames];
            for (int t = 0; t < frames; t++)
            {
                resized[t] = weights.IsEmpty ? 1 : weights[(int)((long)t * weights.Length / frames)];
                if (resized[t] > 0) positive++;
            }
            if (positive < 2)
                return new WeSpeakerEmbedding(Array.AsReadOnly(Array.Empty<float>()),
                    WeSpeakerEmbeddingStatus.InsufficientFrames, frames, positive);
            var encoding = encoder.CreateExecution(ExecutionOptions.Memory);
            var projecting = projection.CreateExecution(ExecutionOptions.Memory);
            try
            {
                var hidden = Execute(encoding, "fbank_features", features, EncodedName, new[] { 1, 2560, frames });
                cancellation.ThrowIfCancellationRequested();
                var pooled = WeSpeakerPooling.Pool(hidden, resized, !weights.IsEmpty, cancellation);
                var vector = Execute(projecting, "pooled", pooled, "embedding", new[] { 1, 256 });
                cancellation.ThrowIfCancellationRequested();
                return new WeSpeakerEmbedding(Array.AsReadOnly(vector.ToArray()),
                    WeSpeakerEmbeddingStatus.Completed, frames, positive);
            }
            finally { encoding.Reset(); projecting.Reset(); }
        }
    }

    // Share one backbone evaluation across the three local speakers. The native pipeline's
    // sparse weighted masks are allowed here; Extract retains its explicit two-frame policy.
    internal WeSpeakerEmbedding[] ExtractPipeline(float[] samples, float[][] masks, CancellationToken cancellation, PipelineRequest? request)
    {
        if (samples.Length != Community1Timeline.WindowSamples || masks.Length != 3
            || masks.Any(m => m is null || m.Length != Community1Timeline.LocalFrames || m.Any(v => v != 0 && v != 1)))
            throw new ArgumentException("Pipeline embeddings require one full audio window and three binary masks.");
        lock (gate)
        {
            if (request is not null) ObjectDisposedException.ThrowIf(request.Disposed, request);
            cancellation.ThrowIfCancellationRequested();
            var features = WeSpeakerAudio.LogMelFilterbank(samples, 16000, cancellation);
            int frames = (features.Dimensions[1] + 7) / 8;
            var encoding = request is null ? encoder.CreateExecution(ExecutionOptions.Memory)
                : (request.Encoding ??= encoder.CreateExecution(ExecutionOptions.Memory));
            var projecting = request is null ? projection.CreateExecution(ExecutionOptions.Memory)
                : (request.Projecting ??= projection.CreateExecution(ExecutionOptions.Memory));
            try
            {
                var hidden = Execute(encoding, "fbank_features", features, EncodedName, new[] { 1, 2560, frames });
                var results = new WeSpeakerEmbedding[3];
                for (int s = 0; s < 3; s++)
                {
                    cancellation.ThrowIfCancellationRequested(); var weights = new float[frames]; int positive = 0;
                    for (int t = 0; t < frames; t++) { weights[t] = masks[s][t * masks[s].Length / frames]; if (weights[t] > 0) positive++; }
                    var pooled = WeSpeakerPooling.PoolPipeline(hidden, weights, cancellation);
                    projecting.Reset();
                    var vector = Execute(projecting, "pooled", pooled, "embedding", new[] { 1, 256 });
                    results[s] = new WeSpeakerEmbedding(Array.AsReadOnly(vector.ToArray()), WeSpeakerEmbeddingStatus.Completed, frames, positive);
                }
                cancellation.ThrowIfCancellationRequested(); return results;
            }
            finally { encoding.Reset(); projecting.Reset(); }
        }
    }

    static ComputationalGraph Load(string path, long budget) => OnnxImport.Load(path, budget)
        ?? throw new InvalidDataException("Could not load WeSpeaker model: " + OnnxImport.LastErrorMessage, OnnxImport.LastErrorCause);

    static void RequireNames(ComputationalGraph graph, string input, string output)
    {
        if (graph.Inputs.Count != 1 || !graph.Inputs.ContainsKey(input)
            || graph.Outputs.Count != 1 || !graph.Outputs.ContainsKey(output))
            throw new NotSupportedException("The graph is not the supported WeSpeaker split export.");
    }

    static Tensor<float> Execute(GraphExecution execution, string inputName, Tensor<float> input,
        string outputName, int[] shape)
    {
        if (!execution.Execute(new Dictionary<string, ITensor> { [inputName] = input }, true,
            ExecutionProvider.CPU, ExecutionOptions.Memory))
            throw new InvalidDataException(execution.LastErrorMessage, execution.LastErrorCause);
        if (!execution.Outputs.TryGetValue(outputName, out var output) || output is not Tensor<float> tensor
            || !tensor.Dimensions.SequenceEqual(shape))
            throw new InvalidDataException("Unexpected WeSpeaker output: " + outputName);
        foreach (float value in tensor.ToArray())
            if (!float.IsFinite(value)) throw new InvalidDataException("Nonfinite WeSpeaker output: " + outputName);
        return tensor;
    }

    internal WeSpeakerEmbedding[] ExtractPipeline(float[] samples, float[][] masks, CancellationToken cancellation) =>
        ExtractPipeline(samples, masks, cancellation, null);

    // Contexts belong to one diarization embedding loop, never to the model instance.
    internal PipelineRequest CreatePipelineRequest() => new PipelineRequest(this);

    internal sealed class PipelineRequest : IDisposable
    {
        readonly WeSpeakerEmbedder owner;
        internal GraphExecution? Encoding, Projecting;
        internal bool Disposed;

        internal PipelineRequest(WeSpeakerEmbedder owner) => this.owner = owner;

        internal WeSpeakerEmbedding[] ExtractPipeline(float[] samples, float[][] masks, CancellationToken cancellation) =>
            owner.ExtractPipeline(samples, masks, cancellation, this);

        public void Dispose()
        {
            lock (owner.gate)
            {
                if (Disposed) return;
                try { Encoding?.Reset(); Projecting?.Reset(); }
                finally { Encoding = null; Projecting = null; Disposed = true; }
            }
        }
    }

}
