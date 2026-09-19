namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;

/// <summary>Owned Community-1 speaker groups. Labels are row-major [Chunks, LocalSpeakers].</summary>
/// <remarks>Label -2 means inactive, invalid or unassigned. Centroids are unnormalized 256-value
/// vectors, indexed by the other labels. An empty centroid list means no usable training data.</remarks>
public sealed record Community1ClusteringResult(IReadOnlyList<int> Labels,
    IReadOnlyList<IReadOnlyList<double>> Centroids, int Chunks, int LocalSpeakers, int TrainingEmbeddings);

/// <summary>Managed automatic speaker clustering for the pyannote Community-1 configuration.</summary>
/// <remarks>Uses centroid linkage, the prepared PLDA transform, VBx refinement and constrained
/// assignment. It does not extract audio windows, estimate speech activity or create timelines.
/// Forced speaker counts are not supported by this automatic-clustering component.</remarks>
public sealed class Community1Clusterer
{
    public const int MaximumTrainingEmbeddings = 4096;
    public const int MaximumEmbeddings = 12288;
    public const int MaximumActivityValues = 8 * 1024 * 1024;
    readonly Community1Parameters parameters;

    /// <summary>Loads a local prepared Community-1 PLDA JSON asset, without Python or downloads.</summary>
    public Community1Clusterer(string preparedModelPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(preparedModelPath);
        parameters = new Community1Parameters(preparedModelPath);
    }

    /// <summary>Groups chunk/local-speaker embeddings using binary activity [chunks, frames, localSpeakers].</summary>
    /// <remarks>Supply one embedding result per chunk/local-speaker row, with one through three
    /// local speakers. Completed vectors must have 256 finite values; InsufficientFrames must
    /// have no values and is excluded from training and assignment. Training requires at least
    /// 20 percent clean single-speaker frames. No usable training data returns only -2 labels;
    /// one usable vector defines one cluster. Requests own their scratch and may run concurrently.
    /// Hierarchy storage grows quadratically with the training count; explicit limits bound it.</remarks>
    public Community1ClusteringResult Cluster(IReadOnlyList<WeSpeakerEmbedding> embeddings,
        Tensor<bool> activity, CancellationToken cancellation)
    {
        ArgumentNullException.ThrowIfNull(embeddings); ArgumentNullException.ThrowIfNull(activity);
        cancellation.ThrowIfCancellationRequested();
        if (activity.Rank != 3 || activity.Dimensions[0] < 1 || activity.Dimensions[1] < 1
            || activity.Dimensions[2] < 1 || activity.Dimensions[2] > 3)
            throw new ArgumentException("Activity must be [chunks, frames, 1..3 local speakers].", nameof(activity));
        int chunks = activity.Dimensions[0], frames = activity.Dimensions[1], speakers = activity.Dimensions[2];
        long rowCount = (long)chunks * speakers;
        if (rowCount > MaximumEmbeddings || rowCount * frames > MaximumActivityValues || embeddings.Count != rowCount)
            throw new ArgumentException("Embedding/activity count mismatch or request limit exceeded.");
        int rows = (int)rowCount;
        var values = new float[rows * 256]; var valid = new bool[rows]; var active = new bool[rows]; var mask = activity.ToArray();
        for (int i = 0; i < rows; i++)
        {
            cancellation.ThrowIfCancellationRequested(); var entry = embeddings[i];
            if (entry is null || entry.Values is null) throw new ArgumentException("Missing embedding result.", nameof(embeddings));
            if (entry.Status == WeSpeakerEmbeddingStatus.InsufficientFrames)
            {
                if (entry.Values.Count != 0) throw new ArgumentException("Insufficient embedding must have no vector.", nameof(embeddings));
                continue;
            }
            if (entry.Status != WeSpeakerEmbeddingStatus.Completed || entry.Values.Count != 256)
                throw new ArgumentException("A completed embedding must contain 256 values.", nameof(embeddings));
            double norm = 0;
            for (int d = 0; d < 256; d++)
            {
                float value = entry.Values[d];
                if (!float.IsFinite(value)) throw new ArgumentException("Embedding values must be finite.", nameof(embeddings));
                values[i * 256 + d] = value; norm += (double)value * value;
            }
            if (!(norm > 0)) throw new ArgumentException("A completed embedding must have nonzero norm.", nameof(embeddings));
            valid[i] = true;
        }
        var training = new List<int>();
        for (int ch = 0; ch < chunks; ch++)
        {
            cancellation.ThrowIfCancellationRequested(); var clean = new int[speakers];
            for (int t = 0; t < frames; t++)
            {
                int count = 0, selected = 0;
                for (int s = 0; s < speakers; s++) if (mask[(ch * frames + t) * speakers + s]) { count++; selected = s; active[ch * speakers + s] = true; }
                if (count == 1) clean[selected]++;
            }
            for (int s = 0; s < speakers; s++) if (valid[ch * speakers + s] && (long)clean[s] * 5 >= frames) training.Add(ch * speakers + s);
        }
        if (training.Count > MaximumTrainingEmbeddings) throw new ArgumentException("At most 4096 training embeddings are supported.");
        var labels = Enumerable.Repeat(-2, rows).ToArray();
        if (training.Count == 0)
            return new Community1ClusteringResult(Array.AsReadOnly(labels), Array.AsReadOnly(Array.Empty<IReadOnlyList<double>>()), chunks, speakers, 0);
        var train = new float[training.Count * 256];
        for (int i = 0; i < training.Count; i++) Array.Copy(values, training[i] * 256, train, i * 256, 256);
        double[] centers; int clusters;
        if (training.Count == 1)
        {
            centers = train.Select(v => (double)v).ToArray(); clusters = 1;
            for (int i = 0; i < rows; i++) if (active[i] && valid[i]) labels[i] = 0;
        }
        else
        {
            var normalized = Community1Math.Normalize(train, training.Count, 256);
            var hierarchy = Community1Math.Hierarchy(normalized, training.Count, 256, .6, cancellation);
            var transformed = parameters.Transform(train, training.Count, cancellation);
            var state = Community1Math.Refine(transformed, training.Count, 128, parameters.Phi, hierarchy.Labels, cancellation);
            var kept = Enumerable.Range(0, state.Priors.Length).Where(s => state.Priors[s] > 1e-7).ToArray(); clusters = kept.Length;
            if (clusters == 0) throw new InvalidDataException("VBx retained no cluster.");
            centers = new double[clusters * 256];
            for (int s = 0; s < clusters; s++)
            {
                cancellation.ThrowIfCancellationRequested(); double weight = 0;
                for (int i = 0; i < training.Count; i++) weight += state.Responsibilities[i * state.Priors.Length + kept[s]];
                for (int d = 0; d < 256; d++)
                {
                    double sum = 0;
                    for (int i = 0; i < training.Count; i++) sum += state.Responsibilities[i * state.Priors.Length + kept[s]] * train[i * 256 + d];
                    centers[s * 256 + d] = sum / weight;
                }
            }
            var norms = new double[clusters];
            for (int s = 0; s < clusters; s++)
            {
                for (int d = 0; d < 256; d++) norms[s] += centers[s * 256 + d] * centers[s * 256 + d];
                if (!(norms[s] > 0) || !double.IsFinite(norms[s])) throw new InvalidDataException("Invalid cluster centroid norm.");
            }
            var scores = new double[rows * clusters]; double minimum = double.PositiveInfinity;
            for (int i = 0; i < rows; i++)
            {
                cancellation.ThrowIfCancellationRequested(); if (!valid[i]) continue;
                double norm = 0; for (int d = 0; d < 256; d++) norm += (double)values[i * 256 + d] * values[i * 256 + d];
                for (int s = 0; s < clusters; s++)
                {
                    double dot = 0; for (int d = 0; d < 256; d++) dot += values[i * 256 + d] * centers[s * 256 + d];
                    scores[i * clusters + s] = 2 - Math.Clamp(1 - dot / Math.Sqrt(norm * norms[s]), 0, 2);
                    minimum = Math.Min(minimum, scores[i * clusters + s]);
                }
            }
            for (int i = 0; i < rows; i++) if (!active[i] || !valid[i]) for (int s = 0; s < clusters; s++) scores[i * clusters + s] = minimum - 1;
            var chunkScores = new double[speakers * clusters];
            for (int ch = 0; ch < chunks; ch++)
            {
                cancellation.ThrowIfCancellationRequested(); Array.Copy(scores, ch * chunkScores.Length, chunkScores, 0, chunkScores.Length);
                var assigned = Community1Math.Match(chunkScores, speakers, clusters);
                for (int s = 0; s < speakers; s++) if (active[ch * speakers + s] && valid[ch * speakers + s]) labels[ch * speakers + s] = assigned[s];
            }
        }
        // Canonical labels follow first assigned occurrence; unused centroids follow in original order.
        var order = Enumerable.Repeat(-1, clusters).ToArray(); int next = 0;
        for (int i = 0; i < rows; i++) if (labels[i] >= 0) { int old = labels[i]; if (order[old] < 0) order[old] = next++; labels[i] = order[old]; }
        for (int s = 0; s < clusters; s++) if (order[s] < 0) order[s] = next++;
        var owned = new IReadOnlyList<double>[clusters];
        for (int s = 0; s < clusters; s++)
        {
            var vector = new double[256]; Array.Copy(centers, s * 256, vector, 0, 256);
            if (vector.Any(v => !double.IsFinite(v))) throw new InvalidDataException("Nonfinite centroid.");
            owned[order[s]] = Array.AsReadOnly(vector);
        }
        cancellation.ThrowIfCancellationRequested();
        return new Community1ClusteringResult(Array.AsReadOnly(labels), Array.AsReadOnly(owned), chunks, speakers, training.Count);
    }
}
