namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

/// <summary>An independently owned [direction, input, gate] transpose of a constant LSTM weight.</summary>
internal sealed record PackedLstmWeight(string SourceName, DenseTensor<float> Source,
    float[] SourceArray, int[] Shape, float[] Values);

/// <summary>Bounded preparation for the observed one-direction, input/hidden-size-640 decoder.</summary>
internal static class GraphLstmPacking
{
    internal const int HiddenSize = 640;
    internal const int Elements = 4 * HiddenSize * HiddenSize;
    internal const long WeightBytes = (long)Elements * sizeof(float);

    readonly record struct Source(string Name, DenseTensor<float> Tensor, float[] Array);
    readonly record struct Pair(Source W, Source R);

    static bool Eligible(Node node) => node.Op == OpType.LSTM && Node.IsStandardDomain(node.Domain)
        && node.Inputs.Length >= 3 && node.GetInt("hidden_size", null) == HiddenSize
        && node.GetInt("layout", 0) == 0 && node.Attr("direction", "forward") == "forward";

    static float[]? Storage(DenseTensor<float> tensor)
    {
        var shape = tensor.Dimensions;
        if (shape.Length != 3 || shape[0] != 1 || shape[1] != 4 * HiddenSize || shape[2] != HiddenSize
            || !GraphConvPacking.Standard(tensor)
            || !MemoryMarshal.TryGetArray(tensor.Buffer, out ArraySegment<float> window)
            || window.Array is null || window.Offset != 0 || window.Count != window.Array.Length) return null;
        return window.Array;
    }

    static List<Pair> Candidates(ComputationalGraph graph)
    {
        var consumers = new Dictionary<string, bool>(StringComparer.Ordinal);
        foreach (var node in graph.Nodes)
        {
            var inputs = GraphCaptures.NodeInputs(node);
            for (int i = 0; i < inputs.Length; i++)
            {
                string name = inputs[i]; if (string.IsNullOrEmpty(name)) continue;
                bool eligible = (i == 1 || i == 2) && Eligible(node);
                consumers[name] = eligible && (!consumers.TryGetValue(name, out bool prior) || prior);
            }
        }
        var sources = new Dictionary<string, Source>(StringComparer.Ordinal);
        foreach (var item in consumers)
        {
            if (!item.Value || graph.Inputs.ContainsKey(item.Key) || graph.Outputs.ContainsKey(item.Key)
                || !graph.Initializers.TryGetValue(item.Key, out var value)
                || value is not DenseTensor<float> dense || Storage(dense) is not float[] array) continue;
            sources.Add(item.Key, new Source(item.Key, dense, array));
        }
        var pairs = new List<Pair>();
        foreach (var node in graph.Nodes)
        {
            if (!Eligible(node) || !sources.TryGetValue(node.Inputs[1], out var w)
                || !sources.TryGetValue(node.Inputs[2], out var r)) continue;
            // One array has one mapping. Different tensor wrappers retain the
            // reference-identity fallback instead of overwriting another pair.
            if (ReferenceEquals(w.Array, r.Array) && !ReferenceEquals(w.Tensor, r.Tensor)) continue;
            pairs.Add(new Pair(w, r));
        }
        return pairs;
    }

    static bool Fresh(PackedLstmWeight record, Source source) =>
        ReferenceEquals(record.Source, source.Tensor) && ReferenceEquals(record.SourceArray, source.Array)
        && record.Shape.AsSpan().SequenceEqual(source.Tensor.Dimensions) && record.Values.Length == Elements;

    /// <summary>Prune complete pairs against the capacity left after other retained maps.</summary>
    internal static long PruneAndBytes(ComputationalGraph graph, long alreadyRetained)
    {
        if (alreadyRetained < 0 || alreadyRetained > graph.MaximumPackedWeightBytes)
            throw new ArgumentOutOfRangeException(nameof(alreadyRetained));
        var pairs = Candidates(graph);
        var sources = new Dictionary<string, Source>(StringComparer.Ordinal);
        foreach (var pair in pairs) { sources[pair.W.Name] = pair.W; sources[pair.R.Name] = pair.R; }
        var keep = new HashSet<float[]>(); long bytes = 0;
        bool Valid(Source source) => graph.PackedLstmWeights.TryGetValue(source.Array, out var record)
            && sources.TryGetValue(record.SourceName, out var original)
            && Fresh(record, original) && Fresh(record, source);
        foreach (var pair in pairs)
        {
            if (!Valid(pair.W) || !Valid(pair.R)) continue;
            long needed = keep.Contains(pair.W.Array) ? 0 : WeightBytes;
            if (!ReferenceEquals(pair.W.Array, pair.R.Array) && !keep.Contains(pair.R.Array)) needed += WeightBytes;
            if (needed > graph.MaximumPackedWeightBytes - alreadyRetained - bytes) continue;
            keep.Add(pair.W.Array); keep.Add(pair.R.Array); bytes += needed;
        }
        foreach (var array in graph.PackedLstmWeights.Keys.ToArray())
            if (!keep.Contains(array)) graph.PackedLstmWeights.Remove(array);
        return bytes;
    }

    static PackedLstmWeight Prepare(Source source)
    {
        var values = new float[Elements]; var input = source.Tensor.Buffer.Span;
        for (int k = 0; k < HiddenSize; k++)
        for (int o = 0; o < 4 * HiddenSize; o++)
            values[k * 4 * HiddenSize + o] = input[o * HiddenSize + k];
        return new PackedLstmWeight(source.Name, source.Tensor, source.Array, source.Tensor.Dimensions.ToArray(), values);
    }

    /// <summary>Prepare both missing arrays before publishing either; all maps share the existing graph cap.</summary>
    internal static void PackWeights(ComputationalGraph graph)
    {
        foreach (var pair in Candidates(graph))
        {
            graph.PackedLstmWeights.TryGetValue(pair.W.Array, out var w);
            graph.PackedLstmWeights.TryGetValue(pair.R.Array, out var r);
            if (w is not null && !Fresh(w, pair.W) || r is not null && !Fresh(r, pair.R)) continue;
            bool shared = ReferenceEquals(pair.W.Array, pair.R.Array);
            long needed = (w is null ? WeightBytes : 0) + (!shared && r is null ? WeightBytes : 0);
            if (needed == 0 || needed > graph.MaximumPackedWeightBytes - graph.RetainedPackedWeightBytes) continue;
            var preparedW = w ?? Prepare(pair.W);
            var preparedR = shared ? preparedW : r ?? Prepare(pair.R);
            if (w is null) graph.PackedLstmWeights.Add(pair.W.Array, preparedW);
            if (!shared && r is null) graph.PackedLstmWeights.Add(pair.R.Array, preparedR);
            graph.RetainedPackedWeightBytes += needed;
        }
    }

    internal static float[]? Resolve(IReadOnlyDictionary<float[], PackedLstmWeight>? map, Tensor<float> weight)
    {
        if (map is null || map.Count == 0 || weight is not DenseTensor<float> dense
            || Storage(dense) is not float[] array || !map.TryGetValue(array, out var record)
            || !Fresh(record, new Source(record.SourceName, dense, array))) return null;
        return record.Values;
    }
}
