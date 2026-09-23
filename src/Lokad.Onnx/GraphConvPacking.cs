namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

/// <summary>An immutable prepared convolution clone owned by a prepared graph.</summary>
internal sealed record PackedConvWeight(string SourceName, DenseTensor<float> Source,
    float[] SourceArray, int[] Shape, int Lanes, float[] Values)
{
    internal float[]? WinogradValues { get; init; }
}

internal static class GraphConvPacking
{
    internal static int Lanes => Avx2.IsSupported && Fma.IsSupported ? (Avx512F.IsSupported ? 16 : 8) : 0;

    internal static bool Standard(DenseTensor<float> value) => !value.IsReversedStride
        && value.Buffer.Length == value.Length
        && value.strides.SequenceEqual(ArrayUtilities.GetStrides(value.dimensions));

    static float[]? Storage(DenseTensor<float> value) => Standard(value)
        && MemoryMarshal.TryGetArray(value.Buffer, out ArraySegment<float> window)
        && window.Array is not null && window.Offset == 0 && window.Count == window.Array.Length
        ? window.Array : null;

    static bool Shape(DenseTensor<float> weight, int lanes, out long elements)
    {
        elements = 0;
        if (lanes == 0 || weight.Rank != 4) return false;
        var d = weight.Dimensions;
        int m = d[0], c = d[1];
        if (m < 32 || m % 16 != 0 || c < 16 || c % 16 != 0 || d[2] != 3 || d[3] != 3) return false;
        long rounded = ((long)m + 2 * lanes - 1) / (2 * lanes) * (2 * lanes);
        // Bound before multiplication, including zero-filled output-channel lanes.
        long maxElements = Math.Min(Array.MaxLength, GraphPacking.MaxPackedBytes / sizeof(float));
        if (rounded > maxElements / 9 / c) return false;
        elements = rounded * c * 9;
        return (long)m * c * 9 == weight.Length;
    }

    // The optional representation and the direct fallback share the per-weight cap.
    internal static bool WinogradShape(int c, int m, long directElements, out int elements)
    {
        elements = 0;
        if (c < 16 || c % 16 != 0 || m < 32 || m % 16 != 0 || directElements < 1) return false;
        long maximum = Math.Min(Array.MaxLength, GraphPacking.MaxPackedBytes / sizeof(float) - directElements);
        if (maximum < 0 || m > maximum / 16 / c) return false;
        elements = 16 * c * m;
        return true;
    }

    static HashSet<string> WinogradConsumers(ComputationalGraph graph)
    {
        var result = new HashSet<string>(StringComparer.Ordinal);
        foreach (var node in graph.Nodes)
        {
            if (node.Op is not (OpType.Conv or OpType.ConvRelu) || node.GetInt("group", 1) != 1) continue;
            var inputs = GraphCaptures.NodeInputs(node);
            var strides = node.Ints("strides"); var dilations = node.Ints("dilations");
            if (inputs.Length < 2 || string.IsNullOrEmpty(inputs[1])
                || strides is not null && (strides.Length != 2 || strides[0] != 1 || strides[1] != 1)
                || dilations is not null && (dilations.Length != 2 || dilations[0] != 1 || dilations[1] != 1)) continue;
            result.Add(inputs[1]);
        }
        return result;
    }

    static Dictionary<string, (DenseTensor<float> Tensor, float[] Array, long Elements)> Candidates(ComputationalGraph graph)
    {
        var result = new Dictionary<string, (DenseTensor<float>, float[], long)>(StringComparer.Ordinal);
        int lanes = Lanes; if (lanes == 0) return result;
        var consumers = new Dictionary<string, bool>(StringComparer.Ordinal);
        foreach (var node in graph.Nodes)
        {
            var inputs = GraphCaptures.NodeInputs(node);
            for (int i = 0; i < inputs.Length; i++)
            {
                string input = inputs[i]; if (string.IsNullOrEmpty(input)) continue;
                bool eligible = i == 1 && node.Op is OpType.Conv or OpType.ConvRelu;
                consumers[input] = eligible && (!consumers.TryGetValue(input, out bool prior) || prior);
            }
        }
        foreach (var pair in consumers)
        {
            if (!pair.Value || graph.Inputs.ContainsKey(pair.Key) || graph.Outputs.ContainsKey(pair.Key)) continue;
            if (!graph.Initializers.TryGetValue(pair.Key, out var value) || value is not DenseTensor<float> dense) continue;
            if (!Shape(dense, lanes, out long elements) || Storage(dense) is not float[] array) continue;
            result[pair.Key] = (dense, array, elements);
        }
        return result;
    }

    // Matrix packing starts with this residency, then convolution packing uses
    // its remainder. The public graph limit covers both maps, including refresh.
    internal static long PruneAndBytes(ComputationalGraph graph)
    {
        var current = Candidates(graph); var winograd = WinogradConsumers(graph); long retained = 0;
        foreach (var pair in graph.PackedConvWeights.ToArray())
        {
            var record = pair.Value; long bytes = record.Values.LongLength * sizeof(float);
            if (!current.TryGetValue(record.SourceName, out var source)
                || !ReferenceEquals(source.Tensor, record.Source) || !ReferenceEquals(source.Array, record.SourceArray)
                || !ReferenceEquals(pair.Key, source.Array) || record.Lanes != Lanes
                || !record.Shape.AsSpan().SequenceEqual(source.Tensor.Dimensions)
                || source.Elements != record.Values.LongLength || bytes > graph.MaximumPackedWeightBytes - retained)
            {
                graph.PackedConvWeights.Remove(pair.Key);
                continue;
            }
            retained += bytes;
            if (record.WinogradValues is not null)
            {
                long optionalBytes = record.WinogradValues.LongLength * sizeof(float);
                if (winograd.Contains(record.SourceName)
                    && WinogradShape(record.Shape[1], record.Shape[0], source.Elements, out int elements)
                    && record.WinogradValues.Length == elements
                    && optionalBytes <= graph.MaximumPackedWeightBytes - retained)
                    retained += optionalBytes;
                else graph.PackedConvWeights[pair.Key] = record with { WinogradValues = null };
            }
        }
        return retained;
    }

    internal static void PackWeights(ComputationalGraph graph)
    {
        foreach (var pair in Candidates(graph))
        {
            var source = pair.Value;
            if (graph.PackedConvWeights.ContainsKey(source.Array)) continue;
            long bytes = source.Elements * sizeof(float);
            if (bytes > graph.MaximumPackedWeightBytes - graph.RetainedPackedWeightBytes) continue;
            var shape = source.Tensor.Dimensions.ToArray(); int lanes = Lanes;
            var values = ConvBlockedSpatial.Prepare(source.Tensor.Buffer.Span, shape[1], shape[0], lanes);
            graph.PackedConvWeights.Add(source.Array, new PackedConvWeight(pair.Key, source.Tensor, source.Array, shape, lanes, values));
            graph.RetainedPackedWeightBytes += bytes;
        }
        var winograd = WinogradConsumers(graph);
        foreach (var pair in graph.PackedConvWeights.ToArray())
        {
            var record = pair.Value;
            if (record.WinogradValues is not null || !winograd.Contains(record.SourceName)
                || !WinogradShape(record.Shape[1], record.Shape[0], record.Values.LongLength, out int elements)) continue;
            long bytes = (long)elements * sizeof(float);
            if (bytes > graph.MaximumPackedWeightBytes - graph.RetainedPackedWeightBytes) continue;
            var values = ConvBlockedSpatial.PrepareWinograd(record.Source.Buffer.Span, record.Shape[1], record.Shape[0], record.Lanes);
            if (values is null) continue;
            graph.PackedConvWeights[pair.Key] = record with { WinogradValues = values };
            graph.RetainedPackedWeightBytes += bytes;
        }
    }

    internal static float[]? Resolve(IReadOnlyDictionary<float[], PackedConvWeight>? map, Tensor<float> weight, int lanes)
        => ResolveRecord(map, weight, lanes)?.Values;

    internal static PackedConvWeight? ResolveRecord(IReadOnlyDictionary<float[], PackedConvWeight>? map, Tensor<float> weight, int lanes)
    {
        if (map is null || map.Count == 0 || weight is not DenseTensor<float> dense
            || !Shape(dense, lanes, out long elements) || Storage(dense) is not float[] array
            || !map.TryGetValue(array, out var record) || !ReferenceEquals(record.Source, dense)
            || !ReferenceEquals(record.SourceArray, array) || record.Lanes != lanes
            || !record.Shape.AsSpan().SequenceEqual(dense.Dimensions) || record.Values.LongLength != elements) return null;
        if (record.WinogradValues is not null
            && (!WinogradShape(record.Shape[1], record.Shape[0], elements, out int count)
                || record.WinogradValues.Length != count)) return null;
        return record;
    }
}
