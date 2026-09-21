namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;

/// <summary>Private sharing between the two immutable decoder plans owned by one transcriber.</summary>
internal static class WhisperDecoderWeights
{
    // This helper is only called before the transcriber creates execution contexts.
    // Neither graph nor its initializer storage is exposed by the public audio API.
    // The returned number is logical shared payload, not reclaimed heap or RSS.
    internal static long Share(ComputationalGraph first, ComputationalGraph past)
    {
        if (ReferenceEquals(first, past)) return 0;
        past.InvalidatePreparation();
        var candidates = new Dictionary<(int Length, string Shape, string Digest), List<DenseTensor<float>>>();
        foreach (var entry in first.Initializers)
        {
            if (!Eligible(first, entry.Key, entry.Value, out var tensor)) continue;
            var key = Key(tensor);
            if (!candidates.TryGetValue(key, out var bucket)) candidates.Add(key, bucket = new List<DenseTensor<float>>());
            bucket.Add(tensor);
        }
        long shared = 0;
        foreach (var entry in past.Initializers.ToArray())
        {
            if (!Eligible(past, entry.Key, entry.Value, out var tensor)) continue;
            if (!candidates.TryGetValue(Key(tensor), out var bucket)) continue;
            foreach (var canonical in bucket)
            {
                // Hashes nominate candidates; equality of all bytes is the contract.
                if (!MemoryMarshal.AsBytes(canonical.Buffer.Span).SequenceEqual(MemoryMarshal.AsBytes(tensor.Buffer.Span))) continue;
                MemoryMarshal.TryGetArray(canonical.Buffer, out ArraySegment<float> canonicalArray);
                MemoryMarshal.TryGetArray(tensor.Buffer, out ArraySegment<float> oldArray);
                if (!ReferenceEquals(canonicalArray.Array, oldArray.Array))
                {
                    past.Initializers[entry.Key] = new DenseTensor<float>(canonical.Buffer, tensor.Dimensions) { Name = tensor.Name };
                    shared += (long)tensor.Length * sizeof(float);
                }
                break;
            }
        }
        return shared;
    }

    static (int Length, string Shape, string Digest) Key(DenseTensor<float> tensor) =>
        (checked((int)tensor.Length), string.Join(",", tensor.Dimensions.ToArray()),
            Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(tensor.Buffer.Span))));

    static bool Eligible(ComputationalGraph graph, string name, ITensor value, [NotNullWhen(true)] out DenseTensor<float>? tensor)
    {
        tensor = value as DenseTensor<float>;
        if (tensor is null || tensor.Length < 1024 || tensor.IsReversedStride || tensor.Buffer.Length != tensor.Length
            || name.StartsWith("folded:", StringComparison.Ordinal) || name.StartsWith("packed:", StringComparison.Ordinal)
            || graph.Inputs.ContainsKey(name) || graph.Outputs.ContainsKey(name)
            || graph.InputDescs.Any(d => d.Name == name) || graph.OutputDescs.Any(d => d.Name == name)) return false;
        int stride = 1;
        for (int axis = tensor.Rank - 1; axis >= 0; axis--)
        {
            if (tensor.Strides[axis] != stride) return false;
            stride = checked(stride * tensor.Dimensions[axis]);
        }
        return MemoryMarshal.TryGetArray(tensor.Buffer, out ArraySegment<float> array)
            && array.Array is not null && array.Offset == 0 && array.Count == array.Array.Length;
    }
}
