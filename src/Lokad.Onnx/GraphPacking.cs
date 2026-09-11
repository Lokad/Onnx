namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// A MatMul weight initializer with a prepared panel-packed clone.
/// The packed payload is owned immutably by the prepared plan: replacing
/// the source initializer is detected per use and falls back to the
/// unpacked path, while structural edits require InvalidatePreparation,
/// which drops the clone like the folded transposes.
/// </summary>
internal sealed record PackedMatMulWeight(
    string SourceName,
    ITensor SourceRef,
    long SourceLength,
    float[] SourceArray,
    string PackedName,
    DenseTensor<float> Packed);

/// <summary>
/// Builds panel-packed clones of eligible MatMul weight initializers.
/// Dispatch resolves them through the per-graph mapping at call time, so
/// every fallback path keeps reading original row-major bytes by
/// construction. Follows the FoldedTransposes ownership precedent.
/// </summary>
internal static class GraphPacking
{
    internal const string PackedPrefix = "packed:";

    /// <summary>Upper axis bound of measured packed-kernel territory (P46: 4096, covering GPT-2 c_proj at n=3072; census shows no other model edge in range).</summary>
    internal const int MaxPackedAxis = 4096;

    /// <summary>Upper packed-clone size in bytes (P34).</summary>
    /// <remarks>The reduction axis stays bounded while the panel axis scales linearly (proven to 1570 panels), so total bytes bound residency instead.</remarks>
    internal const long MaxPackedBytes = 512L * 1024 * 1024;

    /// <summary>Resolves a MatMul edge name to a folded-transpose prepared tensor (P34).</summary>
    /// <remarks>Folded outputs materialize under a generated name, so the consuming edge name never hits Initializers directly; the prepared bytes are stable and guarded exactly like initializer sources downstream.</remarks>
    static bool TryResolveFoldedSource(ComputationalGraph graph, string edgeName, out ITensor? init)
    {
        init = null;
        foreach (var node in graph.Nodes)
        {
            if (node.Op != OpType.Transpose || node.Outputs is null) continue;
            bool produces = false;
            foreach (var o in node.Outputs) if (o == edgeName) { produces = true; break; }
            if (!produces) continue;
            if (graph.FoldedTransposes.TryGetValue(node.Name, out var fold)
                && graph.Initializers.TryGetValue(fold.PreparedName, out var prepared)
                && prepared is DenseTensor<float>)
            {
                init = prepared;
                return true;
            }
            return false;
        }
        return false;
    }
    /// <summary>
    /// Packs eligible MatMul B-side weight initializers of the graph.
    /// Eligible means float32, rank 2, reduction axis below MaxPackedAxis with total bytes below MaxPackedBytes, plain layout (Gemm transB must be 0),
    /// never a graph input or output, and consumed only as MatMul input 1.
    /// Fresh records reuse verified clones; stale records are dropped and
    /// rebuilt. Returns the live packed count.
    /// </summary>
    internal static int PackMatMulWeights(ComputationalGraph graph)
    {
        var consumers = new Dictionary<string, bool>(StringComparer.Ordinal);
        foreach (var node in graph.Nodes)
        {
            if (node.Inputs is null) continue;
            for (int i = 0; i < node.Inputs.Length; i++)
            {
                string input = node.Inputs[i];
                if (string.IsNullOrEmpty(input)) continue;
                bool eligible = (node.Op == OpType.MatMul && i == 1) || (node.Op == OpType.Gemm && i == 1 && (node.GetInt("transB", 0) ?? 0) == 0);
                if (consumers.TryGetValue(input, out bool prior)) consumers[input] = prior && eligible;
                else consumers[input] = eligible;
            }
        }
        var current = new Dictionary<string, (ITensor tensor, float[] array)>(StringComparer.Ordinal);
        foreach (var kv in consumers)
        {
            if (!kv.Value) continue;
            if (graph.Inputs.ContainsKey(kv.Key) || graph.Outputs.ContainsKey(kv.Key)) continue;
            if (!graph.Initializers.TryGetValue(kv.Key, out var init) && !TryResolveFoldedSource(graph, kv.Key, out init)) continue;
            if (init is not DenseTensor<float> dense || init.ElementType != TensorElementType.Float) continue;
            if (init.Rank != 2 || dense.IsReversedStride) continue;
            if (!dense.strides.SequenceEqual(ArrayUtilities.GetStrides(dense.dimensions))) continue;
            int[] dims = init.Dims;
            if (dims.Length != 2) continue;
            int n = dims[0], k = dims[1];
            if (n < 1 || k < 1 || n >= MaxPackedAxis || (long)n * k > MaxPackedBytes) continue;
            if (!System.Runtime.InteropServices.MemoryMarshal.TryGetArray(dense.Buffer, out System.ArraySegment<float> window)
                || window.Array is null || window.Offset != 0 || window.Count != dense.Buffer.Length) continue;
            current[kv.Key] = (init, window.Array);
        }
        int live = 0;
        var stale = new List<float[]>();
        foreach (var kv in graph.PackedWeights)
        {
            var rec = kv.Value;
            if (current.TryGetValue(rec.SourceName, out var cur)
                && ReferenceEquals(cur.tensor, rec.SourceRef) && cur.tensor.Length == rec.SourceLength
                && ReferenceEquals(cur.array, rec.SourceArray)
                && graph.Initializers.TryGetValue(rec.PackedName, out var held)
                && ReferenceEquals(held, rec.Packed))
            {
                live++;
                continue;
            }
            stale.Add(kv.Key);
        }
        foreach (float[] key in stale)
        {
            if (graph.PackedWeights.TryGetValue(key, out var rec))
            {
                if (graph.Initializers.TryGetValue(rec.PackedName, out var held) && ReferenceEquals(held, rec.Packed))
                    graph.Initializers.Remove(rec.PackedName);
                graph.PackedWeights.Remove(key);
            }
        }
        foreach (var kv in current)
        {
            bool already = false;
            foreach (var rec in graph.PackedWeights.Values)
            {
                if (rec.SourceName == kv.Key && ReferenceEquals(rec.SourceRef, kv.Value.tensor)) { already = true; break; }
            }
            if (already) continue;
            string packedName = PackedPrefix + kv.Key;
            if (graph.Initializers.ContainsKey(packedName) || graph.Inputs.ContainsKey(packedName)) continue;
            int n = kv.Value.tensor.Dims[0], k = kv.Value.tensor.Dims[1];
            var panel = new float[(long)n * k];
            var dense = (DenseTensor<float>)kv.Value.tensor;
            unsafe
            {
                using var sh = dense.Buffer.Pin();
                using var ph = new Memory<float>(panel).Pin();
                MathOps.PackPanelsB(n, k, (float*)sh.Pointer, (float*)ph.Pointer);
            }
            var packed = new DenseTensor<float>(new Memory<float>(panel), new int[] { n, k });
            packed.Name = packedName;
            graph.Initializers[packedName] = packed;
            graph.PackedWeights[kv.Value.array] = new PackedMatMulWeight(kv.Key, kv.Value.tensor, kv.Value.tensor.Length, kv.Value.array, packedName, packed);
            live++;
        }
        return live;
    }
    /// <summary>
    /// Resolves a B-side operand to its packed clone when the unwrapped
    /// source is a fresh mapped weight. Broadcast views resolve through
    /// their source; anything else falls back to the unpacked path.
    /// </summary>
    internal static DenseTensor<float>? ResolvePacked(IReadOnlyDictionary<float[], PackedMatMulWeight>? map, Tensor<float> y)
    {
        if (map is null || map.Count == 0) return null;
        Tensor<float> core = y;
        while (core is BroadcastedTensor<float> view) core = view.source;
        if (core is DenseTensor<float> dense
            && map.TryGetValue(DenseArray(dense), out var rec)
            && ReferenceEquals(rec.SourceArray, DenseArray(dense))
            && TrailingDimsMatch(y, rec.Packed))
        {
            return rec.Packed;
        }
        Tensor<float> direct = y;
        while (direct is BroadcastedTensor<float> vw) direct = vw.source;
        if (direct is DenseTensor<float> named && named.Name is not null && named.Name.StartsWith(PackedPrefix, StringComparison.Ordinal))
        {
            foreach (var cand in map.Values)
            {
                if (cand.PackedName == named.Name && ReferenceEquals(cand.Packed, named)) return cand.Packed;
            }
        }
        return null;
    }

    static float[]? DenseArray(DenseTensor<float> dense)
    {
        if (!System.Runtime.InteropServices.MemoryMarshal.TryGetArray(dense.Buffer, out System.ArraySegment<float> window)
            || window.Array is null || window.Offset != 0) return null;
        return window.Array;
    }
    /// <summary>
    /// The operand trailing two dims must equal the packed dims, so the
    /// packed kernel reads the panels it was built from.
    /// </summary>
    static bool TrailingDimsMatch(Tensor<float> y, DenseTensor<float> packed)
    {
        int r = y.Rank;
        if (r < 2) return false;
        int[] pd = packed.Dimensions.ToArray();
        if (pd.Length != 2) return false;
        return y.Dimensions[r - 2] == pd[0] && y.Dimensions[r - 1] == pd[1];
    }
}
