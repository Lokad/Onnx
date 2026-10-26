using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Lokad.Onnx.Optimization;

/// <summary>
/// G02-M1: pure-constant folding over data-movement ops. Nodes whose every input resolves
/// to a standard-domain Constant are executed through the exact runtime path (Node.Execute
/// on a scratch graph seeded with the known constants) and replaced in place by Constant nodes, so
/// folding cannot disagree with runtime execution by construction. Initializers never seed
/// folds (overridable inputs, replacement and in-place mutation must keep working), and graph
/// outputs never fold. Output sizes are estimated from input dimensions before any
/// execution (sound upper bounds; checked arithmetic refuses overflow), each fold is
/// capped at one megabyte, and one invocation stops adding folds past eight megabytes
/// total. Rounds repeat with a fresh counter, so a full load is bounded by rounds
/// (at most MaxRounds) times the per-invocation cap; every node folds at most once.
/// Only data-movement ops
/// fold here (Shape, Gather, Unsqueeze, Concat, Reshape, Transpose, Slice): their results
/// are bit-identical under every execution option, unlike arithmetic kernels whose float
/// reduction order may vary by mode. Graph outputs never fold (structure preserved), fused
/// nodes are left alone, and results over one megabyte are skipped. Dead producers left
/// behind are swept, unused Constants dropped, and bitwise-identical Constant payloads
/// deduplicated (bit comparison, so negative and positive zero never merge).
/// </summary>
internal static class ConstFold
{
    public const long MaxFoldBytes = 1024L * 1024;

    /// <summary>Aggregate estimated fold payload per pass. Bounds total preparation
    /// work and retained constants however many foldable chains a file carries.</summary>
    public const long MaxTotalFoldBytes = 8 * MaxFoldBytes;

    static readonly HashSet<OpType> Foldable = new HashSet<OpType>
    {
        OpType.Shape, OpType.Gather, OpType.Unsqueeze, OpType.Concat,
        OpType.Reshape, OpType.Transpose, OpType.Slice,
    };

    internal static int FoldConstants(ComputationalGraph graph, GraphFacts facts)
    {
        var known = new Dictionary<string, ITensor>(StringComparer.Ordinal);
        var scratch = new ComputationalGraph();
        long foldedBytes = 0;
        foreach (var kv in facts.Constants) { known[kv.Key] = kv.Value; scratch.Initializers[kv.Key] = kv.Value; }
        int folded = 0;
        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            var node = graph.Nodes[i];
            if (node.IsFused) continue;
            if (!Foldable.Contains(node.Op)) continue;
            if (!Node.IsStandardDomain(node.Domain)) continue;
            if (node.Outputs is null || node.Outputs.Length != 1) continue;
            string output = node.Outputs[0];
            if (string.IsNullOrEmpty(output)) continue;
            if (facts.IsGraphOutput(output)) continue;
            if (node.Inputs is null) continue;
            bool ready = true;
            foreach (var inp in node.Inputs)
            {
                if (string.IsNullOrEmpty(inp) || !known.ContainsKey(inp)) { ready = false; break; }
            }
            if (!ready) continue;
            long estimate = EstimateFoldBytes(node, known);
            if (estimate < 0 || estimate > MaxFoldBytes) continue;
            if (foldedBytes + estimate > MaxTotalFoldBytes) continue;
            var probe = node;
            OpResult r;
            try { r = probe.Execute(scratch, ExecutionProvider.CPU, null); }
            catch (Exception e) when (!Runtime.IsFatal(e)) { continue; }
            if (r.Status != OpStatus.Success || r.Outputs is null || r.Outputs.Length != 1 || r.Outputs[0] is null) continue;
            long bytes = ByteCount(r.Outputs[0]);
            if (bytes < 0 || bytes > MaxFoldBytes) continue;
            var foldedNode = node;
            foldedNode.Op = OpType.Constant;
            foldedNode.OpTypeName = OpType.Constant.ToString();
            foldedNode.Inputs = Array.Empty<string>();
            foldedNode.Attributes = new Dictionary<string, object> { ["value"] = r.Outputs[0] };
            graph.Nodes[i] = foldedNode;
            known[output] = r.Outputs[0];
            scratch.Initializers[output] = r.Outputs[0];
            folded++;
            foldedBytes += estimate;
        }
        int before = graph.Nodes.Count;
        SweepDeadNodes(graph, facts);
        DedupeConstants(graph, facts);
        int swept = before - graph.Nodes.Count;
        if (folded == 0 && swept == 0) return 0;
        return folded + swept;
    }

    internal static void SweepDeadNodes(ComputationalGraph graph, GraphFacts facts)
    {
        // Earlier folds may have changed producer positions. Build from the
        // current list and leave malformed duplicate producers for validation.
        var producers = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            foreach (var output in graph.Nodes[i].Outputs ?? Array.Empty<string>())
            {
                if (!string.IsNullOrEmpty(output) && !producers.TryAdd(output, i)) return;
            }
        }
        var live = new bool[graph.Nodes.Count];
        var pending = new Stack<int>();
        foreach (var output in facts.GraphOutputs)
            if (producers.TryGetValue(output, out int producer)) pending.Push(producer);
        while (pending.TryPop(out int index))
        {
            if (live[index]) continue;
            live[index] = true;
            foreach (var input in GraphCaptures.NodeInputs(graph.Nodes[index]))
                if (!string.IsNullOrEmpty(input) && producers.TryGetValue(input, out int producer) && !live[producer])
                    pending.Push(producer);
        }
        int kept = 0;
        for (int i = 0; i < graph.Nodes.Count; i++)
            if (live[i]) graph.Nodes[kept++] = graph.Nodes[i];
        graph.Nodes.RemoveRange(kept, graph.Nodes.Count - kept);
    }

    static void DedupeConstants(ComputationalGraph graph, GraphFacts facts)
    {
        var groups = new Dictionary<(TensorElementType Type, string Shape, int Hash), List<(string Name, Array Data)>>();
        var rename = new Dictionary<string, string>(StringComparer.Ordinal);
        long retainedBytes = 0;
        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            var node = graph.Nodes[i];
            if (node.Op != OpType.Constant || node.IsFused || !Node.IsStandardDomain(node.Domain)
                || node.Outputs is null || node.Outputs.Length != 1) continue;
            string output = node.Outputs[0];
            if (string.IsNullOrEmpty(output) || facts.IsGraphOutput(output) || facts.CapturedValues.Contains(output)) continue;
            if (node.Attributes is null || !node.Attributes.TryGetValue("value", out var v) || !(v is ITensor t)) continue;
            long estimate = ByteCount(t);
            if (estimate < 0 || estimate > MaxFoldBytes) continue;
            // One logical snapshot per candidate, never one per comparison.
            var data = t.ToArray();
            if (!TryGetBits(data, out var bits) || bits.Length > MaxFoldBytes) continue;
            var hash = new HashCode();
            hash.AddBytes(bits);
            var key = (t.ElementType, string.Join("x", t.Dims), hash.ToHashCode());
            if (groups.TryGetValue(key, out var survivors))
            {
                foreach (var survivor in survivors)
                {
                    // A hash collision is not proof of equal payloads.
                    if (TryGetBits(survivor.Data, out var candidate) && bits.SequenceEqual(candidate))
                    {
                        rename[output] = survivor.Name;
                        break;
                    }
                }
            }
            if (rename.ContainsKey(output)) continue;
            // Bound pass-local snapshots, including actual widened Complex storage.
            // A full budget still permits duplicates of already retained survivors.
            if (bits.Length > MaxTotalFoldBytes - retainedBytes) continue;
            if (survivors is null) groups.Add(key, survivors = new());
            survivors.Add((output, data));
            retainedBytes += bits.Length;
        }
        if (rename.Count == 0) return;
        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            var node = graph.Nodes[i];
            if (node.Inputs is null) continue;
            bool touched = false;
            var inputs = (string[])node.Inputs.Clone();
            for (int k = 0; k < inputs.Length; k++)
            {
                if (!string.IsNullOrEmpty(inputs[k]) && rename.TryGetValue(inputs[k], out var to)) { inputs[k] = to; touched = true; }
            }
            if (touched) { node.Inputs = inputs; graph.Nodes[i] = node; }
        }
        SweepDeadNodes(graph, facts);
    }

    static long ElementWidth(TensorElementType t)
    {
        switch (t)
        {
            case TensorElementType.Bool:
            case TensorElementType.Int8:
            case TensorElementType.UInt8: return 1;
            case TensorElementType.Int16:
            case TensorElementType.UInt16: return 2;
            case TensorElementType.Int32:
            case TensorElementType.UInt32:
            case TensorElementType.Float: return 4;
            case TensorElementType.Int64:
            case TensorElementType.UInt64:
            case TensorElementType.Double: return 8;
            default: return 8;
        }
    }

    /// <summary>Sound upper bound on a foldable node's output bytes from input
    /// dimensions alone: exact for same-count ops and Concat, an over-approximation
    /// for Slice/Gather. Negative signals refusal (unknown shapes, bad attributes,
    /// unchecked overflow). Runs before probe execution so hostile constants cannot
    /// force large allocations; the post-execution ByteCount check stays as backstop.</summary>
    static long EstimateFoldBytes(Node node, Dictionary<string, ITensor> known)
    {
        try
        {
            checked
            {
                switch (node.Op)
                {
                    case OpType.Shape:
                        if (node.Inputs is null || node.Inputs.Length < 1) return -1;
                        if (!known.TryGetValue(node.Inputs[0], out var sdata)) return -1;
                        return 8L * sdata.Dims.Length;
                    case OpType.Gather:
                        if (node.Inputs is null || node.Inputs.Length < 2) return -1;
                        if (!known.TryGetValue(node.Inputs[0], out var gdata)) return -1;
                        if (!known.TryGetValue(node.Inputs[1], out var gidx)) return -1;
                        int axis = node.Int("axis", null) ?? 0;
                        int rank = gdata.Dims.Length;
                        if (axis < 0) axis += rank;
                        if (axis < 0 || axis >= rank) return -1;
                        long tail = gdata.Length;
                        int dim = gdata.Dims[axis];
                        if (dim > 1) tail = gdata.Length / dim;
                        else if (dim < 1) return 0;
                        return gidx.Length * tail * ElementWidth(gdata.ElementType);
                    case OpType.Unsqueeze:
                    case OpType.Transpose:
                    case OpType.Reshape:
                    case OpType.Slice:
                        if (node.Inputs is null || node.Inputs.Length < 1) return -1;
                        if (!known.TryGetValue(node.Inputs[0], out var data)) return -1;
                        long n = 1;
                        foreach (var d in data.Dims)
                        {
                            if (d < 0) return -1;
                            n *= d;
                        }
                        return n * ElementWidth(data.ElementType);
                    case OpType.Concat:
                        if (node.Inputs is null || node.Inputs.Length < 1) return -1;
                        long total = 0;
                        foreach (var inp in node.Inputs)
                        {
                            if (string.IsNullOrEmpty(inp) || !known.TryGetValue(inp, out var part)) return -1;
                            total += ByteCount(part);
                            if (total < 0) return -1;
                        }
                        return total;
                    default:
                        return -1;
                }
            }
        }
        catch (OverflowException) { return -1; }
    }

    static long ByteCount(ITensor t)
    {
        try
        {
            checked
            {
                long n = 1;
                foreach (var d in t.Dims)
                {
                    if (d < 0) return -1;
                    n *= d;
                    if (n > MaxFoldBytes) return n * ElementWidth(t.ElementType);
                }
                return n * ElementWidth(t.ElementType);
            }
        }
        catch (OverflowException) { return -1; }
    }

    // ToArray supplies logical order. Use actual unmanaged storage widths:
    // CLR Complex contains two doubles, matching the Complex128 representation.
    static bool TryGetBits(Array data, out ReadOnlySpan<byte> bits)
    {
        switch (data)
        {
            case bool[] v: bits = MemoryMarshal.AsBytes(v.AsSpan()); return true;
            case sbyte[] v: bits = MemoryMarshal.AsBytes(v.AsSpan()); return true;
            case byte[] v: bits = v; return true;
            case short[] v: bits = MemoryMarshal.AsBytes(v.AsSpan()); return true;
            case ushort[] v: bits = MemoryMarshal.AsBytes(v.AsSpan()); return true;
            case int[] v: bits = MemoryMarshal.AsBytes(v.AsSpan()); return true;
            case uint[] v: bits = MemoryMarshal.AsBytes(v.AsSpan()); return true;
            case long[] v: bits = MemoryMarshal.AsBytes(v.AsSpan()); return true;
            case ulong[] v: bits = MemoryMarshal.AsBytes(v.AsSpan()); return true;
            case Half[] v: bits = MemoryMarshal.AsBytes(v.AsSpan()); return true;
            case BFloat16[] v: bits = MemoryMarshal.AsBytes(v.AsSpan()); return true;
            case float[] v: bits = MemoryMarshal.AsBytes(v.AsSpan()); return true;
            case double[] v: bits = MemoryMarshal.AsBytes(v.AsSpan()); return true;
            case Complex[] v: bits = MemoryMarshal.AsBytes(v.AsSpan()); return true;
            default: bits = default; return false;
        }
    }
}
