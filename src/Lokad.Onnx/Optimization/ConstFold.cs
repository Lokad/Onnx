using System;
using System.Collections;
using System.Collections.Generic;

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

    static readonly object FoldLock = new object();
    static bool registered;

    public static void RegisterConstFoldPass()
    {
        lock (FoldLock)
        {
            if (registered) return;
            GraphOptimizer.AddPass(new GraphOptimizer.Pass(
                "constfold",
                (graph, facts) =>
                {
                    var rewritten = new List<int>();
                    int n = FoldConstants(graph, facts, rewritten);
                    var result = new GraphOptimizer.PassResult(n > 0, n);
                    result.Nodes.AddRange(rewritten);
                    if (n > 0) result.Notes.Add("folded " + n + " constant subgraphs");
                    return result;
                }));
            registered = true;
        }
    }

    internal static int FoldConstants(ComputationalGraph graph, GraphFacts facts, List<int> rewritten)
    {
        var known = new Dictionary<string, ITensor>(StringComparer.Ordinal);
        var scratch = new ComputationalGraph();
        long foldedBytes = 0;
        foreach (var kv in facts.Constants) { known[kv.Key] = kv.Value; scratch.Initializers[kv.Key] = kv.Value; }
        var drop = new HashSet<int>();
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
            rewritten.Add(i);
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
        for (int round = 0; round < graph.Nodes.Count + 1; round++)
        {
            var consumers = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                foreach (var inp in graph.Nodes[i].Inputs ?? Array.Empty<string>())
                {
                    if (string.IsNullOrEmpty(inp)) continue;
                    consumers[inp] = consumers.TryGetValue(inp, out var c) ? c + 1 : 1;
                }
            }
            var drop = new List<int>();
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                bool live = false;
                foreach (var o in graph.Nodes[i].Outputs ?? Array.Empty<string>())
                {
                    if (string.IsNullOrEmpty(o)) continue;
                    if (facts.IsGraphOutput(o) || consumers.ContainsKey(o)) { live = true; break; }
                }
                if (!live) drop.Add(i);
            }
            if (drop.Count == 0) return;
            var dead = new HashSet<int>(drop);
            var kept = new List<Node>(graph.Nodes.Count - drop.Count);
            for (int i = 0; i < graph.Nodes.Count; i++) if (!dead.Contains(i)) kept.Add(graph.Nodes[i]);
            graph.Nodes.Clear();
            graph.Nodes.AddRange(kept);
        }
    }

    static void DedupeConstants(ComputationalGraph graph, GraphFacts facts)
    {
        var groups = new Dictionary<string, List<int>>();
        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            var node = graph.Nodes[i];
            if (node.Op != OpType.Constant || node.Outputs is null || node.Outputs.Length != 1) continue;
            string output = node.Outputs[0];
            if (string.IsNullOrEmpty(output) || facts.IsGraphOutput(output)) continue;
            if (node.Attributes is null || !node.Attributes.TryGetValue("value", out var v) || !(v is ITensor t)) continue;
            if (ByteCount(t) > MaxFoldBytes) continue;
            string key = t.ElementType.ToString() + "|" + string.Join("x", t.Dims);
            if (!groups.TryGetValue(key, out var list)) { list = new List<int>(); groups[key] = list; }
            list.Add(i);
        }
        var rename = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var group in groups.Values)
        {
            var survivors = new List<int> { group[0] };
            for (int a = 1; a < group.Count; a++)
            {
                string dup = graph.Nodes[group[a]].Outputs[0];
                var dupAttrs = graph.Nodes[group[a]].Attributes;
                if (dupAttrs is null) continue;
                if (!dupAttrs.TryGetValue("value", out var dupObj) || dupObj is not ITensor dupTensor) continue;
                foreach (int s in survivors)
                {
                    string survivor = graph.Nodes[s].Outputs[0];
                    var survivorAttrs = graph.Nodes[s].Attributes;
                    if (survivorAttrs is null) continue;
                    if (!survivorAttrs.TryGetValue("value", out var survivorObj) || survivorObj is not ITensor survivorTensor) continue;
                    if (BitsEqual(survivorTensor, dupTensor)) { rename[dup] = survivor; break; }
                }
                if (!rename.ContainsKey(dup)) survivors.Add(group[a]);
            }
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

    static bool BitsEqual(ITensor a, ITensor b)
    {
        if (a.ElementType != b.ElementType) return false;
        var da = a.ToArray();
        var db = b.ToArray();
        if (da.Length != db.Length) return false;
        switch (a.ElementType)
        {
            case TensorElementType.Float:
                var fa = (float[])da;
                var fb = (float[])db;
                for (int i = 0; i < fa.Length; i++)
                    if (BitConverter.SingleToInt32Bits(fa[i]) != BitConverter.SingleToInt32Bits(fb[i])) return false;
                return true;
            case TensorElementType.Double:
                var xa = (double[])da;
                var xb = (double[])db;
                for (int i = 0; i < xa.Length; i++)
                    if (BitConverter.DoubleToInt64Bits(xa[i]) != BitConverter.DoubleToInt64Bits(xb[i])) return false;
                return true;
            default:
                return StructuralComparisons.StructuralEqualityComparer.Equals(da, db);
        }
    }
}
