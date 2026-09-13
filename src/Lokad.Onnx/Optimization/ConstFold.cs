using System;
using System.Collections;
using System.Collections.Generic;

namespace Lokad.Onnx.Optimization;

/// <summary>
/// G02-M1: pure-constant folding over data-movement ops. Nodes whose every input resolves
/// to a constant are executed through the exact runtime path (Node.Execute on a scratch
/// graph seeded with the known constants) and replaced in place by Constant nodes, so
/// folding cannot disagree with runtime execution by construction. Only data-movement ops
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

    static int FoldConstants(ComputationalGraph graph, GraphFacts facts, List<int> rewritten)
    {
        var known = new Dictionary<string, ITensor>(StringComparer.Ordinal);
        var scratch = new ComputationalGraph();
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
            if (node.Inputs is null) continue;
            bool ready = true;
            foreach (var inp in node.Inputs)
            {
                if (string.IsNullOrEmpty(inp) || !known.ContainsKey(inp)) { ready = false; break; }
            }
            if (!ready) continue;
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
            for (int a = 1; a < group.Count; a++)
            {
                string survivor = graph.Nodes[group[0]].Outputs[0];
                string dup = graph.Nodes[group[a]].Outputs[0];
                var survivorTensor = (ITensor)graph.Nodes[group[0]].Attributes!["value"];
                var dupTensor = (ITensor)graph.Nodes[group[a]].Attributes!["value"];
                if (BitsEqual(survivorTensor, dupTensor)) rename[dup] = survivor;
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

    static long ByteCount(ITensor t)
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
