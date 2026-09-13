using System;
using System.Collections.Generic;

namespace Lokad.Onnx.Optimization;

/// <summary>
/// G02-M3: self-shape Reshape lowering to zero-copy constants. Every E5 attention
/// Reshape rebuilds its shape as Concat(Unsqueeze(Gather(Shape(data), 0)),
/// Unsqueeze(Gather(Shape(data), 1)), literals...), where data is the Reshape own
/// data input. By the Reshape contract (allowzero off), a 0 entry copies that dim
/// from the data input, so the whole chain is bit-identically the constant vector
/// [0, 0, literals...] for every input length. The pass rewires each matching
/// Reshape to one shared Constant per distinct vector and sweeps the dead chains.
/// Only provenance-proven equalities merge: a Gather must index Shape(data) of the
/// same tensor at exactly its Concat position, and every literal must be an int64
/// Constant node (never an overridable initializer). No length is ever frozen: the
/// new constants carry zeros, never observed extents.
/// </summary>
internal static class ShapeZeroCopy
{
    static readonly object PassLock = new object();
    static bool registered;

    public static void RegisterShapeZeroCopyPass()
    {
        lock (PassLock)
        {
            if (registered) return;
            GraphOptimizer.AddPass(new GraphOptimizer.Pass(
                "reshape-zerocopy",
                (graph, facts) =>
                {
                    var rewritten = new List<int>();
                    int n = RewriteSelfShapes(graph, facts, rewritten, out int rewired, out int swept);
                    var result = new GraphOptimizer.PassResult(n > 0, n);
                    result.Nodes.AddRange(rewritten);
                    if (n > 0) result.Notes.Add("rewrote " + rewired + " reshape shape inputs; swept " + swept + " dead nodes");
                    return result;
                }));
            registered = true;
        }
    }

    static int RewriteSelfShapes(ComputationalGraph graph, GraphFacts facts, List<int> rewritten, out int rewired, out int swept)
    {
        var nodes = graph.Nodes;
        var shared = new Dictionary<string, string>(StringComparer.Ordinal);
        var pending = new List<KeyValuePair<long[], string>>();
        rewired = 0; swept = 0;
        for (int i = 0; i < nodes.Count; i++)
        {
            long[]? vector;
            try { vector = MatchSelfShape(nodes[i], graph, facts); }
            catch (Exception e) when (!Runtime.IsFatal(e)) { continue; }
            if (vector is null) continue;
            string key = string.Join(",", vector);
            if (!shared.TryGetValue(key, out var name))
            {
                name = FreshName(graph, shared);
                shared[key] = name;
                pending.Add(new KeyValuePair<long[], string>(vector, name));
            }
            var node = nodes[i];
            node.Inputs = new string[] { node.Inputs[0], name };
            nodes[i] = node;
            rewritten.Add(i);
            rewired++;
        }
        if (rewired == 0) return 0;
        int opset = graph.Opset.TryGetValue("", out var v) ? v : -1;
        for (int p = pending.Count - 1; p >= 0; p--)
        {
            var tensor = DenseTensor<long>.OfValues(pending[p].Key);
            tensor.Name = pending[p].Value;
            var constant = new Node
            {
                Name = pending[p].Value,
                ID = pending[p].Value.GetHashCode(),
                Attributes = new Dictionary<string, object> { ["value"] = tensor },
                Op = OpType.Constant,
                OpTypeName = OpType.Constant.ToString(),
                Domain = "",
                OpsetVersion = opset,
                IsFused = false,
                Inputs = Array.Empty<string>(),
                Outputs = new string[] { pending[p].Value },
            };
            nodes.Insert(0, constant);
            graph.IntermediateOutputs[pending[p].Value] = null;
        }
        int before = nodes.Count;
        ConstFold.SweepDeadNodes(graph, facts);
        swept = before - nodes.Count; return rewired + swept;
    }

    static string FreshName(ComputationalGraph graph, Dictionary<string, string> shared)
    {
        for (int n = 0; ; n++)
        {
            string candidate = "__reshape_zero_" + n;
            if (graph.IntermediateOutputs.ContainsKey(candidate)) continue;
            if (graph.Outputs.ContainsKey(candidate)) continue;
            if (graph.Inputs.ContainsKey(candidate)) continue;
            if (graph.Initializers.ContainsKey(candidate)) continue;
            if (shared.ContainsValue(candidate)) continue;
            bool clash = false;
            foreach (var node in graph.Nodes)
            {
                foreach (var o in node.Outputs ?? Array.Empty<string>())
                {
                    if (o == candidate) { clash = true; break; }
                }
                if (clash) break;
                foreach (var inp in node.Inputs ?? Array.Empty<string>())
                {
                    if (inp == candidate) { clash = true; break; }
                }
                if (clash) break;
            }
            if (!clash) return candidate;
        }
    }

    static long[]? MatchSelfShape(Node r, ComputationalGraph graph, GraphFacts facts)
    {
        if (r.IsFused) return null;
        if (!Node.IsStandardDomain(r.Domain)) return null;
        if (r.Op != OpType.Reshape) return null;
        if (r.Inputs is null || r.Inputs.Length != 2) return null;
        if (r.Outputs is null || r.Outputs.Length != 1) return null;
        string data = r.Inputs[0];
        string shapeIn = r.Inputs[1];
        if (string.IsNullOrEmpty(data) || string.IsNullOrEmpty(shapeIn)) return null;
        if (r.GetReshapeAllowZero()) return null;
        if (facts.Constants.ContainsKey(shapeIn)) return null;
        if (!facts.Producer.TryGetValue(shapeIn, out int ci)) return null;
        if (ci < 0 || ci >= graph.Nodes.Count) return null;
        var c = graph.Nodes[ci];
        if (c.IsFused || !Node.IsStandardDomain(c.Domain) || c.Op != OpType.Concat) return null;
        if (c.Inputs is null || c.Inputs.Length < 1) return null;
        if (c.RequiredInt("axis") != 0) return null;
        var vector = new List<long>(c.Inputs.Length);
        int pos = 0;
        bool seenDynamic = false;
        foreach (var entry in c.Inputs)
        {
            if (string.IsNullOrEmpty(entry)) return null;
            if (TryMatchDynamicEntry(entry, pos, data, graph, facts))
            {
                vector.Add(0);
                seenDynamic = true;
                pos++;
                continue;
            }
            long? lit = ReadSingletonInt64(entry, graph, facts);
            if (!lit.HasValue) return null;
            vector.Add(lit.Value);
            pos++;
        }
        if (!seenDynamic) return null;
        return vector.ToArray();
    }

    static bool TryMatchDynamicEntry(string entry, int pos, string data, ComputationalGraph graph, GraphFacts facts)
    {
        if (!facts.Producer.TryGetValue(entry, out int ui)) return false;
        if (ui < 0 || ui >= graph.Nodes.Count) return false;
        var u = graph.Nodes[ui];
        if (u.IsFused || !Node.IsStandardDomain(u.Domain) || u.Op != OpType.Unsqueeze) return false;
        if (u.Inputs is null || u.Inputs.Length != 1) return false;
        if (u.Outputs is null || u.Outputs.Length != 1) return false;
        int[]? axes;
        try { axes = u.Ints("axes"); }
        catch (ArgumentException) { return false; }
        if (axes is null || axes.Length != 1 || axes[0] != 0) return false;
        string gathered = u.Inputs[0];
        if (string.IsNullOrEmpty(gathered)) return false;
        if (!facts.Producer.TryGetValue(gathered, out int gi)) return false;
        if (gi < 0 || gi >= graph.Nodes.Count) return false;
        var g = graph.Nodes[gi];
        if (g.IsFused || !Node.IsStandardDomain(g.Domain) || g.Op != OpType.Gather) return false;
        if (g.Inputs is null || g.Inputs.Length != 2) return false;
        if (g.Outputs is null || g.Outputs.Length != 1) return false;
        int? axis;
        try { axis = g.Int("axis", null); }
        catch (ArgumentException) { return false; }
        if (axis.HasValue && axis.Value != 0) return false;
        string shaped = g.Inputs[0];
        string idxName = g.Inputs[1];
        if (string.IsNullOrEmpty(shaped) || string.IsNullOrEmpty(idxName)) return false;
        if (!facts.Producer.TryGetValue(shaped, out int si)) return false;
        if (si < 0 || si >= graph.Nodes.Count) return false;
        var s = graph.Nodes[si];
        if (s.IsFused || !Node.IsStandardDomain(s.Domain) || s.Op != OpType.Shape) return false;
        if (s.Inputs is null || s.Inputs.Length != 1) return false;
        if (!string.Equals(s.Inputs[0], data, StringComparison.Ordinal)) return false;
        try
        {
            if (s.Int("start", null).HasValue) return false;
            if (s.Int("end", null).HasValue) return false;
        }
        catch (ArgumentException) { return false; }
        long? idx = ReadInt64Constant(idxName, graph, facts);
        if (!idx.HasValue || idx.Value != pos) return false;
        var idxTensor = ConstantTensorOf(idxName, graph, facts);
        if (idxTensor is null || idxTensor.ElementType != TensorElementType.Int64) return false;
        if (idxTensor.Rank != 0) return false;
        return true;
    }

    static ITensor? ConstantTensorOf(string name, ComputationalGraph graph, GraphFacts facts)
    {
        if (!facts.Producer.TryGetValue(name, out int pi)) return null;
        if (pi < 0 || pi >= graph.Nodes.Count) return null;
        var node = graph.Nodes[pi];
        if (node.IsFused || !Node.IsStandardDomain(node.Domain) || node.Op != OpType.Constant) return null;
        if (node.Outputs is null || node.Outputs.Length != 1) return null;
        if (!string.Equals(node.Outputs[0], name, StringComparison.Ordinal)) return null;
        if (node.Attributes is null || !node.Attributes.TryGetValue("value", out var v) || !(v is ITensor t)) return null;
        return t;
    }

    static long? ReadInt64Constant(string name, ComputationalGraph graph, GraphFacts facts)
    {
        var t = ConstantTensorOf(name, graph, facts);
        if (t is null || t.ElementType != TensorElementType.Int64) return null;
        bool scalar = t.Rank == 0;
        bool singleton = t.Rank == 1 && t.Dims.Length == 1 && t.Dims[0] == 1;
        if (!scalar && !singleton) return null;
        try { return Convert.ToInt64(t.GetValue(0)); }
        catch (Exception e) when (!Runtime.IsFatal(e)) { return null; }
    }

    static long? ReadSingletonInt64(string name, ComputationalGraph graph, GraphFacts facts)
    {
        // Literals must be rank-1 singletons like the Unsqueeze outputs they sit
        // beside: a scalar literal would make the original Concat throw, so the
        // chain stays to preserve even that failure.
        var t = ConstantTensorOf(name, graph, facts);
        if (t is null || t.ElementType != TensorElementType.Int64) return null;
        if (t.Rank != 1 || t.Dims.Length != 1 || t.Dims[0] != 1) return null;
        return ReadInt64Constant(name, graph, facts);
    }
}
