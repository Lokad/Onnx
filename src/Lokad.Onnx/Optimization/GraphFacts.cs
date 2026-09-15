using System;
using System.Collections.Generic;

namespace Lokad.Onnx.Optimization;

/// <summary>
/// Immutable-after-build snapshot of provable load-time facts over one node list.
/// Passes read facts through pure lookups; nothing here mutates the graph. The snapshot
/// is rebuilt whenever a pass reports a change, so facts never describe a newer list.
/// Dtype proving mirrors the historical per-fusion rules exactly (same accept/refuse answers);
/// only the plumbing is shared. Rank, dimensions and constants cover what load can prove
/// cheaply: model inputs, initializer rank/dtype (never foldable values) and
/// standard-domain Constant nodes. Produced tensors without a
/// static source stay unknown rather than guessed.
/// </summary>
internal sealed class GraphFacts
{
    public Dictionary<string, int> Producer { get; } = new Dictionary<string, int>(StringComparer.Ordinal);
    public Dictionary<string, List<int>> Consumers { get; } = new Dictionary<string, List<int>>(StringComparer.Ordinal);
    public HashSet<string> GraphOutputs { get; } = new HashSet<string>(StringComparer.Ordinal);
    public Dictionary<string, TensorElementType> Dtypes { get; } = new Dictionary<string, TensorElementType>();
    public Dictionary<string, int[]> KnownDims { get; } = new Dictionary<string, int[]>();
    public Dictionary<string, ITensor> Constants { get; } = new Dictionary<string, ITensor>(StringComparer.Ordinal);

    GraphFacts() { }

    public static GraphFacts Build(ComputationalGraph graph)
    {
        var facts = new GraphFacts();
        var nodes = new List<Node>(graph.Nodes);
        for (int i = 0; i < nodes.Count; i++)
        {
            foreach (var o in nodes[i].Outputs)
            {
                if (!string.IsNullOrEmpty(o)) facts.Producer[o] = i;
            }
        }
        for (int i = 0; i < nodes.Count; i++)
        {
            foreach (var input in nodes[i].Inputs)
            {
                if (string.IsNullOrEmpty(input)) continue;
                if (!facts.Consumers.TryGetValue(input, out var list))
                {
                    list = new List<int>();
                    facts.Consumers[input] = list;
                }
                list.Add(i);
            }
        }
        foreach (var k in graph.Outputs.Keys)
        {
            if (!string.IsNullOrEmpty(k)) facts.GraphOutputs.Add(k);
        }
        foreach (var desc in graph.InputDescs)
        {
            if (desc is null || string.IsNullOrEmpty(desc.Name)) continue;
            facts.Dtypes[desc.Name] = desc.ElementType;
            facts.KnownDims[desc.Name] = desc.Dims;
        }
        foreach (var kv in graph.Inputs)
        {
            if (kv.Value is not null && !string.IsNullOrEmpty(kv.Key))
            {
                facts.Dtypes[kv.Key] = kv.Value.ElementType;
                facts.KnownDims[kv.Key] = kv.Value.Dims;
            }
        }
        foreach (var kv in graph.Initializers)
        {
            if (kv.Value is null || string.IsNullOrEmpty(kv.Key)) continue;
            facts.Dtypes[kv.Key] = kv.Value.ElementType;
            facts.KnownDims[kv.Key] = kv.Value.Dims;
        }
        foreach (var kv in facts.Producer)
        {
            var node = nodes[kv.Value];
            // Fold sources are standard-domain Constant nodes only. Initializers stay
            // out even when immutable today: graph inputs may override them, callers may
            // replace their contents, and preparation invalidation cannot resurrect an
            // erased node. Custom-domain "Constant" nodes are not constants either.
            if (node.IsFused || !Node.IsStandardDomain(node.Domain)) continue;
            if (node.Op == OpType.Constant && node.Outputs is not null && node.Outputs.Length == 1
                && node.Attributes is not null && node.Attributes.TryGetValue("value", out var v) && v is ITensor t)
            {
                facts.Dtypes[kv.Key] = t.ElementType;
                facts.KnownDims[kv.Key] = t.Dims;
                facts.Constants[kv.Key] = t;
            }
        }
        var memo = new Dictionary<string, TensorElementType?>();
        foreach (var name in facts.Producer.Keys)
        {
            var dtype = ProveDtype(graph, nodes, facts.Producer, name, new HashSet<string>(StringComparer.Ordinal), memo);
            if (dtype.HasValue) facts.Dtypes[name] = dtype.Value;
        }
        return facts;
    }

    public bool IsSingleConsumer(string name) =>
        Consumers.TryGetValue(name, out var list) && list.Count == 1;

    public bool IsGraphOutput(string name) => GraphOutputs.Contains(name);

    static TensorElementType? ProveDtype(ComputationalGraph graph, List<Node> nodes, Dictionary<string, int> producer, string name, HashSet<string> visiting, Dictionary<string, TensorElementType?> memo)
    {
        if (string.IsNullOrEmpty(name)) return null;
        if (graph.Inputs.TryGetValue(name, out var gi) && gi is not null) return gi.ElementType;
        foreach (var desc in graph.InputDescs) if (desc is not null && desc.Name == name) return desc.ElementType;
        if (graph.Initializers.TryGetValue(name, out var ti) && ti is not null) return ti.ElementType;
        if (memo.TryGetValue(name, out var cached)) return cached;
        if (!producer.TryGetValue(name, out var pi)) return null;
        if (pi < 0 || pi >= nodes.Count) return null;
        if (!visiting.Add(name)) return null;
        TensorElementType? result = ProveNodeDtype(graph, nodes, producer, nodes[pi], visiting, memo);
        visiting.Remove(name);
        memo[name] = result;
        return result;
    }

    static TensorElementType? ProveNodeDtype(ComputationalGraph graph, List<Node> nodes, Dictionary<string, int> producer, Node pn, HashSet<string> visiting, Dictionary<string, TensorElementType?> memo)
    {
        if (pn.IsFused)
        {
            if (!Node.IsStandardDomain(pn.Domain)) return null;
            if (pn.Op == OpType.ConvRelu || pn.Op == OpType.AddRelu || pn.Op == OpType.BiasGelu || pn.Op == OpType.GemmGelu)
            {
                // Fused epilogues keep producer inputs verbatim, so prove under
                // the producer rule with a defused copy (Node is a value type).
                // The defused copy must also satisfy the producer arity gate:
                // BiasGelu carries [data, bias] while plain Gelu takes one
                // input, so truncate (data is first by matcher contract).
                var qn = pn;
                qn.IsFused = false;
                qn.Op = pn.Op == OpType.ConvRelu ? OpType.Conv : pn.Op == OpType.AddRelu ? OpType.Add : pn.Op == OpType.BiasGelu ? OpType.Gelu : OpType.Gemm;
                if (pn.Op == OpType.BiasGelu && qn.Inputs is not null && qn.Inputs.Length > 1)
                    qn.Inputs = new string[] { qn.Inputs[0] };
                return ProveNodeDtype(graph, nodes, producer, qn, visiting, memo);
            }
            if (pn.Op == OpType.ScaledMatMul)
            {
                // Defuse to the MatMul rule over the data/weight inputs; the
                // scalar scale must prove the same dtype (the Mul rule).
                if (pn.Inputs is null || pn.Inputs.Length != 3) return null;
                var qn = pn;
                qn.IsFused = false;
                qn.Op = OpType.MatMul;
                qn.Inputs = new string[] { pn.Inputs[0], pn.Inputs[1] };
                var madd = ProveNodeDtype(graph, nodes, producer, qn, visiting, memo);
                if (!madd.HasValue) return null;
                var third = ProveDtype(graph, nodes, producer, pn.Inputs[2], visiting, memo);
                if (!third.HasValue || third.Value != madd.Value) return null;
                return madd;
            }
            if (pn.Op != OpType.LayerNormalization && pn.Op != OpType.RotaryEmbedding && pn.Op != OpType.Gelu) return null;
            if (pn.Inputs is null || pn.Inputs.Length < 1) return null;
            return ProveDtype(graph, nodes, producer, pn.Inputs[0], visiting, memo);
        }
        if (pn.Op == OpType.Constant)
        {
            if (pn.Attributes is not null && pn.Attributes.TryGetValue("value", out var v) && v is ITensor t) return t.ElementType;
            return null;
        }
        if (pn.Op == OpType.Cast)
        {
            int? to = pn.Int("to", null);
            if (!to.HasValue) return null;
            return Enum.IsDefined(typeof(TensorElementType), to.Value) ? (TensorElementType)to.Value : null;
        }
        // Fused epilogues keep their producer inputs verbatim and Relu
        // preserves dtype, so they prove exactly under the producer rule.
        // These sit before the fusability gate on purpose: without them every
        // fact downstream of a fused node degrades to unknown, silently
        // disabling later matchers (found when Add+Relu fusion starved).
        if (pn.Op == OpType.ConvRelu)
        {
            if (pn.Inputs is null || pn.Inputs.Length < 2) return null;
            var first = ProveDtype(graph, nodes, producer, pn.Inputs[0], visiting, memo);
            var second = ProveDtype(graph, nodes, producer, pn.Inputs[1], visiting, memo);
            if (!first.HasValue || !second.HasValue || first.Value != second.Value) return null;
            if (pn.Inputs.Length >= 3 && !string.IsNullOrEmpty(pn.Inputs[2]))
            {
                var third = ProveDtype(graph, nodes, producer, pn.Inputs[2], visiting, memo);
                if (!third.HasValue || third.Value != first.Value) return null;
            }
            return first;
        }
        if (pn.Op == OpType.AddRelu)
        {
            if (pn.Inputs is null || pn.Inputs.Length != 2) return null;
            var left = ProveDtype(graph, nodes, producer, pn.Inputs[0], visiting, memo);
            var right = ProveDtype(graph, nodes, producer, pn.Inputs[1], visiting, memo);
            if (!left.HasValue || !right.HasValue || left.Value != right.Value) return null;
            return left;
        }
        if (!IsFusableParticipant(pn)) return null;
        if (pn.Op == OpType.Add || pn.Op == OpType.Sub || pn.Op == OpType.Mul || pn.Op == OpType.Div || pn.Op == OpType.Pow)
        {
            if (pn.Inputs is null || pn.Inputs.Length != 2) return null;
            var left = ProveDtype(graph, nodes, producer, pn.Inputs[0], visiting, memo);
            var right = ProveDtype(graph, nodes, producer, pn.Inputs[1], visiting, memo);
            if (!left.HasValue || !right.HasValue || left.Value != right.Value) return null;
            return left;
        }
        if (pn.Op == OpType.SplitToSequence)
        {
            // Sequences carry no tensor dtype themselves; SequenceAt
            // resolves through the split source below.
            return null;
        }
        if (pn.Op == OpType.SequenceAt)
        {
            // Element dtype equals the split source dtype: chunks preserve
            // dtype by construction (per-dtype ChunkCopy). Resolves only
            // through a visible SplitToSequence producer of the exact
            // sequence edge; anything else stays unknown rather than guessed.
            // Trust class matches the existing rules (same ProveDtype
            // machinery over initializers/descriptors/Constants).
            if (pn.Inputs is null || pn.Inputs.Length != 2) return null;
            string seq = pn.Inputs[0];
            if (string.IsNullOrEmpty(seq) || !producer.TryGetValue(seq, out var spi)) return null;
            if (spi < 0 || spi >= nodes.Count) return null;
            var split = nodes[spi];
            if (split.Op != OpType.SplitToSequence || split.Outputs is null || split.Outputs.Length != 1 || split.Outputs[0] != seq) return null;
            if (split.Inputs is null || split.Inputs.Length < 1 || string.IsNullOrEmpty(split.Inputs[0])) return null;
            return ProveDtype(graph, nodes, producer, split.Inputs[0], visiting, memo);
        }
        if (pn.Op == OpType.Concat)
        {
            // Mirrors the historical rule exactly, including its vacuous truth: empty inputs
            // are skipped, and all-empty (but present) inputs prove float like the original.
            if (pn.Inputs is null || pn.Inputs.Length < 1) return null;
            TensorElementType? first = null;
            foreach (var inp in pn.Inputs)
            {
                if (string.IsNullOrEmpty(inp)) continue;
                var next = ProveDtype(graph, nodes, producer, inp, visiting, memo);
                if (!next.HasValue) return null;
                if (!first.HasValue) first = next;
                else if (next.Value != first.Value) return null;
            }
            return first ?? TensorElementType.Float;
        }
        if (pn.Op == OpType.MatMul)
        {
            if (pn.Inputs is null || pn.Inputs.Length != 2) return null;
            var left = ProveDtype(graph, nodes, producer, pn.Inputs[0], visiting, memo);
            var right = ProveDtype(graph, nodes, producer, pn.Inputs[1], visiting, memo);
            if (!left.HasValue || !right.HasValue || left.Value != right.Value) return null;
            return left;
        }
        if (pn.Op == OpType.Conv || pn.Op == OpType.Gemm)
        {
            if (pn.Inputs is null || pn.Inputs.Length < 2) return null;
            var first = ProveDtype(graph, nodes, producer, pn.Inputs[0], visiting, memo);
            var second = ProveDtype(graph, nodes, producer, pn.Inputs[1], visiting, memo);
            if (!first.HasValue || !second.HasValue || first.Value != second.Value) return null;
            if (pn.Inputs.Length >= 3 && !string.IsNullOrEmpty(pn.Inputs[2]))
            {
                var third = ProveDtype(graph, nodes, producer, pn.Inputs[2], visiting, memo);
                if (!third.HasValue || third.Value != first.Value) return null;
            }
            return first;
        }
        if (pn.Op == OpType.Where)
        {
            if (pn.Inputs is null || pn.Inputs.Length != 3) return null;
            var branch = ProveDtype(graph, nodes, producer, pn.Inputs[1], visiting, memo);
            var other = ProveDtype(graph, nodes, producer, pn.Inputs[2], visiting, memo);
            if (!branch.HasValue || !other.HasValue || branch.Value != other.Value) return null;
            return branch;
        }
        if (pn.Op == OpType.Range)
        {
            if (pn.Inputs is null || pn.Inputs.Length != 3) return null;
            var first = ProveDtype(graph, nodes, producer, pn.Inputs[0], visiting, memo);
            if (!first.HasValue) return null;
            for (int i = 1; i < 3; i++)
            {
                var next = ProveDtype(graph, nodes, producer, pn.Inputs[i], visiting, memo);
                if (!next.HasValue || next.Value != first.Value) return null;
            }
            return first;
        }
        if (pn.Op == OpType.ConstantOfShape)
        {
            if (pn.Attributes is not null && pn.Attributes.TryGetValue("value", out var v) && v is ITensor ct && ct.ElementType == TensorElementType.Float) return TensorElementType.Float;
            return null;
        }
        switch (pn.Op)
        {
            case OpType.Gather:
            case OpType.Split:
            case OpType.SplitToSequence:
            case OpType.SequenceAt:
            case OpType.Tile:
            case OpType.GlobalAveragePool:
            case OpType.MaxPool:
            case OpType.Sqrt:
            case OpType.ReduceMean:
            case OpType.ReduceSum:
            case OpType.ReduceMax:
            case OpType.Transpose:
            case OpType.Squeeze:
            case OpType.Unsqueeze:
            case OpType.Reshape:
            case OpType.Slice:
            case OpType.Resize:
            case OpType.Expand:
            case OpType.Relu:
            case OpType.Gelu:
            case OpType.Tanh:
            case OpType.Erf:
            case OpType.Softmax:
            case OpType.Neg:
            case OpType.Abs:
            case OpType.Cos:
            case OpType.Sin:
            case OpType.LayerNormalization:
            case OpType.RotaryEmbedding:
                if (pn.Inputs is null || pn.Inputs.Length < 1 || string.IsNullOrEmpty(pn.Inputs[0])) return null;
                return ProveDtype(graph, nodes, producer, pn.Inputs[0], visiting, memo);
            default:
                return null;
        }
    }

    static bool IsFusableParticipant(Node pn)
    {
        if (pn.IsFused) return false;
        if (!CPUExecutionProvider.SupportsNode(pn)) return false;
        return true;
    }
}
