namespace Lokad.Onnx;

using System;
using System.Collections.Generic;

/// <summary>
/// Conservative preparation-time constant handling (M3): literal Constant
/// nodes with execution-internal outputs become initializers, so repeated
/// runs share the attribute storage instead of cloning it per execution.
/// The safety posture matches existing initializers exactly: converted names
/// are never graph outputs (callers could mutate exposed storage) and never
/// fed inputs (user values win), and node consumers already treat
/// initializer storage as read-only. Only top-level nodes convert; branch
/// subgraphs keep the per-execution clone until constant evaluation recurses
/// through branch plans.
/// </summary>
internal static class GraphConstants
{
    /// <summary>
    /// Folds top-level literal Constant nodes into initializers. Only direct
    /// ITensor "value" attributes qualify: sequences, scalar spellings, and
    /// missing attributes keep the node. Custom-domain constants keep the node so domain guards keep blocking fusion and dispatch. Graph outputs, fed names, existing
    /// initializer collisions, and nodes with inputs keep the per-execution
    /// clone. Idempotent: converted nodes are gone and converted names are
    /// initializers, so a second pass finds nothing to do. Returns the number
    /// of nodes folded.
    /// </summary>
    internal static int FoldLiteralConstants(ComputationalGraph graph)
    {
        int folded = 0;
        for (int i = graph.Nodes.Count - 1; i >= 0; i--)
        {
            var node = graph.Nodes[i];
            if (node.Op != OpType.Constant) continue;
            if (!Node.IsStandardDomain(node.Domain)) continue;
            if (node.Inputs is not null && node.Inputs.Length != 0) continue;
            if (node.Outputs is null || node.Outputs.Length != 1) continue;
            string output = node.Outputs[0];
            if (string.IsNullOrEmpty(output)) continue;
            if (node.Attributes is null) continue;
            if (!node.Attributes.TryGetValue("value", out var attr)) continue;
            if (attr is not ITensor tensor || attr is TensorSequence) continue;
            if (graph.Outputs.ContainsKey(output)) continue;
            if (graph.Inputs.ContainsKey(output)) continue;
            if (graph.Initializers.ContainsKey(output)) continue;
            tensor.Name = output;
            graph.Initializers[output] = tensor;
            graph.Nodes.RemoveAt(i);
            folded++;
        }
        return folded;
    }

    /// <summary>Prepared computed-constant record: the removed node blueprint for restore, the folded output, the source identity it was built from, and the plan-owned value.</summary>
    internal sealed record FoldedComputation(
        int Index,
        Node Blueprint,
        string Output,
        (string Name, ITensor Ref, long Length)[] Sources,
        ITensor Value);

    /// <summary>Constant-only ops eligible for preparation-time evaluation (M3).</summary>
    /// <remarks>Every member is pure and deterministic; shape members are exact under all options, while arithmetic members evaluate under the graph preparation options (matching default execution, and inside the numerical gates under custom per-call options). Random, stateful, control-flow, and multi-output ops are excluded by construction; data-dependent inputs fail the initializer precondition below.</remarks>
    internal static readonly HashSet<OpType> FoldableOps = new HashSet<OpType>
    {
        OpType.ConstantOfShape,
        OpType.Concat,
        OpType.Reshape,
        OpType.Slice,
        OpType.Transpose,
        OpType.Cast,
        OpType.Add,
        OpType.Sub,
        OpType.Mul,
        OpType.Div,
        OpType.Sin,
        OpType.Cos,
        OpType.Abs,
        OpType.Neg,
        OpType.Clip,
        OpType.Gather,
        OpType.Unsqueeze,
        OpType.MatMul,
    };

    /// <summary>Maximum elements of one folded value: bounds preparation residency.</summary>
    internal const long MaxFoldedElements = 16L * 1024 * 1024;

    /// <summary>
    /// Evaluates constant-only chains to fixpoint: any eligible node whose non-empty inputs all resolve to initializers executes once via the standard provider path and its fresh output becomes a plan-owned initializer while the node is removed. Runs after literal folding so folded literals feed chains, and before packing so folded weights pack. Skipped failures keep the node for run time, exactly as before.
    /// </summary>

    internal static int FoldComputedConstants(ComputationalGraph graph)
    {
        int folded = 0;
        bool progress;
        do
        {
            progress = false;
            for (int i = graph.Nodes.Count - 1; i >= 0; i--)
            {
                if (TryFoldComputedNode(graph, i))
                {
                    folded++;
                    progress = true;
                }
            }
        } while (progress);
        return folded;
    }

    static bool TryFoldComputedNode(ComputationalGraph graph, int index)
    {
        var node = graph.Nodes[index];
        if (!FoldableOps.Contains(node.Op)) return false;
        if (node.IsFused) return false;
        if (!Node.IsStandardDomain(node.Domain)) return false;
        if (node.Inputs is null) return false;
        if (node.Outputs is null || node.Outputs.Length != 1) return false;
        string output = node.Outputs[0];
        if (string.IsNullOrEmpty(output)) return false;
        if (graph.Outputs.ContainsKey(output)) return false;
        if (graph.Inputs.ContainsKey(output)) return false;
        if (graph.Initializers.ContainsKey(output)) return false;
        if (graph.FoldedComputations.ContainsKey(output)) return false;
        if (node.Op == OpType.Transpose && !FollowsFoldedInput(graph, node)) return false;
        var sources = new List<(string Name, ITensor Ref, long Length)>();
        foreach (string name in node.Inputs)
        {
            if (string.IsNullOrEmpty(name)) continue;
            if (graph.Inputs.ContainsKey(name)) return false;
            if (!graph.Initializers.TryGetValue(name, out var init)) return false;
            sources.Add((name, init, init.Length));
        }
        OpResult result;
        try
        {
            result = node.Execute(graph, ExecutionProvider.CPU, graph.Options);
        }
        catch (Exception ex) when (ex is ArgumentException || ex is InvalidOperationException || ex is NotSupportedException)
        {
            return false;
        }
        if (result.Status != OpStatus.Success || result.Outputs is null || result.Outputs.Length != 1) return false;
        var value = result.Outputs[0];
        if (value is null) return false;
        if (value.Length > MaxFoldedElements) return false;
        value.Name = output;
        graph.Initializers[output] = value;
        graph.FoldedComputations[output] = new FoldedComputation(index, node, output, sources.ToArray(), value);
        graph.Nodes.RemoveAt(index);
        return true;
    }

    /// <summary>Transpose ownership split: direct Transpose-over-initializer stays with the dedicated transpose-fold machinery (eager plus per-run source validation), so computed folding only takes mid-chain Transposes whose input is already a folded value. The dedicated path never sees removed nodes, and re-runs never ping-pong: folded inputs only come from this map.</summary>
    static bool FollowsFoldedInput(ComputationalGraph graph, Node node)
    {
        if (node.Inputs is null) return false;
        foreach (string name in node.Inputs)
        {
            if (!string.IsNullOrEmpty(name) && graph.FoldedComputations.ContainsKey(name)) return true;
        }
        return false;
    }

    /// <summary>
    /// Revalidates folded computations against current initializers. Unchanged sources keep their folds; a replaced source or an overridden folded output triggers a full restore (blueprints back, tracked values dropped) so the fold loop rebuilds from current values. User overrides of folded outputs survive as plain initializers.
    /// </summary>
    internal static void RevalidateFoldedComputations(ComputationalGraph graph)
    {
        if (graph.FoldedComputations.Count == 0) return;
        foreach (var rec in graph.FoldedComputations.Values)
        {
            if (!graph.Initializers.TryGetValue(rec.Output, out var cur) || !ReferenceEquals(cur, rec.Value))
            {
                RebuildFoldedComputations(graph);
                return;
            }
            foreach (var source in rec.Sources)
            {
                if (!graph.Initializers.TryGetValue(source.Name, out var init) || !ReferenceEquals(init, source.Ref) || init.Length != source.Length)
                {
                    RebuildFoldedComputations(graph);
                    return;
                }
            }
        }
    }

    /// <summary>Restores removed blueprint nodes in recorded order and drops tracked folded values, returning the graph to its unfolded structure.</summary>
    internal static void RebuildFoldedComputations(ComputationalGraph graph)
    {
        if (graph.FoldedComputations.Count == 0) return;
        var recs = new List<FoldedComputation>(graph.FoldedComputations.Values);
        recs.Sort((a, b) => a.Index.CompareTo(b.Index));
        graph.FoldedComputations.Clear();
        foreach (var rec in recs)
        {
            if (graph.Initializers.TryGetValue(rec.Output, out var held) && ReferenceEquals(held, rec.Value))
                graph.Initializers.Remove(rec.Output);
        }
        foreach (var rec in recs)
        {
            int at = rec.Index < 0 ? graph.Nodes.Count : Math.Min(rec.Index, graph.Nodes.Count);
            graph.Nodes.Insert(at, rec.Blueprint);
        }
    }

}