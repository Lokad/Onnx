namespace Lokad.Onnx;

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
}

