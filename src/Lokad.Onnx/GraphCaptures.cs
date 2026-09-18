namespace Lokad.Onnx;

/// <summary>Lexical reads shared by optimization, preparation and execution lifetime analysis.</summary>
internal static class GraphCaptures
{
    internal static string[] NodeInputs(Node node)
    {
        if (node.Attributes is not null)
            foreach (var value in node.Attributes.Values)
                if (value is ComputationalGraph)
                    return NodeInputs(node, new HashSet<ComputationalGraph>(ReferenceEqualityComparer.Instance));
        return node.Inputs ?? Array.Empty<string>();
    }

    static string[] NodeInputs(Node node, HashSet<ComputationalGraph> visiting)
    {
        List<string>? reads = null;
        if (node.Attributes is not null)
            foreach (var value in node.Attributes.Values)
                if (value is ComputationalGraph branch)
                {
                    reads ??= new List<string>(node.Inputs ?? Array.Empty<string>());
                    foreach (string name in FreeVariables(branch, visiting))
                        if (!reads.Contains(name)) reads.Add(name);
                }
        return reads is null ? node.Inputs ?? Array.Empty<string>() : reads.ToArray();
    }

    internal static string[] FreeVariables(ComputationalGraph graph) =>
        FreeVariables(graph, new HashSet<ComputationalGraph>(ReferenceEqualityComparer.Instance));

    static string[] FreeVariables(ComputationalGraph graph, HashSet<ComputationalGraph> visiting)
    {
        if (!visiting.Add(graph)) throw new InvalidOperationException("Cyclic graph attributes are not supported.");
        try
        {
            var local = new HashSet<string>(graph.Inputs.Keys, StringComparer.Ordinal);
            foreach (var desc in graph.InputDescs) local.Add(desc.Name);
            local.UnionWith(graph.Initializers.Keys);
            foreach (var node in graph.Nodes)
                if (node.Outputs is not null) local.UnionWith(node.Outputs);
            var reads = new HashSet<string>(StringComparer.Ordinal);
            foreach (var node in graph.Nodes) reads.UnionWith(NodeInputs(node, visiting));
            reads.UnionWith(graph.Outputs.Keys);
            foreach (var desc in graph.OutputDescs) reads.Add(desc.Name);
            reads.ExceptWith(local);
            reads.Remove("");
            return reads.Order(StringComparer.Ordinal).ToArray();
        }
        finally { visiting.Remove(graph); }
    }
}
