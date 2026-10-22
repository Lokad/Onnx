namespace Lokad.Onnx;

internal static partial class GraphFusion
{

    internal static int FuseRmsNorm(ComputationalGraph graph, Optimization.GraphFacts facts)
    {
        var removed = new HashSet<int>();
        int count = 0;
        for (int index = 0; index < graph.Nodes.Count; index++)
        {
            var last = graph.Nodes[index];
            if (removed.Contains(index) || !Binary(last, OpType.Mul)) continue;
            int normalizedIndex = Producer(last.Inputs[0], OpType.Mul);
            if (normalizedIndex < 0) continue;
            var normalized = graph.Nodes[normalizedIndex];
            string x = normalized.Inputs[0];
            int inverseIndex = Producer(normalized.Inputs[1], OpType.Pow);
            if (inverseIndex < 0) continue;
            var inverse = graph.Nodes[inverseIndex];
            int addIndex = Producer(inverse.Inputs[0], OpType.Add);
            if (addIndex < 0) continue;
            var add = graph.Nodes[addIndex];
            int meanIndex = Producer(add.Inputs[0], OpType.ReduceMean);
            if (meanIndex < 0) continue;
            var mean = graph.Nodes[meanIndex];
            if (mean.OpsetVersion < 18 || string.IsNullOrEmpty(mean.Inputs[1])) continue;
            int keep, noop;
            try { keep = mean.Int("keepdims", null) ?? 1; noop = mean.Int("noop_with_empty_axes", null) ?? 0; }
            catch (Exception error) when (error is ArgumentException or FormatException or InvalidCastException or OverflowException) { continue; }
            if (keep != 1) continue;
            int squareIndex = Producer(mean.Inputs[0], OpType.Mul);
            if (squareIndex < 0) continue;
            var square = graph.Nodes[squareIndex];
            if (square.Inputs[0] != x || square.Inputs[1] != x) continue;
            if (!KnownFloat(x) || !KnownFloat(add.Inputs[1]) || !KnownFloat(inverse.Inputs[1]) || !KnownFloat(last.Inputs[1])) continue;
            var dead = new[] { squareIndex, meanIndex, addIndex, inverseIndex, normalizedIndex };
            var consumers = new[] { meanIndex, addIndex, inverseIndex, normalizedIndex, index };
            bool privateValues = true;
            for (int i = 0; i < dead.Length; i++)
            {
                string output = graph.Nodes[dead[i]].Outputs[0];
                if (facts.GraphOutputs.Contains(output) || facts.CapturedValues.Contains(output)
                    || !facts.Consumers.TryGetValue(output, out var uses) || uses.Count != 1 || uses[0] != consumers[i])
                { privateValues = false; break; }
            }
            if (!privateValues) continue;
            last.Op = OpType.FusedRmsNormalization;
            last.OpTypeName = last.Op.ToString();
            last.IsFused = true;
            last.Domain = "";
            int version = StandardOpset(graph);
            if (version >= 0) last.OpsetVersion = version;
            last.Inputs = new[] { x, mean.Inputs[1], add.Inputs[1], inverse.Inputs[1], last.Inputs[1] };
            last.Attributes = new Dictionary<string, object> { ["keepdims"] = keep, ["noop_with_empty_axes"] = noop };
            graph.Nodes[index] = last;
            foreach (int deadIndex in dead) removed.Add(deadIndex);
            count++;
        }
        if (removed.Count > 0)
        {
            var kept = new List<Node>(graph.Nodes.Count - removed.Count);
            for (int i = 0; i < graph.Nodes.Count; i++) if (!removed.Contains(i)) kept.Add(graph.Nodes[i]);
            graph.Nodes.Clear();
            graph.Nodes.AddRange(kept);
        }
        return count;

        bool Binary(Node node, OpType op) => node.Op == op && IsFusableParticipant(node)
            && node.Inputs.Length == 2 && node.Outputs.Length == 1 && node.Inputs.All(s => !string.IsNullOrEmpty(s))
            && !string.IsNullOrEmpty(node.Outputs[0]);

        int Producer(string name, OpType op)
        {
            if (!facts.Producer.TryGetValue(name, out int producer) || removed.Contains(producer)) return -1;
            return Binary(graph.Nodes[producer], op) ? producer : -1;
        }

        bool KnownFloat(string name) => facts.Dtypes.TryGetValue(name, out var dtype) && dtype == TensorElementType.Float;
    }
}
