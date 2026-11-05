namespace Lokad.Onnx;

internal static partial class GraphFusion
{
    internal static int FuseHarmonicSum(ComputationalGraph graph, Optimization.GraphFacts facts)
    {
        // Moving live operand reads to the final node requires single assignment.
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var node in graph.Nodes)
        foreach (string output in node.Outputs)
            if (!string.IsNullOrEmpty(output) && (!names.Add(output)
                || graph.Inputs.ContainsKey(output) || graph.Initializers.ContainsKey(output))) return 0;
        var removed = new HashSet<int>();
        int count = 0;
        for (int final = 0; final < graph.Nodes.Count; final++)
        {
            var sum = graph.Nodes[final];
            if (sum.Op != OpType.ReduceSum || !IsFusableParticipant(sum) || sum.OpsetVersion < 13
                || sum.Inputs.Length != 2 || sum.Outputs.Length != 1
                || !Attributes(sum, "keepdims", "noop_with_empty_axes")) continue;
            int product = Producer(sum.Inputs[0], OpType.Mul, 2);
            if (product < 0) continue;
            int sin = Producer(graph.Nodes[product].Inputs[0], OpType.Sin, 1);
            int amplitude = Producer(graph.Nodes[product].Inputs[1], OpType.Mul, 2);
            if (sin < 0 || amplitude < 0) continue;
            int phase = Producer(graph.Nodes[sin].Inputs[0], OpType.Mul, 2);
            int cast = Producer(graph.Nodes[amplitude].Inputs[0], OpType.Cast, 1, "to");
            if (phase < 0 || cast < 0 || graph.Nodes[cast].GetInt("to", -1) != (int)TensorElementType.Float) continue;
            int less = Producer(graph.Nodes[cast].Inputs[0], OpType.Less, 2);
            if (less < 0) continue;
            int frequency = Producer(graph.Nodes[less].Inputs[0], OpType.Mul, 2);
            if (frequency < 0) continue;
            string[] inputs = [graph.Nodes[frequency].Inputs[0], graph.Nodes[frequency].Inputs[1],
                graph.Nodes[less].Inputs[1], graph.Nodes[amplitude].Inputs[1],
                graph.Nodes[phase].Inputs[0], graph.Nodes[phase].Inputs[1], sum.Inputs[1]];
            if (inputs.Take(6).Any(name => !facts.Dtypes.TryGetValue(name, out var dtype) || dtype != TensorElementType.Float)) continue;
            int[] members = [frequency, less, cast, amplitude, phase, sin, product];
            if (members.Distinct().Count() != members.Length || members.Any(i => i >= final || removed.Contains(i))) continue;
            sum.Op = OpType.FusedHarmonicSum; sum.OpTypeName = nameof(OpType.FusedHarmonicSum);
            sum.IsFused = true; sum.Domain = ""; sum.Inputs = inputs;
            graph.Nodes[final] = sum;
            foreach (int index in members) removed.Add(index);
            count++;
        }
        if (removed.Count > 0)
        {
            var kept = new List<Node>(graph.Nodes.Count - removed.Count);
            for (int i = 0; i < graph.Nodes.Count; i++) if (!removed.Contains(i)) kept.Add(graph.Nodes[i]);
            graph.Nodes.Clear(); graph.Nodes.AddRange(kept);
        }
        return count;

        int Producer(string name, OpType op, int arity, params string[] attributes)
        {
            if (!facts.Producer.TryGetValue(name, out int index) || removed.Contains(index)) return -1;
            var node = graph.Nodes[index];
            return node.Op == op && IsFusableParticipant(node) && node.Inputs.Length == arity
                && node.Inputs.All(s => !string.IsNullOrEmpty(s)) && node.Outputs.Length == 1 && node.Outputs[0] == name
                && Attributes(node, attributes) && facts.IsSingleConsumer(name)
                && !facts.GraphOutputs.Contains(name) && !facts.CapturedValues.Contains(name) ? index : -1;
        }
        static bool Attributes(Node node, params string[] allowed) => node.Attributes is null || node.Attributes.Keys.All(allowed.Contains);
    }
}
