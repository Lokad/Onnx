namespace Lokad.Onnx;

using System.Collections.Generic;

internal static partial class GraphFusion
{
    // Keep the ordered, adjacent form: no intervening binding can change x.
    internal static int FuseSigmoidMul(ComputationalGraph graph, Optimization.GraphFacts facts)
    {
        var drop = new HashSet<int>();
        for (int i = 1; i < graph.Nodes.Count; i++)
        {
            var sigmoid = graph.Nodes[i - 1]; var mul = graph.Nodes[i];
            if (sigmoid.Op != OpType.Sigmoid || mul.Op != OpType.Mul
                || !IsFusableParticipant(sigmoid) || !IsFusableParticipant(mul)
                || sigmoid.Inputs.Length != 1 || sigmoid.Outputs.Length != 1
                || mul.Inputs.Length != 2 || mul.Outputs.Length != 1
                || sigmoid.Attributes is { Count: > 0 } || mul.Attributes is { Count: > 0 }) continue;
            string x = sigmoid.Inputs[0], mid = sigmoid.Outputs[0];
            // The fused kernel repeats Sigmoid's runtime type validation, so
            // an unknown static dtype needs no new assumption.
            if (string.IsNullOrEmpty(x) || string.IsNullOrEmpty(mid) || x == mid
                || mul.Inputs[0] != x || mul.Inputs[1] != mid
                || facts.GraphOutputs.Contains(mid) || facts.CapturedValues.Contains(mid)
                || graph.Inputs.ContainsKey(mid) || graph.Initializers.ContainsKey(mid) || !facts.IsSingleConsumer(mid)
                || facts.Dtypes.TryGetValue(x, out var dtype)
                    && dtype is not (TensorElementType.Float or TensorElementType.Double)) continue;
            mul.Op = OpType.FusedSigmoidMul; mul.OpTypeName = nameof(OpType.FusedSigmoidMul);
            mul.IsFused = true; mul.Domain = ""; mul.Inputs = new[] { x };
            graph.Nodes[i] = mul; drop.Add(i - 1);
        }
        if (drop.Count != 0)
        {
            var kept = new List<Node>(graph.Nodes.Count - drop.Count);
            for (int i = 0; i < graph.Nodes.Count; i++) if (!drop.Contains(i)) kept.Add(graph.Nodes[i]);
            graph.Nodes.Clear(); graph.Nodes.AddRange(kept);
        }
        return drop.Count;
    }
}
