namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;
using Lokad.Onnx.Optimization;

internal static partial class GraphFusion
{
    // One admitted region, with live weights/mask/pads/fill and private intermediates.
    // Require single assignment so moving the reads to the final slot cannot
    // observe an intervening rebinding of the original input or a weight.
    internal static int FuseConformerConv(ComputationalGraph graph, GraphFacts facts, Func<string, string, bool>? canOwn = null)
    {
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var node in graph.Nodes)
            foreach (string output in node.Outputs)
                if (!string.IsNullOrEmpty(output) && (!names.Add(output)
                    || graph.Inputs.ContainsKey(output) || graph.Initializers.ContainsKey(output))) return 0;
        var drop = new HashSet<int>();
        var replacements = new Dictionary<int, Node>();
        for (int final = 0; final < graph.Nodes.Count; final++)
        {
            if (!TryConformerRegion(graph, facts, final, out var indices, out var inputs)) continue;
            if (indices.Any(drop.Contains)) continue;
            if (canOwn is not null && (!canOwn(inputs[2], graph.Nodes[final].Name) || !canOwn(inputs[5], graph.Nodes[final].Name))) continue;
            var replacement = graph.Nodes[final];
            replacement.Op = OpType.FusedConformerConv;
            replacement.OpTypeName = nameof(OpType.FusedConformerConv);
            replacement.IsFused = true;
            replacement.Domain = "";
            replacement.Attributes = null;
            replacement.Inputs = inputs;
            replacements.Add(final, replacement);
            foreach (int index in indices) drop.Add(index);
        }
        if (replacements.Count == 0) return 0;
        var kept = new List<Node>(graph.Nodes.Count - drop.Count + replacements.Count);
        for (int i = 0; i < graph.Nodes.Count; i++)
            if (replacements.TryGetValue(i, out var replacement)) kept.Add(replacement);
            else if (!drop.Contains(i)) kept.Add(graph.Nodes[i]);
        graph.Nodes.Clear(); graph.Nodes.AddRange(kept);
        return replacements.Count;
    }

    static bool TryConformerRegion(ComputationalGraph graph, GraphFacts facts, int final, out int[] indices, out string[] inputs)
    {
        indices = Array.Empty<int>(); inputs = Array.Empty<string>();
        var nodes = graph.Nodes;
        var last = nodes[final];
        if (!ConformerTranspose(last)) return false;
        int Producer(string name, OpType op)
        {
            if (!facts.Producer.TryGetValue(name, out int index)) return -1;
            var node = nodes[index];
            if (node.Op != op || node.Outputs.Length != 1 || node.Outputs[0] != name
                || !facts.IsSingleConsumer(name) || facts.GraphOutputs.Contains(name) || facts.CapturedValues.Contains(name)) return -1;
            return index;
        }
        int pw2 = Producer(last.Inputs[0], OpType.Conv);
        if (pw2 < 0 || !ConformerConvNode(nodes[pw2], 1, 1, false)) return false;
        int silu = Producer(nodes[pw2].Inputs[0], OpType.FusedSigmoidMul);
        if (silu < 0 || !nodes[silu].IsFused || !Node.IsStandardDomain(nodes[silu].Domain)
            || nodes[silu].Inputs.Length != 1 || !ConformerAttributes(nodes[silu])) return false;
        int dw = Producer(nodes[silu].Inputs[0], OpType.Conv);
        if (dw < 0 || !ConformerConvNode(nodes[dw], 1024, 9, true)) return false;
        int pad = Producer(nodes[dw].Inputs[0], OpType.Pad);
        if (pad < 0 || !IsFusableParticipant(nodes[pad]) || !ConformerAttributes(nodes[pad], "mode")
            || nodes[pad].Attr("mode", "constant") != "constant"
            || nodes[pad].Inputs.Length is < 2 or > 3
            || nodes[pad].Inputs.Length == 3 && !string.IsNullOrEmpty(nodes[pad].Inputs[2])) return false;
        int where = Producer(nodes[pad].Inputs[0], OpType.Where);
        if (where < 0 || !IsFusableParticipant(nodes[where]) || nodes[where].Inputs.Length != 3
            || !ConformerAttributes(nodes[where])) return false;
        int mul = Producer(nodes[where].Inputs[2], OpType.Mul);
        if (mul < 0 || !IsFusableParticipant(nodes[mul]) || nodes[mul].Inputs.Length != 2 || !ConformerAttributes(nodes[mul])) return false;
        int sigmoid = Producer(nodes[mul].Inputs[1], OpType.Sigmoid);
        if (sigmoid < 0 || !IsFusableParticipant(nodes[sigmoid]) || nodes[sigmoid].Inputs.Length != 1
            || !ConformerAttributes(nodes[sigmoid]) || !facts.Producer.TryGetValue(nodes[mul].Inputs[0], out int split)) return false;
        var splitNode = nodes[split];
        if (splitNode.Op != OpType.Split || !IsFusableParticipant(splitNode) || splitNode.Inputs.Length != 1
            || splitNode.Outputs.Length != 2 || splitNode.Outputs[0] != nodes[mul].Inputs[0]
            || splitNode.Outputs[1] != nodes[sigmoid].Inputs[0] || splitNode.GetInt("axis", 0) != 1
            || !ConformerAttributes(splitNode, "axis")) return false;
        foreach (string output in splitNode.Outputs)
            if (!facts.IsSingleConsumer(output) || facts.GraphOutputs.Contains(output) || facts.CapturedValues.Contains(output)) return false;
        int pw1 = Producer(splitNode.Inputs[0], OpType.Conv);
        if (pw1 < 0 || !ConformerConvNode(nodes[pw1], 1, 1, false)) return false;
        int first = Producer(nodes[pw1].Inputs[0], OpType.Transpose);
        if (first < 0 || !ConformerTranspose(nodes[first])) return false;
        bool Weight(string name, params int[] shape) => graph.Initializers.TryGetValue(name, out var value)
            && value is Tensor<float> && value.Dims.SequenceEqual(shape);
        if (!Weight(nodes[pw1].Inputs[1], 2048, 1024, 1) || !Weight(nodes[pw2].Inputs[1], 1024, 1024, 1)
            || !Weight(nodes[dw].Inputs[1], 1024, 1, 9) || !Weight(nodes[dw].Inputs[2], 1024)) return false;

        indices = new[] { first, pw1, split, sigmoid, mul, where, pad, dw, silu, pw2, final };
        for (int i = 1; i < indices.Length; i++) if (indices[i] <= indices[i - 1]) return false;
        var members = new HashSet<int>(indices);
        var parameters = new HashSet<int>();
        var pending = new Stack<string>(new[] { nodes[where].Inputs[0], nodes[where].Inputs[1], nodes[pad].Inputs[1] });
        while (pending.Count != 0)
        {
            if (!facts.Producer.TryGetValue(pending.Pop(), out int index) || index < first) continue;
            if (index >= final || members.Contains(index)) return false;
            if (!parameters.Add(index)) continue;
            var parameter = nodes[index];
            if (parameter.Op is not (OpType.Constant or OpType.ConstantOfShape or OpType.Cast or OpType.Unsqueeze
                or OpType.Concat or OpType.Reshape or OpType.Slice or OpType.Transpose)
                || !IsFusableParticipant(parameter)) return false;
            foreach (string name in parameter.Inputs) if (!string.IsNullOrEmpty(name)) pending.Push(name);
        }
        for (int i = first; i <= final; i++)
            if (!members.Contains(i) && !parameters.Contains(i)) return false;
        inputs = new[] { nodes[first].Inputs[0], nodes[where].Inputs[0], nodes[pw1].Inputs[1],
            nodes[dw].Inputs[1], nodes[dw].Inputs[2], nodes[pw2].Inputs[1], nodes[pad].Inputs[1], nodes[where].Inputs[1] };
        return true;
    }

    static bool ConformerAttributes(Node node, params string[] allowed) =>
        node.Attributes is null || node.Attributes.Keys.All(key => allowed.Contains(key));

    static bool ConformerTranspose(Node node) => node.Op == OpType.Transpose && IsFusableParticipant(node)
        && node.Inputs.Length == 1 && node.Outputs.Length == 1 && ConformerAttributes(node, "perm")
        && node.Ints("perm") is { } perm && perm.SequenceEqual(new[] { 0, 2, 1 });

    static bool ConformerConvNode(Node node, int group, int kernel, bool bias)
    {
        if (!IsFusableParticipant(node) || node.Inputs.Length != (bias ? 3 : 2)
            || node.Outputs.Length != 1 || !ConformerAttributes(node, "auto_pad", "dilations", "group", "kernel_shape", "pads", "strides")
            || node.GetInt("group", 1) != group || node.Attr("auto_pad", "NOTSET") != "NOTSET") return false;
        return (node.Ints("kernel_shape") is not { } ks || ks.SequenceEqual(new[] { kernel }))
            && (node.Ints("dilations") is not { } dilations || dilations.SequenceEqual(new[] { 1 }))
            && (node.Ints("strides") is not { } strides || strides.SequenceEqual(new[] { 1 }))
            && (node.Ints("pads") is not { } pads || pads.SequenceEqual(new[] { 0, 0 }));
    }

}
