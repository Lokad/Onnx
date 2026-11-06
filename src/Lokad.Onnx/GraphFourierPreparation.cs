namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Lokad.Onnx.Optimization;

internal static class GraphFourierPreparation
{
    internal static IReadOnlyDictionary<int, PreparedFourierProjection[]>? Build(ComputationalGraph graph)
    {
        if (!graph.Options.Tensor.UseSimd || !graph.Options.Tensor.UseIntrinsics || !Vector.IsHardwareAccelerated) return null;
        long remaining = Math.Min(graph.MaximumPackedWeightBytes, GraphPacking.MaxPackedBytes) - graph.RetainedPackedWeightBytes;
        if (remaining <= 0) return null;
        var result = new Dictionary<int, List<PreparedFourierProjection>>();
        var visited = new HashSet<string>(StringComparer.Ordinal);
        GraphFacts? facts = null;
        foreach (var node in graph.Nodes)
        {
            if (node.Op != OpType.Conv || node.IsFused || !Node.IsStandardDomain(node.Domain)
                || node.Inputs.Length != 2 || !visited.Add(node.Inputs[1])) continue;
            try
            {
                if (node.GetInt("group", 1) != 1 || node.Ints("pads")?.Any(p => p != 0) == true
                    || node.Ints("dilations")?.Any(d => d != 1) == true) continue;
                string name = node.Inputs[1];
                if (graph.Initializers.TryGetValue(name, out var weight))
                {
                    if (weight is not DenseTensor<float> dense || !GraphConvPacking.Standard(dense)
                        || dense.Rank is not (3 or 4) || dense.Dimensions[1] != 1
                        || dense.Rank == 4 && dense.Dimensions[2] != 1) continue;
                    int length = dense.Dimensions[^1];
                    if (!PreparedFourierProjection.SupportedLength(length) || dense.Dimensions[0] != 2 * (length / 2 + 1)) continue;
                    if (PreparedFourierProjection.RequiredBytes(length) > remaining) continue;
                    var plan = PreparedFourierProjection.Hann(length);
                    if (!plan.Matches(dense.Buffer.Span))
                    {
                        plan = PreparedFourierProjection.FloatWindow(length, dense.Buffer.Span[..length]);
                        if (plan is null || !plan.Matches(dense.Buffer.Span)) continue;
                    }
                    Add(plan);
                }
                else
                {
                    var kernel = node.Ints("kernel_shape");
                    int length = kernel is { Length: 1 } ? kernel[0]
                        : kernel is { Length: 2 } && kernel[0] == 1 ? kernel[1] : 0;
                    if (!PreparedFourierProjection.SupportedLength(length) || PreparedFourierProjection.RequiredBytes(length) > remaining) continue;
                    facts ??= GraphFacts.Build(graph);
                    var names = new HashSet<string>(StringComparer.Ordinal);
                    var pending = new Stack<(string Name, int Depth)>();
                    pending.Push((name, 0));
                    while (pending.Count != 0)
                    {
                        var item = pending.Pop();
                        if (item.Depth > 4 || !names.Add(item.Name)) continue;
                        if (graph.Initializers.TryGetValue(item.Name, out var candidate))
                        {
                            if (candidate is not DenseTensor<float> window || !GraphConvPacking.Standard(window)
                                || window.Length != length || window.Rank is not (1 or 2)
                                || window.Rank == 2 && window.Dimensions[0] != 1) continue;
                            if (PreparedFourierProjection.RequiredBytes(length) > remaining) break;
                            var plan = PreparedFourierProjection.FloatWindow(length, window.Buffer.Span);
                            if (plan is not null) Add(plan);
                        }
                        else if (facts.Producer.TryGetValue(item.Name, out int producer))
                        {
                            var parent = graph.Nodes[producer];
                            if (Node.IsStandardDomain(parent.Domain)
                                && (!parent.IsFused && parent.Op is OpType.Unsqueeze or OpType.Mul or OpType.Concat
                                    || parent.IsFused && parent.Op == OpType.FusedFourierWeights))
                                foreach (string input in parent.Inputs) if (!string.IsNullOrEmpty(input)) pending.Push((input, item.Depth + 1));
                        }
                    }
                }
            }
            catch (Exception ex) when (!Runtime.IsFatal(ex)) { /* An unproved template uses ordinary Conv. */ }
        }
        return result.Count == 0 ? null : result.ToDictionary(pair => pair.Key, pair => pair.Value.ToArray());

        void Add(PreparedFourierProjection plan)
        {
            if (!result.TryGetValue(plan.Length, out var list)) result[plan.Length] = list = new();
            if (list.Any(existing => existing.SameWeights(plan)) || plan.RetainedBytes > remaining) return;
            list.Add(plan);
            remaining -= plan.RetainedBytes;
            graph.RetainedPackedWeightBytes += plan.RetainedBytes;
        }
    }
}
