using System;
using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx
{
    internal static class GraphFusion
    {
        public static int FuseLayerNormPatterns(ComputationalGraph graph)
        {
            var producer = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                foreach (var o in graph.Nodes[i].Outputs)
                {
                    if (!string.IsNullOrEmpty(o)) producer[o] = i;
                }
            }
            var consumers = new Dictionary<string, List<int>>(StringComparer.Ordinal);
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                foreach (var input in graph.Nodes[i].Inputs)
                {
                    if (string.IsNullOrEmpty(input)) continue;
                    if (!consumers.TryGetValue(input, out var list))
                    {
                        list = new List<int>();
                        consumers[input] = list;
                    }
                    list.Add(i);
                }
            }

            var drop = new HashSet<int>();
            int fused = 0;
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (graph.Nodes[i].Op != OpType.Div) continue;
                if (TryMatchLayerNorm(graph, producer, consumers, i, out var dead, out var epsConst))
                {
                    foreach (var d in dead) drop.Add(d);
                    if (epsConst >= 0)
                    {
                        bool ConstOnlyFeedsDead(string output) =>
                            !consumers.TryGetValue(output, out var uses) || uses.All(u => drop.Contains(u));
                        var cnode = graph.Nodes[epsConst];
                        if (cnode.Outputs.All(ConstOnlyFeedsDead)) drop.Add(epsConst);
                    }
                    fused++;
                }
            }
            if (drop.Count > 0)
            {
                var kept = new List<Node>(graph.Nodes.Count - drop.Count);
                for (int i = 0; i < graph.Nodes.Count; i++)
                {
                    if (!drop.Contains(i)) kept.Add(graph.Nodes[i]);
                }
                graph.Nodes.Clear();
                graph.Nodes.AddRange(kept);
            }
            return fused;
        }

        static bool TryMatchLayerNorm(
            ComputationalGraph graph,
            Dictionary<string, int> producer,
            Dictionary<string, List<int>> consumers,
            int divIndex,
            out List<int> dead,
            out int epsConst)
        {
            dead = new List<int>();
            epsConst = -1;
            var div = graph.Nodes[divIndex];
            if (div.Inputs.Length != 2) return false;
            int OnlyConsumer(string output, OpType op)
            {
                if (!consumers.TryGetValue(output, out var uses) || uses.Count != 1) return -1;
                return graph.Nodes[uses[0]].Op == op ? uses[0] : -1;
            }
            int ProducerOf(string input, OpType op)
            {
                if (string.IsNullOrEmpty(input) || !producer.TryGetValue(input, out var pi)) return -1;
                return graph.Nodes[pi].Op == op ? pi : -1;
            }
            float? ScalarOf(string input, out int constNode)
            {
                constNode = -1;
                if (string.IsNullOrEmpty(input)) return null;
                if (graph.Initializers.TryGetValue(input, out var init))
                {
                    var arr = init.ToArray();
                    if (arr.Length == 1) return Convert.ToSingle(arr.GetValue(0));
                    return null;
                }
                int pi = ProducerOf(input, OpType.Constant);
                if (pi < 0) return null;
                var tensor = ConstantValue(graph.Nodes[pi]);
                if (tensor is null) return null;
                var vals = tensor.ToArray();
                if (vals.Length != 1) return null;
                constNode = pi;
                return Convert.ToSingle(vals.GetValue(0));
            }
            bool IsLastAxisKeepDims(Node rm)
            {
                if (rm.Inputs.Length != 1) return false;
                var axes = rm.Ints("axes");
                if (axes is null || axes.Length != 1 || axes[0] != -1) return false;
                return (rm.Int("keepdims") ?? 1) == 1;
            }
            bool IsRankOne(string name)
            {
                return graph.Initializers.TryGetValue(name, out var t) && t.Rank == 1;
            }

            foreach (int di in new[] { 0, 1 })
            {
                int sub = ProducerOf(div.Inputs[di], OpType.Sub);
                int sqrt = ProducerOf(div.Inputs[1 - di], OpType.Sqrt);
                if (sub < 0 || sqrt < 0) continue;
                var subNode = graph.Nodes[sub];
                if (subNode.Inputs.Length != 2) continue;
                var sqrtNode = graph.Nodes[sqrt];
                if (sqrtNode.Inputs.Length != 1) continue;
                int addEps = ProducerOf(sqrtNode.Inputs[0], OpType.Add);
                if (addEps < 0 || graph.Nodes[addEps].Inputs.Length != 2) continue;
                var addEpsNode = graph.Nodes[addEps];
                int rm2 = -1;
                float? eps = null;
                int epsC = -1;
                foreach (var input in addEpsNode.Inputs)
                {
                    int r = ProducerOf(input, OpType.ReduceMean);
                    if (r >= 0 && rm2 < 0) { rm2 = r; continue; }
                    float? v = ScalarOf(input, out int cn);
                    if (v.HasValue && !eps.HasValue) { eps = v.Value; epsC = cn; }
                }
                if (rm2 < 0 || !eps.HasValue) continue;
                if (!IsLastAxisKeepDims(graph.Nodes[rm2])) continue;
                var pow = ProducerOf(graph.Nodes[rm2].Inputs[0], OpType.Pow);
                if (pow < 0 || graph.Nodes[pow].Inputs.Length != 2) continue;
                var powNode = graph.Nodes[pow];
                int sub2 = -1;
                float? exp = null;
                foreach (var input in powNode.Inputs)
                {
                    int s = ProducerOf(input, OpType.Sub);
                    if (s >= 0 && sub2 < 0) { sub2 = s; continue; }
                    float? v = ScalarOf(input, out _);
                    if (v.HasValue && !exp.HasValue) exp = v.Value;
                }
                if (sub2 != sub || !exp.HasValue || exp.Value != 2f) continue;
                int rm1 = -1;
                string? x = null;
                foreach (var input in subNode.Inputs)
                {
                    int r = ProducerOf(input, OpType.ReduceMean);
                    if (r >= 0 && rm1 < 0) { rm1 = r; continue; }
                    x ??= input;
                }
                if (rm1 < 0 || x is null) continue;
                if (!IsLastAxisKeepDims(graph.Nodes[rm1])) continue;
                int mu = OnlyConsumer(div.Outputs.Length == 1 ? div.Outputs[0] : "", OpType.Mul);
                if (mu < 0 || graph.Nodes[mu].Inputs.Length != 2) continue;
                var muNode = graph.Nodes[mu];
                string gamma = muNode.Inputs[0] == div.Outputs[0] ? muNode.Inputs[1] : muNode.Inputs[0];
                if (!IsRankOne(gamma)) continue;
                int ab = OnlyConsumer(muNode.Outputs.Length == 1 ? muNode.Outputs[0] : "", OpType.Add);
                if (ab < 0 || graph.Nodes[ab].Inputs.Length != 1 + 1) continue;
                var abNode = graph.Nodes[ab];
                string beta = abNode.Inputs[0] == muNode.Outputs[0] ? abNode.Inputs[1] : abNode.Inputs[0];
                if (!IsRankOne(beta)) continue;
                var deadSet = new HashSet<int> { sub, pow, rm1, rm2, addEps, sqrt, divIndex, mu };
                foreach (var d in deadSet)
                {
                    foreach (var o in graph.Nodes[d].Outputs)
                    {
                        if (string.IsNullOrEmpty(o)) continue;
                        if (o == abNode.Outputs[0]) continue;
                        if (consumers.TryGetValue(o, out var uses) && uses.Any(u => !deadSet.Contains(u) && u != ab)) return false;
                    }
                }
                dead.AddRange(deadSet);
                epsConst = epsC;
                var ln = abNode;
                ln.Op = OpType.LayerNormalization;
                ln.Inputs = new[] { x, gamma, beta };
                ln.Attributes = new Dictionary<string, object> { { "axis", -1 }, { "epsilon", eps.Value } };
                graph.Nodes[ab] = ln;
                return true;
            }
            return false;
        }

        static ITensor? ConstantValue(Node node)
        {
            if (node.Attributes is not null && node.Attributes.TryGetValue("value", out var v) && v is ITensor t) return t;
            return null;
        }
    }
}
