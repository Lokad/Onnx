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

        public static int FuseRopePatterns(ComputationalGraph graph)
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
                if (graph.Nodes[i].Op != OpType.Add) continue;
                if (drop.Contains(i)) continue;
                if (TryMatchRope(graph, producer, consumers, i, out var dead))
                {
                    foreach (var d in dead) drop.Add(d);
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

        static bool TryMatchRope(
            ComputationalGraph graph,
            Dictionary<string, int> producer,
            Dictionary<string, List<int>> consumers,
            int addIndex,
            out List<int> dead)
        {
            dead = new List<int>();
            var add = graph.Nodes[addIndex];
            if (add.Inputs.Length != 2) return false;
            int ProducerOf(string input, OpType op)
            {
                if (string.IsNullOrEmpty(input) || !producer.TryGetValue(input, out var pi)) return -1;
                return graph.Nodes[pi].Op == op ? pi : -1;
            }
            int OnlyConsumer(string output, OpType op)
            {
                if (!consumers.TryGetValue(output, out var uses) || uses.Count != 1) return -1;
                return graph.Nodes[uses[0]].Op == op ? uses[0] : -1;
            }
            long[]? LongsOf(string input)
            {
                if (string.IsNullOrEmpty(input)) return null;
                if (graph.Initializers.TryGetValue(input, out var init)) return ToLongs(init);
                int pi = ProducerOf(input, OpType.Constant);
                if (pi < 0) return null;
                var tensor = ConstantValue(graph.Nodes[pi]);
                if (tensor is null) return null;
                return ToLongs(tensor);
            }
            static long[]? ToLongs(ITensor t)
            {
                if (t.ElementType == TensorElementType.Int64) return (long[])t.ToArray();
                if (t.ElementType == TensorElementType.Int32) return ((int[])t.ToArray()).Select(v => (long)v).ToArray();
                return null;
            }
            int m1 = ProducerOf(add.Inputs[0], OpType.Mul);
            int m2 = ProducerOf(add.Inputs[1], OpType.Mul);
            if (m1 < 0 || m2 < 0 || m1 == m2) return false;
            if (graph.Nodes[m1].Inputs.Length != 2 || graph.Nodes[m2].Inputs.Length != 2) return false;
            if (graph.Nodes[m1].Outputs.Length != 1 || graph.Nodes[m2].Outputs.Length != 1) return false;

            if (TryRopeBranch(graph, producer, consumers, addIndex, m1, m2, ProducerOf, OnlyConsumer, LongsOf, out dead)) return true;
            if (TryRopeBranch(graph, producer, consumers, addIndex, m2, m1, ProducerOf, OnlyConsumer, LongsOf, out dead)) return true;
            return false;
        }

        static bool TryRopeBranch(
            ComputationalGraph graph,
            Dictionary<string, int> producer,
            Dictionary<string, List<int>> consumers,
            int addIndex,
            int mxIdx,
            int mrIdx,
            Func<string, OpType, int> producerOf,
            Func<string, OpType, int> onlyConsumer,
            Func<string, long[]?> longsOf,
            out List<int> dead)
        {
            dead = new List<int>();
            var add = graph.Nodes[addIndex];
            var mx = graph.Nodes[mxIdx];
            var mr = graph.Nodes[mrIdx];
            for (int qi = 0; qi < 2; qi++)
            {
                string x = mx.Inputs[qi];
                string c = mx.Inputs[1 - qi];
                int cosIdx = producerOf(c, OpType.Cos);

                if (cosIdx < 0) continue;
                var cosNode = graph.Nodes[cosIdx];
                if (cosNode.Inputs.Length != 1) continue;
                if (!consumers.TryGetValue(x, out var xuses)) continue;
                var sliceUses = xuses.Where(u => graph.Nodes[u].Op == OpType.Slice).ToList();
                if (sliceUses.Count != 2) continue;
                for (int ri = 0; ri < 2; ri++)
                {
                    string ccIn = mr.Inputs[ri];
                    string sIn = mr.Inputs[1 - ri];
                    int sinIdx = producerOf(sIn, OpType.Sin);

                    if (sinIdx < 0) continue;
                    var sinNode = graph.Nodes[sinIdx];
                    if (sinNode.Inputs.Length != 1) continue;
                    if (cosNode.Inputs[0] != sinNode.Inputs[0]) continue;
                    int concatIdx = producerOf(ccIn, OpType.Concat);
                    if (concatIdx < 0) continue;
                    var cnode = graph.Nodes[concatIdx];
                    if (cnode.Inputs.Length != 2 || cnode.Outputs.Length != 1) continue;
                    int? caxis = cnode.Int("axis");
                    if (!caxis.HasValue) continue;
                    int negIdx = producerOf(cnode.Inputs[0], OpType.Neg);
                    if (negIdx < 0) continue;
                    var nnode = graph.Nodes[negIdx];
                    if (nnode.Inputs.Length != 1 || nnode.Outputs.Length != 1) continue;
                    if (TryRopeSlices(graph, producerOf, onlyConsumer, longsOf, sliceUses, cnode, negIdx, concatIdx, mrIdx, mxIdx, addIndex, out int sFirst, out int sSecond, out int half, out int axis))
                    {
                        var deadSet = new HashSet<int> { sFirst, sSecond, negIdx, concatIdx, mxIdx, mrIdx };
                        foreach (var d in deadSet)
                        {
                            foreach (var o in graph.Nodes[d].Outputs)
                            {
                                if (string.IsNullOrEmpty(o)) continue;
                                if (o == add.Outputs[0]) continue;
                                if (consumers.TryGetValue(o, out var uses) && uses.Any(u => !deadSet.Contains(u) && u != addIndex)) return false;
                            }
                        }
                        dead.AddRange(deadSet);
                        var fused = add;
                        fused.Op = OpType.RotaryEmbedding;
                        fused.Inputs = new[] { x, c, sIn };
                        fused.Attributes = new Dictionary<string, object> { { "half", half }, { "axis", axis }, { "concatAxis", caxis.Value } };
                        graph.Nodes[addIndex] = fused;
                        return true;
                    }
                }
            }
            return false;
        }

        static bool TryRopeSlices(
            ComputationalGraph graph,
            Func<string, OpType, int> producerOf,
            Func<string, OpType, int> onlyConsumer,
            Func<string, long[]?> longsOf,
            System.Collections.Generic.List<int> sliceUses,
            Node cnode,
            int negIdx,
            int concatIdx,
            int mrIdx,
            int mxIdx,
            int addIndex,
            out int sFirst,
            out int sSecond,
            out int half,
            out int axis)
        {
            sFirst = -1;
            sSecond = -1;
            half = 0;
            axis = 0;
            var nnode = graph.Nodes[negIdx];
            int sNeg = producerOf(nnode.Inputs[0], OpType.Slice);
            if (sNeg < 0 || !sliceUses.Contains(sNeg)) return false;
            foreach (int first in sliceUses)
            {
                if (first == sNeg) continue;
                if (!SliceBounds(graph, longsOf, first, out long s0, out long e0, out long ax, out long st)) continue;
                if (!SliceBounds(graph, longsOf, sNeg, out long s1, out long e1, out long ax2, out long st2)) continue;
                if (s0 != 0 || st != 1 || st2 != 1 || ax != ax2) continue;
                if (e1 != long.MaxValue) continue;
                if (e0 <= 0 || e0 > int.MaxValue || s1 != e0) continue;
                if (ax < int.MinValue || ax > int.MaxValue) continue;
                var firstNode = graph.Nodes[first];
                var secondNode = graph.Nodes[sNeg];
                if (firstNode.Outputs.Length != 1 || secondNode.Outputs.Length != 1) continue;


                if (onlyConsumer(firstNode.Outputs[0], OpType.Concat) != concatIdx) continue;
                if (onlyConsumer(secondNode.Outputs[0], OpType.Neg) != negIdx) continue;
                if (onlyConsumer(nnode.Outputs[0], OpType.Concat) != concatIdx) continue;
                var concatOut = cnode.Outputs[0];
                var mxOut = graph.Nodes[mxIdx].Outputs;
                var mrOut = graph.Nodes[mrIdx].Outputs;
                if (mxOut.Length != 1 || mrOut.Length != 1) continue;
                if (onlyConsumer(concatOut, OpType.Mul) != mrIdx) continue;

                if (onlyConsumer(mxOut[0], OpType.Add) != addIndex) continue;
                if (onlyConsumer(mrOut[0], OpType.Add) != addIndex) continue;
                if (producerOf(cnode.Inputs[1], OpType.Slice) != first) continue;
                sFirst = first;
                sSecond = sNeg;
                half = (int)e0;
                axis = (int)ax;
                return true;
            }
            return false;
        }

        static bool SliceBounds(
            ComputationalGraph graph,
            Func<string, long[]?> longsOf,
            int sliceIdx,
            out long starts,
            out long ends,
            out long axes,
            out long steps)
        {
            starts = 0;
            ends = 0;
            axes = 0;
            steps = 0;
            var node = graph.Nodes[sliceIdx];
            if (node.Inputs.Length != 5) return false;
            var s = longsOf(node.Inputs[1]);
            var e = longsOf(node.Inputs[2]);
            var a = longsOf(node.Inputs[3]);
            var st = longsOf(node.Inputs[4]);
            if (s is null || e is null || a is null || st is null) return false;
            if (s.Length != 1 || e.Length != 1 || a.Length != 1 || st.Length != 1) return false;
            starts = s[0];
            ends = e[0];
            axes = a[0];
            steps = st[0];
            return true;
        }
        static ITensor? ConstantValue(Node node)
        {
            if (node.Attributes is not null && node.Attributes.TryGetValue("value", out var v) && v is ITensor t) return t;
            return null;
        }
    }
}
