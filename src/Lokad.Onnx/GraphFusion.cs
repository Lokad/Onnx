using System;
using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx
{
    internal static class GraphFusion
    {
        sealed class UseIndex
        {
            public Dictionary<string, int> Producer = new Dictionary<string, int>(StringComparer.Ordinal);
            public Dictionary<string, List<int>> Consumers = new Dictionary<string, List<int>>(StringComparer.Ordinal);
            public HashSet<string> Outputs = new HashSet<string>(StringComparer.Ordinal);
        }

        static UseIndex BuildIndex(ComputationalGraph graph)
        {
            var idx = new UseIndex();
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                foreach (var o in graph.Nodes[i].Outputs)
                {
                    if (!string.IsNullOrEmpty(o)) idx.Producer[o] = i;
                }
            }
            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                foreach (var input in graph.Nodes[i].Inputs)
                {
                    if (string.IsNullOrEmpty(input)) continue;
                    if (!idx.Consumers.TryGetValue(input, out var list))
                    {
                        list = new List<int>();
                        idx.Consumers[input] = list;
                    }
                    list.Add(i);
                }
            }
            foreach (var k in graph.Outputs.Keys)
            {
                if (!string.IsNullOrEmpty(k)) idx.Outputs.Add(k);
            }
            return idx;
        }

        public static int FuseLayerNormPatterns(ComputationalGraph graph)
        {
            var idx = BuildIndex(graph);
            var producer = idx.Producer;
            var consumers = idx.Consumers;
            var outputs = idx.Outputs;

            var drop = new HashSet<int>();
            int fused = 0;

            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (graph.Nodes[i].Op != OpType.Div) continue;
                if (drop.Contains(i)) continue;
                if (TryMatchLayerNorm(graph, producer, consumers, outputs, drop, i, out var dead, out var epsConst))
                {
                    foreach (var d in dead) drop.Add(d);
                    if (epsConst >= 0)
                    {
                        bool ConstOnlyFeedsDead(string output) =>
                            !outputs.Contains(output) && (!consumers.TryGetValue(output, out var uses) || uses.All(u => drop.Contains(u)));
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
            HashSet<string> outputs,
            HashSet<int> drop,
            int divIndex,
            out List<int> dead,
            out int epsConst)
        {
            dead = new List<int>();
            epsConst = -1;
            var div = graph.Nodes[divIndex];
            if (div.Inputs.Length != 2) return false;
            if (div.Outputs.Length != 1) return false;
            int OnlyConsumer(string output, OpType op)
            {
                if (outputs.Contains(output)) return -1;
                if (!consumers.TryGetValue(output, out var uses)) return -1;
                var live = new List<int>();
                foreach (var u in uses) if (!drop.Contains(u)) live.Add(u);
                if (live.Count != 1) return -1;
                return graph.Nodes[live[0]].Op == op ? live[0] : -1;
            }
            int ProducerOf(string input, OpType op)
            {
                if (string.IsNullOrEmpty(input) || !producer.TryGetValue(input, out var pi)) return -1;
                if (drop.Contains(pi)) return -1;
                return graph.Nodes[pi].Op == op ? pi : -1;
            }
            float? ScalarFloatOf(string input, out int constNode)
            {
                constNode = -1;
                if (string.IsNullOrEmpty(input)) return null;
                if (graph.Initializers.TryGetValue(input, out var init))
                {
                    if (init.ElementType != TensorElementType.Float) return null;
                    var arr = init.ToArray();
                    if (arr.Length != 1) return null;
                    float v = Convert.ToSingle(arr.GetValue(0));
                    if (!float.IsFinite(v)) return null;
                    return v;
                }
                int pi = ProducerOf(input, OpType.Constant);
                if (pi < 0) return null;
                var cnode = graph.Nodes[pi];
                if (cnode.Outputs.Length != 1) return null;
                var tensor = ConstantValue(cnode);
                if (tensor is null) return null;
                if (tensor.ElementType != TensorElementType.Float) return null;
                var vals = tensor.ToArray();
                if (vals.Length != 1) return null;
                float fv = Convert.ToSingle(vals.GetValue(0));
                if (!float.IsFinite(fv)) return null;
                constNode = pi;
                return fv;
            }
            bool IsLastAxisKeepDims(Node rm)
            {
                if (rm.Inputs.Length != 1) return false;
                if (rm.Outputs.Length != 1) return false;
                var axes = rm.Ints("axes");
                if (axes is null || axes.Length != 1 || axes[0] != -1) return false;
                return (rm.Int("keepdims", null) ?? 1) == 1;
            }
            bool IsFloatRankOne(string name, int expectedLength)
            {
                if (!graph.Initializers.TryGetValue(name, out var t)) return false;
                if (t.ElementType != TensorElementType.Float) return false;
                if (t.Rank != 1) return false;
                if (t.Length < 1) return false;
                if (expectedLength > 0 && (int)t.Length != expectedLength) return false;
                return true;
            }
            bool IsKnownFloat(string name)
            {
                if (graph.Inputs.TryGetValue(name, out var gi)) return gi.ElementType == TensorElementType.Float;
                if (graph.Initializers.TryGetValue(name, out var ti)) return ti.ElementType == TensorElementType.Float;
                return true;
            }
            int LastDimOf(string name)
            {
                if (graph.Inputs.TryGetValue(name, out var gi) && gi.Dims.Length > 0) return gi.Dims[gi.Dims.Length - 1];
                if (graph.Initializers.TryGetValue(name, out var ti) && ti.Dims.Length > 0) return ti.Dims[ti.Dims.Length - 1];
                return -1;
            }
            bool OpsetSupportsAttributeReduceMean()
            {
                if (graph.Opset.TryGetValue("", out var v)) return v < 18;
                return false;
            }

            if (!OpsetSupportsAttributeReduceMean()) return false;
            int sub = ProducerOf(div.Inputs[0], OpType.Sub);
            int sqrt = ProducerOf(div.Inputs[1], OpType.Sqrt);
            if (sub < 0 || sqrt < 0) return false;
            var subNode = graph.Nodes[sub];
            if (subNode.Inputs.Length != 2) return false;
            if (subNode.Outputs.Length != 1) return false;
            var sqrtNode = graph.Nodes[sqrt];
            if (sqrtNode.Inputs.Length != 1) return false;
            if (sqrtNode.Outputs.Length != 1) return false;
            string xPos = subNode.Inputs[0];
            string mPos = subNode.Inputs[1];
            int rm1 = ProducerOf(mPos, OpType.ReduceMean);
            if (rm1 < 0) return false;
            var rm1Node = graph.Nodes[rm1];
            if (rm1Node.Inputs.Length != 1) return false;
            if (rm1Node.Outputs.Length != 1) return false;
            if (rm1Node.Inputs[0] != xPos) return false;
            if (!IsKnownFloat(xPos)) return false;
            if (!IsLastAxisKeepDims(rm1Node)) return false;
            int addEps = ProducerOf(sqrtNode.Inputs[0], OpType.Add);
            if (addEps < 0) return false;
            var addEpsNode = graph.Nodes[addEps];
            if (addEpsNode.Inputs.Length != 2) return false;
            if (addEpsNode.Outputs.Length != 1) return false;
            int rm2 = -1;
            float? eps = null;
            int epsC = -1;
            foreach (var input in addEpsNode.Inputs)
            {
                int r = ProducerOf(input, OpType.ReduceMean);
                if (r >= 0 && rm2 < 0) { rm2 = r; continue; }
                float? v = ScalarFloatOf(input, out int cn);
                if (v.HasValue && !eps.HasValue) { eps = v.Value; epsC = cn; }
            }
            if (rm2 < 0 || !eps.HasValue) return false;
            if (eps.Value < 0f) return false;
            var rm2Node = graph.Nodes[rm2];
            if (rm2Node.Inputs.Length != 1) return false;
            if (rm2Node.Outputs.Length != 1) return false;
            if (!IsLastAxisKeepDims(rm2Node)) return false;
            var pow = ProducerOf(rm2Node.Inputs[0], OpType.Pow);
            if (pow < 0) return false;
            var powNode = graph.Nodes[pow];
            if (powNode.Inputs.Length != 2) return false;
            if (powNode.Outputs.Length != 1) return false;
            if (ProducerOf(powNode.Inputs[0], OpType.Sub) != sub) return false;
            float? exp = ScalarFloatOf(powNode.Inputs[1], out _);
            if (!exp.HasValue || exp.Value != 2f) return false;
            if (subNode.Outputs[0] != powNode.Inputs[0]) return false;
            if (subNode.Outputs[0] != div.Inputs[0]) return false;
            int mu = OnlyConsumer(div.Outputs[0], OpType.Mul);
            if (mu < 0) return false;
            var muNode = graph.Nodes[mu];
            if (muNode.Inputs.Length != 2) return false;
            if (muNode.Outputs.Length != 1) return false;
            string gamma = muNode.Inputs[0] == div.Outputs[0] ? muNode.Inputs[1] : muNode.Inputs[0];
            int lastDim = LastDimOf(xPos);
            if (!IsFloatRankOne(gamma, lastDim)) return false;
            int ab = OnlyConsumer(muNode.Outputs[0], OpType.Add);
            if (ab < 0) return false;
            var abNode = graph.Nodes[ab];
            if (abNode.Inputs.Length != 2) return false;
            if (abNode.Outputs.Length != 1) return false;
            string beta = abNode.Inputs[0] == muNode.Outputs[0] ? abNode.Inputs[1] : abNode.Inputs[0];
            if (!IsFloatRankOne(beta, lastDim)) return false;
            if (drop.Contains(ab)) return false;
            var deadSet = new HashSet<int> { sub, pow, rm1, rm2, addEps, sqrt, divIndex, mu };
            foreach (var dd in deadSet) if (drop.Contains(dd)) return false;
            foreach (var d in deadSet)
            {
                foreach (var o in graph.Nodes[d].Outputs)
                {
                    if (string.IsNullOrEmpty(o)) continue;
                    if (o == abNode.Outputs[0]) continue;
                    if (outputs.Contains(o)) return false;
                    if (consumers.TryGetValue(o, out var uses) && uses.Any(u => !deadSet.Contains(u) && u != ab && !drop.Contains(u))) return false;
                }
            }
            dead.AddRange(deadSet);
            epsConst = epsC;
            var ln = abNode;
            ln.Op = OpType.LayerNormalization;
            ln.OpTypeName = OpType.LayerNormalization.ToString();
            ln.IsFused = true;
            ln.Inputs = new[] { xPos, gamma, beta };
            ln.Attributes = new Dictionary<string, object> { { "axis", -1 }, { "epsilon", eps.Value } };
            graph.Nodes[ab] = ln;
            return true;
        }

        public static int FuseRopePatterns(ComputationalGraph graph)
        {
            var idx = BuildIndex(graph);
            var producer = idx.Producer;
            var consumers = idx.Consumers;
            var outputs = idx.Outputs;

            var drop = new HashSet<int>();
            int fused = 0;

            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (graph.Nodes[i].Op != OpType.Add) continue;
                if (drop.Contains(i)) continue;
                if (TryMatchRope(graph, producer, consumers, outputs, drop, i, out var dead))
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
            HashSet<string> outputs,
            HashSet<int> drop,
            int addIndex,
            out List<int> dead)
        {
            dead = new List<int>();
            var add = graph.Nodes[addIndex];
            if (add.Inputs.Length != 2) return false;
            if (add.Outputs.Length != 1) return false;
            int ProducerOf(string input, OpType op)
            {
                if (string.IsNullOrEmpty(input) || !producer.TryGetValue(input, out var pi)) return -1;
                if (drop.Contains(pi)) return -1;
                return graph.Nodes[pi].Op == op ? pi : -1;
            }
            int OnlyConsumer(string output, OpType op)
            {
                if (outputs.Contains(output)) return -1;
                if (!consumers.TryGetValue(output, out var uses)) return -1;
                var live = new List<int>();
                foreach (var u in uses) if (!drop.Contains(u)) live.Add(u);
                if (live.Count != 1) return -1;
                return graph.Nodes[live[0]].Op == op ? live[0] : -1;
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

            bool TryRopeBranch(
                int mxIdx,
                int mrIdx,
                out List<int> branchDead)
            {
                branchDead = new List<int>();
                var branchAdd = graph.Nodes[addIndex];
                var mx = graph.Nodes[mxIdx];
                var mr = graph.Nodes[mrIdx];

                bool TryRopeSlices(
                    System.Collections.Generic.List<int> sliceUses,
                    Node cnode,
                    int negIdx,
                    int concatIdx,
                    out int sFirst,
                    out int sSecond,
                    out int half,
                    out int axis)
                {
                        bool SliceBounds(
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
                            var s = LongsOf(node.Inputs[1]);
                            var e = LongsOf(node.Inputs[2]);
                            var a = LongsOf(node.Inputs[3]);
                            var stp = LongsOf(node.Inputs[4]);
                            if (s is null || e is null || a is null || stp is null) return false;
                            if (s.Length != 1 || e.Length != 1 || a.Length != 1 || stp.Length != 1) return false;
                            starts = s[0];
                            ends = e[0];
                            axes = a[0];
                            steps = stp[0];
                            return true;
                        }
                    sFirst = -1;
                    sSecond = -1;
                    half = 0;
                    axis = 0;
                    var nnode = graph.Nodes[negIdx];
                    int sNeg = ProducerOf(nnode.Inputs[0], OpType.Slice);
                    if (sNeg < 0 || !sliceUses.Contains(sNeg)) return false;
                    foreach (int first in sliceUses)
                    {
                        if (first == sNeg) continue;
                        if (!SliceBounds(first, out long s0, out long e0, out long ax, out long st)) continue;
                        if (!SliceBounds(sNeg, out long s1, out long e1, out long ax2, out long st2)) continue;
                        if (s0 != 0 || st != 1 || st2 != 1 || ax != ax2) continue;
                        if (e1 != long.MaxValue) continue;
                        if (e0 <= 0 || e0 > int.MaxValue || s1 != e0) continue;
                        if (ax < int.MinValue || ax > int.MaxValue) continue;
                        var firstNode = graph.Nodes[first];
                        var secondNode = graph.Nodes[sNeg];
                        if (firstNode.Outputs.Length != 1 || secondNode.Outputs.Length != 1) continue;


                        if (OnlyConsumer(firstNode.Outputs[0], OpType.Concat) != concatIdx) continue;
                        if (OnlyConsumer(secondNode.Outputs[0], OpType.Neg) != negIdx) continue;
                        if (OnlyConsumer(nnode.Outputs[0], OpType.Concat) != concatIdx) continue;
                        var concatOut = cnode.Outputs[0];
                        var mxOut = graph.Nodes[mxIdx].Outputs;
                        var mrOut = graph.Nodes[mrIdx].Outputs;
                        if (mxOut.Length != 1 || mrOut.Length != 1) continue;
                        if (OnlyConsumer(concatOut, OpType.Mul) != mrIdx) continue;

                        if (OnlyConsumer(mxOut[0], OpType.Add) != addIndex) continue;
                        if (OnlyConsumer(mrOut[0], OpType.Add) != addIndex) continue;
                        if (ProducerOf(cnode.Inputs[1], OpType.Slice) != first) continue;
                        sFirst = first;
                        sSecond = sNeg;
                        half = (int)e0;
                        axis = (int)ax;
                        return true;
                    }
                    return false;
                }

                for (int qi = 0; qi < 2; qi++)
                {
                    string x = mx.Inputs[qi];
                    string c = mx.Inputs[1 - qi];
                    int cosIdx = ProducerOf(c, OpType.Cos);

                    if (cosIdx < 0) continue;
                    var cosNode = graph.Nodes[cosIdx];
                    if (cosNode.Inputs.Length != 1) continue;
                    if (!consumers.TryGetValue(x, out var xuses)) continue;
                    var sliceUses = xuses.Where(u => !drop.Contains(u) && graph.Nodes[u].Op == OpType.Slice).ToList();
                    if (sliceUses.Count != 2) continue;
                    if (outputs.Contains(x)) continue;
                    for (int ri = 0; ri < 2; ri++)
                    {
                        string ccIn = mr.Inputs[ri];
                        string sIn = mr.Inputs[1 - ri];
                        int sinIdx = ProducerOf(sIn, OpType.Sin);

                        if (sinIdx < 0) continue;
                        var sinNode = graph.Nodes[sinIdx];
                        if (sinNode.Inputs.Length != 1) continue;
                        if (cosNode.Inputs[0] != sinNode.Inputs[0]) continue;
                        int concatIdx = ProducerOf(ccIn, OpType.Concat);
                        if (concatIdx < 0) continue;
                        var cnode = graph.Nodes[concatIdx];
                        if (cnode.Inputs.Length != 2 || cnode.Outputs.Length != 1) continue;
                        int? caxis = cnode.Int("axis", null);
                        if (!caxis.HasValue) continue;
                        int negIdx = ProducerOf(cnode.Inputs[0], OpType.Neg);
                        if (negIdx < 0) continue;
                        var nnode = graph.Nodes[negIdx];
                        if (nnode.Inputs.Length != 1 || nnode.Outputs.Length != 1) continue;
                        if (TryRopeSlices(sliceUses, cnode, negIdx, concatIdx, out int sFirst, out int sSecond, out int half, out int axis))
                        {
                            var deadSet = new HashSet<int> { sFirst, sSecond, negIdx, concatIdx, mxIdx, mrIdx };
                            foreach (var dd in deadSet) if (drop.Contains(dd)) return false;
                            if (drop.Contains(addIndex)) return false;
                            foreach (var d in deadSet)
                            {
                                foreach (var o in graph.Nodes[d].Outputs)
                                {
                                    if (string.IsNullOrEmpty(o)) continue;
                                    if (o == branchAdd.Outputs[0]) continue;
                                    if (outputs.Contains(o)) return false;
                                    if (consumers.TryGetValue(o, out var uses) && uses.Any(u => !deadSet.Contains(u) && u != addIndex && !drop.Contains(u))) return false;
                                }
                            }
                            branchDead.AddRange(deadSet);
                            var fused = branchAdd;
                            fused.Op = OpType.RotaryEmbedding;
                            fused.OpTypeName = OpType.RotaryEmbedding.ToString();
                            fused.IsFused = true;
                            fused.Inputs = new[] { x, c, sIn };
                            fused.Attributes = new Dictionary<string, object> { { "half", half }, { "axis", axis }, { "concatAxis", caxis.Value } };
                            graph.Nodes[addIndex] = fused;
                            return true;
                        }
                    }
                }
                return false;
            }

            int m1 = ProducerOf(add.Inputs[0], OpType.Mul);
            int m2 = ProducerOf(add.Inputs[1], OpType.Mul);
            if (m1 < 0 || m2 < 0 || m1 == m2) return false;
            if (graph.Nodes[m1].Inputs.Length != 2 || graph.Nodes[m2].Inputs.Length != 2) return false;
            if (graph.Nodes[m1].Outputs.Length != 1 || graph.Nodes[m2].Outputs.Length != 1) return false;

            if (TryRopeBranch(m1, m2, out dead)) return true;
            if (TryRopeBranch(m2, m1, out dead)) return true;
            return false;
        }

        static ITensor? ConstantValue(Node node)
        {
            if (node.Attributes is not null && node.Attributes.TryGetValue("value", out var v) && v is ITensor t) return t;
            return null;
        }
    }
}
