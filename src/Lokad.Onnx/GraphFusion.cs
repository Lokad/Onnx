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

        static bool IsFusableParticipant(Node node)
        {
            if (node.IsFused) return false;
            if (!CPUExecutionProvider.SupportsNode(node)) return false;
            return true;
        }

        static int StandardOpset(ComputationalGraph graph)
        {
            if (graph.Opset.TryGetValue("", out var v)) return v;
            if (graph.Opset.TryGetValue("ai.onnx", out var a)) return a;
            return -1;
        }

        static bool IsStandardAttributeReduceMean(Node node)
        {
            return IsFusableParticipant(node) && node.OpsetVersion < 18;
        }

        static bool IsStandardSlice(Node node)
        {
            return IsFusableParticipant(node) && node.OpsetVersion >= 10;
        }

        static bool IsProvenFloat(ComputationalGraph graph, Dictionary<string, int> producer, string name)
        {
            return IsProvenFloatInner(graph, producer, name, new HashSet<string>(StringComparer.Ordinal), new Dictionary<string, bool>(StringComparer.Ordinal));
        }

        static bool IsProvenFloatInner(ComputationalGraph graph, Dictionary<string, int> producer, string name, HashSet<string> visiting, Dictionary<string, bool> memo)
        {
            if (string.IsNullOrEmpty(name)) return false;
            if (graph.Inputs.TryGetValue(name, out var gi) && gi is not null) return gi.ElementType == TensorElementType.Float;
            foreach (var desc in graph.InputDescs) if (desc.Name == name) return desc.ElementType == TensorElementType.Float;
            if (graph.Initializers.TryGetValue(name, out var ti)) return ti.ElementType == TensorElementType.Float;
            if (memo.TryGetValue(name, out var cached)) return cached;
            if (!producer.TryGetValue(name, out var pi)) return false;
            if (pi < 0 || pi >= graph.Nodes.Count) return false;
            if (!visiting.Add(name)) return false;
            bool result = ProveFloat(graph, producer, graph.Nodes[pi], visiting, memo);
            visiting.Remove(name);
            memo[name] = result;
            return result;
        }

        static bool ProveFloat(ComputationalGraph graph, Dictionary<string, int> producer, Node pn, HashSet<string> visiting, Dictionary<string, bool> memo)
        {
            if (pn.IsFused)
            {
                if (!Node.IsStandardDomain(pn.Domain)) return false;
                // Epilogue-fused nodes keep their input dtype (the epilogue is
                // an elementwise tail), so they prove exactly like their
                // unfused data inputs. Every attribute-epilogue fusion must
                // extend this list or downstream float proofs silently fail.
                if (pn.Op == OpType.Mul)
                {
                    if (pn.Inputs.Length != 2) return false;
                    return IsProvenFloatInner(graph, producer, pn.Inputs[0], visiting, memo) && IsProvenFloatInner(graph, producer, pn.Inputs[1], visiting, memo);
                }
                if (pn.Op != OpType.LayerNormalization && pn.Op != OpType.RotaryEmbedding && pn.Op != OpType.Gelu && pn.Op != OpType.Conv) return false;
                if (pn.Inputs.Length < 1) return false;
                return IsProvenFloatInner(graph, producer, pn.Inputs[0], visiting, memo);
            }
            if (!IsFusableParticipant(pn)) return false;
            if (pn.Op == OpType.Constant)
            {
                var ct = ConstantValue(pn);
                return ct is not null && ct.ElementType == TensorElementType.Float;
            }
            if (pn.Op == OpType.Cast)
            {
                int? to = pn.Int("to", null);
                return to.HasValue && to.Value == (int)TensorElementType.Float;
            }
            switch (pn.Op)
            {
                case OpType.Add:
                case OpType.Sub:
                case OpType.Mul:
                case OpType.Div:
                case OpType.Pow:
                    if (pn.Inputs.Length != 2) return false;
                    return IsProvenFloatInner(graph, producer, pn.Inputs[0], visiting, memo) && IsProvenFloatInner(graph, producer, pn.Inputs[1], visiting, memo);
                case OpType.Concat:
                    if (pn.Inputs.Length < 1) return false;
                    foreach (var inp in pn.Inputs)
                    {
                        if (string.IsNullOrEmpty(inp)) continue;
                        if (!IsProvenFloatInner(graph, producer, inp, visiting, memo)) return false;
                    }
                    return true;
                case OpType.MatMul:
                    if (pn.Inputs.Length != 2) return false;
                    return IsProvenFloatInner(graph, producer, pn.Inputs[0], visiting, memo) && IsProvenFloatInner(graph, producer, pn.Inputs[1], visiting, memo);
                case OpType.Conv:
                    if (pn.Inputs.Length < 2) return false;
                    if (!IsProvenFloatInner(graph, producer, pn.Inputs[0], visiting, memo) || !IsProvenFloatInner(graph, producer, pn.Inputs[1], visiting, memo)) return false;
                    if (pn.Inputs.Length >= 3 && !string.IsNullOrEmpty(pn.Inputs[2])) return IsProvenFloatInner(graph, producer, pn.Inputs[2], visiting, memo);
                    return true;
                case OpType.Where:
                    if (pn.Inputs.Length != 3) return false;
                    return IsProvenFloatInner(graph, producer, pn.Inputs[1], visiting, memo) && IsProvenFloatInner(graph, producer, pn.Inputs[2], visiting, memo);
                case OpType.Range:
                    if (pn.Inputs.Length != 3) return false;
                    return IsProvenFloatInner(graph, producer, pn.Inputs[0], visiting, memo) && IsProvenFloatInner(graph, producer, pn.Inputs[1], visiting, memo) && IsProvenFloatInner(graph, producer, pn.Inputs[2], visiting, memo);
                case OpType.Gemm:
                    if (pn.Inputs.Length < 2) return false;
                    if (!IsProvenFloatInner(graph, producer, pn.Inputs[0], visiting, memo) || !IsProvenFloatInner(graph, producer, pn.Inputs[1], visiting, memo)) return false;
                    if (pn.Inputs.Length >= 3 && !string.IsNullOrEmpty(pn.Inputs[2])) return IsProvenFloatInner(graph, producer, pn.Inputs[2], visiting, memo);
                    return true;
                case OpType.ConstantOfShape:
                {
                    // Output dtype comes from the value attribute alone; the shape input is int64 by spec and says nothing.
                    var cv = ConstantValue(pn);
                    return cv is not null && cv.ElementType == TensorElementType.Float;
                }
                case OpType.Gather:
                case OpType.Split:
                case OpType.SplitToSequence:
                case OpType.SequenceAt:
                case OpType.Tile:
                case OpType.GlobalAveragePool:
                case OpType.MaxPool:
                case OpType.Sqrt:
                case OpType.ReduceMean:
                case OpType.ReduceSum:
                case OpType.ReduceMax:
                case OpType.Transpose:
                case OpType.Squeeze:
                case OpType.Unsqueeze:
                case OpType.Reshape:
                case OpType.Slice:
                case OpType.Resize:
                case OpType.Expand:
                case OpType.Relu:
                case OpType.Gelu:
                case OpType.Tanh:
                case OpType.Erf:
                case OpType.Softmax:
                case OpType.Neg:
                case OpType.Abs:
                case OpType.Cos:
                case OpType.Sin:
                case OpType.Sigmoid:
                case OpType.Floor:
                case OpType.Pad:
                case OpType.LayerNormalization:
                case OpType.RotaryEmbedding:
                    if (pn.Inputs.Length < 1 || string.IsNullOrEmpty(pn.Inputs[0])) return false;
                    return IsProvenFloatInner(graph, producer, pn.Inputs[0], visiting, memo);
                default:
                    return false;
            }
        }

        static int FloatRankOneLength(ComputationalGraph graph, string name)
        {
            if (!graph.Initializers.TryGetValue(name, out var t)) return -1;
            if (t.ElementType != TensorElementType.Float) return -1;
            if (t.Rank != 1) return -1;
            if (t.Length < 1 || t.Length > int.MaxValue) return -1;
            return (int)t.Length;
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
                if (!IsFusableParticipant(graph.Nodes[i])) continue;
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
            if (!IsFusableParticipant(div)) return false;
            if (div.Inputs.Length != 2) return false;
            if (div.Outputs.Length != 1) return false;
            int OnlyConsumer(string output, OpType op)
            {
                if (outputs.Contains(output)) return -1;
                if (!consumers.TryGetValue(output, out var uses)) return -1;
                var live = new List<int>();
                foreach (var u in uses) if (!drop.Contains(u)) live.Add(u);
                if (live.Count != 1) return -1;
                var cand = graph.Nodes[live[0]];
                if (cand.Op != op || !IsFusableParticipant(cand)) return -1;
                return live[0];
            }
            int ProducerOf(string input, OpType op)
            {
                if (string.IsNullOrEmpty(input) || !producer.TryGetValue(input, out var pi)) return -1;
                if (drop.Contains(pi)) return -1;
                var cand = graph.Nodes[pi];
                if (cand.Op != op || !IsFusableParticipant(cand)) return -1;
                return pi;
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
            bool IsKnownFloat(string name)
            {
                return IsProvenFloat(graph, producer, name);
            }
            int LastDimOf(string name)
            {
                if (graph.Inputs.TryGetValue(name, out var gi) && gi is not null && gi.Dims.Length > 0) return gi.Dims[gi.Dims.Length - 1];
                foreach (var desc in graph.InputDescs) if (desc.Name == name && desc.Dims.Length > 0) return desc.Dims[desc.Dims.Length - 1];
                if (graph.Initializers.TryGetValue(name, out var ti) && ti.Dims.Length > 0) return ti.Dims[ti.Dims.Length - 1];
                return -1;
            }
            bool OpsetSupportsAttributeReduceMean()
            {
                if (graph.Opset.TryGetValue("", out var v)) return v < 18;
                if (graph.Opset.TryGetValue("ai.onnx", out var a)) return a < 18;
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
            if (!IsStandardAttributeReduceMean(rm1Node)) return false;
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
            if (!IsStandardAttributeReduceMean(rm2Node)) return false;
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
            int gammaLen = FloatRankOneLength(graph, gamma);
            if (gammaLen <= 0) return false;
            if (lastDim > 0 && gammaLen != lastDim) return false;
            if (lastDim == 0) return false;
            int ab = OnlyConsumer(muNode.Outputs[0], OpType.Add);
            if (ab < 0) return false;
            var abNode = graph.Nodes[ab];
            if (abNode.Inputs.Length != 2) return false;
            if (abNode.Outputs.Length != 1) return false;
            string beta = abNode.Inputs[0] == muNode.Outputs[0] ? abNode.Inputs[1] : abNode.Inputs[0];
            int betaLen = FloatRankOneLength(graph, beta);
            if (betaLen <= 0) return false;
            if (lastDim > 0 && betaLen != lastDim) return false;
            if (lastDim == 0) return false;
            if (lastDim < 0 && gammaLen != betaLen) return false;
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
            if (!IsFusableParticipant(abNode)) return false;
            dead.AddRange(deadSet);
            epsConst = epsC;
            var ln = abNode;
            ln.Op = OpType.LayerNormalization;
            ln.OpTypeName = OpType.LayerNormalization.ToString();
            ln.Domain = "";
            int stdv = StandardOpset(graph);
            if (stdv >= 0) ln.OpsetVersion = stdv;
            ln.IsFused = true;
            ln.Inputs = new[] { xPos, gamma, beta };
            ln.Attributes = new Dictionary<string, object> { { "axis", -1 }, { "epsilon", eps.Value } };
            graph.Nodes[ab] = ln;
            return true;
        }

        /// <summary>
        /// Fuses exact-GELU erf chains (Div by sqrt(2), Erf, plus one, times
        /// x, times one half) into native exact Gelu nodes. ORT fuses the same
        /// pattern, so the unfused float chain accumulates a few ulps per
        /// application against the fused reference; the native kernel matches
        /// ORT bit-identically.
        /// </summary>
        public static int FuseGeluPatterns(ComputationalGraph graph)
        {
            var idx = BuildIndex(graph);
            var producer = idx.Producer;
            var consumers = idx.Consumers;
            var outputs = idx.Outputs;

            var drop = new HashSet<int>();
            int fused = 0;

            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (graph.Nodes[i].Op != OpType.Erf) continue;
                if (!IsFusableParticipant(graph.Nodes[i])) continue;
                if (drop.Contains(i)) continue;
                if (TryMatchGelu(graph, producer, consumers, outputs, drop, i, out var dead, out var constNodes, out var survivor))
                {
                    foreach (var d in dead) drop.Add(d);
                    foreach (var cn in constNodes)
                    {
                        // The surviving fused node still names its half
                        // constant, which no longer needs a producer.
                        bool ConstOnlyFeedsDead(string output) =>
                            !outputs.Contains(output) && (!consumers.TryGetValue(output, out var uses) || uses.All(u => u == survivor || drop.Contains(u)));
                        var cnode = graph.Nodes[cn];
                        if (cnode.Outputs.All(ConstOnlyFeedsDead)) drop.Add(cn);
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

        static bool TryMatchGelu(
            ComputationalGraph graph,
            Dictionary<string, int> producer,
            Dictionary<string, List<int>> consumers,
            HashSet<string> outputs,
            HashSet<int> drop,
            int erfIndex,
            out List<int> dead,
            out List<int> constNodes,
            out int survivor)
        {
            dead = new List<int>();
            constNodes = new List<int>();
            survivor = -1;
            var erf = graph.Nodes[erfIndex];
            if (!IsFusableParticipant(erf)) return false;
            if (erf.Inputs.Length != 1) return false;
            if (erf.Outputs.Length != 1) return false;
            int OnlyConsumer(string output, OpType op)
            {
                if (outputs.Contains(output)) return -1;
                if (!consumers.TryGetValue(output, out var uses)) return -1;
                var live = new List<int>();
                foreach (var u in uses) if (!drop.Contains(u)) live.Add(u);
                if (live.Count != 1) return -1;
                var cand = graph.Nodes[live[0]];
                if (cand.Op != op || !IsFusableParticipant(cand)) return -1;
                return live[0];
            }
            int ProducerOf(string input, OpType op)
            {
                if (string.IsNullOrEmpty(input) || !producer.TryGetValue(input, out var pi)) return -1;
                if (drop.Contains(pi)) return -1;
                var cand = graph.Nodes[pi];
                if (cand.Op != op || !IsFusableParticipant(cand)) return -1;
                return pi;
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
            int div = ProducerOf(erf.Inputs[0], OpType.Div);
            if (div < 0) return false;
            var divNode = graph.Nodes[div];
            if (divNode.Inputs.Length != 2) return false;
            if (divNode.Outputs.Length != 1) return false;
            // Division is not commutative: the activated value divides first.
            string xPos = divNode.Inputs[0];
            if (!IsProvenFloat(graph, producer, xPos)) return false;
            float? c0 = ScalarFloatOf(divNode.Inputs[1], out int c0n);
            if (!c0.HasValue || c0.Value != 1.4142135f) return false;
            int add = OnlyConsumer(erf.Outputs[0], OpType.Add);
            if (add < 0) return false;
            var addNode = graph.Nodes[add];
            if (addNode.Inputs.Length != 2) return false;
            if (addNode.Outputs.Length != 1) return false;
            string addOther = addNode.Inputs[0] == erf.Outputs[0] ? addNode.Inputs[1] : addNode.Inputs[0];
            if (addOther == erf.Outputs[0]) return false;
            float? c1 = ScalarFloatOf(addOther, out int c1n);
            if (!c1.HasValue || c1.Value != 1f) return false;
            int mul = OnlyConsumer(addNode.Outputs[0], OpType.Mul);
            if (mul < 0) return false;
            var mulNode = graph.Nodes[mul];
            if (mulNode.Inputs.Length != 2) return false;
            if (mulNode.Outputs.Length != 1) return false;
            if (!mulNode.Inputs.Contains(addNode.Outputs[0])) return false;
            string mulOther = mulNode.Inputs[0] == addNode.Outputs[0] ? mulNode.Inputs[1] : mulNode.Inputs[0];
            if (mulOther != xPos) return false;
            int mul1 = OnlyConsumer(mulNode.Outputs[0], OpType.Mul);
            if (mul1 < 0) return false;
            var mul1Node = graph.Nodes[mul1];
            if (mul1Node.Inputs.Length != 2) return false;
            if (mul1Node.Outputs.Length != 1) return false;
            if (!mul1Node.Inputs.Contains(mulNode.Outputs[0])) return false;
            string mul1Other = mul1Node.Inputs[0] == mulNode.Outputs[0] ? mul1Node.Inputs[1] : mul1Node.Inputs[0];
            if (mul1Other == mulNode.Outputs[0]) return false;
            float? c2 = ScalarFloatOf(mul1Other, out int c2n);
            if (!c2.HasValue || c2.Value != 0.5f) return false;
            if (drop.Contains(mul1)) return false;
            var deadSet = new HashSet<int> { div, erfIndex, add, mul };
            foreach (var dd in deadSet) if (drop.Contains(dd)) return false;
            foreach (var d in deadSet)
            {
                foreach (var o in graph.Nodes[d].Outputs)
                {
                    if (string.IsNullOrEmpty(o)) continue;
                    if (o == mul1Node.Outputs[0]) continue;
                    if (outputs.Contains(o)) return false;
                    if (consumers.TryGetValue(o, out var uses) && uses.Any(u => !deadSet.Contains(u) && u != mul1 && !drop.Contains(u))) return false;
                }
            }
            if (!IsFusableParticipant(mul1Node)) return false;
            dead.AddRange(deadSet);
            survivor = mul1;
            if (c0n >= 0) constNodes.Add(c0n);
            if (c1n >= 0) constNodes.Add(c1n);
            if (c2n >= 0) constNodes.Add(c2n);
            var gelu = mul1Node;
            gelu.Op = OpType.Gelu;
            gelu.OpTypeName = OpType.Gelu.ToString();
            gelu.Domain = "";
            int stdv = StandardOpset(graph);
            if (stdv >= 0) gelu.OpsetVersion = stdv;
            gelu.IsFused = true;
            gelu.Inputs = new[] { xPos };
            gelu.Attributes = new Dictionary<string, object>();
            graph.Nodes[mul1] = gelu;
            return true;
        }

        /// <summary>
        static bool TryMatchGeluTanh(
            ComputationalGraph graph,
            Dictionary<string, int> producer,
            Dictionary<string, List<int>> consumers,
            HashSet<string> outputs,
            HashSet<int> drop,
            int tanhIndex,
            out List<int> dead,
            out List<int> constNodes,
            out int survivor)
        {
            dead = new List<int>();
            constNodes = new List<int>();
            survivor = -1;
            var tanh = graph.Nodes[tanhIndex];
            if (!IsFusableParticipant(tanh)) return false;
            if (tanh.Inputs.Length != 1) return false;
            if (tanh.Outputs.Length != 1) return false;
            int OnlyConsumer(string output, OpType op)
            {
                if (outputs.Contains(output)) return -1;
                if (!consumers.TryGetValue(output, out var uses)) return -1;
                var live = new List<int>();
                foreach (var u in uses) if (!drop.Contains(u)) live.Add(u);
                if (live.Count != 1) return -1;
                var cand = graph.Nodes[live[0]];
                if (cand.Op != op || !IsFusableParticipant(cand)) return -1;
                return live[0];
            }
            int ProducerOf(string input, OpType op)
            {
                if (string.IsNullOrEmpty(input) || !producer.TryGetValue(input, out var pi)) return -1;
                if (drop.Contains(pi)) return -1;
                var cand = graph.Nodes[pi];
                if (cand.Op != op || !IsFusableParticipant(cand)) return -1;
                return pi;
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
            int mul2 = ProducerOf(tanh.Inputs[0], OpType.Mul);
            if (mul2 < 0) return false;
            var mul2Node = graph.Nodes[mul2];
            if (mul2Node.Inputs.Length != 2) return false;
            if (mul2Node.Outputs.Length != 1) return false;
            if (mul2Node.Outputs[0] != tanh.Inputs[0]) return false;
            string mul2A = mul2Node.Inputs[0], mul2B = mul2Node.Inputs[1];
            float? cS = ScalarFloatOf(mul2A, out int cSn);
            string addOut = mul2B;
            if (!cS.HasValue || cS.Value != 0.7978846f)
            {
                cS = ScalarFloatOf(mul2B, out cSn);
                addOut = mul2A;
                if (!cS.HasValue || cS.Value != 0.7978846f) return false;
            }
            int add = ProducerOf(addOut, OpType.Add);
            if (add < 0) return false;
            var addNode = graph.Nodes[add];
            if (addNode.Inputs.Length != 2) return false;
            if (addNode.Outputs.Length != 1) return false;
            if (addNode.Outputs[0] != addOut) return false;
            string addX = addNode.Inputs[0] == addOut ? addNode.Inputs[1] : addNode.Inputs[0];
            if (addX == addOut) return false;
            string mul1Out = addNode.Inputs[0] == addX ? addNode.Inputs[1] : addNode.Inputs[0];
            if (!IsProvenFloat(graph, producer, addX)) return false;
            if (mul1Out == addX) return false;
            int mul1 = ProducerOf(mul1Out, OpType.Mul);
            if (mul1 < 0) return false;
            var mul1Node = graph.Nodes[mul1];
            if (mul1Node.Inputs.Length != 2) return false;
            if (mul1Node.Outputs.Length != 1) return false;
            if (mul1Node.Outputs[0] != mul1Out) return false;
            string mul1A = mul1Node.Inputs[0], mul1B = mul1Node.Inputs[1];
            float? cQ = ScalarFloatOf(mul1A, out int cQn);
            string powOut = mul1B;
            if (!cQ.HasValue || cQ.Value != 0.044715f)
            {
                cQ = ScalarFloatOf(mul1B, out cQn);
                powOut = mul1A;
                if (!cQ.HasValue || cQ.Value != 0.044715f) return false;
            }
            int pow = ProducerOf(powOut, OpType.Pow);
            if (pow < 0) return false;
            var powNode = graph.Nodes[pow];
            if (powNode.Inputs.Length != 2) return false;
            if (powNode.Outputs.Length != 1) return false;
            if (powNode.Outputs[0] != powOut) return false;
            if (powNode.Inputs[0] != addX) return false;
            float? cE = ScalarFloatOf(powNode.Inputs[1], out int cEn);
            if (!cE.HasValue || cE.Value != 3f) return false;
            int add1 = OnlyConsumer(tanh.Outputs[0], OpType.Add);
            if (add1 < 0) return false;
            var add1Node = graph.Nodes[add1];
            if (add1Node.Inputs.Length != 2) return false;
            if (add1Node.Outputs.Length != 1) return false;
            if (!add1Node.Inputs.Contains(tanh.Outputs[0])) return false;
            string add1Other = add1Node.Inputs[0] == tanh.Outputs[0] ? add1Node.Inputs[1] : add1Node.Inputs[0];
            if (add1Other == tanh.Outputs[0]) return false;
            float? c1 = ScalarFloatOf(add1Other, out int c1n);
            if (!c1.HasValue || c1.Value != 1f) return false;
            int mul3 = OnlyConsumer(add1Node.Outputs[0], OpType.Mul);
            if (mul3 < 0) return false;
            var mul3Node = graph.Nodes[mul3];
            if (mul3Node.Inputs.Length != 2) return false;
            if (mul3Node.Outputs.Length != 1) return false;
            if (!mul3Node.Inputs.Contains(add1Node.Outputs[0])) return false;
            string mul0Out = mul3Node.Inputs[0] == add1Node.Outputs[0] ? mul3Node.Inputs[1] : mul3Node.Inputs[0];
            if (mul0Out == add1Node.Outputs[0]) return false;
            int mul0 = ProducerOf(mul0Out, OpType.Mul);
            if (mul0 < 0) return false;
            var mul0Node = graph.Nodes[mul0];
            if (mul0Node.Inputs.Length != 2) return false;
            if (mul0Node.Outputs.Length != 1) return false;
            if (mul0Node.Outputs[0] != mul0Out) return false;
            int c0nA = -1;
            bool mul0Ok = (mul0Node.Inputs[0] == addX && ScalarFloatOf(mul0Node.Inputs[1], out c0nA) is float hA && hA == 0.5f) || (mul0Node.Inputs[1] == addX && ScalarFloatOf(mul0Node.Inputs[0], out c0nA) is float hB && hB == 0.5f);
            if (!mul0Ok) return false;
            var deadSet = new HashSet<int> { mul0, pow, mul1, add, mul2, tanhIndex, add1 };
            foreach (var d in deadSet) if (drop.Contains(d)) return false;
            foreach (var d in deadSet)
            {
                foreach (var o in graph.Nodes[d].Outputs)
                {
                    if (string.IsNullOrEmpty(o)) continue;
                    if (o == mul3Node.Outputs[0]) continue;
                    if (outputs.Contains(o)) return false;
                    if (consumers.TryGetValue(o, out var uses) && uses.Any(u => !deadSet.Contains(u) && u != mul3 && !drop.Contains(u))) return false;
                }
            }
            if (!IsFusableParticipant(mul3Node)) return false;
            dead.AddRange(deadSet);
            survivor = mul3;
            if (cSn >= 0) constNodes.Add(cSn);
            if (cQn >= 0) constNodes.Add(cQn);
            if (cEn >= 0) constNodes.Add(cEn);
            if (c1n >= 0) constNodes.Add(c1n);
            if (c0nA >= 0) constNodes.Add(c0nA);
            var gelu = mul3Node;
            gelu.Op = OpType.Gelu;
            gelu.OpTypeName = OpType.Gelu.ToString();
            gelu.Domain = "";
            int stdv = StandardOpset(graph);
            if (stdv >= 0) gelu.OpsetVersion = stdv;
            gelu.IsFused = true;
            gelu.Inputs = new[] { addX };
            gelu.Attributes = new Dictionary<string, object> { ["approximate"] = "tanh" };
            graph.Nodes[mul3] = gelu;
            return true;
        }

        /// Fuses tanh-approximate GELU chains (x times one half; cube; times 0.044715; plus x; times sqrt(2/pi); Tanh; plus one; times) into native Gelu nodes carrying approximate=tanh.
        /// </summary>
        public static int FuseGeluTanhPatterns(ComputationalGraph graph)
        {
            var idx = BuildIndex(graph);
            var producer = idx.Producer;
            var consumers = idx.Consumers;
            var outputs = idx.Outputs;

            var drop = new HashSet<int>();
            int fused = 0;

            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (graph.Nodes[i].Op != OpType.Tanh) continue;
                if (!IsFusableParticipant(graph.Nodes[i])) continue;
                if (drop.Contains(i)) continue;
                if (TryMatchGeluTanh(graph, producer, consumers, outputs, drop, i, out var dead, out var constNodes, out var survivor))
                {
                    foreach (var d in dead) drop.Add(d);
                    foreach (var cn in constNodes)
                    {
                        bool ConstOnlyFeedsDead(string output) =>
                            !outputs.Contains(output) && (!consumers.TryGetValue(output, out var uses) || uses.All(u => u == survivor || drop.Contains(u)));
                        var cnode = graph.Nodes[cn];
                        if (cnode.Outputs.All(ConstOnlyFeedsDead)) drop.Add(cn);
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

        /// <summary>
        /// Fuses Conv followed by a single-consumer ReLU into the convolution
        /// node with a fused epilogue (fuse_relu attribute): the bias/activation
        /// tail runs in the same output pass, so the pair costs no extra tensor
        /// pass and matches the unfused result bit for bit (same add order and
        /// max, including signed zero and NaN). Only rank-4 float weights take
        /// part: the rank-3 adapter path has no epilogue, and anything else
        /// keeps the unfused nodes. Graph outputs on either side veto fusion.
        /// </summary>
        public static int FuseConvReluPatterns(ComputationalGraph graph)
        {
            var idx = BuildIndex(graph);
            var producer = idx.Producer;
            var consumers = idx.Consumers;
            var outputs = idx.Outputs;

            var drop = new HashSet<int>();
            int fused = 0;

            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (graph.Nodes[i].Op != OpType.Relu) continue;
                if (!IsFusableParticipant(graph.Nodes[i])) continue;
                if (drop.Contains(i)) continue;
                if (TryMatchConvRelu(graph, producer, consumers, outputs, drop, i, out int convIndex))
                {
                    var conv = graph.Nodes[convIndex];
                    conv.Attributes ??= new Dictionary<string, object>();
                    conv.Attributes["fuse_relu"] = 1;
                    conv.IsFused = true;
                    graph.Nodes[convIndex] = conv;
                    string convOut = conv.Outputs[0];
                    string reluOut = graph.Nodes[i].Outputs[0];
                    if (consumers.TryGetValue(reluOut, out var uses))
                    {
                        foreach (var u in uses)
                        {
                            if (drop.Contains(u)) continue;
                            var n = graph.Nodes[u];
                            n.Inputs = n.Inputs.Select(x => x == reluOut ? convOut : x).ToArray();
                            graph.Nodes[u] = n;
                        }
                    }
                    drop.Add(i);
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

        static bool TryMatchConvRelu(
            ComputationalGraph graph,
            Dictionary<string, int> producer,
            Dictionary<string, List<int>> consumers,
            HashSet<string> outputs,
            HashSet<int> drop,
            int reluIndex,
            out int convIndex)
        {
            convIndex = -1;
            var relu = graph.Nodes[reluIndex];
            if (!IsFusableParticipant(relu)) return false;
            if (relu.Inputs.Length != 1 || relu.Outputs.Length != 1) return false;
            string convOut = relu.Inputs[0];
            string reluOut = relu.Outputs[0];
            if (string.IsNullOrEmpty(convOut) || string.IsNullOrEmpty(reluOut)) return false;
            if (outputs.Contains(reluOut) || outputs.Contains(convOut)) return false;
            if (!producer.TryGetValue(convOut, out int ci) || drop.Contains(ci)) return false;
            var conv = graph.Nodes[ci];
            if (conv.Op != OpType.Conv || !IsFusableParticipant(conv)) return false;
            if (conv.Outputs.Length != 1 || conv.Outputs[0] != convOut) return false;
            if (!consumers.TryGetValue(convOut, out var uses)) return false;
            int live = 0;
            foreach (var u in uses) if (!drop.Contains(u)) live++;
            if (live != 1) return false;
            if (conv.Inputs.Length < 2 || string.IsNullOrEmpty(conv.Inputs[1])) return false;
            if (!graph.Initializers.TryGetValue(conv.Inputs[1], out var w)) return false;
            if (w.Rank != 4) return false;
            convIndex = ci;
            return true;
        }

        /// <summary>
        /// Fuses Sigmoid feeding a single-consumer Mul into the multiply node
        /// with a fused sigmoid epilogue (fuse_sigmoid attribute naming the
        /// activated input): the pair costs one tensor pass instead of two and
        /// matches the unfused chain within float rounding (never
        /// bit-identical). Covers both Swish x*sigmoid(x) and gated
        /// a*sigmoid(b) halves; only one sigmoid leg fuses per Mul, float
        /// proven on both sides, and a Sigmoid held as a graph output vetoes.
        /// </summary>
        public static int FuseSigmoidMulPatterns(ComputationalGraph graph)
        {
            var idx = BuildIndex(graph);
            var producer = idx.Producer;
            var consumers = idx.Consumers;
            var outputs = idx.Outputs;

            var drop = new HashSet<int>();
            int fused = 0;

            for (int j = 0; j < graph.Nodes.Count; j++)
            {
                if (graph.Nodes[j].Op != OpType.Mul) continue;
                if (!IsFusableParticipant(graph.Nodes[j])) continue;
                if (drop.Contains(j)) continue;
                if (TryMatchSigmoidMul(graph, producer, consumers, outputs, drop, j, out int leg))
                {
                    var mul = graph.Nodes[j];
                    mul.Attributes ??= new Dictionary<string, object>();
                    mul.Attributes["fuse_sigmoid"] = leg;
                    mul.IsFused = true;
                    graph.Nodes[j] = mul;
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

        static bool TryMatchSigmoidMul(
            ComputationalGraph graph,
            Dictionary<string, int> producer,
            Dictionary<string, List<int>> consumers,
            HashSet<string> outputs,
            HashSet<int> drop,
            int mulIndex,
            out int leg)
        {
            leg = -1;
            var mul = graph.Nodes[mulIndex];
            if (!IsFusableParticipant(mul)) return false;
            if (mul.Inputs.Length != 2 || mul.Outputs.Length != 1) return false;
            for (int k = 0; k < 2; k++)
            {
                string sname = mul.Inputs[k];
                string other = mul.Inputs[1 - k];
                if (string.IsNullOrEmpty(sname) || string.IsNullOrEmpty(other)) continue;
                if (outputs.Contains(sname)) continue;
                if (!producer.TryGetValue(sname, out int si) || drop.Contains(si)) continue;
                var sig = graph.Nodes[si];
                if (sig.Op != OpType.Sigmoid || !IsFusableParticipant(sig)) continue;
                if (sig.Inputs.Length != 1 || sig.Outputs.Length != 1 || sig.Outputs[0] != sname) continue;
                string sin = sig.Inputs[0];
                if (string.IsNullOrEmpty(sin)) continue;
                if (!consumers.TryGetValue(sname, out var uses)) continue;
                int live = 0;
                foreach (var u in uses) if (!drop.Contains(u)) live++;
                if (live != 1) continue;
                if (!IsProvenFloat(graph, producer, sin)) continue;
                if (!IsProvenFloat(graph, producer, other)) continue;
                mul.Inputs[k] = sin;
                graph.Nodes[mulIndex] = mul;
                drop.Add(si);
                leg = k;
                return true;
            }
            return false;
        }

        /// <summary>Fuses MatMul followed by a scalar-constant Mul into the MatMul node with a fuse_scale epilogue (M3).</summary>
        /// <remarks>Only float MatMul pairs with a single live consumer qualify; the scale must come from a single-use standard Constant node holding one float (import-time form). The surviving MatMul keeps its op, inputs, and float proofs; dispatch routes it to a scaled provider that runs the identical product plus one scale pass over the owned destination. That pass is bit-identical to the removed Mul: same product, same single rounding per element (multiplication commutes), with no bias term at all. No tolerance is involved. Removes the Mul dispatch and its M-by-N intermediate per site.</remarks>
        public static int FuseMatMulScalePatterns(ComputationalGraph graph)
        {
            var idx = BuildIndex(graph);
            var producer = idx.Producer;
            var consumers = idx.Consumers;
            var outputs = idx.Outputs;

            var drop = new HashSet<int>();
            int fused = 0;

            for (int j = 0; j < graph.Nodes.Count; j++)
            {
                if (graph.Nodes[j].Op != OpType.Mul) continue;
                if (!IsFusableParticipant(graph.Nodes[j])) continue;
                if (drop.Contains(j)) continue;
                if (TryMatchMatMulScale(graph, producer, consumers, outputs, drop, j, out float scale, out int mmIndex, out int scaleIndex))
                {
                    var mm = graph.Nodes[mmIndex];
                    var mul = graph.Nodes[j];
                    mm.Attributes ??= new Dictionary<string, object>();
                    mm.Attributes["fuse_scale"] = scale;
                    mm.Outputs = new[] { mul.Outputs[0] };
                    graph.Nodes[mmIndex] = mm;
                    drop.Add(j);
                    if (scaleIndex >= 0) drop.Add(scaleIndex);
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

        static bool TryMatchMatMulScale(
            ComputationalGraph graph,
            Dictionary<string, int> producer,
            Dictionary<string, List<int>> consumers,
            HashSet<string> outputs,
            HashSet<int> drop,
            int mulIndex,
            out float scale,
            out int mmIndex,
            out int scaleIndex)
        {
            scale = 1f;
            mmIndex = -1;
            scaleIndex = -1;
            var mul = graph.Nodes[mulIndex];
            if (!IsFusableParticipant(mul)) return false;
            if (mul.Inputs.Length != 2 || mul.Outputs.Length != 1) return false;
            if (string.IsNullOrEmpty(mul.Outputs[0])) return false;
            for (int k = 0; k < 2; k++)
            {
                string mmOut = mul.Inputs[k];
                string sname = mul.Inputs[1 - k];
                if (string.IsNullOrEmpty(mmOut) || string.IsNullOrEmpty(sname)) continue;
                if (outputs.Contains(mmOut)) continue;
                if (!producer.TryGetValue(mmOut, out int mi) || drop.Contains(mi)) continue;
                var mm = graph.Nodes[mi];
                if (mm.Op != OpType.MatMul || !IsFusableParticipant(mm)) continue;
                if (mm.Inputs.Length != 2 || mm.Outputs.Length != 1 || mm.Outputs[0] != mmOut) continue;
                if (!consumers.TryGetValue(mmOut, out var uses)) continue;
                int live = 0;
                foreach (var u in uses) if (!drop.Contains(u)) live++;
                if (live != 1) continue;
                if (!IsProvenFloat(graph, producer, mm.Inputs[0])) continue;
                if (!IsProvenFloat(graph, producer, mm.Inputs[1])) continue;
                if (!TryMatMulScaleValue(graph, producer, consumers, outputs, drop, sname, out scale, out scaleIndex)) continue;
                mmIndex = mi;
                return true;
            }
            return false;
        }

        static bool TryMatMulScaleValue(
            ComputationalGraph graph,
            Dictionary<string, int> producer,
            Dictionary<string, List<int>> consumers,
            HashSet<string> outputs,
            HashSet<int> drop,
            string sname,
            out float scale,
            out int scaleIndex)
        {
            scale = 1f;
            scaleIndex = -1;
            if (string.IsNullOrEmpty(sname) || outputs.Contains(sname)) return false;
            if (!producer.TryGetValue(sname, out int si) || drop.Contains(si)) return false;
            var sn = graph.Nodes[si];
            if (sn.Op != OpType.Constant || !IsFusableParticipant(sn)) return false;
            if (sn.Inputs.Length != 0 || sn.Outputs.Length != 1 || sn.Outputs[0] != sname) return false;
            if (!consumers.TryGetValue(sname, out var uses)) return false;
            int live = 0;
            foreach (var u in uses) if (!drop.Contains(u)) live++;
            if (live != 1) return false;
            var cv = ConstantValue(sn);
            if (cv is not Tensor<float> ctf || ctf.Length != 1) return false;
            scale = ctf.ToArray()[0];
            scaleIndex = si;
            return true;
        }


        /// <summary>Fuses maximal blocked-execution regions: Conv/Relu/Add chains with single-use internal links become one region head carrying the step plan (M5).</summary>
        /// <remarks>Runs after the Conv+Relu fusion, so surviving Relu nodes are post-Add or multi-use links and fused Convs carry their epilogue in fuse_relu. Regions need at least two Convs (a lone Conv is the refuted per-layer sandwich); fan-out, graph outputs, and dead ends terminate them. Filter packing happens here, all-or-nothing per region.</remarks>
        public static int FuseBlockedRegionPatterns(ComputationalGraph graph)
        {
            var idx = BuildIndex(graph);
            var producer = idx.Producer;
            var consumers = idx.Consumers;
            var outputs = idx.Outputs;

            var drop = new HashSet<int>();
            int fused = 0;

            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (drop.Contains(i)) continue;
                if (TryMatchBlockedRegion(graph, producer, consumers, outputs, drop, i,
                    out var spec, out var members, out var biasDrops) && spec.Steps.Length > 0)
                {
                    int convs = 0;
                    foreach (var step in spec.Steps) if (step.Kind == BlockedRegionStepKind.Conv) convs++;
                    if (convs < 2) continue;
                    var packs = new Dictionary<string, PreparedBlockedFilter>(StringComparer.Ordinal);
                    bool ok = true;
                    foreach (var step in spec.Steps)
                    {
                        if (step.Kind != BlockedRegionStepKind.Conv || step.FilterName is null) continue;
                        if (!TryPackRegionFilter(graph, step.FilterName, packs)) { ok = false; break; }
                    }
                    if (!ok) continue;
                    foreach (var kv in biasDrops)
                    {
                        if (!graph.Initializers.ContainsKey(kv.Key))
                            graph.Initializers[kv.Key] = kv.Value;
                    }
                    foreach (var kv in packs) graph.BlockedFilters[kv.Key] = kv.Value;
                    var head = graph.Nodes[i];
                    head.Attributes ??= new Dictionary<string, object>();
                    head.Attributes["fuse_region"] = spec;
                    head.IsFused = true;
                    head.Outputs = new[] { spec.OutputName };
                    graph.Nodes[i] = head;
                    foreach (int m in members) drop.Add(m);
                    foreach (var kv in biasDrops)
                    {
                        if (producer.TryGetValue(kv.Key, out int bi) && !drop.Contains(bi))
                        {
                            if (consumers.TryGetValue(kv.Key, out var uses))
                            {
                                // The region head still names the bias input but reads the materialized initializer, so it does not pin the node.
                                bool shared = false;
                                foreach (var u in uses) if (u != i && !drop.Contains(u)) { shared = true; break; }
                                if (!shared) drop.Add(bi);
                            }
                            else drop.Add(bi);
                        }
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

        static bool TryMatchBlockedRegion(
            ComputationalGraph graph,
            Dictionary<string, int> producer,
            Dictionary<string, List<int>> consumers,
            HashSet<string> outputs,
            HashSet<int> drop,
            int start,
            out BlockedRegionSpec spec,
            out List<int> members,
            out Dictionary<string, ITensor> biasDrops)
        {
            spec = new BlockedRegionSpec("", new BlockedRegionStep[0], "");
            members = new List<int>();
            biasDrops = new Dictionary<string, ITensor>(StringComparer.Ordinal);
            if (!TryRegionConv(graph, producer, start, drop, biasDrops,
                out string data, out string filter, out string? bias, out bool relu, out string firstOut))
                return false;
            var steps = new List<BlockedRegionStep>
            {
                new BlockedRegionStep(BlockedRegionStepKind.Conv, firstOut, data, null, filter, bias, relu),
            };
            string cur = firstOut;
            while (true)
            {
                if (outputs.Contains(cur)) break;
                if (!consumers.TryGetValue(cur, out var uses)) break;
                int live = -1;
                int n = 0;
                foreach (var u in uses) if (!drop.Contains(u)) { n++; live = u; }
                if (n != 1) break;
                var nx = graph.Nodes[live];
                if (nx.Op == OpType.Relu && IsFusableParticipant(nx)
                    && nx.Inputs.Length == 1 && nx.Outputs.Length == 1 && nx.Inputs[0] == cur
                    && !string.IsNullOrEmpty(nx.Outputs[0]))
                {
                    if (steps.Count > 0 && steps[steps.Count - 1].Kind == BlockedRegionStepKind.Conv
                        && steps[steps.Count - 1].Output == cur)
                    {
                        var prev = steps[steps.Count - 1];
                        if (!prev.ReluAfter) steps[steps.Count - 1] = prev with { ReluAfter = true };
                    }
                    else
                    {
                        steps.Add(new BlockedRegionStep(BlockedRegionStepKind.Relu, nx.Outputs[0], cur, null, null, null, false));
                    }
                    members.Add(live);
                    cur = nx.Outputs[0];
                    continue;
                }
                if (nx.Op == OpType.Conv && TryRegionConv(graph, producer, live, drop, biasDrops,
                    out string cdata, out string cfilter, out string? cbias, out bool crelu, out string cout)
                    && cdata == cur)
                {
                    steps.Add(new BlockedRegionStep(BlockedRegionStepKind.Conv, cout, cdata, null, cfilter, cbias, crelu));
                    members.Add(live);
                    cur = cout;
                    continue;
                }
                if (nx.Op == OpType.Add && IsFusableParticipant(nx)
                    && nx.Inputs.Length == 2 && nx.Outputs.Length == 1 && !string.IsNullOrEmpty(nx.Outputs[0])
                    && (nx.Inputs[0] == cur || nx.Inputs[1] == cur))
                {
                    string skip = nx.Inputs[0] == cur ? nx.Inputs[1] : nx.Inputs[0];
                    if (string.IsNullOrEmpty(skip) || !IsProvenFloat(graph, producer, skip)) break;
                    steps.Add(new BlockedRegionStep(BlockedRegionStepKind.Add, nx.Outputs[0], cur, skip, null, null, false));
                    members.Add(live);
                    cur = nx.Outputs[0];
                    continue;
                }
                break;
            }
            spec = new BlockedRegionSpec(steps[0].DataInput, steps.ToArray(), cur);
            return true;
        }

        static bool TryRegionConv(
            ComputationalGraph graph,
            Dictionary<string, int> producer,
            int index,
            HashSet<int> drop,
            Dictionary<string, ITensor> biasDrops,
            out string data,
            out string filter,
            out string? bias,
            out bool reluAfter,
            out string output)
        {
            data = "";
            filter = "";
            bias = null;
            reluAfter = false;
            output = "";
            if (drop.Contains(index)) return false;
            var node = graph.Nodes[index];
            if (node.Op != OpType.Conv) return false;
            if (!Node.IsStandardDomain(node.Domain) || !CPUExecutionProvider.SupportsNode(node)) return false;
            if (node.IsFused)
            {
                if (node.Attributes is null || !node.Attributes.TryGetValue("fuse_relu", out var fr) || !(fr is int fri && fri == 1))
                    return false;
                reluAfter = true;
            }
            if (node.Inputs.Length < 2 || node.Outputs.Length != 1) return false;
            if (string.IsNullOrEmpty(node.Inputs[0]) || string.IsNullOrEmpty(node.Inputs[1])) return false;
            if (string.IsNullOrEmpty(node.Outputs[0])) return false;
            var ks = node.Ints("kernel_shape") ?? new int[] { 3, 3 };
            if (ks.Length != 2 || ks[0] != 3 || ks[1] != 3) return false;
            var st = node.Ints("strides") ?? new int[] { 1, 1 };
            if (st.Length != 2 || st[0] != 1 || st[1] != 1) return false;
            var di = node.Ints("dilations") ?? new int[] { 1, 1 };
            if (di.Length != 2 || di[0] != 1 || di[1] != 1) return false;
            var pa = node.Ints("pads");
            if (pa is null || pa.Length != 4 || pa[0] != 1 || pa[1] != 1 || pa[2] != 1 || pa[3] != 1) return false;
            if (node.Attributes is not null && node.Attributes.TryGetValue("auto_pad", out var ap)
                && ap is string aps && !string.IsNullOrEmpty(aps) && aps != "NOTSET") return false;
            int grp = node.GetInt("group", null) ?? 1;
            if (grp != 1) return false;
            if (!graph.Initializers.TryGetValue(node.Inputs[1], out var w) || w is not Tensor<float> wf || wf.Rank != 4) return false;
            var wd = wf.Dimensions;
            if (wd.Length != 4 || wd[2] != 3 || wd[3] != 3) return false;
            int m = wd[0], c = wd[1];
            if (m % 16 != 0 || c % 16 != 0 || m < 16 || c < 16) return false;
            if (!IsProvenFloat(graph, producer, node.Inputs[0])) return false;
            string? biasName = null;
            if (node.Inputs.Length >= 3 && !string.IsNullOrEmpty(node.Inputs[2]))
            {
                biasName = node.Inputs[2];
                if (graph.Initializers.TryGetValue(biasName, out var bi) && bi is Tensor<float> bf && bf.Length == m)
                {
                }
                else if (!TryRegionConstantBias(graph, biasName, m, biasDrops))
                {
                    return false;
                }
            }
            data = node.Inputs[0];
            filter = node.Inputs[1];
            bias = biasName;
            output = node.Outputs[0];
            return true;
        }

        static bool TryRegionConstantBias(
            ComputationalGraph graph,
            string name,
            int m,
            Dictionary<string, ITensor> biasDrops)
        {
            if (biasDrops.ContainsKey(name)) return true;
            if (graph.Outputs.ContainsKey(name) || graph.Inputs.ContainsKey(name)) return false;
            if (graph.Initializers.ContainsKey(name)) return false;
            foreach (var node in graph.Nodes)
            {
                if (node.Op != OpType.Constant || node.Inputs.Length != 0) continue;
                if (node.Outputs.Length != 1 || node.Outputs[0] != name) continue;
                if (!Node.IsStandardDomain(node.Domain)) return false;
                if (node.Attributes is null || !node.Attributes.TryGetValue("value", out var v)) return false;
                if (v is not Tensor<float> tf || tf.Length != m) return false;
                biasDrops[name] = tf;
                return true;
            }
            return false;
        }

        static bool TryPackRegionFilter(
            ComputationalGraph graph,
            string filter,
            Dictionary<string, PreparedBlockedFilter> packs)
        {
            if (packs.ContainsKey(filter)) return true;
            if (!graph.Initializers.TryGetValue(filter, out var w) || w is not Tensor<float> wf || wf.Rank != 4) return false;
            var wd = wf.Dimensions;
            if (wd.Length != 4 || wd[2] != 3 || wd[3] != 3) return false;
            int m = wd[0], c = wd[1];
            if (m % 16 != 0 || c % 16 != 0) return false;
            float[] packed;
            if (w is DenseTensor<float> dense)
                packed = MathOpsConvBlocked.PackBlockedFilter(dense.Buffer.Span, m, c);
            else
                packed = MathOpsConvBlocked.PackBlockedFilter(wf.ToArray(), m, c);
            packs[filter] = new PreparedBlockedFilter(packed, w, w.Length, m, c);
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
                if (!IsFusableParticipant(graph.Nodes[i])) continue;
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
            if (!IsFusableParticipant(add)) return false;
            if (add.Inputs.Length != 2) return false;
            if (add.Outputs.Length != 1) return false;
            int ProducerOf(string input, OpType op)
            {
                if (string.IsNullOrEmpty(input) || !producer.TryGetValue(input, out var pi)) return -1;
                if (drop.Contains(pi)) return -1;
                var cand = graph.Nodes[pi];
                if (cand.Op != op || !IsFusableParticipant(cand)) return -1;
                return pi;
            }
            int OnlyConsumer(string output, OpType op)
            {
                if (outputs.Contains(output)) return -1;
                if (!consumers.TryGetValue(output, out var uses)) return -1;
                var live = new List<int>();
                foreach (var u in uses) if (!drop.Contains(u)) live.Add(u);
                if (live.Count != 1) return -1;
                var cand = graph.Nodes[live[0]];
                if (cand.Op != op || !IsFusableParticipant(cand)) return -1;
                return live[0];
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
                            if (!IsStandardSlice(node)) return false;
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
                    if (!IsProvenFloat(graph, producer, x)) continue;
                    if (!IsProvenFloat(graph, producer, c)) continue;
                    if (!consumers.TryGetValue(x, out var xuses)) continue;
                    var sliceUses = xuses.Where(u => !drop.Contains(u) && graph.Nodes[u].Op == OpType.Slice && IsStandardSlice(graph.Nodes[u])).ToList();
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
                        if (!IsProvenFloat(graph, producer, sIn)) continue;
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
                            if (!IsFusableParticipant(branchAdd)) return false;
                            branchDead.AddRange(deadSet);
                            var fused = branchAdd;
                            fused.Op = OpType.RotaryEmbedding;
                            fused.OpTypeName = OpType.RotaryEmbedding.ToString();
                            fused.Domain = "";
                            int rstdv = StandardOpset(graph);
                            if (rstdv >= 0) fused.OpsetVersion = rstdv;
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
