using System;
using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx
{
    internal static class GraphFusion
    {
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

        static int FloatRankOneLength(ComputationalGraph graph, string name)
        {
            if (!graph.Initializers.TryGetValue(name, out var t)) return -1;
            if (t.ElementType != TensorElementType.Float) return -1;
            if (t.Rank != 1) return -1;
            if (t.Length < 1 || t.Length > int.MaxValue) return -1;
            return (int)t.Length;
        }

        static readonly object PassLock = new object();
        static bool layerNormRegistered;

        public static void RegisterLayerNormPass()
        {
            lock (PassLock)
            {
                if (layerNormRegistered) return;
                Optimization.GraphOptimizer.AddPass(new Optimization.GraphOptimizer.Pass(
                    "layernorm",
                    (graph, facts) =>
                    {
                        var rewritten = new List<int>();
                        int n = FuseLayerNormPass(graph, facts, rewritten);
                        var result = new Optimization.GraphOptimizer.PassResult(n > 0, n);
                        result.Nodes.AddRange(rewritten);
                        if (n > 0) result.Notes.Add("fused " + n + " LayerNorm patterns");
                        return result;
                    }));
                layerNormRegistered = true;
            }
        }
        public static int FuseLayerNormPass(ComputationalGraph graph, Optimization.GraphFacts facts, List<int> rewritten)
        {
            var drop = new HashSet<int>();
            int fused = 0;

            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (graph.Nodes[i].Op != OpType.Div) continue;
                if (!IsFusableParticipant(graph.Nodes[i])) continue;
                if (drop.Contains(i)) continue;
                if (TryMatchLayerNorm(graph, facts, drop, rewritten, i, out var dead, out var epsConst))
                {
                    foreach (var d in dead) drop.Add(d);
                    if (epsConst >= 0)
                    {
                        bool ConstOnlyFeedsDead(string output) =>
                            !facts.GraphOutputs.Contains(output) && (!facts.Consumers.TryGetValue(output, out var uses) || uses.All(u => drop.Contains(u)));
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
            Optimization.GraphFacts facts,
            HashSet<int> drop,
            List<int> rewritten,
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
                if (facts.GraphOutputs.Contains(output)) return -1;
                if (!facts.Consumers.TryGetValue(output, out var uses)) return -1;
                var live = new List<int>();
                foreach (var u in uses) if (!drop.Contains(u)) live.Add(u);
                if (live.Count != 1) return -1;
                var cand = graph.Nodes[live[0]];
                if (cand.Op != op || !IsFusableParticipant(cand)) return -1;
                return live[0];
            }
            int ProducerOf(string input, OpType op)
            {
                if (string.IsNullOrEmpty(input) || !facts.Producer.TryGetValue(input, out var pi)) return -1;
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
                return facts.Dtypes.TryGetValue(name, out var proven) && proven == TensorElementType.Float;
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
                    if (facts.GraphOutputs.Contains(o)) return false;
                    if (facts.Consumers.TryGetValue(o, out var uses) && uses.Any(u => !deadSet.Contains(u) && u != ab && !drop.Contains(u))) return false;
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
            rewritten.Add(ab);
            return true;
        }

        /// <summary>
        /// Fuses exact-GELU erf chains (Div by sqrt(2), Erf, plus one, times
        /// x, times one half) into native exact Gelu nodes. ORT fuses the same
        /// pattern, so the unfused float chain accumulates a few ulps per
        /// application against the fused reference; the native kernel matches
        /// ORT bit-identically.
        /// </summary>
        static readonly object GeluPassLock = new object();
        static bool geluRegistered;

        public static void RegisterGeluPass()
        {
            lock (GeluPassLock)
            {
                if (geluRegistered) return;
                Optimization.GraphOptimizer.AddPass(new Optimization.GraphOptimizer.Pass(
                    "gelu",
                    (graph, facts) =>
                    {
                        var rewritten = new List<int>();
                        int n = FuseGeluPass(graph, facts, rewritten);
                        var result = new Optimization.GraphOptimizer.PassResult(n > 0, n);
                        result.Nodes.AddRange(rewritten);
                        if (n > 0) result.Notes.Add("fused " + n + " exact-GELU patterns");
                        return result;
                    }));
                geluRegistered = true;
            }
        }

        public static int FuseGeluPass(ComputationalGraph graph, Optimization.GraphFacts facts, List<int> rewritten)
        {
            var drop = new HashSet<int>();
            int fused = 0;

            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (graph.Nodes[i].Op != OpType.Erf) continue;
                if (!IsFusableParticipant(graph.Nodes[i])) continue;
                if (drop.Contains(i)) continue;
                if (TryMatchGelu(graph, facts, drop, i, out var dead, out var constNodes, out var survivor))
                {
                    foreach (var d in dead) drop.Add(d);
                    foreach (var cn in constNodes)
                    {
                        // The surviving fused node still names its half
                        // constant, which no longer needs a producer.
                        bool ConstOnlyFeedsDead(string output) =>
                            !facts.GraphOutputs.Contains(output) && (!facts.Consumers.TryGetValue(output, out var uses) || uses.All(u => u == survivor || drop.Contains(u)));
                        var cnode = graph.Nodes[cn];
                        if (cnode.Outputs.All(ConstOnlyFeedsDead)) drop.Add(cn);
                    }
                    if (survivor >= 0) rewritten.Add(survivor);
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
            Optimization.GraphFacts facts,
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
                if (facts.GraphOutputs.Contains(output)) return -1;
                if (!facts.Consumers.TryGetValue(output, out var uses)) return -1;
                var live = new List<int>();
                foreach (var u in uses) if (!drop.Contains(u)) live.Add(u);
                if (live.Count != 1) return -1;
                var cand = graph.Nodes[live[0]];
                if (cand.Op != op || !IsFusableParticipant(cand)) return -1;
                return live[0];
            }
            int ProducerOf(string input, OpType op)
            {
                if (string.IsNullOrEmpty(input) || !facts.Producer.TryGetValue(input, out var pi)) return -1;
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
            if (!(facts.Dtypes.TryGetValue(xPos, out var xdt) && xdt == TensorElementType.Float)) return false;
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
                    if (facts.GraphOutputs.Contains(o)) return false;
                    if (facts.Consumers.TryGetValue(o, out var uses) && uses.Any(u => !deadSet.Contains(u) && u != mul1 && !drop.Contains(u))) return false;
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
            Optimization.GraphFacts facts,
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
                if (facts.GraphOutputs.Contains(output)) return -1;
                if (!facts.Consumers.TryGetValue(output, out var uses)) return -1;
                var live = new List<int>();
                foreach (var u in uses) if (!drop.Contains(u)) live.Add(u);
                if (live.Count != 1) return -1;
                var cand = graph.Nodes[live[0]];
                if (cand.Op != op || !IsFusableParticipant(cand)) return -1;
                return live[0];
            }
            int ProducerOf(string input, OpType op)
            {
                if (string.IsNullOrEmpty(input) || !facts.Producer.TryGetValue(input, out var pi)) return -1;
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
            if (!(facts.Dtypes.TryGetValue(addX, out var addDt) && addDt == TensorElementType.Float)) return false;
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
                    if (facts.GraphOutputs.Contains(o)) return false;
                    if (facts.Consumers.TryGetValue(o, out var uses) && uses.Any(u => !deadSet.Contains(u) && u != mul3 && !drop.Contains(u))) return false;
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
        static readonly object GeluTanhPassLock = new object();
        static bool geluTanhRegistered;

        public static void RegisterGeluTanhPass()
        {
            lock (GeluTanhPassLock)
            {
                if (geluTanhRegistered) return;
                Optimization.GraphOptimizer.AddPass(new Optimization.GraphOptimizer.Pass(
                    "gelu-tanh",
                    (graph, facts) =>
                    {
                        var rewritten = new List<int>();
                        int n = FuseGeluTanhPass(graph, facts, rewritten);
                        var result = new Optimization.GraphOptimizer.PassResult(n > 0, n);
                        result.Nodes.AddRange(rewritten);
                        if (n > 0) result.Notes.Add("fused " + n + " tanh-GELU patterns");
                        return result;
                    }));
                geluTanhRegistered = true;
            }
        }
        public static int FuseGeluTanhPass(ComputationalGraph graph, Optimization.GraphFacts facts, List<int> rewritten)
        {
            var drop = new HashSet<int>();
            int fused = 0;

            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (graph.Nodes[i].Op != OpType.Tanh) continue;
                if (!IsFusableParticipant(graph.Nodes[i])) continue;
                if (drop.Contains(i)) continue;
                if (TryMatchGeluTanh(graph, facts, drop, i, out var dead, out var constNodes, out var survivor))
                {
                    foreach (var d in dead) drop.Add(d);
                    foreach (var cn in constNodes)
                    {
                        bool ConstOnlyFeedsDead(string output) =>
                            !facts.GraphOutputs.Contains(output) && (!facts.Consumers.TryGetValue(output, out var uses) || uses.All(u => u == survivor || drop.Contains(u)));
                        var cnode = graph.Nodes[cn];
                        if (cnode.Outputs.All(ConstOnlyFeedsDead)) drop.Add(cn);
                    }
                    if (survivor >= 0) rewritten.Add(survivor);
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

        static readonly object ConvReluPassLock = new object();
        static bool convReluRegistered;

        public static void RegisterConvReluPass()
        {
            lock (ConvReluPassLock)
            {
                if (convReluRegistered) return;
                Optimization.GraphOptimizer.AddPass(new Optimization.GraphOptimizer.Pass(
                    "convrelu",
                    (graph, facts) =>
                    {
                        var rewritten = new List<int>();
                        int n = FuseConvReluPass(graph, facts, rewritten);
                        var result = new Optimization.GraphOptimizer.PassResult(n > 0, n);
                        result.Nodes.AddRange(rewritten);
                        if (n > 0) result.Notes.Add("fused " + n + " Conv+Relu epilogues");
                        return result;
                    }));
                convReluRegistered = true;
            }
        }

        public static int FuseConvReluPass(ComputationalGraph graph, Optimization.GraphFacts facts, List<int> rewritten)
        {
            var drop = new HashSet<int>();
            int fused = 0;

            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (graph.Nodes[i].Op != OpType.Relu) continue;
                if (!IsFusableParticipant(graph.Nodes[i])) continue;
                if (drop.Contains(i)) continue;
                if (TryMatchEpilogue(graph, facts, drop, i, OpType.Conv, 2, 3, OpType.ConvRelu, out int producer))
                {
                    drop.Add(i);
                    rewritten.Add(producer);
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

        static readonly object AddReluPassLock = new object();
        static bool addReluRegistered;

        public static void RegisterAddReluPass()
        {
            lock (AddReluPassLock)
            {
                if (addReluRegistered) return;
                Optimization.GraphOptimizer.AddPass(new Optimization.GraphOptimizer.Pass(
                    "addrelu",
                    (graph, facts) =>
                    {
                        var rewritten = new List<int>();
                        int n = FuseAddReluPass(graph, facts, rewritten);
                        var result = new Optimization.GraphOptimizer.PassResult(n > 0, n);
                        result.Nodes.AddRange(rewritten);
                        if (n > 0) result.Notes.Add("fused " + n + " Add+Relu epilogues");
                        return result;
                    }));
                addReluRegistered = true;
            }
        }

        public static int FuseAddReluPass(ComputationalGraph graph, Optimization.GraphFacts facts, List<int> rewritten)
        {
            var drop = new HashSet<int>();
            int fused = 0;

            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (graph.Nodes[i].Op != OpType.Relu) continue;
                if (!IsFusableParticipant(graph.Nodes[i])) continue;
                if (drop.Contains(i)) continue;
                if (TryMatchEpilogue(graph, facts, drop, i, OpType.Add, 2, 2, OpType.AddRelu, out int producer))
                {
                    drop.Add(i);
                    rewritten.Add(producer);
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
        /// Matches Producer followed by Relu where the producer output feeds
        /// nothing else and is no graph output, with proven float32/64 dtype.
        /// Rewrites the producer struct in place to the fused identity carrying
        /// the Relu output name; the caller drops the Relu node.
        /// </summary>
        static bool TryMatchEpilogue(
            ComputationalGraph graph,
            Optimization.GraphFacts facts,
            HashSet<int> drop,
            int reluIndex,
            OpType producerOp,
            int minInputs,
            int maxInputs,
            OpType fusedOp,
            out int producerIndex)
        {
            producerIndex = -1;
            var relu = graph.Nodes[reluIndex];
            if (relu.Op != OpType.Relu || relu.IsFused) return false;
            if (!IsFusableParticipant(relu)) return false;
            if (relu.Inputs.Length != 1 || relu.Outputs.Length != 1) return false;
            if (relu.Attributes is not null && relu.Attributes.Count != 0) return false;
            string mid = relu.Inputs[0];
            string rout = relu.Outputs[0];
            if (string.IsNullOrEmpty(mid) || string.IsNullOrEmpty(rout)) return false;
            if (facts.GraphOutputs.Contains(mid)) return false;
            if (!facts.Producer.TryGetValue(mid, out int pi) || drop.Contains(pi)) return false;
            var prod = graph.Nodes[pi];
            if (prod.Op != producerOp || prod.IsFused) return false;
            if (!IsFusableParticipant(prod)) return false;
            if (prod.Inputs.Length < minInputs || prod.Inputs.Length > maxInputs) return false;
            if (prod.Outputs.Length != 1) return false;
            if (!facts.Consumers.TryGetValue(mid, out var uses)) return false;
            int live = 0;
            foreach (var u in uses) if (!drop.Contains(u)) live++;
            if (live != 1) return false;
            if (!facts.Dtypes.TryGetValue(mid, out var dt)) return false;
            if (dt != TensorElementType.Float && dt != TensorElementType.Double) return false;
            var fused = prod;
            fused.Op = fusedOp;
            fused.OpTypeName = fusedOp.ToString();
            fused.Domain = "";
            int stdv = StandardOpset(graph);
            if (stdv >= 0) fused.OpsetVersion = stdv;
            fused.IsFused = true;
            fused.Outputs = new[] { rout };
            graph.Nodes[pi] = fused;
            producerIndex = pi;
            return true;
        }
        static readonly object BiasGeluPassLock = new object();
        static bool biasGeluRegistered;

        public static void RegisterBiasGeluPass()
        {
            lock (BiasGeluPassLock)
            {
                if (biasGeluRegistered) return;
                Optimization.GraphOptimizer.AddPass(new Optimization.GraphOptimizer.Pass(
                    "biasgelu",
                    (graph, facts) =>
                    {
                        var rewritten = new List<int>();
                        int n = FuseBiasGeluPass(graph, facts, rewritten);
                        var result = new Optimization.GraphOptimizer.PassResult(n > 0, n);
                        result.Nodes.AddRange(rewritten);
                        if (n > 0) result.Notes.Add("fused " + n + " bias+GELU regions");
                        return result;
                    }));
                biasGeluRegistered = true;
            }
        }

        public static int FuseBiasGeluPass(ComputationalGraph graph, Optimization.GraphFacts facts, List<int> rewritten)
        {
            var drop = new HashSet<int>();
            int fused = 0;

            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (graph.Nodes[i].Op != OpType.Gelu) continue;
                if (drop.Contains(i)) continue;
                if (TryMatchBiasGelu(graph, facts, drop, i, out int producer))
                {
                    drop.Add(i);
                    rewritten.Add(producer);
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
        /// Matches Add(data, bias-initializer) feeding exact GELU with no other
        /// consumer and no graph-output exposure, rewriting the Add struct in
        /// place to BiasGelu with data-first inputs; the caller drops the Gelu
        /// node. Float32 only; tanh-approximate and double forms keep legacy paths.
        /// </summary>
        static bool TryMatchBiasGelu(
            ComputationalGraph graph,
            Optimization.GraphFacts facts,
            HashSet<int> drop,
            int geluIndex,
            out int producerIndex)
        {
            producerIndex = -1;
            var gelu = graph.Nodes[geluIndex];
            if (gelu.Op != OpType.Gelu) return false;
            // The consumer may be a native exact GELU or an already-fused one
            // from the earlier gelu pass (same formula, empty attributes); only
            // the producer side must be an unfused participant. Tanh forms
            // decline on attributes below regardless of fused state.
            if (!Node.IsStandardDomain(gelu.Domain)) return false;
            if (gelu.Inputs.Length != 1 || gelu.Outputs.Length != 1) return false;
            string? approx = gelu.Attr<string>("approximate", null);
            if (approx is not null && approx != "none") return false;
            string mid = gelu.Inputs[0];
            string rout = gelu.Outputs[0];
            if (string.IsNullOrEmpty(mid) || string.IsNullOrEmpty(rout)) return false;
            if (facts.GraphOutputs.Contains(mid)) return false;
            if (!facts.Producer.TryGetValue(mid, out int pi) || drop.Contains(pi)) return false;
            var add = graph.Nodes[pi];
            if (add.Op != OpType.Add || add.IsFused) return false;
            if (!IsFusableParticipant(add)) return false;
            if (add.Inputs.Length != 2 || add.Outputs.Length != 1) return false;
            if (!facts.Consumers.TryGetValue(mid, out var uses)) return false;
            int live = 0;
            foreach (var u in uses) if (!drop.Contains(u)) live++;
            if (live != 1) return false;
            if (!facts.Dtypes.TryGetValue(mid, out var dt) || dt != TensorElementType.Float) return false;
            string? data = null;
            string? bias = null;
            foreach (var inp in add.Inputs)
            {
                // Exactly one side must be a rank-one float initializer: the
                // bias. Graph inputs are excluded outright (their values vary
                // per run), and a second constant side declines so constant
                // folding keeps owning all-constant Adds with their own rules.
                if (!string.IsNullOrEmpty(inp) && bias is null && !graph.Inputs.ContainsKey(inp)
                    && graph.Initializers.TryGetValue(inp, out var init)
                    && init.ElementType == TensorElementType.Float && init.Rank == 1 && init.Length >= 1)
                    bias = inp;
                else if (data is null && !string.IsNullOrEmpty(inp))
                    data = inp;
                else
                    return false;
            }
            if (data is null || bias is null) return false;
            if (graph.Initializers.TryGetValue(data, out var dataInit)
                && dataInit.ElementType == TensorElementType.Float && dataInit.Rank == 1)
                return false;
            var fused = add;
            fused.Op = OpType.BiasGelu;
            fused.OpTypeName = OpType.BiasGelu.ToString();
            fused.Domain = "";
            int stdv = StandardOpset(graph);
            if (stdv >= 0) fused.OpsetVersion = stdv;
            fused.IsFused = true;
            fused.Inputs = new[] { data, bias };
            fused.Attributes = new Dictionary<string, object>();
            fused.Outputs = new[] { rout };
            graph.Nodes[pi] = fused;
            producerIndex = pi;
            return true;
        }

        static readonly object GemmGeluPassLock = new object();
        static bool gemmGeluRegistered;

        public static void RegisterGemmGeluPass()
        {
            lock (GemmGeluPassLock)
            {
                if (gemmGeluRegistered) return;
                Optimization.GraphOptimizer.AddPass(new Optimization.GraphOptimizer.Pass(
                    "gemmgelu",
                    (graph, facts) =>
                    {
                        var rewritten = new List<int>();
                        int n = FuseGemmGeluPass(graph, facts, rewritten);
                        var result = new Optimization.GraphOptimizer.PassResult(n > 0, n);
                        result.Nodes.AddRange(rewritten);
                        if (n > 0) result.Notes.Add("fused " + n + " Gemm+GELU regions");
                        return result;
                    }));
                gemmGeluRegistered = true;
            }
        }

        public static int FuseGemmGeluPass(ComputationalGraph graph, Optimization.GraphFacts facts, List<int> rewritten)
        {
            var drop = new HashSet<int>();
            int fused = 0;

            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (graph.Nodes[i].Op != OpType.Gelu) continue;
                if (drop.Contains(i)) continue;
                if (TryMatchGemmGelu(graph, facts, drop, i, out int producer))
                {
                    drop.Add(i);
                    rewritten.Add(producer);
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
        /// Matches Gemm(data, weights[, bias]) feeding exact GELU directly or through
        /// one single-consumer view Reshape, with no other live consumer and no
        /// graph-output exposure on the links. Rewrites the Gemm struct in place to
        /// GemmGelu, drops only the Gelu node, and rewires Gelu-output consumers to
        /// the kept tensor. Float32 only; tanh-approximate forms keep legacy paths.
        /// </summary>
        static bool TryMatchGemmGelu(
            ComputationalGraph graph,
            Optimization.GraphFacts facts,
            HashSet<int> drop,
            int geluIndex,
            out int producerIndex)
        {
            producerIndex = -1;
            var gelu = graph.Nodes[geluIndex];
            if (gelu.Op != OpType.Gelu) return false;
            if (!Node.IsStandardDomain(gelu.Domain)) return false;
            if (gelu.Inputs.Length != 1 || gelu.Outputs.Length != 1) return false;
            string? approx = gelu.Attr<string>("approximate", null);
            if (approx is not null && approx != "none" && approx != "tanh") return false;
            bool tanh = approx == "tanh";
            string mid = gelu.Inputs[0];
            string rout = gelu.Outputs[0];
            if (string.IsNullOrEmpty(mid) || string.IsNullOrEmpty(rout)) return false;
            if (!facts.Producer.TryGetValue(mid, out int mi) || drop.Contains(mi)) return false;
            var middle = graph.Nodes[mi];
            if (middle.IsFused) return false;
            if (!IsFusableParticipant(middle)) return false;
            if (middle.Outputs.Length != 1) return false;
            string kept;
            int gemmIndex;
            if (middle.Op == OpType.Gemm)
            {
                if (middle.Inputs.Length < 2 || middle.Inputs.Length > 3) return false;
                kept = rout;
                gemmIndex = mi;
            }
            else if (middle.Op == OpType.Reshape)
            {
                if (middle.Inputs.Length != 2) return false;
                string data = middle.Inputs[0];
                if (string.IsNullOrEmpty(data)) return false;
                if (!facts.Producer.TryGetValue(data, out int gi) || drop.Contains(gi)) return false;
                var gemm = graph.Nodes[gi];
                if (gemm.Op != OpType.Gemm || gemm.IsFused) return false;
                if (!IsFusableParticipant(gemm)) return false;
                if (gemm.Inputs.Length < 2 || gemm.Inputs.Length > 3 || gemm.Outputs.Length != 1) return false;
                if (facts.GraphOutputs.Contains(data)) return false;
                if (!facts.Consumers.TryGetValue(data, out var dataUses)) return false;
                int dataLive = 0;
                foreach (var u in dataUses) if (!drop.Contains(u)) dataLive++;
                if (dataLive != 1) return false;
                kept = mid;
                gemmIndex = gi;
            }
            else return false;
            var gnode = graph.Nodes[gemmIndex];
            if (!facts.Dtypes.TryGetValue(gnode.Outputs[0], out var dt) || dt != TensorElementType.Float) return false;
            if (facts.GraphOutputs.Contains(gnode.Outputs[0])) return false;
            if (facts.GraphOutputs.Contains(mid)) return false;
            if (!facts.Consumers.TryGetValue(mid, out var uses)) return false;
            int live = 0;
            foreach (var u in uses) if (!drop.Contains(u)) live++;
            if (live != 1) return false;
            var fused = gnode;
            fused.Op = OpType.GemmGelu;
            fused.OpTypeName = OpType.GemmGelu.ToString();
            fused.Domain = "";
            int stdv = StandardOpset(graph);
            if (stdv >= 0) fused.OpsetVersion = stdv;
            fused.IsFused = true;
            if (tanh)
            {
                var attrs = new Dictionary<string, object>(fused.Attributes ?? new Dictionary<string, object>());
                attrs["approximate"] = "tanh";
                fused.Attributes = attrs;
            }
            if (middle.Op == OpType.Gemm)
                fused.Outputs = new[] { rout };
            graph.Nodes[gemmIndex] = fused;
            if (facts.Consumers.TryGetValue(rout, out var routUses))
            {
                foreach (var u in routUses)
                {
                    if (drop.Contains(u)) continue;
                    if (u == geluIndex) continue;
                    var consumer = graph.Nodes[u];
                    if (consumer.Inputs is null) continue;
                    var inputs = (string[])consumer.Inputs.Clone();
                    bool changed = false;
                    for (int k = 0; k < inputs.Length; k++)
                        if (inputs[k] == rout) { inputs[k] = kept; changed = true; }
                    if (changed) { consumer.Inputs = inputs; graph.Nodes[u] = consumer; }
                }
            }
            producerIndex = gemmIndex;
            return true;
        }


        static readonly object RopePassLock = new object();
        static bool ropeRegistered;

        public static void RegisterRopePass()
        {
            lock (RopePassLock)
            {
                if (ropeRegistered) return;
                Optimization.GraphOptimizer.AddPass(new Optimization.GraphOptimizer.Pass(
                    "rope",
                    (graph, facts) =>
                    {
                        var rewritten = new List<int>();
                        int n = FuseRopePass(graph, facts, rewritten);
                        var result = new Optimization.GraphOptimizer.PassResult(n > 0, n);
                        result.Nodes.AddRange(rewritten);
                        if (n > 0) result.Notes.Add("fused " + n + " rotary-embedding patterns");
                        return result;
                    }));
                ropeRegistered = true;
            }
        }

        public static int FuseRopePass(ComputationalGraph graph, Optimization.GraphFacts facts, List<int> rewritten)
        {
            var drop = new HashSet<int>();
            int fused = 0;

            for (int i = 0; i < graph.Nodes.Count; i++)
            {
                if (graph.Nodes[i].Op != OpType.Add) continue;
                if (!IsFusableParticipant(graph.Nodes[i])) continue;
                if (drop.Contains(i)) continue;
                if (TryMatchRope(graph, facts, drop, i, out var dead))
                {
                    foreach (var d in dead) drop.Add(d);
                    rewritten.Add(i);
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
            Optimization.GraphFacts facts,
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
                if (string.IsNullOrEmpty(input) || !facts.Producer.TryGetValue(input, out var pi)) return -1;
                if (drop.Contains(pi)) return -1;
                var cand = graph.Nodes[pi];
                if (cand.Op != op || !IsFusableParticipant(cand)) return -1;
                return pi;
            }
            int OnlyConsumer(string output, OpType op)
            {
                if (facts.GraphOutputs.Contains(output)) return -1;
                if (!facts.Consumers.TryGetValue(output, out var uses)) return -1;
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
                    if (!(facts.Dtypes.TryGetValue(x, out var xdt) && xdt == TensorElementType.Float)) continue;
                    if (!(facts.Dtypes.TryGetValue(c, out var cdt) && cdt == TensorElementType.Float)) continue;
                    if (!facts.Consumers.TryGetValue(x, out var xuses)) continue;
                    var sliceUses = xuses.Where(u => !drop.Contains(u) && graph.Nodes[u].Op == OpType.Slice && IsStandardSlice(graph.Nodes[u])).ToList();
                    if (sliceUses.Count != 2) continue;
                    if (facts.GraphOutputs.Contains(x)) continue;
                    for (int ri = 0; ri < 2; ri++)
                    {
                        string ccIn = mr.Inputs[ri];
                        string sIn = mr.Inputs[1 - ri];
                        int sinIdx = ProducerOf(sIn, OpType.Sin);

                        if (sinIdx < 0) continue;
                        var sinNode = graph.Nodes[sinIdx];
                        if (sinNode.Inputs.Length != 1) continue;
                        if (!(facts.Dtypes.TryGetValue(sIn, out var sdt) && sdt == TensorElementType.Float)) continue;
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
                                    if (facts.GraphOutputs.Contains(o)) return false;
                                    if (facts.Consumers.TryGetValue(o, out var uses) && uses.Any(u => !deadSet.Contains(u) && u != addIndex && !drop.Contains(u))) return false;
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
