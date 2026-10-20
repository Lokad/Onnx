using System;
using System.Collections.Generic;

namespace Lokad.Onnx.Optimization;

/// <summary>
/// Deterministic load-time rewrite runner. Passes are applied in declaration order;
/// facts are rebuilt only when a pass reports a change; a small fixed round cap stops
/// a buggy pass from looping forever. The fixed pass set is immutable.
/// Disable switches let a diagnosis isolate one rewrite by name.
/// </summary>
internal static class GraphOptimizer
{
    public sealed class Pass
    {
        public Pass(string name, Func<ComputationalGraph, GraphFacts, PassResult> run)
        {
            Name = name;
            Run = run;
        }

        public string Name { get; }
        public Func<ComputationalGraph, GraphFacts, PassResult> Run { get; }
    }

    public sealed class PassResult
    {
        public PassResult(bool changed, int rewritten)
        {
            Changed = changed;
            Rewritten = rewritten;
            Nodes = new List<int>();
            Notes = new List<string>();
        }

        public bool Changed { get; }
        public int Rewritten { get; }
        public List<int> Nodes { get; }
        public List<string> Notes { get; }
    }

    public sealed class PassChange
    {
        public PassChange(string pass, int rewritten, List<int> nodes, List<string> notes, TimeSpan elapsed)
        {
            Pass = pass;
            Rewritten = rewritten;
            Nodes = nodes;
            Notes = notes;
            Elapsed = elapsed;
        }

        public string Pass { get; }
        public int Rewritten { get; }
        public List<int> Nodes { get; }
        public List<string> Notes { get; }
        public TimeSpan Elapsed { get; }
    }

    internal static IReadOnlyList<Pass> Passes { get; } = Array.AsReadOnly(new Pass[]
    {
        Rewrite("layernorm", GraphFusion.FuseLayerNormPass, "fused {0} LayerNorm patterns"),
        Rewrite("rmsnorm", GraphFusion.FuseRmsNorm, "fused {0} RMS normalization patterns"),
        Rewrite("rope", GraphFusion.FuseRopePass, "fused {0} rotary-embedding patterns"),
        Rewrite("gelu", GraphFusion.FuseGeluPass, "fused {0} exact-GELU patterns"),
        Rewrite("gelu-tanh", GraphFusion.FuseGeluTanhPass, "fused {0} tanh-GELU patterns"),
        Rewrite("convrelu", GraphFusion.FuseConvReluPass, "fused {0} Conv+Relu epilogues"),
        Rewrite("addrelu", GraphFusion.FuseAddReluPass, "fused {0} Add+Relu epilogues"),
        Rewrite("biasgelu", GraphFusion.FuseBiasGeluPass, "fused {0} bias+GELU regions"),
        Rewrite("gemmgelu", GraphFusion.FuseGemmGeluPass, "fused {0} Gemm+GELU regions"),
        Rewrite("scalematmul", GraphFusion.FuseScaledMatMulPass, "fused {0} scale+MatMul regions"),
        Rewrite("maskedsoftmax", GraphFusion.FuseMaskedSoftmaxPass, "fused {0} mask+softmax regions"),
        Rewrite("constfold", ConstFold.FoldConstants, "folded {0} constant subgraphs"),
        new Pass("reshape-zerocopy", (graph, facts) =>
        {
            var rewritten = new List<int>();
            int count = ShapeZeroCopy.RewriteSelfShapes(graph, facts, rewritten, out int rewired, out int swept);
            var result = new PassResult(count > 0, count);
            result.Nodes.AddRange(rewritten);
            if (count > 0) result.Notes.Add("rewrote " + rewired + " reshape shape inputs; swept " + swept + " dead nodes");
            return result;
        }),
    });

    static Pass Rewrite(string name, Func<ComputationalGraph, GraphFacts, List<int>, int> apply, string note) =>
        new Pass(name, (graph, facts) =>
        {
            var rewritten = new List<int>();
            int count = apply(graph, facts, rewritten);
            var result = new PassResult(count > 0, count);
            result.Nodes.AddRange(rewritten);
            if (count > 0) result.Notes.Add(string.Format(note, count));
            return result;
        });

    public const int MaxRounds = 8;

    public static List<PassChange> Run(ComputationalGraph graph)
    {
        return Run(graph, Array.Empty<string>());
    }

    public static List<PassChange> Run(ComputationalGraph graph, IEnumerable<string> disabled)
    {
        var report = new List<PassChange>();
        var off = new HashSet<string>(disabled, StringComparer.Ordinal);
        var env = Environment.GetEnvironmentVariable("LOKAD_ONNX_DISABLE_PASSES");
        if (!string.IsNullOrWhiteSpace(env))
            foreach (var name in env.Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries)) off.Add(name.Trim());
        bool exhausted = false;
        for (int round = 0; round < MaxRounds; round++)
        {
            var facts = GraphFacts.Build(graph);
            bool changed = false;
            foreach (var pass in Passes)
            {
                if (off.Contains(pass.Name)) continue;
                var sw = Stopwatch.StartNew();
                var result = pass.Run(graph, facts);
                sw.Stop();
                if (result is null || !result.Changed) continue;
                changed = true;
                report.Add(new PassChange(pass.Name, result.Rewritten, result.Nodes, result.Notes, sw.Elapsed));
                // A pass that reports a change may have reordered the node list; later passes
                // in this round must see fresh positions, never the pre-mutation snapshot.
                facts = GraphFacts.Build(graph);
            }
            if (!changed) break;
            if (round == MaxRounds - 1) exhausted = true;
        }
        if (exhausted && Log.IsEnabled(LogLevel.Warn))
            Log.Write(LogLevel.Warn, "Graph optimization hit the {rounds}-round cap with changes still pending; results are deterministic but may be incomplete.", MaxRounds);
        return report;
    }
}

