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
        public Pass(string name, Func<ComputationalGraph, GraphFacts, int> run)
        {
            Name = name;
            Run = run;
        }

        public string Name { get; }
        public Func<ComputationalGraph, GraphFacts, int> Run { get; }
    }

    public sealed class PassChange
    {
        public PassChange(string pass, int rewritten)
        {
            Pass = pass;
            Rewritten = rewritten;
        }

        public string Pass { get; }
        public int Rewritten { get; }
    }

    internal static IReadOnlyList<Pass> Passes { get; } = Array.AsReadOnly(new Pass[]
    {
        new Pass("layernorm", GraphFusion.FuseLayerNormPass),
        new Pass("rmsnorm", GraphFusion.FuseRmsNorm),
        new Pass("rope", GraphFusion.FuseRopePass),
        new Pass("gelu", GraphFusion.FuseGeluPass),
        new Pass("gelu-tanh", GraphFusion.FuseGeluTanhPass),
        new Pass("sigmoid-mul", GraphFusion.FuseSigmoidMul),
        new Pass("convrelu", GraphFusion.FuseConvReluPass),
        new Pass("addrelu", GraphFusion.FuseAddReluPass),
        new Pass("biasgelu", GraphFusion.FuseBiasGeluPass),
        new Pass("gemmgelu", GraphFusion.FuseGemmGeluPass),
        new Pass("scalematmul", GraphFusion.FuseScaledMatMulPass),
        new Pass("maskedsoftmax", GraphFusion.FuseMaskedSoftmaxPass),
        new Pass("constfold", ConstFold.FoldConstants),
        new Pass("reshape-zerocopy", ShapeZeroCopy.RewriteSelfShapes),
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
                int rewritten = pass.Run(graph, facts);
                if (rewritten <= 0) continue;
                changed = true;
                report.Add(new PassChange(pass.Name, rewritten));
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

