using System;
using System.Collections.Generic;

namespace Lokad.Onnx.Optimization;

/// <summary>
/// Deterministic load-time rewrite runner. Passes are applied in declaration order;
/// facts are rebuilt only when a pass reports a change; a small fixed round cap stops
/// a buggy pass from looping forever. With zero registered passes this is a no-op by
/// construction. Disable switches let a diagnosis isolate one rewrite by name.
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
        public PassResult(bool changed)
        {
            Changed = changed;
            Notes = new List<string>();
        }

        public bool Changed { get; }
        public List<string> Notes { get; }
    }

    public sealed class PassChange
    {
        public PassChange(string pass, List<int> nodes, List<string> notes)
        {
            Pass = pass;
            Nodes = nodes;
            Notes = notes;
        }

        public string Pass { get; }
        public List<int> Nodes { get; }
        public List<string> Notes { get; }
    }

    public static readonly List<Pass> Passes = new List<Pass>();

    public static readonly HashSet<string> Disabled = new HashSet<string>(StringComparer.Ordinal);

    public const int MaxRounds = 8;

    public static List<PassChange> Run(ComputationalGraph graph)
    {
        var report = new List<PassChange>();
        for (int round = 0; round < MaxRounds; round++)
        {
            var facts = GraphFacts.Build(graph);
            bool changed = false;
            foreach (var pass in Passes)
            {
                if (Disabled.Contains(pass.Name)) continue;
                var result = pass.Run(graph, facts);
                if (result is null || !result.Changed) continue;
                changed = true;
                report.Add(new PassChange(pass.Name, new List<int>(), result.Notes));
            }
            if (!changed) break;
        }
        return report;
    }
}
