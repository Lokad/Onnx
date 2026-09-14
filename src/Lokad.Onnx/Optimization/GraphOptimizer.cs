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
        public PassChange(string pass, int rewritten, List<int> nodes, List<string> notes)
        {
            Pass = pass;
            Rewritten = rewritten;
            Nodes = nodes;
            Notes = notes;
        }

        public string Pass { get; }
        public int Rewritten { get; }
        public List<int> Nodes { get; }
        public List<string> Notes { get; }
    }

    public static readonly List<Pass> Passes = new List<Pass>();

    /// <summary>Guards pass registration and snapshots: model loads run concurrently across tests.</summary>
    internal static readonly object PassLock = new object();

    public static void AddPass(Pass pass)
    {
        lock (PassLock)
        {
            foreach (var p in Passes) if (p.Name == pass.Name) return;
            Passes.Add(pass);
        }
    }

    public static readonly HashSet<string> Disabled = new HashSet<string>(StringComparer.Ordinal);

    public const int MaxRounds = 8;

    public static List<PassChange> Run(ComputationalGraph graph)
    {
        return Run(graph, Array.Empty<string>());
    }

    public static List<PassChange> Run(ComputationalGraph graph, IEnumerable<string> disabled)
    {
        var report = new List<PassChange>();
        Pass[] active;
        HashSet<string> off;
        lock (PassLock)
        {
            active = Passes.ToArray();
            off = new HashSet<string>(Disabled, StringComparer.Ordinal);
            foreach (var d in disabled) off.Add(d);
        }
        for (int round = 0; round < MaxRounds; round++)
        {
            var facts = GraphFacts.Build(graph);
            bool changed = false;
            foreach (var pass in active)
            {
                if (off.Contains(pass.Name)) continue;
                var result = pass.Run(graph, facts);
                if (result is null || !result.Changed) continue;
                changed = true;
                report.Add(new PassChange(pass.Name, result.Rewritten, result.Nodes, result.Notes));
                // A pass that reports a change may have reordered the node list; later passes
                // in this round must see fresh positions, never the pre-mutation snapshot.
                facts = GraphFacts.Build(graph);
            }
            if (!changed) break;
        }
        return report;
    }
}
