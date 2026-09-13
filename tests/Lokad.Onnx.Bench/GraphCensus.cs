namespace Lokad.Onnx.Bench;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Lokad.Onnx;
using Microsoft.ML.OnnxRuntime;

// B03 graph-region attribution: structural census promoted from the
// artifacts/plan021-audit scratch into benchmark tooling. No inference runs
// here, so no CPU affinity is pinned and no latency is claimed. Output (node
// name/domain/op/inputs/outputs for the original, Lokad-prepared and
// ORT-optimized graphs) feeds region comparison and fusion-attribution work;
// counting nodes never prices them.
internal static class GraphCensus
{
    static readonly string[] CensusModels = new[] { "e5", "dinov3", "resnet50", "gpt2" };

    internal static int RunGraph(string root, Dictionary<string, string[]> assets, string[] args)
    {
        var keys = new List<string>();
        string outDir = Path.Combine(root, "artifacts", "graph-census");
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--out" && i + 1 < args.Length) { outDir = args[++i]; }
            else if (Array.Exists(CensusModels, k => k.Equals(args[i], StringComparison.OrdinalIgnoreCase))) keys.Add(args[i].ToLowerInvariant());
            else { Console.WriteLine("usage: Bench graph [e5 dinov3 resnet50 gpt2] [--out dir]"); return 2; }
        }
        if (keys.Count == 0) keys.AddRange(CensusModels);
        foreach (var key in keys)
        {
            if (!assets.TryGetValue(key, out var files) || !File.Exists(files[0]))
            { Console.WriteLine("missing asset for " + key); return 1; }
        }
        Directory.CreateDirectory(outDir);
        var summary = new List<object>();
        foreach (var key in keys)
        {
            string model = assets[key][0];
            var original = OnnxImport.ParseMetadata(model);
            var graph = OnnxImport.Load(model)
                ?? throw new InvalidOperationException(key + ": load failed: " + OnnxImport.LastErrorMessage);
            string optimized = Path.Combine(outDir, key + "-ort-all.onnx");
            using (var options = new SessionOptions
            {
                IntraOpNumThreads = 1,
                InterOpNumThreads = 1,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                OptimizedModelFilePath = optimized,
            })
            {
                options.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
                options.AddSessionConfigEntry("session.inter_op.allow_spinning", "0");
                using var session = new InferenceSession(model, options);
            }
            var ort = OnnxImport.ParseMetadata(optimized);
            var entry = new
            {
                key,
                model,
                originalNodes = original.Nodes.Count,
                lokadNodes = graph.Nodes.Count,
                ortNodes = ort.Nodes.Count,
                original = original.Nodes.Select(NodeView).ToArray(),
                lokad = graph.Nodes.Select(LokadView).ToArray(),
                ortOptimized = ort.Nodes.Select(NodeView).ToArray(),
            };
            File.WriteAllText(Path.Combine(outDir, key + "-census.json"),
                JsonSerializer.Serialize(entry, new JsonSerializerOptions { WriteIndented = true }));
            summary.Add(new { key, originalNodes = entry.originalNodes, lokadNodes = entry.lokadNodes, ortNodes = entry.ortNodes });
            Console.WriteLine(key + ": original=" + entry.originalNodes + " lokad=" + entry.lokadNodes + " ort=" + entry.ortNodes);
            graph = null;
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
        File.WriteAllText(Path.Combine(outDir, "graph-census.json"),
            JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine("graph-census wrote " + summary.Count + " models to " + outDir);
        return 0;
    }

    static object NodeView(OnnxNode n) => new { name = n.Name, domain = n.Domain, op = n.OpType, inputs = n.Inputs, outputs = n.Outputs };
    static object LokadView(Node n) => new { id = n.ID, name = n.Name, domain = n.Domain, op = n.OpTypeName ?? n.Op.ToString(), fused = n.IsFused, inputs = n.Inputs, outputs = n.Outputs };
}
