namespace Lokad.Onnx.Bench;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Lokad.Onnx;
using Microsoft.ML.OnnxRuntime;

// B03 graph-region attribution: per-op cost tables for identical inputs on both
// engines. Lokad side aggregates Profiler stage timings over profiled executes;
// ORT side aggregates node durations from a session trace (EndProfiling). Every
// run is agreement-gated at 1e-4 before profiling, so costs always describe
// correct execution. Regions (attention blocks, residual bottlenecks) are compared
// by op family, because optimized node names do not correspond one-to-one.
// Decode cases need the two-phase past procedure and stay out for now.
internal static class GraphProfile
{
    internal static int RunProfile(string root, string[] args)
    {
        string? kase = null;
        string outDir = Path.Combine(root, "artifacts", "graph-profile");
        int cpu = 4;
        int reps = 5;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--out" && i + 1 < args.Length) { outDir = args[++i]; }
            else if (args[i] == "--cpu" && i + 1 < args.Length && int.TryParse(args[i + 1], out var c)) { cpu = c; i++; }
            else if (args[i] == "--reps" && i + 1 < args.Length && int.TryParse(args[i + 1], out var k) && k >= 1) { reps = k; i++; }
            else if (kase is null && !args[i].StartsWith("--")) kase = args[i];
            else { Console.WriteLine("usage: Bench profile <case> [--out dir] [--cpu N] [--reps K]"); return 2; }
        }
        if (kase is null) { Console.WriteLine("usage: Bench profile <case> [--out dir] [--cpu N] [--reps K]"); return 2; }
        if (cpu < 0 || cpu >= 64 || cpu >= Environment.ProcessorCount) { Console.WriteLine("invalid --cpu " + cpu); return 2; }
        var startedUtc = DateTime.UtcNow;
        try { int rc = ProfileCase(root, kase, outDir, cpu, reps); SweepStrayTraces(startedUtc); return rc; }
        catch (Exception ex) { Console.WriteLine(kase + ": profile failed: " + ex.GetType().Name + ": " + ex.Message.Split((char)10)[0]); SweepStrayTraces(startedUtc); return 1; }
    }

    static int ProfileCase(string root, string kase, string outDir, int cpu, int reps)
    {
        string model, tokenizer = "";
        Func<string, ITensor[]> build;
        switch (kase)
        {
            case "e5-8tok": model = E5Model(root, out tokenizer); build = n => global::Bench.E5Inputs(n, tokenizer, global::Bench.E5ShortText, 0, 0); break;
            case "e5-30tok": model = E5Model(root, out tokenizer); build = n => global::Bench.E5Inputs(n, tokenizer, "query: " + global::Bench.E5Sentence, 0, 0); break;
            case "e5-128tok": model = E5Model(root, out tokenizer); build = n => global::Bench.E5Inputs(n, tokenizer, E5Long(root), 128, 0); break;
            case "e5-512tok": model = E5Model(root, out tokenizer); build = n => global::Bench.E5Inputs(n, tokenizer, E5Long(root), 512, 0); break;
            case "e5-30pad128": model = E5Model(root, out tokenizer); build = n => global::Bench.E5Inputs(n, tokenizer, "query: " + global::Bench.E5Sentence, 0, 128); break;
            case "dinov3-224": model = Path.Combine(root, "models", "dinov3-vits16", "onnx", "model.onnx"); build = _ => global::Bench.VisionInputs(); break;
            case "resnet50-224": model = Path.Combine(root, "models", "resnet50-onnx", "model.onnx"); build = _ => global::Bench.VisionInputs(); break;
            case "gpt2-1tok": model = Gpt2Model(root); build = _ => global::Bench.Gpt2PrefillInputs(1, false); break;
            case "gpt2-4tok": model = Gpt2Model(root); build = _ => global::Bench.Gpt2PrefillInputs(4, true); break;
            case "gpt2-32tok": model = Gpt2Model(root); build = _ => global::Bench.Gpt2PrefillInputs(32, false); break;
            case "gpt2-128tok": model = Gpt2Model(root); build = _ => global::Bench.Gpt2PrefillInputs(128, false); break;
            case "dinov2-224": Console.WriteLine("dinov2-224 is diverged; profiling it would price incorrect execution. Refusing."); return 1;
            default:
                if (kase.StartsWith("gpt2-dec-")) { Console.WriteLine(kase + ": decode profiling needs the two-phase past procedure; not yet supported."); return 2; }
                Console.WriteLine("unknown case " + kase); return 2;
        }
        if (!File.Exists(model)) { Console.WriteLine("missing asset: " + model); return 1; }
        global::Bench.EnforceSingleCpuAffinity(cpu);
        Directory.CreateDirectory(outDir);
        var tensorOpts = TensorExecutionOptions.Auto with { MaxDegreeOfParallelism = 1 };
        var opts = new ExecutionOptions(OptimizationMode.Speed, tensorOpts);
        var graph = OnnxImport.Load(model)!;
        graph.Prepare();
        using var so = global::Bench.CreateSingleCpuSessionOptions(1);
        so.EnableProfiling = true;
        so.ProfileOutputPathPrefix = Path.Combine(outDir, kase + "-ort-");
        using var session = new InferenceSession(model, so);
        var inNames = session.InputMetadata.Keys.ToArray();
        var outNames = session.OutputMetadata.Keys.ToArray();
        var built = build(kase);
        if (built.Length == 1 && string.IsNullOrEmpty(built[0].Name) && inNames.Length == 1) built[0].Name = inNames[0];
        var named = global::Bench.ToNamed(kase, built, inNames);
        var gate = global::Bench.Validate(kase, graph, session, named, outNames, opts);
        Console.WriteLine(kase + ": agreement gate maxScaled=" + gate.scaled.ToString("E2") + " (tol 1e-4); profiling " + reps + " reps per engine.");
        var ortInputs = global::Bench.BuildOrtInputs(named, inNames);
        try
        {
            var lokStages = new Dictionary<string, double>(StringComparer.Ordinal);
            var lokNodes = new Dictionary<long, double>();
            var lokNodeOps = new Dictionary<long, string>();
            var lokOps = new Dictionary<string, double>(StringComparer.Ordinal);
            double lokTotal = 0;
            var ortOps = new Dictionary<string, double>(StringComparer.Ordinal);
            double ortTotal = 0;
            var sw = new System.Diagnostics.Stopwatch();
            using var ro = new RunOptions();
            for (int r = 0; r < reps; r++)
            {
                if (r % 2 == 0)
                {
                    lokTotal += ProfileLokad(graph, named, opts, lokStages, lokOps, lokNodes, lokNodeOps);
                    sw.Restart(); using (var o = session.Run(ro, ortInputs, outNames)) { sw.Stop(); }
                    ortTotal += sw.Elapsed.TotalMilliseconds;
                }
                else
                {
                    sw.Restart(); using (var o = session.Run(ro, ortInputs, outNames)) { sw.Stop(); }
                    ortTotal += sw.Elapsed.TotalMilliseconds;
                    lokTotal += ProfileLokad(graph, named, opts, lokStages, lokOps, lokNodes, lokNodeOps);
                }
            }
            var mem = new
            {
                allocatedBytes = graph.LastAllocatedBytes,
                gc = graph.LastGcCollections.ToArray(),
                poolNew = graph.LastPoolAllocatedNew,
                poolNewBytes = graph.LastPoolAllocatedNewBytes,
                poolReused = graph.LastPoolReused,
                poolReusedBytes = graph.LastPoolReusedBytes,
                poolReturned = graph.LastPoolReturned,
                poolDropped = graph.LastPoolDropped,
                poolPeakOutstandingBytes = graph.LastPoolPeakOutstandingBytes,
                scratchBytes = graph.LastScratchBytes,
                copyBytes = graph.LastCopyBytes,
                peakLiveBytes = graph.LastPeakLiveBytes,
                retainedPackedWeightBytes = graph.RetainedPackedWeightBytes,
                workingSetBytes = System.Diagnostics.Process.GetCurrentProcess().WorkingSet64,
                gcTotalBytes = GC.GetTotalMemory(false),
            };
            string landed = session.EndProfiling();
            string traceFile = Path.Combine(outDir, kase + "-ort-trace.json");
            if (!string.Equals(landed, traceFile, StringComparison.OrdinalIgnoreCase))
            {
                if (File.Exists(traceFile)) File.Delete(traceFile);
                File.Move(landed, traceFile);
            }
            AggregateOrtTrace(traceFile, ortOps);
            var doc = new
            {
                kase,
                model,
                reps,
                gateScaled = gate.scaled,
                memory = mem,
                lokad = new { totalMs = lokTotal,
                    nodes = lokNodes.OrderByDescending(kv => kv.Value).Select(kv => new { id = kv.Key, op = lokNodeOps[kv.Key], ms = kv.Value }).ToArray(), perOp = lokOps.OrderByDescending(kv => kv.Value).ToDictionary(kv => kv.Key, kv => kv.Value), stages = lokStages.OrderByDescending(kv => kv.Value).ToDictionary(kv => kv.Key, kv => kv.Value) },
                ort = new { totalMs = ortTotal, perOp = ortOps.OrderByDescending(kv => kv.Value).ToDictionary(kv => kv.Key, kv => kv.Value), traceFile },
            };
            File.WriteAllText(Path.Combine(outDir, kase + "-profile.json"),
                JsonSerializer.Serialize(doc, new JsonSerializerOptions { WriteIndented = true }));
            Console.WriteLine("--- lokad per-op ms (profiled, " + reps + " reps) ---");
            foreach (var kv in lokOps.OrderByDescending(kv => kv.Value).Take(12))
                Console.WriteLine("  " + kv.Key + "=" + kv.Value.ToString("F1"));
            Console.WriteLine("--- ort per-op ms (trace, " + reps + " reps) ---");
            foreach (var kv in ortOps.OrderByDescending(kv => kv.Value).Take(12))
                Console.WriteLine("  " + kv.Key + "=" + kv.Value.ToString("F1"));
            Console.WriteLine("memory: peakLive=" + mem.peakLiveBytes + " poolPeakOut=" + mem.poolPeakOutstandingBytes + " scratch=" + mem.scratchBytes + " copy=" + mem.copyBytes + " retainedPacks=" + mem.retainedPackedWeightBytes + " allocMB=" + (mem.allocatedBytes / 1000000.0).ToString("F1"));
            Console.WriteLine("profile wrote " + Path.Combine(outDir, kase + "-profile.json"));
            return 0;
        }
        finally { foreach (var v in ortInputs.Values) v.Dispose(); }
    }

    static string E5Model(string root, out string tokenizer)
    {
        tokenizer = Path.Combine(root, "models", "multilingual-e5-small", "sentencepiece.bpe.model");
        return Path.Combine(root, "models", "multilingual-e5-small", "model.onnx");
    }

    static string Gpt2Model(string root) => Path.Combine(root, "models", "gpt2-onnx", "onnx", "model.onnx");

    static string E5Long(string root) => "query: " + string.Join(" ", Enumerable.Repeat(global::Bench.E5Sentence, 40));

    static double ProfileLokad(ComputationalGraph graph, Dictionary<string, ITensor> named, ExecutionOptions opts, Dictionary<string, double> stages, Dictionary<string, double> ops, Dictionary<long, double> nodes, Dictionary<long, string> nodeOps)
    {
        using (Profiler.BeginExecution(true))
        {
            graph.Reset();
            if (!graph.Execute(named, true, ExecutionProvider.CPU, opts))
                throw new InvalidOperationException("profiled execute failed.");
            graph.Reset();
        }
        double total = 0;
        if (graph.LastProfile is { } profile)
        {
            foreach (var np in profile)
            {
                string op = np.Op.ToString();
                double nodeMs = 0;
                foreach (var s in np.OpsProfile)
                {
                    double ms = s.Time.TotalMilliseconds;
                    nodeMs += ms;
                    string stage = s.Stage.ToString();
                    stages[stage] = stages.TryGetValue(stage, out var acc) ? acc + ms : ms;
                }
                ops[op] = ops.TryGetValue(op, out var oacc) ? oacc + nodeMs : nodeMs;
                nodes[np.NodeId] = nodes.TryGetValue(np.NodeId, out var nacc) ? nacc + nodeMs : nodeMs;
                nodeOps[np.NodeId] = op;
                total += nodeMs;
            }
        }
        return total;
    }

    static void AggregateOrtTrace(string traceFile, Dictionary<string, double> ops)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(traceFile));
        var events = doc.RootElement.ValueKind == JsonValueKind.Array
            ? doc.RootElement
            : doc.RootElement.GetProperty("traceEvents");
        int skipped = 0;
        foreach (var el in events.EnumerateArray())
        {
            if (!el.TryGetProperty("cat", out var kindEl) || kindEl.GetString() != "Node") { skipped++; continue; }
            if (!el.TryGetProperty("dur", out var durEl) || durEl.GetDouble() <= 0) { skipped++; continue; }
            string op = UnknownOp;
            if (el.TryGetProperty("cat", out var catEl) && catEl.GetString() == "Node"
                && el.TryGetProperty("args", out var argsEl) && argsEl.TryGetProperty("op_name", out var opEl))
                op = opEl.GetString() ?? UnknownOp;
            else if (el.TryGetProperty("name", out var nameEl) && nameEl.GetString() is { } nm && nm.Contains(ScopeSep))
                op = nm.Split(new[] { ScopeSep }, StringSplitOptions.None)[0];
            else { skipped++; continue; }
            double ms = durEl.GetDouble() / 1000.0;
            ops[op] = ops.TryGetValue(op, out var acc) ? acc + ms : ms;
        }
        Console.WriteLine("ort trace " + Path.GetFileName(traceFile) + ": priced " + ops.Count + " op families, skipped " + skipped + " events.");
    }

    static void SweepStrayTraces(DateTime startedUtc)
    {
        // ORT drops its trace in the working directory and rewrites a near-empty one on
        // session dispose; the real trace is relocated by ProfileCase. Delete our own duds.
        foreach (var f in Directory.GetFiles(Directory.GetCurrentDirectory(), "onnxruntime_profile_*.json"))
        {
            try { if (File.GetCreationTimeUtc(f) >= startedUtc) File.Delete(f); }
            catch (Exception ex) { Console.WriteLine("stray-trace sweep skipped " + f + ": " + ex.GetType().Name); }
        }
    }

    const string UnknownOp = "(unknown)";
    const string ScopeSep = "::";
}
