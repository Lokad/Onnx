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
        var disabledPasses = new List<string>();
        bool noPool = false;
        bool wall = false;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--out" && i + 1 < args.Length) { outDir = args[++i]; }
            else if (args[i] == "--cpu" && i + 1 < args.Length && int.TryParse(args[i + 1], out var c)) { cpu = c; i++; }
            else if (args[i] == "--reps" && i + 1 < args.Length && int.TryParse(args[i + 1], out var k) && k >= 1) { reps = k; i++; }
            else if (args[i] == "--disable-pass" && i + 1 < args.Length) { disabledPasses.Add(args[++i]); i++; }
            else if (args[i] == "--no-pool") { noPool = true; }
            else if (args[i] == "--wall") { wall = true; }
            else if (kase is null && !args[i].StartsWith("--")) kase = args[i];
            else { Console.WriteLine("usage: Bench profile <case> [--out dir] [--cpu N] [--reps K] [--disable-pass name]..."); return 2; }
        }
        if (kase is null) { Console.WriteLine("usage: Bench profile <case> [--out dir] [--cpu N] [--reps K]"); return 2; }
        // With startup affinity, ProcessorCount is the number of allowed CPUs,
        // not the largest OS CPU ID (taskset -c 2 legitimately reports one CPU).
        // EnforceSingleCpuAffinity below verifies the requested OS mask.
        if (cpu < 0 || cpu >= 64) { Console.WriteLine("invalid --cpu " + cpu); return 2; }
        var startedUtc = DateTime.UtcNow;
        var ownedTraces = new List<string>();
        if (disabledPasses.Count > 0) Console.WriteLine("profile: disabled passes=[" + string.Join(",", disabledPasses) + "]");
        Console.WriteLine("profile: mode=" + (wall ? "wall-clock nodes" : "instrumented stages"));
        Environment.SetEnvironmentVariable("LOKAD_ONNX_DISABLE_PASSES", string.Join(",", disabledPasses));
        try { int rc = ProfileCase(root, kase, outDir, cpu, reps, ownedTraces, noPool, wall); SweepStrayTraces(startedUtc, outDir, kase, ownedTraces); return rc; }
        catch (Exception ex) { Console.WriteLine(kase + ": profile failed: " + ex.GetType().Name + ": " + ex.Message.Split((char)10)[0]); SweepStrayTraces(startedUtc, outDir, kase, ownedTraces); return 1; }
        finally { Environment.SetEnvironmentVariable("LOKAD_ONNX_DISABLE_PASSES", null); }
    }

    static int ProfileCase(string root, string kase, string outDir, int cpu, int reps, List<string> ownedTraces, bool noPool, bool wall)
    {
        string model, tokenizer = "";
        Func<string, ITensor[]> build;
        int decodeLen = 0;
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
            case "gpt2-dec-p1": model = Gpt2Model(root); build = _ => global::Bench.Gpt2PrefillInputs(1, false); decodeLen = 1; break;
            case "gpt2-dec-p32": model = Gpt2Model(root); build = _ => global::Bench.Gpt2PrefillInputs(32, false); decodeLen = 32; break;
            case "gpt2-dec-p128": model = Gpt2Model(root); build = _ => global::Bench.Gpt2PrefillInputs(128, false); decodeLen = 128; break;
            case "gpt2-dec-p512": model = Gpt2Model(root); build = _ => global::Bench.Gpt2PrefillInputs(512, false); decodeLen = 512; break;
            case "dinov2-224": Console.WriteLine("dinov2-224 is diverged; profiling it would price incorrect execution. Refusing."); return 1;
            default:
                Console.WriteLine("unknown case " + kase); return 2;
        }
        if (!File.Exists(model)) { Console.WriteLine("missing asset: " + model); return 1; }
        global::Bench.EnforceSingleCpuAffinity(cpu);
        Directory.CreateDirectory(outDir);
        var tensorOpts = TensorExecutionOptions.Auto with { MaxDegreeOfParallelism = 1, DisableBufferPool = noPool };
        var opts = new ExecutionOptions(OptimizationMode.Speed, tensorOpts);
        var graph = OnnxImport.Load(model)!;
        graph.Prepare();
        using var so = global::Bench.CreateSingleCpuSessionOptions(1);
        so.EnableProfiling = true;
        so.ProfileOutputPathPrefix = Path.Combine(outDir, kase + "-ort-");
        using var session = new InferenceSession(model, so);
        // The agreement gate and decode prefill must not pollute the profiled trace:
        // ORT records every Run of a profiling session, so validation runs on an
        // identically configured session with profiling left disabled.
        using var valSession = new InferenceSession(model, global::Bench.CreateSingleCpuSessionOptions(1));
        var inNames = session.InputMetadata.Keys.ToArray();
        var outNames = session.OutputMetadata.Keys.ToArray();
        Dictionary<string, ITensor> named;
        if (decodeLen > 0)
        {
            var preNamed = global::Bench.ToNamed(kase + "-prefill", global::Bench.Gpt2PrefillInputs(decodeLen, false), inNames);
            var pre = global::Bench.Validate(kase + "-prefill", graph, valSession, preNamed, outNames, opts);
            Console.WriteLine("prefill " + kase + " tokens=" + decodeLen + " maxScaled=" + pre.scaled.ToString("E2") + " maxAbs=" + pre.abs.ToString("E2"));
            named = global::Bench.ToNamed(kase, global::Bench.BuildGpt2DecodeInputs(graph, decodeLen), inNames);
        }
        else
        {
            var built = build(kase);
            if (built.Length == 1 && string.IsNullOrEmpty(built[0].Name) && inNames.Length == 1) built[0].Name = inNames[0];
            named = global::Bench.ToNamed(kase, built, inNames);
        }
        var gate = global::Bench.Validate(kase, graph, valSession, named, outNames, opts);
        Console.WriteLine(kase + ": agreement gate maxScaled=" + gate.scaled.ToString("E2") + " (tol 1e-4); profiling " + reps + " reps per engine.");
        var ortInputs = global::Bench.BuildOrtInputs(named, inNames);
        try
        {
            var lokStages = new Dictionary<string, double>(StringComparer.Ordinal);
            var lokNodes = new Dictionary<long, double>();
            var lokNodeOps = new Dictionary<long, string>();
            var lokNodeCopies = new Dictionary<long, double>();
            var lokNodeCopyXs = new Dictionary<long, double>();
            var lokNodeCopyYs = new Dictionary<long, double>();
            var lokOps = new Dictionary<string, double>(StringComparer.Ordinal);
            var lokOpStages = new Dictionary<string, Dictionary<string, double>>(StringComparer.Ordinal);
            double lokTotal = 0;
            double lokWall = 0;
            var repLokad = new List<double>();
            var repLokadWall = new List<double>();
            var repPhases = new List<ProfilePhases>();
            var controlLokad = new List<double>();
            var controlOrt = new List<double>();
            var repLokadWallNodes = new List<double>();
            var wallNodes = new Dictionary<long, double>();
            var wallNodeOps = new Dictionary<long, string>();
            var wallOps = new Dictionary<string, double>(StringComparer.Ordinal);
            var repOrt = new List<double>();
            var repLokadOps = new List<Dictionary<string, double>>();
            var ortOps = new Dictionary<string, double>(StringComparer.Ordinal);
            double ortTotal = 0;
            var sw = new System.Diagnostics.Stopwatch();
            using var ro = new RunOptions();
            // Adjacent controls use the non-profiled ORT session and a disabled
            // Lokad profiler. These diagnose observer cost; they are not a scored
            // campaign (no convergence or four-process calibration here).
            void ControlLokad()
            {
                using var disabled = Profiler.BeginExecution(false);
                graph.Reset();
                long start = System.Diagnostics.Stopwatch.GetTimestamp();
                bool ok = graph.Execute(named, true, ExecutionProvider.CPU, opts);
                long end = System.Diagnostics.Stopwatch.GetTimestamp();
                if (!ok) throw new InvalidOperationException("unprofiled control execute failed.");
                controlLokad.Add(Milliseconds(start, end));
                graph.Reset();
            }
            void ControlOrt()
            {
                long start = System.Diagnostics.Stopwatch.GetTimestamp();
                using var result = valSession.Run(ro, ortInputs, outNames);
                long end = System.Diagnostics.Stopwatch.GetTimestamp();
                controlOrt.Add(Milliseconds(start, end));
            }
            for (int r = 0; r < reps; r++)
            {
                if (r % 2 == 0) { ControlLokad(); ControlOrt(); }
                else { ControlOrt(); ControlLokad(); }
                if (r % 2 == 0)
                {
                    var repStages = new Dictionary<string, double>(StringComparer.Ordinal);
                    var repOps = new Dictionary<string, double>(StringComparer.Ordinal);
                    var repOpStages = new Dictionary<string, Dictionary<string, double>>(StringComparer.Ordinal);
                    var repNodes = new Dictionary<long, double>();
                    var repNodeOps = new Dictionary<long, string>();
                    var repNodeCopies = new Dictionary<long, double>();
                    var repNodeCopyXs = new Dictionary<long, double>();
                    var repNodeCopyYs = new Dictionary<long, double>();
                    var lok = ProfileLokad(graph, named, opts, repStages, repOps, repNodes, repNodeOps, repOpStages, repNodeCopies, repNodeCopyXs, repNodeCopyYs, wall);
                    MergeProfile(repStages, repOps, repNodes, repNodeOps, repOpStages, repNodeCopies, repNodeCopyXs, repNodeCopyYs, lokStages, lokOps, lokNodes, lokNodeOps, lokOpStages, lokNodeCopies, lokNodeCopyXs, lokNodeCopyYs);
                    lokTotal += lok.nodeMs;
                    lokWall += lok.wallMs;
                    repLokad.Add(lok.nodeMs);
                    repLokadWall.Add(lok.wallMs);
                    repPhases.Add(lok.phases);
                    repLokadWallNodes.Add(lok.wallNodeMs);
                    foreach (var wb in lok.wallBreakdown)
                    {
                        wallNodes[wb.id] = wallNodes.TryGetValue(wb.id, out var wacc) ? wacc + wb.ms : wb.ms;
                        wallNodeOps[wb.id] = wb.op;
                        wallOps[wb.op] = wallOps.TryGetValue(wb.op, out var woacc) ? woacc + wb.ms : wb.ms;
                    }
                    repLokadOps.Add(repOps);
                    sw.Restart(); using (var o = session.Run(ro, ortInputs, outNames)) { sw.Stop(); }
                    ortTotal += sw.Elapsed.TotalMilliseconds;
                    repOrt.Add(sw.Elapsed.TotalMilliseconds);
                }
                else
                {
                    sw.Restart(); using (var o = session.Run(ro, ortInputs, outNames)) { sw.Stop(); }
                    ortTotal += sw.Elapsed.TotalMilliseconds;
                    repOrt.Add(sw.Elapsed.TotalMilliseconds);
                    var repStages = new Dictionary<string, double>(StringComparer.Ordinal);
                    var repOps = new Dictionary<string, double>(StringComparer.Ordinal);
                    var repOpStages = new Dictionary<string, Dictionary<string, double>>(StringComparer.Ordinal);
                    var repNodes = new Dictionary<long, double>();
                    var repNodeOps = new Dictionary<long, string>();
                    var repNodeCopies = new Dictionary<long, double>();
                    var repNodeCopyXs = new Dictionary<long, double>();
                    var repNodeCopyYs = new Dictionary<long, double>();
                    var lok = ProfileLokad(graph, named, opts, repStages, repOps, repNodes, repNodeOps, repOpStages, repNodeCopies, repNodeCopyXs, repNodeCopyYs, wall);
                    MergeProfile(repStages, repOps, repNodes, repNodeOps, repOpStages, repNodeCopies, repNodeCopyXs, repNodeCopyYs, lokStages, lokOps, lokNodes, lokNodeOps, lokOpStages, lokNodeCopies, lokNodeCopyXs, lokNodeCopyYs);
                    lokTotal += lok.nodeMs;
                    lokWall += lok.wallMs;
                    repLokad.Add(lok.nodeMs);
                    repLokadWall.Add(lok.wallMs);
                    repPhases.Add(lok.phases);
                    repLokadWallNodes.Add(lok.wallNodeMs);
                    foreach (var wb in lok.wallBreakdown)
                    {
                        wallNodes[wb.id] = wallNodes.TryGetValue(wb.id, out var wacc) ? wacc + wb.ms : wb.ms;
                        wallNodeOps[wb.id] = wb.op;
                        wallOps[wb.op] = wallOps.TryGetValue(wb.op, out var woacc) ? woacc + wb.ms : wb.ms;
                    }
                    repLokadOps.Add(repOps);
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
            ownedTraces.Add(landed);
            ownedTraces.Add(traceFile);
            double ortNodes = AggregateOrtTrace(traceFile, ortOps, reps);
            var doc = new
            {
                schema = 2,
                timingContract = "execute-only-v2",
                profileMode = wall ? "wall-nodes" : "instrumented-stages",
                interpretation = "Diagnostics only. Execute includes internal profiler collection; Reset, outer scope and report work are separate. Controls are unprofiled but not scored campaign evidence.",
                kase,
                model,
                reps,
                gateScaled = gate.scaled,
                repLokadMs = repLokad.ToArray(),
                repOrtMs = repOrt.ToArray(),
                repLokadExecuteMs = repLokadWall.ToArray(),
                repLokadPhases = repPhases.ToArray(),
                unprofiledControls = new { lokadExecuteMs = controlLokad.ToArray(), ortRunMs = controlOrt.ToArray() },
                repLokadWallNodesMs = repLokadWallNodes.ToArray(),
                repLokadPerOp = repLokadOps.Select(d => d.OrderByDescending(kv => kv.Value).ToDictionary(kv => kv.Key, kv => kv.Value)).ToArray(),
                memory = mem,
                lokad = new { totalMs = lokTotal, executeMs = lokWall, wallNodeMs = repLokadWallNodes.Sum(), wallNodes = wallNodes.OrderByDescending(kv => kv.Value).Select(kv => new { id = kv.Key, op = wallNodeOps[kv.Key], ms = kv.Value }).ToArray(), wallFamilies = wallOps.OrderByDescending(kv => kv.Value).ToDictionary(kv => kv.Key, kv => kv.Value),
                    nodes = lokNodes.OrderByDescending(kv => kv.Value).Select(kv => new { id = kv.Key, op = lokNodeOps[kv.Key], ms = kv.Value, copy = lokNodeCopies.TryGetValue(kv.Key, out var ncv) ? ncv : 0, copyX = lokNodeCopyXs.TryGetValue(kv.Key, out var ncx) ? ncx : 0, copyY = lokNodeCopyYs.TryGetValue(kv.Key, out var ncy) ? ncy : 0 }).ToArray(), perOp = lokOps.OrderByDescending(kv => kv.Value).ToDictionary(kv => kv.Key, kv => kv.Value), stages = lokStages.OrderByDescending(kv => kv.Value).ToDictionary(kv => kv.Key, kv => kv.Value), perOpStages = lokOpStages.ToDictionary(kv => kv.Key, kv => kv.Value.OrderByDescending(sv => sv.Value).ToDictionary(sv => sv.Key, sv => sv.Value)) },
                ort = new { totalMs = ortTotal, nodeMs = ortNodes, perOp = ortOps.OrderByDescending(kv => kv.Value).ToDictionary(kv => kv.Key, kv => kv.Value), traceFile },
            };
            File.WriteAllText(Path.Combine(outDir, kase + "-profile.json"),
                JsonSerializer.Serialize(doc, new JsonSerializerOptions { WriteIndented = true }));
            Console.WriteLine("--- lokad per-op ms (profiled, " + reps + " reps) ---");
            foreach (var kv in lokOps.OrderByDescending(kv => kv.Value).Take(12))
                Console.WriteLine("  " + kv.Key + "=" + kv.Value.ToString("F1"));
            Console.WriteLine("--- lokad Copy-stage ms by op (profiled) ---");
            foreach (var kv in lokOpStages.OrderByDescending(kv => (kv.Value.TryGetValue("Copy", out var c0) ? c0 : 0) + (kv.Value.TryGetValue("CopyX", out var c1) ? c1 : 0) + (kv.Value.TryGetValue("CopyY", out var c2) ? c2 : 0)).Take(12))
            {
                double cx = kv.Value.TryGetValue("CopyX", out var vx) ? vx : 0;
                double cy = kv.Value.TryGetValue("CopyY", out var vy) ? vy : 0;
                double cc = kv.Value.TryGetValue("Copy", out var vc) ? vc : 0;
                Console.WriteLine("  " + kv.Key + " copy=" + cc.ToString("F1") + " copyX=" + cx.ToString("F1") + " copyY=" + cy.ToString("F1"));
            }
            Console.WriteLine("--- ort per-op ms (trace, " + reps + " reps) ---");
            foreach (var kv in ortOps.OrderByDescending(kv => kv.Value).Take(12))
                Console.WriteLine("  " + kv.Key + "=" + kv.Value.ToString("F1"));
            Console.WriteLine("reps lokad=[" + string.Join(",", repLokad.Select(v => v.ToString("F1"))) + "] ort=[" + string.Join(",", repOrt.Select(v => v.ToString("F1"))) + "]");
            Console.WriteLine("reps lokadwallnodes=[" + string.Join(",", repLokadWallNodes.Select(v => v.ToString("F1"))) + "]");
            Console.WriteLine("wall closure nodes/execute=[" + string.Join(",", repLokadWallNodes.Zip(repLokadWall, (a, b) => (a / Math.Max(b, 1e-9)).ToString("F3"))) + "]");
            Console.WriteLine("--- lokad per-op ms (wall, " + reps + " reps) ---");
            foreach (var kv in wallOps.OrderByDescending(kv => kv.Value).Take(12))
                Console.WriteLine("  " + kv.Key + "=" + kv.Value.ToString("F1"));
            Console.WriteLine("totals: lokad nodes=" + lokTotal.ToString("F1") + " execute=" + lokWall.ToString("F1") + "; ort nodes=" + ortNodes.ToString("F1") + " profiledRun=" + ortTotal.ToString("F1"));
            Console.WriteLine("--- lokad top Copy nodes (profiled, id op copyMs totalMs name) ---");
            foreach (var kv in lokNodeCopies.OrderByDescending(kv => kv.Value).Take(10))
            {
                string nm = (kv.Key >= 0 && kv.Key < graph.Nodes.Count) ? (graph.Nodes[(int)kv.Key].Name ?? "") : "";
                double tot = lokNodes.TryGetValue(kv.Key, out var tm) ? tm : 0;
                string oo = lokNodeOps.TryGetValue(kv.Key, out var ooo) ? ooo : "?";
                double txx = lokNodeCopyXs.TryGetValue(kv.Key, out var vxx) ? vxx : 0;
                double tyy = lokNodeCopyYs.TryGetValue(kv.Key, out var vyy) ? vyy : 0;
                Console.WriteLine("  id=" + kv.Key + " op=" + oo + " copy=" + kv.Value.ToString("F1") + " X=" + txx.ToString("F1") + " Y=" + tyy.ToString("F1") + " total=" + tot.ToString("F1") + " " + nm);
            }
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

    static void MergeProfile(Dictionary<string, double> stages, Dictionary<string, double> ops, Dictionary<long, double> nodes, Dictionary<long, string> nodeOps, Dictionary<string, Dictionary<string, double>> opStages, Dictionary<long, double> nodeCopies, Dictionary<long, double> nodeCopyXs, Dictionary<long, double> nodeCopyYs,
        Dictionary<string, double> allStages, Dictionary<string, double> allOps, Dictionary<long, double> allNodes, Dictionary<long, string> allNodeOps, Dictionary<string, Dictionary<string, double>> allOpStages, Dictionary<long, double> allNodeCopies, Dictionary<long, double> allNodeCopyXs, Dictionary<long, double> allNodeCopyYs)
    {
        foreach (var kv in stages) allStages[kv.Key] = allStages.TryGetValue(kv.Key, out var a) ? a + kv.Value : kv.Value;
        foreach (var kv in ops) allOps[kv.Key] = allOps.TryGetValue(kv.Key, out var b) ? b + kv.Value : kv.Value;
        foreach (var kv in nodes) allNodes[kv.Key] = allNodes.TryGetValue(kv.Key, out var c) ? c + kv.Value : kv.Value;
        foreach (var kv in nodeOps) allNodeOps[kv.Key] = kv.Value;
        foreach (var kv in nodeCopies) allNodeCopies[kv.Key] = allNodeCopies.TryGetValue(kv.Key, out var dd) ? dd + kv.Value : kv.Value;
        foreach (var kv in nodeCopyXs) allNodeCopyXs[kv.Key] = allNodeCopyXs.TryGetValue(kv.Key, out var dx) ? dx + kv.Value : kv.Value;
        foreach (var kv in nodeCopyYs) allNodeCopyYs[kv.Key] = allNodeCopyYs.TryGetValue(kv.Key, out var dy) ? dy + kv.Value : kv.Value;
        foreach (var kv in opStages)
        {
            if (!allOpStages.TryGetValue(kv.Key, out var inner)) { inner = new Dictionary<string, double>(StringComparer.Ordinal); allOpStages[kv.Key] = inner; }
            foreach (var sv in kv.Value) inner[sv.Key] = inner.TryGetValue(sv.Key, out var b) ? b + sv.Value : sv.Value;
        }
    }

    sealed record ProfilePhases(double ResetBeforeMs, double ScopeSetupMs, double ExecuteMs,
        double ResetAfterMs, double ScopeDisposeMs, double ReportAggregationMs);

    static double Milliseconds(long start, long end) => (end - start) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

    static (double nodeMs, double wallMs, double wallNodeMs, List<(long id, string op, double ms)> wallBreakdown, ProfilePhases phases) ProfileLokad(ComputationalGraph graph, Dictionary<string, ITensor> named, ExecutionOptions opts, Dictionary<string, double> stages, Dictionary<string, double> ops, Dictionary<long, double> nodes, Dictionary<long, string> nodeOps, Dictionary<string, Dictionary<string, double>> opStages, Dictionary<long, double> nodeCopies, Dictionary<long, double> nodeCopyXs, Dictionary<long, double> nodeCopyYs, bool wallOnly)
    {
        long beforeReset = System.Diagnostics.Stopwatch.GetTimestamp();
        graph.Reset();
        long afterReset = System.Diagnostics.Stopwatch.GetTimestamp();
        var scope = wallOnly ? Profiler.BeginWallExecution() : Profiler.BeginExecution(true);
        long beforeExecute = System.Diagnostics.Stopwatch.GetTimestamp();
        long afterExecute, afterCleanup;
        try
        {
            bool ok = graph.Execute(named, true, ExecutionProvider.CPU, opts);
            afterExecute = System.Diagnostics.Stopwatch.GetTimestamp();
            if (!ok) throw new InvalidOperationException("profiled execute failed.");
            graph.Reset();
            afterCleanup = System.Diagnostics.Stopwatch.GetTimestamp();
        }
        finally { scope.Dispose(); }
        long afterScope = System.Diagnostics.Stopwatch.GetTimestamp();
        var breakdown = new List<(long id, string op, double ms)>();
        double total = 0;
        double wallNodeMs = 0;
        if (wallOnly)
        {
            if (graph.LastWallProfile is { } wprof)
            {
                double freq = (double)System.Diagnostics.Stopwatch.Frequency;
                foreach (var wn in wprof)
                {
                    double wms = (wn.EndTicks - wn.StartTicks) * 1000.0 / freq;
                    breakdown.Add((wn.NodeId, wn.Op.ToString(), wms));
                    wallNodeMs += wms;
                }
            }
        }
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
                if (!opStages.TryGetValue(op, out var osp)) { osp = new Dictionary<string, double>(StringComparer.Ordinal); opStages[op] = osp; }
                foreach (var s in np.OpsProfile)
                {
                    string st2 = s.Stage.ToString();
                    double m2 = s.Time.TotalMilliseconds;
                    osp[st2] = osp.TryGetValue(st2, out var oacc2) ? oacc2 + m2 : m2;
                }
                nodes[np.NodeId] = nodes.TryGetValue(np.NodeId, out var nacc) ? nacc + nodeMs : nodeMs;
                double nodeCopy = 0;
                foreach (var sc in np.OpsProfile)
                {
                    string stc = sc.Stage.ToString();
                    if (stc == "Copy" || stc == "CopyX" || stc == "CopyY") nodeCopy += sc.Time.TotalMilliseconds;
                }
                if (nodeCopy > 0) nodeCopies[np.NodeId] = nodeCopies.TryGetValue(np.NodeId, out var ncacc) ? ncacc + nodeCopy : nodeCopy;
                double nodeCopyX = 0, nodeCopyY = 0;
                foreach (var sc2 in np.OpsProfile)
                {
                    string st4 = sc2.Stage.ToString();
                    if (st4 == "CopyX") nodeCopyX += sc2.Time.TotalMilliseconds;
                    else if (st4 == "CopyY") nodeCopyY += sc2.Time.TotalMilliseconds;
                }
                if (nodeCopyX > 0) nodeCopyXs[np.NodeId] = nodeCopyXs.TryGetValue(np.NodeId, out var nx) ? nx + nodeCopyX : nodeCopyX;
                if (nodeCopyY > 0) nodeCopyYs[np.NodeId] = nodeCopyYs.TryGetValue(np.NodeId, out var ny) ? ny + nodeCopyY : nodeCopyY;
                nodeOps[np.NodeId] = op;
                total += nodeMs;
            }
        }
        long afterReport = System.Diagnostics.Stopwatch.GetTimestamp();
        var phases = new ProfilePhases(Milliseconds(beforeReset, afterReset), Milliseconds(afterReset, beforeExecute),
            Milliseconds(beforeExecute, afterExecute), Milliseconds(afterExecute, afterCleanup),
            Milliseconds(afterCleanup, afterScope), Milliseconds(afterScope, afterReport));
        return (total, phases.ExecuteMs, wallNodeMs, breakdown, phases);
    }

    static double AggregateOrtTrace(string traceFile, Dictionary<string, double> ops, int reps)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(traceFile));
        var events = doc.RootElement.ValueKind == JsonValueKind.Array
            ? doc.RootElement
            : doc.RootElement.GetProperty("traceEvents");
        int skipped = 0;
        int modelRuns = 0;
        foreach (var el in events.EnumerateArray())
        {
            if (el.TryGetProperty("name", out var mrEl) && mrEl.GetString() == "model_run") modelRuns++;
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
        double priced = 0;
        foreach (var v in ops.Values) priced += v;
        Console.WriteLine("ort trace " + Path.GetFileName(traceFile) + ": priced " + ops.Count + " op families, skipped " + skipped + " events, model_runs=" + modelRuns + " (reps=" + reps + ").");
        if (modelRuns != reps)
            throw new InvalidOperationException("ORT trace holds " + modelRuns + " model_run events, expected " + reps + "; the profiled session observed a stray run.");
        return priced;
    }

    static void SweepStrayTraces(DateTime startedUtc, string outDir, string kase, List<string> owned)
    {
        // Ownership rule: recorded EndProfiling identities are exact and always kept.
        // Anything else under our case prefix is deleted only when it is provably a
        // dispose-time byproduct: born during this run AND holding zero Node events
        // (ORT emits a near-empty trace on session dispose). A same-prefix file with
        // real Node content is a foreign run sharing our directory: kept with a loud
        // warning, never priced, never deleted. Creation time alone proves nothing.
        // Working-directory traces are only reported, never deleted: a concurrent
        // process may own them, and creation time cannot tell owners apart.
        var keep = new HashSet<string>(owned, StringComparer.OrdinalIgnoreCase);
        string prefix = kase + "-ort-";
        foreach (var f in Directory.GetFiles(outDir, prefix + "*.json"))
        {
            try
            {
                if (keep.Contains(f)) continue;
                bool bornHere = File.GetCreationTimeUtc(f) >= startedUtc;
                int nodes = bornHere ? CountNodeEvents(f) : -1;
                if (bornHere && nodes == 0) { File.Delete(f); Console.WriteLine("stray-trace removed (empty dispose byproduct): " + f); }
                else Console.WriteLine("stray-trace kept (unowned content, nodes=" + nodes + "): " + f);
            }
            catch (Exception ex) { Console.WriteLine("stray-trace sweep skipped " + f + ": " + ex.GetType().Name); }
        }
        foreach (var f in Directory.GetFiles(Directory.GetCurrentDirectory(), "onnxruntime_profile_*.json"))
        {
            try
            {
                if (File.GetCreationTimeUtc(f) >= startedUtc)
                    Console.WriteLine("stray-trace kept (working directory, owner unknown): " + f);
            }
            catch (Exception ex) { Console.WriteLine("stray-trace sweep skipped " + f + ": " + ex.GetType().Name); }
        }
    }

    static int CountNodeEvents(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var events = doc.RootElement.ValueKind == JsonValueKind.Array
            ? doc.RootElement
            : (doc.RootElement.TryGetProperty("traceEvents", out var te) ? te : default);
        if (events.ValueKind != JsonValueKind.Array) return -1;
        int n = 0;
        foreach (var el in events.EnumerateArray())
            if (el.TryGetProperty("cat", out var kindEl) && kindEl.GetString() == "Node") n++;
        return n;
    }

    const string UnknownOp = "(unknown)";
    const string ScopeSep = "::";
}
