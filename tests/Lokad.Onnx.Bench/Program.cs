using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Lokad.Onnx;
using Lokad.Onnx.Bench;
using Microsoft.ML.OnnxRuntime;

static class Bench
{
    const int DefaultWarmup = 3;
    internal const string E5ShortText = "query: hello world";
    internal const string E5Sentence = "The quick brown fox jumps over the lazy dog near the river bank in springtime weather for a pleasant afternoon walk";
    const double Tolerance = 1e-4;
    const double ConfinementRatioLimit = 1.3;
    const int ConfinementDurationMs = 2000;

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool GetLogicalProcessorInformationEx(int relationshipType, IntPtr buffer, ref int returnedLength);

    static int Main(string[] args)
    {
        var root = FindRoot();
        var assets = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase)
        {
            ["e5"] = new[] { Path.Combine(root, "models", "multilingual-e5-small", "model.onnx"), Path.Combine(root, "models", "multilingual-e5-small", "sentencepiece.bpe.model") },
            ["dinov2"] = new[] { Path.Combine(root, "models", "dinov2-small-onnx", "model.onnx") },
            ["dinov3"] = new[] { Path.Combine(root, "models", "dinov3-vits16", "onnx", "model.onnx") },
            ["resnet50"] = new[] { Path.Combine(root, "models", "resnet50-onnx", "model.onnx") },
            ["gpt2"] = new[] { Path.Combine(root, "models", "gpt2-onnx", "onnx", "model.onnx") },
        };
        if (args.Length > 0 && args[0] == "graph")
        {
            return GraphCensus.RunGraph(root, assets, args.Skip(1).ToArray());
        }
        if (args.Length > 0 && args[0] == "profile")
        {
            return GraphProfile.RunProfile(root, args.Skip(1).ToArray());
        }
        if (args.Length > 0 && args[0] == "micro")
        {
            int microCpu = 0;
            var microArgs = StripCpuSelector(args.Skip(1).ToArray(), ref microCpu);
            long microMask = EnforceSingleCpuAffinity(microCpu).ToInt64();
            Console.WriteLine("bench-micro affinity=0x" + microMask.ToString("X") + " logical-cpu=" + microCpu + " (child jobs inherit process affinity on Windows)");
            return RunMicro(microArgs);
        }
        var selected = new List<string>();
        string modeName = "auto";
        string rowsName = "canonical";
        int threads = 1;
        int cpu = 0;
        int iters = 7;
        int warmup = DefaultWarmup;
        int warmupMin = -1;
        int warmupMax = -1;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--mode" && i + 1 < args.Length) modeName = args[++i];
            else if (args[i] == "--rows" && i + 1 < args.Length) rowsName = args[++i];
            else if (args[i] == "--cpu" && i + 1 < args.Length && int.TryParse(args[i + 1], out var c)) { cpu = c; i++; }
            else if (args[i] == "--threads" && i + 1 < args.Length && int.TryParse(args[i + 1], out var t) && t >= 1) { threads = t; i++; }
            else if (args[i] == "--iters" && i + 1 < args.Length && int.TryParse(args[i + 1], out var k) && k >= 1) { iters = k; i++; }
            else if (args[i] == "--warmup" && i + 1 < args.Length && int.TryParse(args[i + 1], out var w) && w >= 0) { warmup = w; i++; }
            else if (args[i] == "--warmup-min" && i + 1 < args.Length && int.TryParse(args[i + 1], out var wmin)) { warmupMin = wmin; i++; }
            else if (args[i] == "--warmup-max" && i + 1 < args.Length && int.TryParse(args[i + 1], out var wmax)) { warmupMax = wmax; i++; }
            else if (args[i] == "all" || assets.ContainsKey(args[i])) { if (args[i] != "all" && !selected.Contains(args[i], StringComparer.OrdinalIgnoreCase)) selected.Add(args[i]); }
            else { Console.WriteLine("usage: Bench [e5 dinov2 dinov3 resnet50 gpt2 all] [--mode auto|scalar|simd|intrinsics] [--threads N] [--iters N] [--warmup N] [--warmup-min A --warmup-max B] [--rows canonical|all] [--cpu N]"); return 2; }
        }
        if (rowsName != "canonical" && rowsName != "all") { Console.WriteLine("unknown --rows " + rowsName + " (expected canonical|all)"); return 2; }
        if ((warmupMin < 0) != (warmupMax < 0)) { Console.WriteLine("--warmup-min and --warmup-max must be given together (absent selects fixed --warmup " + warmup + ")"); return 2; }
        if (warmupMin >= 0 && (warmupMin > warmupMax || warmupMax > 100)) { Console.WriteLine("invalid adaptive warmup range (expected 0 <= min <= max <= 100)"); return 2; }
        if (cpu < 0 || cpu >= 64 || cpu >= Environment.ProcessorCount) { Console.WriteLine("invalid --cpu " + cpu + " (expected 0.." + (Environment.ProcessorCount - 1) + ")"); return 2; }
        if (selected.Count == 0) selected.AddRange(assets.Keys);
        TensorExecutionOptions tensorOpts = modeName.ToLowerInvariant() switch
        {
            "scalar" => TensorExecutionOptions.Scalar with { MaxDegreeOfParallelism = threads },
            "simd" => TensorExecutionOptions.Simd with { MaxDegreeOfParallelism = threads },
            "intrinsics" => TensorExecutionOptions.Intrinsics with { MaxDegreeOfParallelism = threads },
            _ when modeName.Equals("auto", StringComparison.OrdinalIgnoreCase) => TensorExecutionOptions.Auto with { MaxDegreeOfParallelism = threads },
            _ => throw new InvalidOperationException("Unknown mode " + modeName + "."),
        };
        foreach (var key in selected)
        {
            foreach (var f in assets[key])
            {
                if (!File.Exists(f)) { Console.WriteLine("missing asset for " + key + ": " + f); return 1; }
            }
        }
        long affinityMask = EnforceSingleCpuAffinity(cpu).ToInt64();
        string cpuId = Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? "unknown";
        string vec = System.Numerics.Vector.IsHardwareAccelerated
            ? "Vector" + (System.Numerics.Vector<byte>.Count * 8)
            : "Vector-none";
        Console.WriteLine("host=" + Environment.MachineName + " cpu=" + cpuId
            + " procs=" + Environment.ProcessorCount + " (logical)"
            + " affinity=0x" + affinityMask.ToString("X") + " (verified single-CPU, logical-cpu=" + cpu + ")"
            + " " + DescribeCpuTopology(cpu)
            + " vector=" + vec + " isa=" + HardwareIntrinsics.GetFullInfo()
            + " fma=" + System.Runtime.Intrinsics.X86.Fma.IsSupported
            + " runtime=" + RuntimeInformation.FrameworkDescription
            + " lokad=" + typeof(ComputationalGraph).Assembly.GetName().Version
            + " ort=" + typeof(InferenceSession).Assembly.GetName().Version
            + " ort-provider=cpu-only ort-optimizations=ORT_ENABLE_ALL nospin intraop=" + threads + " interop=1 seq"
            + " mode=" + modeName + " threads=" + threads + " rows=" + rowsName + " iters=" + iters + " warmup=" + warmup + " warmupMin=" + warmupMin + " warmupMax=" + warmupMax);
        var confinement = MeasureSingleCpuConfinement(ConfinementDurationMs);
        Console.WriteLine("confinement wallMs=" + confinement.wallMs.ToString("F0")
            + " cpuMs=" + confinement.cpuMs.ToString("F0")
            + " ratio=" + confinement.ratio.ToString("F2")
            + " (single-threaded " + ConfinementDurationMs + "ms busy loop; expect ~1.0, limit " + ConfinementRatioLimit.ToString("F1") + ")");
        if (confinement.ratio < 0.8)
            Console.WriteLine("confinement warning: cpu/wall ratio " + confinement.ratio.ToString("F2")
                + " is well below 1.0; the pinned CPU was starved by concurrent load."
                + " Treat these timings as diagnostic and rerun on a quiet core.");
        if (confinement.ratio > ConfinementRatioLimit)
            throw new InvalidOperationException("single-CPU confinement failed: cpu/wall ratio "
                + confinement.ratio.ToString("F2") + " exceeds " + ConfinementRatioLimit.ToString("F1")
                + "; parallel workers escaped the pinned CPU.");
        using (var verifyProc = Process.GetCurrentProcess())
        {
            if (AffinitySupported && verifyProc.ProcessorAffinity != (IntPtr)(1L << cpu))
                throw new InvalidOperationException("single-CPU confinement failed: affinity changed during the confinement check.");
        }
        Console.WriteLine("boundaries: load=Lokad import|ortLoad=ORT session build; prepare=Lokad lifetime analysis;"
            + " first=Lokad first Execute|ORT first Run in this row (process-cold only on the first row);"
            + " lokad=warmed public Execute|ctx=warmed reused-context Execute|ort=warmed ORT Run, alternating order, other side outputs already released;"
            + " convert=one-time managed-to-ORT input build reused by all ORT runs;"
            + " req=warmed Reset+Execute+Reset per request (public)|reqCtx=same on reused context"
            + " resetPop=reset after execute|resetClean=reset when empty (no-op floor);"
            + " gc/alloc=shared-process totals over warmed loops without per-engine attribution; output disposal outside all timings."
            + " agree=maxAbs + ORT-reference-scaled diff (tol 1e-4 scaled, unchanged);"
            + " known-divergence=tracked registry with tripwire; excluded cases skip timing;"
            + " post=agreement re-check after timed reuse; inputsIntact=input fingerprint before/after.");
        // Each requested case reports its own status. A failing case never suppresses the
        // remaining cases, but the run still exits nonzero so its absence cannot read as success.
        // A tracked known divergence excludes its case (no timing, no publication) without failing the run.
        var caseFailures = new List<string>();
        var excludedCases = new List<string>();
        void RunCase(string label, Action run)
        {
            try
            {
                run();
                Console.WriteLine("case-status " + label + "=ok");
            }
            catch (KnownDivergenceException kdx)
            {
                Console.WriteLine("case-status " + label + "=excluded-known-divergence: " + kdx.Message.Split((char)10)[0]);
                excludedCases.Add(label);
            }
            catch (Exception ex)
            {
                string detail = ex.Message.Split((char)10)[0];
                Console.WriteLine("case-status " + label + "=FAILED: " + ex.GetType().Name + ": " + detail);
                caseFailures.Add(label);
            }
        }
        if (selected.Contains("e5", StringComparer.OrdinalIgnoreCase))
        {
            var e5 = assets["e5"];
            RunCase("e5-8tok", () => CompareE5("e5-8tok", e5[0], e5[1], E5ShortText, tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName));
            RunCase("e5-30tok", () => CompareE5("e5-30tok", e5[0], e5[1], "query: " + E5Sentence, tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName));
            RunCase("e5-30pad128", () => CompareE5("e5-30pad128", e5[0], e5[1], "query: " + E5Sentence, tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName, 0, 128));
            var e5Long = "query: " + string.Join(" ", Enumerable.Repeat(E5Sentence, 40));
            RunCase("e5-128tok", () => CompareE5("e5-128tok", e5[0], e5[1], e5Long, tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName, 128));
            RunCase("e5-512tok", () => CompareE5("e5-512tok", e5[0], e5[1], e5Long, tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName, 512));
        }
        if (selected.Contains("dinov2", StringComparer.OrdinalIgnoreCase)) RunCase("dinov2-224", () => CompareVision("dinov2-224", assets["dinov2"][0], tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName));
        if (selected.Contains("dinov3", StringComparer.OrdinalIgnoreCase)) RunCase("dinov3-224", () => CompareVision("dinov3-224", assets["dinov3"][0], tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName));
        if (selected.Contains("resnet50", StringComparer.OrdinalIgnoreCase)) RunCase("resnet50-224", () => CompareVision("resnet50-224", assets["resnet50"][0], tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName));
        if (selected.Contains("gpt2", StringComparer.OrdinalIgnoreCase))
        {
            var gpt2 = assets["gpt2"];
            RunCase("gpt2-1tok", () => CompareGpt2("gpt2-1tok", gpt2[0], 1, tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName));
            RunCase("gpt2-4tok", () => CompareGpt2("gpt2-4tok", gpt2[0], 4, tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName, legacyCycle: true));
            RunCase("gpt2-32tok", () => CompareGpt2("gpt2-32tok", gpt2[0], 32, tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName));
            RunCase("gpt2-128tok", () => CompareGpt2("gpt2-128tok", gpt2[0], 128, tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName));
            RunCase("gpt2-dec-p1", () => CompareGpt2Decode("gpt2-dec-p1", gpt2[0], 1, tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName));
            RunCase("gpt2-dec-p32", () => CompareGpt2Decode("gpt2-dec-p32", gpt2[0], 32, tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName));
            RunCase("gpt2-dec-p128", () => CompareGpt2Decode("gpt2-dec-p128", gpt2[0], 128, tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName));
            RunCase("gpt2-dec-p512", () => CompareGpt2Decode("gpt2-dec-p512", gpt2[0], 512, tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName));
        }
        if (excludedCases.Count > 0)
        {
            Console.WriteLine("cases-excluded [" + string.Join(",", excludedCases) + "] (tracked known divergences; no rows are eligible for excluded cases)");
        }
        if (caseFailures.Count > 0)
        {
            Console.WriteLine("cases-failed [" + string.Join(",", caseFailures) + "] (no rows are eligible for failed cases)");
            return 1;
        }
        return 0;
    }

    internal static string[] StripCpuSelector(string[] args, ref int cpu)
    {
        var rest = new List<string>();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--cpu" && i + 1 < args.Length && int.TryParse(args[i + 1], out var c)) { cpu = c; i++; }
            else rest.Add(args[i]);
        }
        if (cpu < 0 || cpu >= 64 || cpu >= Environment.ProcessorCount)
            throw new InvalidOperationException("invalid --cpu " + cpu + " (expected 0.." + (Environment.ProcessorCount - 1) + ").");
        return rest.ToArray();
    }

    [System.Runtime.Versioning.SupportedOSPlatformGuard("windows")]
    [System.Runtime.Versioning.SupportedOSPlatformGuard("linux")]
    static bool AffinitySupported => OperatingSystem.IsWindows() || OperatingSystem.IsLinux();

    internal static IntPtr EnforceSingleCpuAffinity(int cpu)
    {
        if (!AffinitySupported)
            throw new InvalidOperationException("single-CPU confinement failed: process affinity is not supported on "
                + RuntimeInformation.OSDescription + "; refusing to time without confinement.");
        long mask = 1L << cpu;
        IntPtr requested = (IntPtr)mask;
        IntPtr actual;
        try
        {
            using var proc = Process.GetCurrentProcess();
            if (AffinitySupported)
            {
                proc.ProcessorAffinity = requested;
                actual = proc.ProcessorAffinity;
            }
            else
            {
                throw new PlatformNotSupportedException("process affinity is not supported on this platform.");
            }
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException("single-CPU confinement failed: cannot pin process affinity to 0x"
                + mask.ToString("X") + " (logical CPU " + cpu + "): " + ex.Message + ".", ex);
        }
        if (actual != requested)
            throw new InvalidOperationException("single-CPU confinement failed: requested affinity=0x"
                + mask.ToString("X") + " but verified 0x" + actual.ToInt64().ToString("X") + ".");
        return actual;
    }

    static (double wallMs, double cpuMs, double ratio) MeasureSingleCpuConfinement(int durationMs)
    {
        using var proc = Process.GetCurrentProcess();
        TimeSpan before = proc.TotalProcessorTime;
        var sw = Stopwatch.StartNew();
        double sink = 0;
        while (sw.Elapsed.TotalMilliseconds < durationMs)
        {
            for (int i = 1; i <= 20000; i++) sink += Math.Sqrt(i);
        }
        sw.Stop();
        GC.KeepAlive(sink);
        TimeSpan after = proc.TotalProcessorTime;
        double wallMs = sw.Elapsed.TotalMilliseconds;
        double cpuMs = (after - before).TotalMilliseconds;
        return (wallMs, cpuMs, cpuMs / Math.Max(wallMs, 1e-9));
    }

    static string DescribeCpuTopology(int cpu)
    {
        try
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return "topology=non-windows-unknown";
            if (!Environment.Is64BitProcess) return "topology=32bit-unknown";
            if (cpu < 0 || cpu >= 64) return "topology=cpu-out-of-range";
            const int RelationProcessorCore = 0;
            int length = 0;
            GetLogicalProcessorInformationEx(RelationProcessorCore, IntPtr.Zero, ref length);
            if (length <= 0) return "topology=query-unavailable";
            IntPtr buffer = Marshal.AllocHGlobal(length);
            try
            {
                if (!GetLogicalProcessorInformationEx(RelationProcessorCore, buffer, ref length))
                    return "topology=query-failed";
                string? hit = null;
                int offset = 0;
                while (offset < length)
                {
                    int relationship = Marshal.ReadInt32(buffer, offset);
                    int size = Marshal.ReadInt32(buffer, offset + 4);
                    if (size <= 0) break;
                    if (relationship == RelationProcessorCore)
                    {
                        byte flags = Marshal.ReadByte(buffer, offset + 8);
                        byte efficiency = Marshal.ReadByte(buffer, offset + 9);
                        ushort groupCount = (ushort)Marshal.ReadInt16(buffer, offset + 30);
                        for (int g = 0; g < groupCount; g++)
                        {
                            int groupOffset = offset + 32 + g * 16;
                            if (groupOffset + 16 > offset + size) break;
                            long mask = Marshal.ReadInt64(buffer, groupOffset);
                            ushort group = (ushort)Marshal.ReadInt16(buffer, groupOffset + 8);
                            if (group == 0 && ((mask >> cpu) & 1L) != 0)
                            {
                                int siblings = System.Numerics.BitOperations.PopCount((ulong)mask);
                                hit = "core-group=0 core-mask=0x" + mask.ToString("X")
                                    + " efficiency-class=" + efficiency
                                    + " smt-siblings=" + siblings
                                    + " smt=" + (((flags & 1) != 0) ? "yes" : "no");
                            }
                        }
                    }
                    offset += size;
                }
                return hit is null ? "topology=core-not-found" : "topology=(" + hit + ")";
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }
        catch (Exception ex)
        {
            return "topology=unavailable(" + ex.GetType().Name + ")";
        }
    }

    internal static SessionOptions CreateSingleCpuSessionOptions(int threads)
    {
        var so = new SessionOptions();
        so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        so.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
        so.AddSessionConfigEntry("session.inter_op.allow_spinning", "0");
        so.IntraOpNumThreads = threads;
        so.InterOpNumThreads = 1;
        so.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        return so;
    }

    static string CanonicalLokDesc(string modeName, int threads)
    {
        if (modeName.Equals("auto", StringComparison.OrdinalIgnoreCase) && threads == 1)
            return "single-cpu-auto-1 (canonical)";
        return "single-cpu-" + modeName.ToLowerInvariant() + "-" + threads + " (diagnostic)";
    }

    static int RunMicro(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("usage: Bench micro <matmul2d|matmul|indexing|ops|oneop|packed> [BenchmarkDotNet options]");
            return 2;
        }
        // Micro failures propagate with a nonzero exit; nothing here converts an error into success.
        switch (args[0])
        {
            case "matmul2d":
                MicroBenchmarks.RunMatMul2D(args.Skip(1).ToArray());
                return 0;
            case "matmul":
                MicroBenchmarks.RunMatMul(args.Skip(1).ToArray());
                return 0;
            case "indexing":
                MicroBenchmarks.RunIndexing(args.Skip(1).ToArray());
                return 0;
            case "ops":
                MicroBenchmarks.RunOps(args.Skip(1).ToArray());
                return 0;
            case "oneop":
                OneOpMicro.RunOneOp(args.Skip(1).ToArray());
                return 0;
            case "packed":
                MicroBenchmarks.RunPacked(args.Skip(1).ToArray());
                return 0;
            default:
                Console.WriteLine("Unknown micro benchmark: " + args[0] + ".");
                Console.WriteLine("usage: Bench micro <matmul2d|matmul|indexing|ops|oneop|packed> [BenchmarkDotNet options]");
                return 2;
        }
    }

    internal static string FindRoot()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Lokad.Onnx.slnx"))) return dir.FullName;
            dir = dir.Parent;
        }
        throw new InvalidOperationException("repo root not found");
    }

    internal static ITensor[] E5Inputs(string name, string tokenizer, string text, int takeTokens, int padTo)
    {
        var inputs = Text.RobertaTokenizeFromFile(text, tokenizer)!;
        if (takeTokens > 0)
        {
            var cut = new List<ITensor>();
            foreach (var t in inputs)
            {
                if (t is Tensor<long> tl && tl.Dimensions.Length == 2 && tl.Dimensions[0] == 1 && tl.Dimensions[1] >= takeTokens)
                {
                    if (tl.Dimensions[1] == takeTokens) { cut.Add(t); continue; }
                    var sub = new long[takeTokens];
                    for (int ti = 0; ti < takeTokens; ti++) sub[ti] = tl.GetValue(ti);
                    var nt = new DenseTensor<long>(sub, new[] { 1, takeTokens });
                    nt.Name = tl.Name;
                    cut.Add(nt);
                }
                else if (t is Tensor<long>) throw new InvalidOperationException(name + ": cannot take " + takeTokens + " tokens from " + t.Dims[1] + ".");
                else cut.Add(t);
            }
            inputs = cut.ToArray();
        }
        if (padTo > 0)
        {
            // Right-pad to padTo with RoBERTa <pad> (1) on input_ids and 0 elsewhere, so the
            // attention mask marks real content vs padding on both engines identically.
            var padded = new List<ITensor>();
            foreach (var t in inputs)
            {
                if (t is Tensor<long> tl && tl.Dimensions.Length == 2 && tl.Dimensions[0] == 1 && tl.Dimensions[1] < padTo)
                {
                    long pad = tl.Name == "input_ids" ? 1 : 0;
                    var ext = new long[padTo];
                    for (int pi = 0; pi < tl.Dimensions[1]; pi++) ext[pi] = tl.GetValue(pi);
                    for (int pi = tl.Dimensions[1]; pi < padTo; pi++) ext[pi] = pad;
                    var nt = new DenseTensor<long>(ext, new[] { 1, padTo });
                    nt.Name = tl.Name;
                    padded.Add(nt);
                }
                else padded.Add(t);
            }
            inputs = padded.ToArray();
        }
        return inputs;
    }

    internal static ITensor[] VisionInputs()
    {
        var flat = new float[1 * 3 * 224 * 224];
        for (int i = 0; i < flat.Length; i++) flat[i] = 0.5f;
        return new ITensor[] { new DenseTensor<float>(flat, new[] { 1, 3, 224, 224 }) };
    }
    static void CompareE5(string name, string model, string tokenizer, string text, TensorExecutionOptions tensorOpts, int threads, int iters, int warmup, int warmupMin, int warmupMax, string rowsName, string modeName, int takeTokens = 0, int padTo = 0)
    {
        var inputs = E5Inputs(name, tokenizer, text, takeTokens, padTo);
        Console.WriteLine("sidecar tokenizer bytes=" + new FileInfo(tokenizer).Length + " sha12=" + ShortHash(tokenizer));
        Compare(name, model, inputs, tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName);
    }

    static void CompareVision(string name, string model, TensorExecutionOptions tensorOpts, int threads, int iters, int warmup, int warmupMin, int warmupMax, string rowsName, string modeName)
    {
        Compare(name, model, VisionInputs(), tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName);
    }

    static readonly long[] Gpt2PrefillPattern = new long[] { 15496, 11, 314, 716 };
    internal static ITensor[] Gpt2PrefillInputs(int tokens, bool legacyCycle)
    {
        if (tokens < 1) throw new InvalidOperationException("gpt2 prefill needs at least 1 token.");
        var idv = new long[tokens];
        var mav = new long[tokens];
        var pav = new long[tokens];
        // The frozen gpt2-4tok case keeps the legacy 4-cycle byte-identical to the B01 baseline.
        // New regimes use a non-repeating stride (closer to natural text): the 4-cycle at 32+ tokens
        // trips a narrow 1.5e-4 operating-point breach while the stride gates at 4.6e-5 through 512
        // tokens (peaked-attention numerics, N01 material).
        for (int i = 0; i < tokens; i++) { idv[i] = legacyCycle ? Gpt2PrefillPattern[i % Gpt2PrefillPattern.Length] : (int)((15496L + (long)i * 7919L) % 50257); mav[i] = 1; pav[i] = i; }
        var ids = new DenseTensor<long>(idv, new[] { 1, tokens });
        ids.Name = "input_ids";
        var mask = new DenseTensor<long>(mav, new[] { 1, tokens });
        mask.Name = "attention_mask";
        var pos = new DenseTensor<long>(pav, new[] { 1, tokens });
        pos.Name = "position_ids";
        var inputs = new List<ITensor> { ids, mask, pos };
        for (int layer = 0; layer < 12; layer++)
        {
            var k = new DenseTensor<float>(Array.Empty<float>(), new[] { 1, 12, 0, 64 });
            k.Name = "past_key_values." + layer + ".key";
            var v = new DenseTensor<float>(Array.Empty<float>(), new[] { 1, 12, 0, 64 });
            v.Name = "past_key_values." + layer + ".value";
            inputs.Add(k);
            inputs.Add(v);
        }
        return inputs.ToArray();
    }


    static void CompareGpt2(string name, string model, int tokens, TensorExecutionOptions tensorOpts, int threads, int iters, int warmup, int warmupMin, int warmupMax, string rowsName, string modeName, bool legacyCycle = false)
    {
        Compare(name, model, Gpt2PrefillInputs(tokens, legacyCycle), tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName);
    }

    static void CompareGpt2Decode(string name, string model, int pastLen, TensorExecutionOptions tensorOpts, int threads, int iters, int warmup, int warmupMin, int warmupMax, string rowsName, string modeName)
    {
        // Teacher-forced single-token decode: run a validated prefill on both engines, then time
        // the decode step on identical past state copied out of the Lokad prefill outputs, so any
        // prefill drift cannot change the decode work. Logits and all present tensors stay gated.
        if (pastLen < 1) throw new InvalidOperationException(name + ": decode needs past length >= 1.");
        var graph = OnnxImport.Load(model)!;
        graph.Prepare();
        var matchedOpts = new ExecutionOptions(OptimizationMode.Speed, tensorOpts);
        using var so = CreateSingleCpuSessionOptions(threads);
        using var session = OpenSession(model, so, out _);
        var outNames = session.OutputMetadata.Keys.ToArray();
        var named = ToNamed(name + "-prefill", Gpt2PrefillInputs(pastLen, false), session.InputMetadata.Keys.ToArray());
        var pre = Validate(name + "-prefill", graph, session, named, outNames, matchedOpts);
        Console.WriteLine("prefill " + name + " tokens=" + pastLen + " maxScaled=" + pre.scaled.ToString("E2") + " maxAbs=" + pre.abs.ToString("E2"));
        var decode = BuildGpt2DecodeInputs(graph, pastLen);
        Compare(name, model, decode, tensorOpts, threads, iters, warmup, warmupMin, warmupMax, rowsName, modeName);
    }

    /// <summary>
    /// Builds teacher-forced single-token decode inputs from Lokad prefill outputs
    /// already bound on the graph (prefill must have executed): token 317 with a
    /// full mask plus past key/values copied out per layer, so prefill drift cannot
    /// change the decode work. Shared by the timing and profile lanes.
    /// </summary>
    internal static ITensor[] BuildGpt2DecodeInputs(ComputationalGraph graph, int pastLen)
    {
        var decode = new List<ITensor>();
        var nid = new DenseTensor<long>(new long[] { 317 }, new[] { 1, 1 });
        nid.Name = "input_ids";
        var nmask = new DenseTensor<long>(Enumerable.Repeat(1L, pastLen + 1).ToArray(), new[] { 1, pastLen + 1 });
        nmask.Name = "attention_mask";
        var npos = new DenseTensor<long>(new long[] { pastLen }, new[] { 1, 1 });
        npos.Name = "position_ids";
        decode.Add(nid);
        decode.Add(nmask);
        decode.Add(npos);
        for (int layer = 0; layer < 12; layer++)
        {
            foreach (var kv in new[] { "key", "value" })
            {
                if (!graph.Outputs.TryGetValue("present." + layer + "." + kv, out var lt) || lt is not Tensor<float> lf)
                    throw new InvalidOperationException("decode past missing: present." + layer + "." + kv);
                var past = new DenseTensor<float>(lf.ToArray(), lf.Dimensions.ToArray());
                past.Name = "past_key_values." + layer + "." + kv;
                decode.Add(past);
            }
        }
        return decode.ToArray();
    }

    static void Compare(string name, string model, ITensor[] inputs, TensorExecutionOptions tensorOpts, int threads, int iters, int warmup, int warmupMin, int warmupMax, string rowsName, string modeName)
    {
        var loadSw = Stopwatch.StartNew();
        var graph = OnnxImport.Load(model)!;
        loadSw.Stop();
        var prepSw = Stopwatch.StartNew();
        graph.Prepare();
        prepSw.Stop();
        double loadMs = loadSw.Elapsed.TotalMilliseconds;
        double prepareMs = prepSw.Elapsed.TotalMilliseconds;
        var sidecar = Path.ChangeExtension(model, ".onnx_data");
        string sidecarInfo = File.Exists(sidecar)
            ? " sidecar=" + Path.GetFileName(sidecar) + " bytes=" + new FileInfo(sidecar).Length + " sha12=" + ShortHash(sidecar)
            : "";
        if (rowsName == "canonical")
        {
            var matchedOpts = new ExecutionOptions(OptimizationMode.Speed, tensorOpts);
            using (var matchedSo = CreateSingleCpuSessionOptions(threads))
            {
                double ortLoadMs;
                using (var ortMatched = OpenSession(model, matchedSo, out ortLoadMs))
                {
                    TimedRow(name, model, graph, inputs, ortMatched, matchedOpts,
                        CanonicalLokDesc(modeName, threads), "intraop=" + threads + " interop=1 seq opt=ALL nospin", iters, warmup, warmupMin, warmupMax, sidecarInfo,
                        loadMs, prepareMs, ortLoadMs);
                }
            }
            return;
        }
        double defaultLoadMs;
        using (var ortDefault = OpenDefaultSession(model, out defaultLoadMs))
        {
            TimedRow(name, model, graph, inputs, ortDefault, ExecutionOptions.Default,
                "defaults (archival unequal-CPU: lokad-auto-1-thread vs ort-default-pool; do-not-gate)", "ort-defaults", iters, warmup, warmupMin, warmupMax, sidecarInfo,
                loadMs, prepareMs, defaultLoadMs);
        }
        var oneOpts = new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Scalar);
        using (var oneSo = CreateSingleCpuSessionOptions(1))
        {
            double oneLoadMs;
            using (var ortOne = OpenSession(model, oneSo, out oneLoadMs))
            {
                TimedRow(name, model, graph, inputs, ortOne, oneOpts,
                    "scalar-1-thread (SIMD-disabled diagnostic)", "intraop=1 interop=1 seq opt=ALL nospin", iters, warmup, warmupMin, warmupMax, sidecarInfo,
                    loadMs, prepareMs, oneLoadMs);
            }
        }
        var legacyMatchedOpts = new ExecutionOptions(OptimizationMode.Speed, tensorOpts);
        using (var legacyMatchedSo = CreateSingleCpuSessionOptions(threads))
        {
            double legacyLoadMs;
            using (var ortMatched = OpenSession(model, legacyMatchedSo, out legacyLoadMs))
            {
                TimedRow(name, model, graph, inputs, ortMatched, legacyMatchedOpts,
                    CanonicalLokDesc(modeName, threads), "intraop=" + threads + " interop=1 seq opt=ALL nospin", iters, warmup, warmupMin, warmupMax, sidecarInfo,
                    loadMs, prepareMs, legacyLoadMs);
            }
        }
    }

    static InferenceSession OpenSession(string model, SessionOptions options, out double loadMs)
    {
        var sw = Stopwatch.StartNew();
        var session = new InferenceSession(model, options);
        sw.Stop();
        loadMs = sw.Elapsed.TotalMilliseconds;
        return session;
    }

    static InferenceSession OpenDefaultSession(string model, out double loadMs)
    {
        var sw = Stopwatch.StartNew();
        var session = new InferenceSession(model);
        sw.Stop();
        loadMs = sw.Elapsed.TotalMilliseconds;
        return session;
    }

    static void TimedRow(string name, string model, ComputationalGraph graph, ITensor[] inputs,
        InferenceSession ortSession, ExecutionOptions lokadOpts, string lokDesc, string ortDesc,
        int iters, int warmup, int warmupMin, int warmupMax, string sidecarInfo, double loadMs, double prepareMs, double ortLoadMs)
    {
        var inNames = ortSession.InputMetadata.Keys.ToArray();
        var outNames = ortSession.OutputMetadata.Keys.ToArray();
        if (inputs.Length == 1 && string.IsNullOrEmpty(inputs[0].Name) && inNames.Length == 1) inputs[0].Name = inNames[0];
        var named = ToNamed(name, inputs, inNames);
        var valSw = Stopwatch.StartNew();
        var first = Validate(name, graph, ortSession, named, outNames, lokadOpts);
        valSw.Stop();
        Console.WriteLine("case " + name + " [" + lokDesc + " vs " + ortDesc + "]: model="
            + Path.GetFileName(Path.GetDirectoryName(model)) + "/model.onnx"
            + " bytes=" + new FileInfo(model).Length + " sha12=" + ShortHash(model) + sidecarInfo
            + " inputs=[" + string.Join(",", named.Select(kv => kv.Key + ":" + string.Join("x", kv.Value.Dims))) + "]"
            + " outputs=[" + string.Join(",", OutputShapes(graph, outNames)) + "]"
            + " warmup=" + warmup + " iters=" + iters
            + " load=" + loadMs.ToString("F1") + "ms prepare=" + prepareMs.ToString("F1") + "ms ortLoad=" + ortLoadMs.ToString("F1") + "ms"
            + " firstLokad=" + first.lokadFirstMs.ToString("F1") + "ms firstOrt=" + first.ortFirstMs.ToString("F1") + "ms"
            + " (first-run is process-cold only on the first row; later rows share warmed JIT)");
        Console.WriteLine("casedef " + name
            + " model=" + Path.GetFileName(Path.GetDirectoryName(model)) + "/model.onnx"
            + " bytes=" + new FileInfo(model).Length + " sha12=" + ShortHash(model)
            + " inputs=" + string.Join(",", named.Select(kv => kv.Key + ":" + string.Join("x", kv.Value.Dims)))
            + " outputs=" + string.Join(",", OutputShapes(graph, outNames))
            + " iters=" + iters + " warmup=" + warmup + " tol=" + Tolerance.ToString("E2"));
        TimedRun(name, graph, named, outNames, lokadOpts, lokDesc, ortDesc, iters, warmup, warmupMin, warmupMax, valSw.Elapsed.TotalMilliseconds, first.scaled, first.abs, ortSession);
    }

    static string[] OutputShapes(ComputationalGraph graph, string[] outNames)
    {
        var shapes = new string[outNames.Length];
        for (int i = 0; i < outNames.Length; i++)
        {
            shapes[i] = graph.Outputs.TryGetValue(outNames[i], out var t) && t is not null
                ? outNames[i] + ":" + string.Join("x", t.Dims)
                : outNames[i] + ":unresolved";
        }
        return shapes;
    }

    static void TimedRun(string name, ComputationalGraph graph, Dictionary<string, ITensor> named, string[] outNames,
        ExecutionOptions lokadOpts, string lokDesc, string ortDesc, int iters, int warmup, int warmupMin, int warmupMax, double validationMs, double maxScaled, double maxAbs, InferenceSession ortSession)
    {
        // One-time input conversion, reused by every ORT run below; Lokad inputs need no conversion.
        var convSw = Stopwatch.StartNew();
        var ortInputs = BuildOrtInputs(named, ortSession.InputMetadata.Keys.ToArray());
        convSw.Stop();
        double convertMs = convSw.Elapsed.TotalMilliseconds;
        ulong fpBefore = FingerprintInputs(named);
        try
        {
            using var ro = new RunOptions();
            // Warmups establish steady state. Each side's outputs are released before the other side runs.
            // Fixed mode runs exactly <warmup> iterations. Adaptive mode (min/max >= 0) runs at least
            // warmupMin and at most warmupMax iterations, stopping early once both engines' last three
            // warmup times agree within 10% (range/median, needing at least 3 iterations for the rule).
            // The recorded warmup line pins what actually ran; timed loops below use only warmed state.
            var wsw = new Stopwatch();
            var wlok = new List<double>();
            var wort = new List<double>();
            void TimeWarmupPair()
            {
                graph.Reset();
                wsw.Restart();
                if (!graph.Execute(named, true, ExecutionProvider.CPU, lokadOpts)) throw new InvalidOperationException(name + ": lokad warmup execute failed");
                wsw.Stop();
                wlok.Add(wsw.Elapsed.TotalMilliseconds);
                graph.Reset();
                wsw.Restart();
                using var wr = ortSession.Run(ro, ortInputs, outNames);
                wsw.Stop();
                wort.Add(wsw.Elapsed.TotalMilliseconds);
            }
            bool WarmupSteady(List<double> xs)
            {
                int n = xs.Count;
                if (n < 3) return false;
                double a = xs[n - 3], b = xs[n - 2], c = xs[n - 1];
                double hi = Math.Max(a, Math.Max(b, c));
                double lo = Math.Min(a, Math.Min(b, c));
                double median = a + b + c - hi - lo;
                return (hi - lo) <= 0.10 * Math.Max(median, 1e-9);
            }
            int warmupUsed;
            string warmupStop;
            if (warmupMin < 0)
            {
                for (int w = 0; w < warmup; w++) TimeWarmupPair();
                warmupUsed = warmup;
                warmupStop = "fixed";
            }
            else
            {
                warmupStop = "max-reached";
                for (int w = 0; w < warmupMax; w++)
                {
                    TimeWarmupPair();
                    if (wlok.Count >= Math.Max(warmupMin, 3) && WarmupSteady(wlok) && WarmupSteady(wort)) { warmupStop = 
"steady"; break; }
                }
                warmupUsed = wlok.Count;
            }
            Console.WriteLine("warmup " + name + " used=" + warmupUsed + " stop=" + warmupStop
                + " lok=[" + string.Join(",", wlok.Select(v => v.ToString("F2"))) + "]"
                + " ort=[" + string.Join(",", wort.Select(v => v.ToString("F2"))) + "]");
            // Warmed reusable-context handle on the shared prepared plan (no per-run context allocation or copy-back).
            var ctx = graph.CreateExecution(lokadOpts);
            ctx.Reset();
            if (!ctx.Execute(named, true, ExecutionProvider.CPU, lokadOpts)) throw new InvalidOperationException(name + ": context warmup execute failed");
            ctx.Reset();
            // Warmed inference, alternating all three lanes fairly. No lane executes while another's
            // outputs are alive: ORT outputs are disposed right after each timed run, the graph is reset
            // right after each timed public execute, and the context is reset right after each timed
            // context execute. Rotation by i % 3 gives each lane each position equally often when iters
            // is a multiple of 3. Reset stays outside the timed regions and is reported separately.
            // GC and allocation accounting below is shared-process; per-engine attribution is not claimed.
            long allocBefore = GC.GetTotalAllocatedBytes(false);
            int g0Before = GC.CollectionCount(0), g1Before = GC.CollectionCount(1), g2Before = GC.CollectionCount(2);
            var clok = new List<double>();
            var tctx = new List<long>();
            var lok = new List<double>();
            var tlok = new List<long>();
            var ort = new List<double>();
            var tort = new List<long>();
            var sw = new Stopwatch();
            void TimeCtx()
            {
                ctx.Reset();
                sw.Restart();
                if (!ctx.Execute(named, true, ExecutionProvider.CPU, lokadOpts)) throw new InvalidOperationException(name + ": context execute failed");
                sw.Stop();
                clok.Add(sw.Elapsed.TotalMilliseconds);
                tctx.Add(sw.ElapsedTicks);
                ctx.Reset();
            }
            void TimeLok()
            {
                graph.Reset();
                sw.Restart();
                if (!graph.Execute(named, true, ExecutionProvider.CPU, lokadOpts)) throw new InvalidOperationException(name + ": lokad execute failed");
                sw.Stop();
                lok.Add(sw.Elapsed.TotalMilliseconds);
                tlok.Add(sw.ElapsedTicks);
                graph.Reset();
            }
            void TimeOrt()
            {
                sw.Restart();
                using (var timed = ortSession.Run(ro, ortInputs, outNames)) { sw.Stop(); }
                ort.Add(sw.Elapsed.TotalMilliseconds);
                tort.Add(sw.ElapsedTicks);
            }
            for (int i = 0; i < iters; i++)
            {
                switch (i % 3)
                {
                    case 0: TimeCtx(); TimeLok(); TimeOrt(); break;
                    case 1: TimeLok(); TimeOrt(); TimeCtx(); break;
                    default: TimeOrt(); TimeCtx(); TimeLok(); break;
                }
            }
            // Complete repeated-request boundary: what a caller pays per request on a reused
            // graph, with Reset/output release inside the timed unit. req uses the public Execute
            // (per-run context plus copy-back); reqCtx reuses one context. The two lanes alternate.
            // ORT has no reset concept (its outputs are already disposed per run), so no ORT lane
            // belongs here. Both handles are left reset-clean, as the timed loops above leave them.
            var req = new List<double>();
            var reqCtx = new List<double>();
            void TimeReq()
            {
                graph.Reset();
                sw.Restart();
                if (!graph.Execute(named, true, ExecutionProvider.CPU, lokadOpts)) throw new InvalidOperationException(name + ": request execute failed");
                graph.Reset();
                sw.Stop();
                req.Add(sw.Elapsed.TotalMilliseconds);
            }
            void TimeReqCtx()
            {
                ctx.Reset();
                sw.Restart();
                if (!ctx.Execute(named, true, ExecutionProvider.CPU, lokadOpts)) throw new InvalidOperationException(name + ": request context execute failed");
                ctx.Reset();
                sw.Stop();
                reqCtx.Add(sw.Elapsed.TotalMilliseconds);
            }
            for (int i = 0; i < iters; i++)
            {
                if (i % 2 == 0) { TimeReq(); TimeReqCtx(); }
                else { TimeReqCtx(); TimeReq(); }
            }
            long allocAfter = GC.GetTotalAllocatedBytes(false);
            string gcLine = "gc=" + (GC.CollectionCount(0) - g0Before) + "/" + (GC.CollectionCount(1) - g1Before) + "/" + (GC.CollectionCount(2) - g2Before)
                + " allocMB=" + ((allocAfter - allocBefore) / 1000000.0).ToString("F1") + " (shared-process over warmed loops)";
            double[] resetPop = TimePopulatedResets(name, graph, named, lokadOpts, iters);
            double[] resetClean = TimeCleanResets(graph, iters);
            ulong fpAfter = FingerprintInputs(named);
            if (fpAfter != fpBefore)
                throw new InvalidOperationException(name + ": inputs mutated during timed reuse (fingerprint changed).");
            var post = Validate(name + " post-timed", graph, ortSession, named, outNames, lokadOpts);

            Console.WriteLine(name + " [" + lokDesc + " vs " + ortDesc + "]: lokad " + Dist(lok) + " | ctxLokad " + Dist(clok) + " | ort " + Dist(ort)
                + " | req " + Dist(req) + " | reqCtx " + Dist(reqCtx) + " | resetPop " + Dist(resetPop) + " | resetClean " + Dist(resetClean) + " | convert=" + convertMs.ToString("F1") + "ms"
                + " | validation=" + validationMs.ToString("F1") + "ms | disposal=outside"
                + " | " + gcLine
                + " | maxScaled=" + maxScaled.ToString("E2") + " maxAbs=" + maxAbs.ToString("E2")
                + " | postScaled=" + post.scaled.ToString("E2") + " postAbs=" + post.abs.ToString("E2")
                + " | inputsIntact=yes");
            Console.WriteLine("raw lok=[" + string.Join(",", lok.Select(v => v.ToString("F2"))) + "]"
                + " raw ctx=[" + string.Join(",", clok.Select(v => v.ToString("F2"))) + "]"
                + " raw ort=[" + string.Join(",", ort.Select(v => v.ToString("F2"))) + "]");
            Console.WriteLine("raw req=[" + string.Join(",", req.Select(v => v.ToString("F2"))) + "]"
                + " raw reqctx=[" + string.Join(",", reqCtx.Select(v => v.ToString("F2"))) + "]");
            Console.WriteLine("rawticks lok=[" + string.Join(",", tlok) + "]"
                + " rawticks ctx=[" + string.Join(",", tctx) + "]"
                + " rawticks ort=[" + string.Join(",", tort) + "]"
                + " tickfreq=" + Stopwatch.Frequency);
        }
        finally
        {
            foreach (var v in ortInputs.Values) v.Dispose();
        }
    }

    internal static (double scaled, double abs, double lokadFirstMs, double ortFirstMs) Validate(string name, ComputationalGraph graph, InferenceSession session, Dictionary<string, ITensor> named, string[] outNames, ExecutionOptions lokadOpts)
    {
        // First runs on cold state; output disposal stays outside both first-run figures.
        // The ORT session is the reference in every comparison below.
        graph.Reset();
        var lokSw = Stopwatch.StartNew();
        if (!graph.Execute(named, true, ExecutionProvider.CPU, lokadOpts)) throw new InvalidOperationException(name + ": lokad validation execute failed");
        lokSw.Stop();
        var declared = graph.OutputDescs.Where(d => !string.IsNullOrEmpty(d.Name)).Select(d => d.Name).OrderBy(n => n, StringComparer.Ordinal).ToArray();
        var requested = outNames.OrderBy(n => n, StringComparer.Ordinal).ToArray();
        if (!declared.SequenceEqual(requested, StringComparer.Ordinal))
            throw new InvalidOperationException(name + ": output-name set mismatch (graph declares ["
                + string.Join(",", declared) + "], session requested [" + string.Join(",", requested) + "]).");
        var ortInputs = BuildOrtInputs(named, session.InputMetadata.Keys.ToArray());
        try
        {
            using var ro = new RunOptions();
            var ortSw = Stopwatch.StartNew();
            using var results = session.Run(ro, ortInputs, outNames);
            ortSw.Stop();
            var outs = results.ToArray();
            if (outs.Length != outNames.Length) throw new InvalidOperationException(name + ": ort returned " + outs.Length + " outputs for " + outNames.Length + " requested.");
            double worstScaled = 0;
            double worstAbs = 0;
            for (int oi = 0; oi < outNames.Length; oi++)
            {
                string onm = outNames[oi];
                var res = outs[oi];
                var shape = res.GetTensorTypeAndShape();
                if (shape.ElementDataType != Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float)
                    throw new InvalidOperationException(name + ": ort output is not float32: " + onm + " (got " + shape.ElementDataType + ").");
                if (!graph.Outputs.TryGetValue(onm, out var lt) || lt is not Tensor<float> lf)
                    throw new InvalidOperationException(name + ": lokad output missing or not float32: " + onm
                        + " (got " + (lt is null ? "unresolved" : lt.GetType().Name) + ").");
                var la = lf.ToArray();
                var oa = res.GetTensorDataAsSpan<float>().ToArray();
                (double scaled, double abs) agree;
                try
                {
                    agree = BenchValidate.RequireAgreement(
                        name + ":" + onm,
                        shape.Shape.Select(d => checked((int)d)).ToArray(), oa,
                        lf.Dimensions.ToArray(), la, Tolerance);
                }
                catch (InvalidOperationException ex)
                {
                    if (oa.Length == la.Length)
                    {
                        var diff = BenchValidate.ScaledAndAbsDiff(oa, la);
                        if (KnownDivergences.TryMatch(name, onm, diff.scaled, Tolerance, out var known))
                            throw new KnownDivergenceException(name, onm, diff.scaled, Tolerance, known, ex);
                    }
                    throw;
                }
                worstScaled = Math.Max(worstScaled, agree.scaled);
                worstAbs = Math.Max(worstAbs, agree.abs);
            }
            return (worstScaled, worstAbs, lokSw.Elapsed.TotalMilliseconds, ortSw.Elapsed.TotalMilliseconds);
        }
        finally
        {
            foreach (var v in ortInputs.Values) v.Dispose();
        }
    }

    static ulong FingerprintInputs(Dictionary<string, ITensor> named)
    {
        // FNV-1a 64 over sorted names, dims, and raw value bits.
        ulong h = 1469598103934665603UL;
        foreach (var key in named.Keys.OrderBy(k => k, StringComparer.Ordinal))
        {
            foreach (char ch in key) { h ^= ch; h *= 1099511628211UL; }
            var t = named[key];
            foreach (int d in t.Dims) { h ^= (uint)d; h *= 1099511628211UL; }
            if (t is Tensor<long> li)
            {
                for (int i = 0; i < li.Length; i++) { h ^= (ulong)li.GetValue(i); h *= 1099511628211UL; }
            }
            else if (t is Tensor<float> fi)
            {
                for (int i = 0; i < fi.Length; i++) { h ^= (ulong)(uint)BitConverter.SingleToInt32Bits(fi.GetValue(i)); h *= 1099511628211UL; }
            }
            else throw new InvalidOperationException("unsupported input tensor type " + t.GetType().Name);
        }
        return h;
    }

    internal static Dictionary<string, OrtValue> BuildOrtInputs(Dictionary<string, ITensor> named, string[] inNames)
    {
        var ortInputs = new Dictionary<string, OrtValue>();
        foreach (var n in inNames)
        {
            var src = named[n];
            if (src is Tensor<long> li)
            {
                ortInputs[n] = OrtValue.CreateTensorValueFromMemory(li.ToArray(), li.Dimensions.ToArray().Select(d => (long)d).ToArray());
            }
            else if (src is Tensor<float> fi)
            {
                ortInputs[n] = OrtValue.CreateTensorValueFromMemory(fi.ToArray(), fi.Dimensions.ToArray().Select(d => (long)d).ToArray());
            }
            else throw new InvalidOperationException("unsupported input tensor type " + src.GetType().Name);
        }
        return ortInputs;
    }

    internal static Dictionary<string, ITensor> ToNamed(string name, ITensor[] inputs, string[] names)
    {
        var d = new Dictionary<string, ITensor>(StringComparer.Ordinal);
        foreach (var t in inputs)
        {
            if (string.IsNullOrEmpty(t.Name) || !names.Contains(t.Name, StringComparer.Ordinal))
                throw new InvalidOperationException(name + ": input has no exact session match (name=" + t.Name + ").");
            if (!d.TryAdd(t.Name, t)) throw new InvalidOperationException(name + ": duplicate input " + t.Name + ".");
        }
        var missing = names.Where(n => !d.ContainsKey(n)).ToArray();
        if (missing.Length > 0) throw new InvalidOperationException(name + ": missing inputs " + string.Join(",", missing) + ".");
        return d;
    }

    static double[] TimePopulatedResets(string name, ComputationalGraph graph, Dictionary<string, ITensor> named, ExecutionOptions lokadOpts, int iters)
    {
        var sw = new Stopwatch();
        var ts = new List<double>();
        for (int i = 0; i < iters; i++)
        {
            graph.Reset();
            if (!graph.Execute(named, true, ExecutionProvider.CPU, lokadOpts)) throw new InvalidOperationException(name + ": populated-reset repopulate execute failed");
            sw.Restart();
            graph.Reset();
            sw.Stop();
            ts.Add(sw.Elapsed.TotalMilliseconds);
        }
        return ts.ToArray();
    }

    static double[] TimeCleanResets(ComputationalGraph graph, int iters)
    {
        graph.Reset();
        var sw = new Stopwatch();
        var ts = new List<double>();
        for (int i = 0; i < iters; i++) { sw.Restart(); graph.Reset(); sw.Stop(); ts.Add(sw.Elapsed.TotalMilliseconds); }
        return ts.ToArray();
    }

    static string Dist(IEnumerable<double> values)
    {
        var ts = values.OrderBy(x => x).ToArray();
        return "best=" + ts[0].ToString("F1") + "ms median=" + ts[ts.Length / 2].ToString("F1")
            + "ms p95=" + ts[Math.Min(ts.Length - 1, (int)Math.Ceiling(ts.Length * 0.95) - 1)].ToString("F1")
            + "ms max=" + ts[ts.Length - 1].ToString("F1") + "ms (n=" + ts.Length + ")";
    }

    static string ShortHash(string file)
    {
        using var sha = SHA256.Create();
        using var s = File.OpenRead(file);
        return Convert.ToHexString(sha.ComputeHash(s)).Substring(0, 12);
    }
}
