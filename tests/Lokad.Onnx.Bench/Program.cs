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
    const int Warm = 3;
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
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--mode" && i + 1 < args.Length) modeName = args[++i];
            else if (args[i] == "--rows" && i + 1 < args.Length) rowsName = args[++i];
            else if (args[i] == "--cpu" && i + 1 < args.Length && int.TryParse(args[i + 1], out var c)) { cpu = c; i++; }
            else if (args[i] == "--threads" && i + 1 < args.Length && int.TryParse(args[i + 1], out var t) && t >= 1) { threads = t; i++; }
            else if (args[i] == "--iters" && i + 1 < args.Length && int.TryParse(args[i + 1], out var k) && k >= 1) { iters = k; i++; }
            else if (args[i] == "all" || assets.ContainsKey(args[i])) { if (args[i] != "all" && !selected.Contains(args[i], StringComparer.OrdinalIgnoreCase)) selected.Add(args[i]); }
            else { Console.WriteLine("usage: Bench [e5 dinov2 dinov3 resnet50 gpt2 all] [--mode auto|scalar|simd|intrinsics] [--threads N] [--iters N] [--rows canonical|all] [--cpu N]"); return 2; }
        }
        if (rowsName != "canonical" && rowsName != "all") { Console.WriteLine("unknown --rows " + rowsName + " (expected canonical|all)"); return 2; }
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
            + " mode=" + modeName + " threads=" + threads + " rows=" + rowsName + " iters=" + iters + " warmup=" + Warm);
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
        if (selected.Contains("e5", StringComparer.OrdinalIgnoreCase))
        {
            var e5 = assets["e5"];
            CompareE5("e5-8tok", e5[0], e5[1], "query: hello world", tensorOpts, threads, iters, rowsName, modeName);
            CompareE5("e5-30tok", e5[0], e5[1], "query: The quick brown fox jumps over the lazy dog near the river bank in springtime weather for a pleasant afternoon walk", tensorOpts, threads, iters, rowsName, modeName);
        }
        if (selected.Contains("dinov2", StringComparer.OrdinalIgnoreCase)) CompareVision("dinov2-224", assets["dinov2"][0], tensorOpts, threads, iters, rowsName, modeName);
        if (selected.Contains("dinov3", StringComparer.OrdinalIgnoreCase)) CompareVision("dinov3-224", assets["dinov3"][0], tensorOpts, threads, iters, rowsName, modeName);
        if (selected.Contains("resnet50", StringComparer.OrdinalIgnoreCase)) CompareVision("resnet50-224", assets["resnet50"][0], tensorOpts, threads, iters, rowsName, modeName);
        if (selected.Contains("gpt2", StringComparer.OrdinalIgnoreCase)) CompareGpt2("gpt2-4tok", assets["gpt2"][0], tensorOpts, threads, iters, rowsName, modeName);
        return 0;
    }

    static string[] StripCpuSelector(string[] args, ref int cpu)
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

    static IntPtr EnforceSingleCpuAffinity(int cpu)
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

    static SessionOptions CreateSingleCpuSessionOptions(int threads)
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
            Console.WriteLine("usage: Bench micro <matmul2d|matmul|indexing|ops> [BenchmarkDotNet options]");
            return 2;
        }
        try
        {
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
                default:
                    Console.WriteLine("Unknown micro benchmark: " + args[0] + ".");
                    Console.WriteLine("usage: Bench micro <matmul2d|matmul|indexing|ops> [BenchmarkDotNet options]");
                    return 2;
            }
        }
        catch (InvalidOperationException e)
        {
            if (e.Message == "Sequence contains no elements")
            {
                return 0;
            }
            throw;
        }
    }

    static string FindRoot()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Lokad.Onnx.slnx"))) return dir.FullName;
            dir = dir.Parent;
        }
        throw new InvalidOperationException("repo root not found");
    }

    static void CompareE5(string name, string model, string tokenizer, string text, TensorExecutionOptions tensorOpts, int threads, int iters, string rowsName, string modeName)
    {
        var inputs = Text.RobertaTokenizeFromFile(text, tokenizer)!;
        Console.WriteLine("sidecar tokenizer bytes=" + new FileInfo(tokenizer).Length + " sha12=" + ShortHash(tokenizer));
        Compare(name, model, inputs, tensorOpts, threads, iters, rowsName, modeName);
    }

    static void CompareVision(string name, string model, TensorExecutionOptions tensorOpts, int threads, int iters, string rowsName, string modeName)
    {
        var flat = new float[1 * 3 * 224 * 224];
        for (int i = 0; i < flat.Length; i++) flat[i] = 0.5f;
        var input = new DenseTensor<float>(flat, new[] { 1, 3, 224, 224 });
        Compare(name, model, new ITensor[] { input }, tensorOpts, threads, iters, rowsName, modeName);
    }

    static void CompareGpt2(string name, string model, TensorExecutionOptions tensorOpts, int threads, int iters, string rowsName, string modeName)
    {
        var ids = new DenseTensor<long>(new long[] { 15496, 11, 314, 716 }, new[] { 1, 4 });
        ids.Name = "input_ids";
        var mask = new DenseTensor<long>(new long[] { 1, 1, 1, 1 }, new[] { 1, 4 });
        mask.Name = "attention_mask";
        var pos = new DenseTensor<long>(new long[] { 0, 1, 2, 3 }, new[] { 1, 4 });
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
        Compare(name, model, inputs.ToArray(), tensorOpts, threads, iters, rowsName, modeName);
    }

    static void Compare(string name, string model, ITensor[] inputs, TensorExecutionOptions tensorOpts, int threads, int iters, string rowsName, string modeName)
    {
        var graph = OnnxImport.Load(model)!;
        var sidecar = Path.ChangeExtension(model, ".onnx_data");
        string sidecarInfo = File.Exists(sidecar)
            ? " sidecar=" + Path.GetFileName(sidecar) + " bytes=" + new FileInfo(sidecar).Length + " sha12=" + ShortHash(sidecar)
            : "";
        if (rowsName == "canonical")
        {
            var matchedOpts = new ExecutionOptions(OptimizationMode.Speed, tensorOpts);
            using (var matchedSo = CreateSingleCpuSessionOptions(threads))
            {
                using var ortMatched = new InferenceSession(model, matchedSo);
                TimedRow(name, model, graph, inputs, ortMatched, matchedOpts,
                    CanonicalLokDesc(modeName, threads), "intraop=" + threads + " interop=1 seq opt=ALL nospin", iters, sidecarInfo);
            }
            return;
        }
        using (var ortDefault = new InferenceSession(model))
        {
            TimedRow(name, model, graph, inputs, ortDefault, ExecutionOptions.Default,
                "defaults (archival unequal-CPU: lokad-auto-1-thread vs ort-default-pool; do-not-gate)", "ort-defaults", iters, sidecarInfo);
        }
        var oneOpts = new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Scalar);
        using (var oneSo = CreateSingleCpuSessionOptions(1))
        {
            using var ortOne = new InferenceSession(model, oneSo);
            TimedRow(name, model, graph, inputs, ortOne, oneOpts,
                "scalar-1-thread (SIMD-disabled diagnostic)", "intraop=1 interop=1 seq opt=ALL nospin", iters, sidecarInfo);
        }
        var legacyMatchedOpts = new ExecutionOptions(OptimizationMode.Speed, tensorOpts);
        using (var legacyMatchedSo = CreateSingleCpuSessionOptions(threads))
        {
            using var ortMatched = new InferenceSession(model, legacyMatchedSo);
            TimedRow(name, model, graph, inputs, ortMatched, legacyMatchedOpts,
                CanonicalLokDesc(modeName, threads), "intraop=" + threads + " interop=1 seq opt=ALL nospin", iters, sidecarInfo);
        }
    }

    static void TimedRow(string name, string model, ComputationalGraph graph, ITensor[] inputs,
        InferenceSession ortSession, ExecutionOptions lokadOpts, string lokDesc, string ortDesc,
        int iters, string sidecarInfo)
    {
        var inNames = ortSession.InputMetadata.Keys.ToArray();
        var outNames = ortSession.OutputMetadata.Keys.ToArray();
        if (inputs.Length == 1 && string.IsNullOrEmpty(inputs[0].Name) && inNames.Length == 1) inputs[0].Name = inNames[0];
        var named = ToNamed(name, inputs, inNames);
        var valSw = Stopwatch.StartNew();
        double worst = Validate(name, graph, ortSession, named, outNames, lokadOpts);
        valSw.Stop();
        Console.WriteLine("case " + name + " [" + lokDesc + " vs " + ortDesc + "]: model="
            + Path.GetFileName(Path.GetDirectoryName(model)) + "/model.onnx"
            + " bytes=" + new FileInfo(model).Length + " sha12=" + ShortHash(model) + sidecarInfo
            + " inputs=[" + string.Join(",", named.Select(kv => kv.Key + ":" + string.Join("x", kv.Value.Dims))) + "]"
            + " outputs=[" + string.Join(",", OutputShapes(graph, outNames)) + "]"
            + " warmup=" + Warm + " iters=" + iters);
        TimedRun(name, graph, named, outNames, lokadOpts, lokDesc, ortDesc, iters, valSw.Elapsed.TotalMilliseconds, worst, ortSession);
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
        ExecutionOptions lokadOpts, string lokDesc, string ortDesc, int iters, double validationMs, double maxDiff, InferenceSession ortSession)
    {
        var convSw = Stopwatch.StartNew();
        var ortInputs = BuildOrtInputs(named, ortSession.InputMetadata.Keys.ToArray());
        convSw.Stop();
        try
        {
            using var ro = new RunOptions();
            double[] resets = TimeResets(graph, iters);
            for (int w = 0; w < Warm; w++)
            {
                graph.Reset();
                if (!graph.Execute(named, true, ExecutionProvider.CPU, lokadOpts)) throw new InvalidOperationException(name + ": lokad warmup execute failed");
                using var wr = ortSession.Run(ro, ortInputs, outNames);
            }
            var lok = new List<double>();
            var ort = new List<double>();
            var sw = new Stopwatch();
            for (int i = 0; i < iters; i++)
            {
                if (i % 2 == 0)
                {
                    sw.Restart();
                    using var first = ortSession.Run(ro, ortInputs, outNames);
                    sw.Stop();
                    ort.Add(sw.Elapsed.TotalMilliseconds);
                    graph.Reset();
                    sw.Restart();
                    if (!graph.Execute(named, true, ExecutionProvider.CPU, lokadOpts)) throw new InvalidOperationException(name + ": lokad execute failed");
                    sw.Stop();
                    lok.Add(sw.Elapsed.TotalMilliseconds);
                }
                else
                {
                    graph.Reset();
                    sw.Restart();
                    if (!graph.Execute(named, true, ExecutionProvider.CPU, lokadOpts)) throw new InvalidOperationException(name + ": lokad execute failed");
                    sw.Stop();
                    lok.Add(sw.Elapsed.TotalMilliseconds);
                    sw.Restart();
                    using var second = ortSession.Run(ro, ortInputs, outNames);
                    sw.Stop();
                    ort.Add(sw.Elapsed.TotalMilliseconds);
                }
            }
            Console.WriteLine(name + " [" + lokDesc + " vs " + ortDesc + "]: lokad " + Dist(lok) + " | ort " + Dist(ort)
                + " | reset " + Dist(resets) + " | convert=" + convSw.Elapsed.TotalMilliseconds.ToString("F1") + "ms"
                + " | validation=" + validationMs.ToString("F1") + "ms | disposal=outside"
                + " | maxdiff=" + maxDiff.ToString("E2"));
            Console.WriteLine("raw lok=[" + string.Join(",", lok.Select(v => v.ToString("F2"))) + "]"
                + " raw ort=[" + string.Join(",", ort.Select(v => v.ToString("F2"))) + "]");
        }
        finally
        {
            foreach (var v in ortInputs.Values) v.Dispose();
        }
    }

    static double Validate(string name, ComputationalGraph graph, InferenceSession session, Dictionary<string, ITensor> named, string[] outNames, ExecutionOptions lokadOpts)
    {
        graph.Reset();
        if (!graph.Execute(named, true, ExecutionProvider.CPU, lokadOpts)) throw new InvalidOperationException(name + ": lokad validation execute failed");
        var ortInputs = BuildOrtInputs(named, session.InputMetadata.Keys.ToArray());
        try
        {
            using var ro = new RunOptions();
            using var results = session.Run(ro, ortInputs, outNames);
            var outs = results.ToArray();
            if (outs.Length != outNames.Length) throw new InvalidOperationException(name + ": ort returned " + outs.Length + " outputs for " + outNames.Length + " requested.");
            double worst = 0;
            for (int oi = 0; oi < outNames.Length; oi++)
            {
                string onm = outNames[oi];
                var res = outs[oi];
                var shape = res.GetTensorTypeAndShape();
                if (shape.ElementDataType != Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float)
                    throw new InvalidOperationException(name + ": ort output is not float32: " + onm + ".");
                if (!graph.Outputs.TryGetValue(onm, out var lt) || lt is not Tensor<float> lf)
                    throw new InvalidOperationException(name + ": lokad output missing or not float32: " + onm);
                var la = lf.ToArray();
                var oa = res.GetTensorDataAsSpan<float>().ToArray();
                worst = Math.Max(worst, BenchValidate.RequireAgreement(
                    name + ":" + onm, lf.Dimensions.ToArray(), la,
                    shape.Shape.Select(d => checked((int)d)).ToArray(), oa, Tolerance));
            }
            return worst;
        }
        finally
        {
            foreach (var v in ortInputs.Values) v.Dispose();
        }
    }

    static Dictionary<string, OrtValue> BuildOrtInputs(Dictionary<string, ITensor> named, string[] inNames)
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

    static Dictionary<string, ITensor> ToNamed(string name, ITensor[] inputs, string[] names)
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

    static double[] TimeResets(ComputationalGraph graph, int iters)
    {
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
