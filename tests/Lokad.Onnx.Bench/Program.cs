using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Lokad.Onnx;
using Microsoft.ML.OnnxRuntime;

static class Bench
{
    const int Warm = 3;
    const double Tolerance = 1e-4;

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
        var selected = new List<string>();
        string modeName = "auto";
        int threads = Environment.ProcessorCount;
        int iters = 7;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--mode" && i + 1 < args.Length) modeName = args[++i];
            else if (args[i] == "--threads" && i + 1 < args.Length && int.TryParse(args[i + 1], out var t) && t >= 1) { threads = t; i++; }
            else if (args[i] == "--iters" && i + 1 < args.Length && int.TryParse(args[i + 1], out var k) && k >= 1) { iters = k; i++; }
            else if (args[i] == "all" || assets.ContainsKey(args[i])) { if (args[i] != "all" && !selected.Contains(args[i], StringComparer.OrdinalIgnoreCase)) selected.Add(args[i]); }
            else { Console.WriteLine("usage: Bench [e5 dinov2 dinov3 resnet50 gpt2 all] [--mode auto|scalar|simd|intrinsics] [--threads N] [--iters N]"); return 2; }
        }
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
        Console.WriteLine("host=" + Environment.MachineName + " procs=" + Environment.ProcessorCount
            + " fma=" + System.Runtime.Intrinsics.X86.Fma.IsSupported
            + " runtime=" + RuntimeInformation.FrameworkDescription
            + " ort=" + typeof(InferenceSession).Assembly.GetName().Version
            + " mode=" + modeName + " threads=" + threads + " iters=" + iters);
        if (selected.Contains("e5", StringComparer.OrdinalIgnoreCase))
        {
            var e5 = assets["e5"];
            CompareE5("e5-8tok", e5[0], e5[1], "query: hello world", tensorOpts, threads, iters);
            CompareE5("e5-30tok", e5[0], e5[1], "query: The quick brown fox jumps over the lazy dog near the river bank in springtime weather for a pleasant afternoon walk", tensorOpts, threads, iters);
        }
        if (selected.Contains("dinov2", StringComparer.OrdinalIgnoreCase)) CompareVision("dinov2-224", assets["dinov2"][0], tensorOpts, threads, iters);
        if (selected.Contains("dinov3", StringComparer.OrdinalIgnoreCase)) CompareVision("dinov3-224", assets["dinov3"][0], tensorOpts, threads, iters);
        if (selected.Contains("resnet50", StringComparer.OrdinalIgnoreCase)) CompareVision("resnet50-224", assets["resnet50"][0], tensorOpts, threads, iters);
        if (selected.Contains("gpt2", StringComparer.OrdinalIgnoreCase)) CompareGpt2("gpt2-4tok", assets["gpt2"][0], tensorOpts, threads, iters);
        return 0;
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

    static void CompareE5(string name, string model, string tokenizer, string text, TensorExecutionOptions tensorOpts, int threads, int iters)
    {
        var inputs = Text.RobertaTokenizeFromFile(text, tokenizer)!;
        // The tokenizer names its outputs; verify the exact session match instead of
        // silently falling back to positions.
        using var probe = new InferenceSession(model);
        var inNames = probe.InputMetadata.Keys.ToArray();
        if (inputs.Length != inNames.Length) throw new InvalidOperationException(name + ": tokenized " + inputs.Length + " inputs but the session wants " + inNames.Length + ".");
        for (int i = 0; i < inputs.Length; i++)
            if (inputs[i].Name != inNames[i]) throw new InvalidOperationException(name + ": input " + i + " is '" + inputs[i].Name + "', session wants '" + inNames[i] + "'.");
        Compare(name, model, inputs, tensorOpts, threads, iters);
    }

    static void CompareVision(string name, string model, TensorExecutionOptions tensorOpts, int threads, int iters)
    {
        var flat = new float[1 * 3 * 224 * 224];
        for (int i = 0; i < flat.Length; i++) flat[i] = 0.5f;
        var input = new DenseTensor<float>(flat, new[] { 1, 3, 224, 224 });
        using var sessionProbe = new InferenceSession(model);
        input.Name = sessionProbe.InputMetadata.Keys.First();
        Compare(name, model, new ITensor[] { input }, tensorOpts, threads, iters);
    }

    static void CompareGpt2(string name, string model, TensorExecutionOptions tensorOpts, int threads, int iters)
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
        Compare(name, model, inputs.ToArray(), tensorOpts, threads, iters);
    }

    static void Compare(string name, string model, ITensor[] inputs, TensorExecutionOptions tensorOpts, int threads, int iters)
    {
        var graph = OnnxImport.Load(model)!;
        using var ortDefault = new InferenceSession(model);
        var inNames = ortDefault.InputMetadata.Keys.ToArray();
        var outNames = ortDefault.OutputMetadata.Keys.ToArray();
        var named = ToNamed(name, inputs, inNames);
        Console.WriteLine("case " + name + ": model=" + Path.GetFileName(Path.GetDirectoryName(model)) + "/model.onnx"
            + " bytes=" + new FileInfo(model).Length + " sha12=" + ShortHash(model)
            + " inputs=[" + string.Join(",", named.Select(kv => kv.Key + ":" + string.Join("x", kv.Value.Dims))) + "]"
            + " outputs=[" + string.Join(",", outNames) + "]");
        var valSw = Stopwatch.StartNew();
        double maxDiff = Validate(name, graph, ortDefault, named, outNames, ExecutionOptions.Default);
        var matchedOpts = new ExecutionOptions(OptimizationMode.Speed, tensorOpts);
        maxDiff = Math.Max(maxDiff, Validate(name, graph, ortDefault, named, outNames, matchedOpts));
        valSw.Stop();
        double copyMs = valSw.Elapsed.TotalMilliseconds;
        TimedRun(name, graph, named, outNames, ExecutionOptions.Default, "defaults", "ort-defaults", iters, copyMs, maxDiff, ortDefault);
        using var ortMatched = MatchedSession(model, threads);
        TimedRun(name, graph, named, outNames, matchedOpts, "mode-threads=" + threads, "intraop=" + threads + " interop=1 seq", iters, copyMs, maxDiff, ortMatched);
    }

    static void TimedRun(string name, ComputationalGraph graph, Dictionary<string, ITensor> named, string[] outNames,
        ExecutionOptions lokadOpts, string lokDesc, string ortDesc, int iters, double copyMs, double maxDiff, InferenceSession ortSession)
    {
        var ortInputs = BuildOrtInputs(named, ortSession.InputMetadata.Keys.ToArray());
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
                graph.Reset();
                sw.Restart();
                if (!graph.Execute(named, true, ExecutionProvider.CPU, lokadOpts)) throw new InvalidOperationException(name + ": lokad execute failed");
                sw.Stop();
                lok.Add(sw.Elapsed.TotalMilliseconds);
                sw.Restart();
                using var results = ortSession.Run(ro, ortInputs, outNames);
                sw.Stop();
                ort.Add(sw.Elapsed.TotalMilliseconds);
            }
            Console.WriteLine(name + " [" + lokDesc + " vs " + ortDesc + "]: lokad " + Dist(lok) + " | ort " + Dist(ort)
                + " | reset " + Dist(resets) + " | copy=" + copyMs.ToString("F1") + "ms"
                + " | maxdiff=" + maxDiff.ToString("E2"));
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
                if (!graph.Outputs.TryGetValue(onm, out var lt) || lt is not Tensor<float> lf)
                    throw new InvalidOperationException(name + ": lokad output missing or not float32: " + onm);
                var la = lf.ToArray();
                var oa = res.GetTensorDataAsSpan<float>().ToArray();
                if (la.Length != oa.Length) throw new InvalidOperationException(name + ": output length mismatch on " + onm);
                for (int i = 0; i < la.Length; i++)
                {
                    double rel = Math.Abs(la[i] - oa[i]) / (1.0 + Math.Abs(oa[i]));
                    if (rel > worst) worst = rel;
                }
            }
            if (worst > Tolerance) throw new InvalidOperationException(name + ": outputs diverge (max rel diff " + worst.ToString("E2") + ").");
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

    static InferenceSession MatchedSession(string model, int threads)
    {
        var so = new SessionOptions();
        so.IntraOpNumThreads = threads;
        so.InterOpNumThreads = 1;
        so.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        return new InferenceSession(model, so);
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
