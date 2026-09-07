using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Lokad.Onnx;
using Microsoft.ML.OnnxRuntime;

static class Bench
{
    const int Warm = 3;
    const int Iters = 7;

    static int Main(string[] args)
    {
        var root = FindRoot();
        var e5 = Path.Combine(root, "models", "multilingual-e5-small", "model.onnx");
        var tok = Path.Combine(root, "models", "multilingual-e5-small", "sentencepiece.bpe.model");
        var v2 = Path.Combine(root, "models", "dinov2-small-onnx", "model.onnx");
        var v3 = Path.Combine(root, "models", "dinov3-vits16", "onnx", "model.onnx");
        var rn = Path.Combine(root, "models", "resnet50-onnx", "model.onnx");
        var g2 = Path.Combine(root, "models", "gpt2-onnx", "onnx", "model.onnx");
        foreach (var f in new[] { e5, tok, v2, v3, rn, g2 })
        {
            if (!File.Exists(f)) { Console.WriteLine("missing asset, skipping all: " + f); return 1; }
        }
        Console.WriteLine("host=" + Environment.MachineName + " fma=" + System.Runtime.Intrinsics.X86.Fma.IsSupported);
        CompareE5("e5-8tok", e5, tok, "query: hello world");
        CompareE5("e5-30tok", e5, tok, "query: The quick brown fox jumps over the lazy dog near the river bank in springtime weather for a pleasant afternoon walk");
        CompareVision("dinov2-224", v2);
        CompareVision("dinov3-224", v3);
        CompareVision("resnet50-224", rn);
        CompareGpt2("gpt2-4tok", g2);
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

    static void CompareE5(string name, string model, string tokenizer, string text)
    {
        var inputs = Text.RobertaTokenizeFromFile(text, tokenizer)!;
        Compare(name, model, inputs);
    }

    static void CompareVision(string name, string model)
    {
        var flat = new float[1 * 3 * 224 * 224];
        for (int i = 0; i < flat.Length; i++) flat[i] = 0.5f;
        var input = new DenseTensor<float>(flat, new[] { 1, 3, 224, 224 });
        using var sessionProbe = new InferenceSession(model);
        input.Name = sessionProbe.InputMetadata.Keys.First();
        Compare(name, model, new ITensor[] { input });
    }

    static void CompareGpt2(string name, string model)
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
        Compare(name, model, inputs.ToArray());
    }

    static void Compare(string name, string model, ITensor[] inputs)
    {
        var graph = OnnxImport.Load(model)!;
        using var session = new InferenceSession(model);
        var inNames = session.InputMetadata.Keys.ToArray();
        var named = ToNamed(inputs, inNames);
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
        try
        {
            var outName = session.OutputMetadata.Keys.First();
            double lokFirst = double.NaN, ortFirst = double.NaN;
            using var runOptions = new RunOptions();
            var outNames = new[] { outName };
            double[] lok = TimeIt(() => {
                graph.Reset();
                if (!graph.Execute(named, true)) throw new InvalidOperationException("lokad execute failed");
                var o = (Tensor<float>)graph.Outputs[outName];
                lokFirst = o.ToArray()[0];
            });
            double[] ort = TimeIt(() => {
                using var results = session.Run(runOptions, ortInputs, outNames);
                var o = results.First().GetTensorDataAsSpan<float>().ToArray();
                ortFirst = o[0];
            });
            Console.WriteLine(name + ": lokad best=" + Best(lok).ToString("F1") + "ms median=" + Median(lok).ToString("F1") + "ms | ort best=" + Best(ort).ToString("F1") + "ms median=" + Median(ort).ToString("F1") + "ms | first lok=" + lokFirst.ToString("F5") + " ort=" + ortFirst.ToString("F5"));
        }
        finally
        {
            foreach (var v in ortInputs.Values) v.Dispose();
        }
    }

    static Dictionary<string, ITensor> ToNamed(ITensor[] inputs, string[] names)
    {
        var d = new Dictionary<string, ITensor>();
        foreach (var t in inputs)
        {
            var match = names.FirstOrDefault(n => n == t.Name) ?? names.First(n => !d.ContainsKey(n));
            d[match] = t;
        }
        return d;
    }

    static double[] TimeIt(Action run)
    {
        for (int i = 0; i < Warm; i++) run();
        var sw = new Stopwatch();
        var ts = new List<double>();
        for (int i = 0; i < Iters; i++) { sw.Restart(); run(); sw.Stop(); ts.Add(sw.Elapsed.TotalMilliseconds); }
        return ts.ToArray();
    }

    static double Best(double[] ts) => ts.Min();
    static double Median(double[] ts) => ts.OrderBy(x => x).ElementAt(ts.Length / 2);
}
