using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Text;
using Lokad.Onnx;

static class OpDump
{
    static string Sanitize(string name)
    {
        var sb = new StringBuilder(name.Length);
        foreach (var c in name)
            sb.Append(char.IsLetterOrDigit(c) || c == '_' || c == '-' || c == '.' ? c : '_');
        return sb.ToString();
    }

    static ITensor ReadTensor(string name, string path)
    {
        var lines = File.ReadAllText(path).Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
        var head = lines[0].Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
        var dtype = head[0];
        int rank = int.Parse(head[1], CultureInfo.InvariantCulture);
        var dims = new int[rank];
        for (int i = 0; i < rank; i++) dims[i] = int.Parse(head[2 + i], CultureInfo.InvariantCulture);
        var vals = string.Join(" ", lines.Skip(1)).Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
        if (dtype == "float32")
        {
            var t = new DenseTensor<float>(dims);
            var sp = t.Buffer.Span;
            for (int i = 0; i < vals.Length; i++) sp[i] = float.Parse(vals[i], CultureInfo.InvariantCulture);
            t.Name = name;
            return t;
        }
        if (dtype == "float64")
        {
            var t = new DenseTensor<double>(dims);
            var sp = t.Buffer.Span;
            for (int i = 0; i < vals.Length; i++) sp[i] = double.Parse(vals[i], CultureInfo.InvariantCulture);
            t.Name = name;
            return t;
        }
        if (dtype == "int64")
        {
            var t = new DenseTensor<long>(dims);
            var sp = t.Buffer.Span;
            for (int i = 0; i < vals.Length; i++) sp[i] = long.Parse(vals[i], CultureInfo.InvariantCulture);
            t.Name = name;
            return t;
        }
        throw new InvalidOperationException("unsupported dtype: " + dtype);
    }

    static void WriteTensor(string path, ITensor t)
    {
        var sb = new StringBuilder();
        if (t is Tensor<float> tf)
        {
            sb.Append("float32 ").Append(tf.Dimensions.Length);
            foreach (var d in tf.Dimensions) sb.Append(' ').Append(d);
            sb.AppendLine();
            sb.AppendLine(string.Join(" ", tf.ToArray().Select(v => ((double)v).ToString("R", CultureInfo.InvariantCulture))));
        }
        else if (t is Tensor<double> td)
        {
            sb.Append("float64 ").Append(td.Dimensions.Length);
            foreach (var d in td.Dimensions) sb.Append(' ').Append(d);
            sb.AppendLine();
            sb.AppendLine(string.Join(" ", td.ToArray().Select(v => v.ToString("R", CultureInfo.InvariantCulture))));
        }
        else if (t is Tensor<long> tl)
        {
            sb.Append("int64 ").Append(tl.Dimensions.Length);
            foreach (var d in tl.Dimensions) sb.Append(' ').Append(d);
            sb.AppendLine();
            sb.AppendLine(string.Join(" ", tl.ToArray().Select(v => v.ToString(CultureInfo.InvariantCulture))));
        }
        else
        {
            throw new InvalidOperationException("unsupported output tensor type: " + t.GetType().Name);
        }
        File.WriteAllText(path, sb.ToString(), new UTF8Encoding(false));
    }

    static int Main(string[] args)
    {
        string? model = null, mode = null, outdir = null;
        var fed = new List<Tuple<string, string>>();
        for (int i = 0; i + 1 < args.Length; i += 2)
        {
            if (args[i] == "--model") model = args[i + 1];
            else if (args[i] == "--mode") mode = args[i + 1];
            else if (args[i] == "--outdir") outdir = args[i + 1];
            else if (args[i] == "--input")
            {
                int eq = args[i + 1].IndexOf('=');
                fed.Add(Tuple.Create(args[i + 1].Substring(0, eq), args[i + 1].Substring(eq + 1)));
            }
        }
        if (model is null || mode is null || outdir is null || fed.Count == 0) { Console.Error.WriteLine("Usage: OpDump --model m.onnx --mode scalar|simd|intrinsics --input name=file [--input ...] --outdir dir"); return 1; }
        bool useSimd, useIntrinsics;
        if (mode == "scalar") { useSimd = false; useIntrinsics = false; }
        else if (mode == "simd") { useSimd = true; useIntrinsics = false; }
        else if (mode == "intrinsics")
        {
            if (!Fma.IsSupported) { Console.Error.WriteLine("OpDump error: FMA not supported"); return 3; }
            useSimd = true; useIntrinsics = true;
        }
        else { Console.Error.WriteLine("OpDump error: unknown mode " + mode); return 1; }
        var graph = OnnxImport.Load(model);
        if (graph is null) { Console.Error.WriteLine("OpDump error: load failed"); return 4; }
        var dict = new Dictionary<string, ITensor>();
        foreach (var kv in fed) dict[kv.Item1] = ReadTensor(kv.Item1, kv.Item2);
        var opts = new ExecutionOptions(OptimizationMode.Speed, new TensorExecutionOptions(useSimd, useIntrinsics, 1));
        if (!graph.Execute(dict, true, ExecutionProvider.CPU, opts)) { Console.Error.WriteLine("OpDump error: exec failed: " + graph.LastErrorMessage); return 4; }
        Directory.CreateDirectory(outdir);
        foreach (var kv in graph.Outputs)
            WriteTensor(Path.Combine(outdir, "dotnet_" + Sanitize(kv.Key) + ".txt"), kv.Value);
        Console.WriteLine("OpDump ok: outputs=" + graph.Outputs.Count);
        return 0;
    }
}
