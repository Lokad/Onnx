using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Runtime.Intrinsics.X86;
using Lokad.Onnx;

static class E5Runner
{
    const string ExpectedModelHash = "CA456C06B3A9505DDFD9131408916DD79290368331E7D76BB621F1CBA6BC8665";
    const string ExpectedTokenizerHash = "CFC8146ABE2A0488E9E2A0C56DE7952F7C11AB059ECA145A0A727AFCE0DB2865";

    static string Sha256Of(string path)
    {
        return Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path)));
    }

    static string Fail(string message, int code)
    {
        Console.Error.WriteLine("E5Runner error: " + message);
        return code.ToString();
    }

    static int Main(string[] args)
    {
        string? model = null, tokenizer = null, cases = null, mode = null, output = null;
        for (int i = 0; i + 1 < args.Length; i += 2)
        {
            switch (args[i])
            {
                case "--model": model = args[i + 1]; break;
                case "--tokenizer": tokenizer = args[i + 1]; break;
                case "--cases": cases = args[i + 1]; break;
                case "--mode": mode = args[i + 1]; break;
                case "--output": output = args[i + 1]; break;
            }
        }
        if (model is null || tokenizer is null || cases is null || mode is null || output is null)
        {
            Console.Error.WriteLine("Usage: E5Runner --model <model.onnx> --tokenizer <sentencepiece.bpe.model> --cases <cases.json> --mode scalar|simd|intrinsics --output <out.json>");
            return 1;
        }
        if (!File.Exists(model)) { Console.Error.WriteLine("E5Runner error: model not found: " + model); return 2; }
        if (!File.Exists(tokenizer)) { Console.Error.WriteLine("E5Runner error: tokenizer not found: " + tokenizer); return 2; }
        if (!File.Exists(cases)) { Console.Error.WriteLine("E5Runner error: cases not found: " + cases); return 2; }
        string modelHash = Sha256Of(model);
        string tokenizerHash = Sha256Of(tokenizer);
        if (!modelHash.Equals(ExpectedModelHash, StringComparison.OrdinalIgnoreCase))
        {
            Console.Error.WriteLine("E5Runner error: model hash mismatch: " + modelHash);
            return 2;
        }
        if (!tokenizerHash.Equals(ExpectedTokenizerHash, StringComparison.OrdinalIgnoreCase))
        {
            Console.Error.WriteLine("E5Runner error: tokenizer hash mismatch: " + tokenizerHash);
            return 2;
        }
        bool useSimd, useIntrinsics;
        switch (mode)
        {
            case "scalar": useSimd = false; useIntrinsics = false; break;
            case "simd": useSimd = true; useIntrinsics = false; break;
            case "intrinsics":
                if (!Fma.IsSupported)
                {
                    Console.Error.WriteLine("E5Runner error: intrinsics requested but x86 FMA is not supported on this machine.");
                    return 3;
                }
                useSimd = true; useIntrinsics = true; break;
            default:
                Console.Error.WriteLine("E5Runner error: unknown mode: " + mode);
                return 1;
        }
        var execOptions = new ExecutionOptions(OptimizationMode.Speed, new TensorExecutionOptions(useSimd, useIntrinsics));

        List<(string id, string text)> caseList = new List<(string, string)>();
        using (var doc = JsonDocument.Parse(File.ReadAllText(cases)))
        {
            foreach (var el in doc.RootElement.GetProperty("cases").EnumerateArray())
            {
                caseList.Add((el.GetProperty("id").GetString() ?? "", el.GetProperty("text").GetString() ?? ""));
            }
        }
        caseList.Sort((a, b) => string.CompareOrdinal(a.id, b.id));

        var graph = OnnxImport.Load(model);
        if (graph is null)
        {
            Console.Error.WriteLine("E5Runner error: could not load model.");
            return 4;
        }

        var sb = new StringBuilder();
        sb.Append("{\"schemaVersion\":1");
        sb.Append(",\"modelHash\":\"").Append(modelHash).Append("\"");
        sb.Append(",\"tokenizerHash\":\"").Append(tokenizerHash).Append("\"");
        sb.Append(",\"requestedMode\":\"").Append(mode).Append("\"");
        sb.Append(",\"observed\":{\"useSimd\":").Append(useSimd ? "true" : "false");
        sb.Append(",\"useIntrinsics\":").Append(useIntrinsics ? "true" : "false").Append("}");
        sb.Append(",\"cases\":[");
        bool firstCase = true;
        foreach (var (id, text) in caseList)
        {
            var inputs = Text.RobertaTokenizeFromFile(text, tokenizer);
            if (inputs is null)
            {
                Console.Error.WriteLine("E5Runner error: tokenization failed for case: " + id);
                return 4;
            }
            if (!graph.Execute(inputs, true, ExecutionProvider.CPU, execOptions))
            {
                Console.Error.WriteLine("E5Runner error: inference failed for case " + id + ": " + graph.LastErrorMessage);
                return 4;
            }
            ITensor? hidden = null;
            foreach (var kv in graph.Outputs)
            {
                if (kv.Key.Contains("last_hidden_state", StringComparison.Ordinal)) hidden = kv.Value;
            }
            hidden ??= graph.Outputs.Values.OfType<Tensor<float>>().FirstOrDefault();
            if (hidden is not Tensor<float> h)
            {
                Console.Error.WriteLine("E5Runner error: last_hidden_state is not a float tensor for case: " + id);
                return 4;
            }
            var mask = (Tensor<long>)inputs.Single(t => t.Name == "attention_mask");
            float[] embedding = PooledEmbedding(h, mask);
            if (!firstCase) sb.Append(",");
            firstCase = false;
            sb.Append("{\"id\":\"").Append(id).Append("\",\"tensors\":[");
            bool firstTensor = true;
            foreach (var t in inputs.OrderBy(t => t.Name, StringComparer.Ordinal))
            {
                if (!firstTensor) sb.Append(",");
                firstTensor = false;
                AppendTensor(sb, t);
            }
            sb.Append(",");
            AppendTensor(sb, h, "last_hidden_state");
            sb.Append(",");
            AppendEmbedding(sb, embedding, h.Dimensions[h.Dimensions.Length - 1]);
            sb.Append("]}");
            foreach (var v in h.ToArray())
            {
                if (!float.IsFinite(v))
                {
                    Console.Error.WriteLine("E5Runner error: non-finite output for case: " + id);
                    return 4;
                }
            }
        }
        sb.Append("]}");
        var outDir = Path.GetDirectoryName(Path.GetFullPath(output));
        if (!string.IsNullOrEmpty(outDir)) Directory.CreateDirectory(outDir);
        File.WriteAllText(output, sb.ToString(), new UTF8Encoding(false));
        Console.WriteLine("E5Runner ok: mode=" + mode + " cases=" + caseList.Count + " output=" + output);
        return 0;
    }

    static float[] PooledEmbedding(Tensor<float> hidden, Tensor<long> mask)
    {
        int rank = hidden.Dimensions.Length;
        int seq = hidden.Dimensions[rank - 2];
        int width = hidden.Dimensions[rank - 1];
        float[] flat = hidden.ToArray();
        long[] m = mask.ToArray();
        double[] acc = new double[width];
        double count = 0;
        for (int s = 0; s < seq; s++)
        {
            double w = m[s];
            count += w;
            for (int h = 0; h < width; h++) acc[h] += flat[s * width + h] * w;
        }
        if (count == 0) count = 1;
        double norm = 0;
        float[] emb = new float[width];
        for (int h = 0; h < width; h++)
        {
            emb[h] = (float)(acc[h] / count);
            norm += (double)emb[h] * emb[h];
        }
        norm = Math.Sqrt(norm);
        if (norm == 0) norm = 1;
        for (int h = 0; h < width; h++) emb[h] = (float)(emb[h] / norm);
        return emb;
    }

    static void AppendTensor(StringBuilder sb, ITensor t)
    {
        AppendTensor(sb, t, t.Name);
    }

    static void AppendTensor(StringBuilder sb, ITensor t, string name)
    {
        sb.Append("{\"name\":\"").Append(name).Append("\"");
        if (t is Tensor<long> tl)
        {
            sb.Append(",\"dtype\":\"int64\"");
            AppendShape(sb, tl.Dimensions);
            sb.Append(",\"values\":[");
            bool first = true;
            foreach (var v in tl.ToArray())
            {
                if (!first) sb.Append(",");
                first = false;
                sb.Append(v.ToString(CultureInfo.InvariantCulture));
            }
            sb.Append("]}");
        }
        else if (t is Tensor<float> tf)
        {
            sb.Append(",\"dtype\":\"float32\"");
            AppendShape(sb, tf.Dimensions);
            sb.Append(",\"values\":[");
            bool first = true;
            foreach (var v in tf.ToArray())
            {
                if (!first) sb.Append(",");
                first = false;
                sb.Append(((double)v).ToString("R", CultureInfo.InvariantCulture));
            }
            sb.Append("]}");
        }
        else
        {
            throw new InvalidOperationException("Unsupported tensor type for " + name + ": " + t.GetType().FullName);
        }
    }

    static void AppendEmbedding(StringBuilder sb, float[] embedding, int width)
    {
        sb.Append("{\"name\":\"embedding\",\"dtype\":\"float32\",\"shape\":[").Append(width).Append("],\"values\":[");
        bool first = true;
        foreach (var v in embedding)
        {
            if (!first) sb.Append(",");
            first = false;
            sb.Append(((double)v).ToString("R", CultureInfo.InvariantCulture));
        }
        sb.Append("]}");
    }

    static void AppendShape(StringBuilder sb, ReadOnlySpan<int> dims)
    {
        sb.Append(",\"shape\":[");
        bool first = true;
        foreach (var d in dims)
        {
            if (!first) sb.Append(",");
            first = false;
            sb.Append(d.ToString(CultureInfo.InvariantCulture));
        }
        sb.Append("]");
    }
}
