using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Lokad.Onnx;

static class Program
{
    static int Main(string[] args)
    {
        try { Run(args); return 0; }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }

    static void Run(string[] args)
    {
        if (args.Length > 0 && args[0] == "api") { ApiRunner.Run(args.Skip(1).ToArray()); return; }
        var arguments = new Dictionary<string, string>(StringComparer.Ordinal);
        if (args.Length % 2 != 0) throw new ArgumentException("Expected --model, --reference, --output, and optional --mode/--case pairs.");
        for (int i = 0; i < args.Length; i += 2)
        {
            if (args[i] is not ("--model" or "--reference" or "--output" or "--mode" or "--case"))
                throw new ArgumentException("Unknown argument: " + args[i]);
            arguments.Add(args[i], args[i + 1]);
        }
        string modelPath = arguments["--model"], referencePath = arguments["--reference"], output = arguments["--output"];
        string mode = arguments.GetValueOrDefault("--mode", "default");
        var options = mode switch
        {
            "scalar" => new ExecutionOptions(OptimizationMode.Memory, new TensorExecutionOptions(false, false, 1)),
            "simd" => new ExecutionOptions(OptimizationMode.Memory, new TensorExecutionOptions(true, false, 1)),
            "default" => ExecutionOptions.Memory,
            _ => throw new ArgumentException("Unknown mode: " + mode),
        };
        Directory.CreateDirectory(output);
        var graph = OnnxImport.Load(modelPath, 64L * 1024 * 1024)
            ?? throw new InvalidOperationException(OnnxImport.LastErrorMessage, OnnxImport.LastErrorCause);
        var execution = graph.CreateExecution(options);
        using var reference = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(referencePath, "manifest.json")));
        var results = new List<object>();
        foreach (var item in reference.RootElement.GetProperty("cases").EnumerateArray())
        {
            string id = item.GetProperty("id").GetString()!;
            if (arguments.TryGetValue("--case", out string? selected) && selected != id) continue;
            string caseDirectory = Path.Combine(referencePath, id);
            var ids = ReadTensor<long>(caseDirectory, item.GetProperty("input_ids"), "<i8");
            var mask = ReadTensor<long>(caseDirectory, item.GetProperty("attention_mask"), "<i8");
            long[] originalIds = ids.ToArray(), originalMask = mask.ToArray();
            var inputs = new Dictionary<string, ITensor> { ["input_ids"] = ids, ["attention_mask"] = mask };
            try
            {
                if (!execution.Execute(inputs, true, ExecutionProvider.CPU, options))
                    throw new InvalidOperationException(id + ": " + execution.LastErrorMessage);
                var hidden = execution.Outputs["last_hidden_state"] as Tensor<float>
                    ?? throw new InvalidDataException("Missing float hidden states");
                if (!originalIds.SequenceEqual(ids.ToArray()) || !originalMask.SequenceEqual(mask.ToArray()))
                    throw new InvalidDataException("Execution mutated input tensors");
                string destination = Path.Combine(output, id);
                Directory.CreateDirectory(destination);
                WriteNpy(Path.Combine(destination, "last_hidden_state.npy"), hidden.ToArray(), hidden.Dimensions.ToArray(), "<f4");
                for (int dimension = 128; dimension <= 768; dimension = dimension == 512 ? 768 : dimension * 2)
                {
                    float[] embedding = Pool(hidden, mask, dimension);
                    WriteNpy(Path.Combine(destination, $"embedding_{dimension}.npy"), embedding, new[] { ids.Dimensions[0], dimension }, "<f4");
                    if (dimension == 768) break;
                }
                results.Add(new { id, shape = hidden.Dimensions.ToArray() });
                Console.WriteLine("PASS managed execution " + id);
            }
            finally { execution.Reset(); }
        }
        if (results.Count == 0) throw new ArgumentException("No cases selected");
        File.WriteAllText(Path.Combine(output, "manifest.json"), JsonSerializer.Serialize(new
        {
            schemaVersion = 1, modelSha256 = Hash(modelPath), mode, cases = results,
            coreSha256 = Hash(typeof(OnnxImport).Assembly.Location),
            runnerSha256 = Hash(typeof(Program).Assembly.Location),
            referenceSha256 = Hash(Path.Combine(referencePath, "manifest.json")),
            runtime = Environment.Version.ToString(),
        }, new JsonSerializerOptions { WriteIndented = true }));
    }

    static float[] Pool(Tensor<float> hidden, Tensor<long> mask, int dimensions)
    {
        int batch = hidden.Dimensions[0], length = hidden.Dimensions[1], width = hidden.Dimensions[2];
        var result = new float[checked(batch * dimensions)];
        for (int b = 0; b < batch; b++)
        {
            int count = 0;
            for (int t = 0; t < length; t++)
            {
                if (mask.GetValue(b * length + t) == 0) continue;
                count++;
                for (int d = 0; d < dimensions; d++)
                    result[b * dimensions + d] += hidden.GetValue((b * length + t) * width + d);
            }
            if (count == 0) throw new InvalidDataException("Empty token mask");
            float squaredNorm = 0;
            for (int d = 0; d < dimensions; d++)
            {
                float value = result[b * dimensions + d] /= count;
                squaredNorm += value * value;
            }
            float norm = MathF.Sqrt(squaredNorm);
            if (!float.IsFinite(norm) || norm <= 1e-12f) throw new InvalidDataException("Invalid vector norm");
            for (int d = 0; d < dimensions; d++) result[b * dimensions + d] /= norm;
        }
        return result;
    }

    static string Hash(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    static DenseTensor<T> ReadTensor<T>(string directory, JsonElement info, string descriptor) where T : unmanaged
    {
        string path = Path.Combine(directory, info.GetProperty("file").GetString()!);
        if (Hash(path) != info.GetProperty("sha256").GetString()) throw new InvalidDataException("Input artifact changed: " + path);
        byte[] data = File.ReadAllBytes(path);
        if (!data.AsSpan(0, 6).SequenceEqual(new byte[] { 147, 78, 85, 77, 80, 89 }) || data[6] != 1 || data[7] != 0)
            throw new InvalidDataException("Expected little-endian NumPy v1 array");
        int headerLength = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(8, 2));
        string header = Encoding.ASCII.GetString(data, 10, headerLength);
        if (!header.Contains("'descr': '" + descriptor + "'", StringComparison.Ordinal) || !header.Contains("'fortran_order': False", StringComparison.Ordinal))
            throw new InvalidDataException("Unexpected NumPy dtype/order");
        int[] shape = info.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        string tuple = Regex.Match(header, @"'shape':\s*\(([^)]*)\)").Groups[1].Value;
        int[] headerShape = tuple.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(x => int.Parse(x.Trim(), System.Globalization.CultureInfo.InvariantCulture)).ToArray();
        if (!shape.SequenceEqual(headerShape)) throw new InvalidDataException("NumPy shape differs from manifest");
        var values = MemoryMarshal.Cast<byte, T>(data.AsSpan(10 + headerLength)).ToArray();
        if (shape.Aggregate(1L, (a, b) => checked(a * b)) != values.LongLength) throw new InvalidDataException("NumPy byte count mismatch");
        return new DenseTensor<T>(values, shape);
    }

    internal static void WriteNpy<T>(string path, T[] values, int[] shape, string descriptor) where T : unmanaged
    {
        string header = "{'descr': '" + descriptor + "', 'fortran_order': False, 'shape': (" + string.Join(", ", shape) + (shape.Length == 1 ? "," : "") + "), }";
        int padding = (16 - (10 + header.Length + 1) % 16) % 16;
        byte[] text = Encoding.ASCII.GetBytes(header + new string(' ', padding) + "\n");
        using var stream = File.Create(path);
        stream.Write(new byte[] { 147, 78, 85, 77, 80, 89, 1, 0 });
        using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);
        writer.Write(checked((ushort)text.Length));
        writer.Write(text);
        stream.Write(MemoryMarshal.AsBytes(values.AsSpan()));
    }
}
