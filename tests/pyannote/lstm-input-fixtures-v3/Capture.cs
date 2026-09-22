using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

internal record ArrayRecord(string file, int[] shape, int values, long bytes, string sha256);
internal record CallRecord(string name, int index, string node, int opset, Dictionary<string, object>? attributes,
    string[] input_names, string[] output_names, ArrayRecord?[] inputs, ArrayRecord[] outputs);
internal static class Capture
{
    static void Require(bool value, string message) { if (!value) throw new InvalidDataException(message); }
    static string Sha(string path) { using var f = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(f)); }
    static string Hash(float[] values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
    static float[] Read(string path) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(path)).ToArray();

    static int Main(string[] args)
    {
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux()) throw new PlatformNotSupportedException();
        Require(args.Length == 2, "specification new-output-directory");
        Require(Environment.ProcessorCount == 1 && Process.GetCurrentProcess().ProcessorAffinity.ToInt64() == 4, "CPU2 before CLR startup");
        Require(!Environment.GetEnvironmentVariables().Keys.Cast<string>().Any(k => k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase)
            || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)), "No overrides");
        using var document = JsonDocument.Parse(File.ReadAllBytes(args[0])); var spec = document.RootElement;
        Require(Sha(typeof(ComputationalGraph).Assembly.Location) == spec.GetProperty("core").GetString(), "Core identity");
        Require(Sha(typeof(Community1Diarizer).Assembly.Location) == spec.GetProperty("data").GetString(), "Data identity");
        Require(!Directory.Exists(args[1]), "Refuse existing output"); Directory.CreateDirectory(args[1]);
        string output = Path.GetFullPath(args[1]); var stored = new Dictionary<string, ArrayRecord>(); var hashes = new Dictionary<string, string>();
        ArrayRecord Save(string key, Tensor<float> tensor)
        {
            var values = tensor.ToArray(); Require(values.All(float.IsFinite), "Finite captured tensor");
            string hash = Hash(values); int[] shape = tensor.Dimensions.ToArray();
            if (stored.TryGetValue(key, out var old))
            {
                Require(old.sha256 == hash && old.shape.SequenceEqual(shape), "Exact capture repeat"); return old;
            }
            if (!hashes.TryGetValue(hash, out string? file))
            {
                file = hashes.Count.ToString("D4") + ".f32";
                File.WriteAllBytes(Path.Combine(output, file), MemoryMarshal.AsBytes(values.AsSpan()).ToArray()); hashes.Add(hash, file);
            }
            var item = new ArrayRecord(file, shape, values.Length, values.LongLength * 4, hash); stored.Add(key, item); return item;
        }
        string model = spec.GetProperty("model").GetString()!;
        Require(Sha(model) == spec.GetProperty("model_sha256").GetString(), "Model identity");
        var graph = OnnxImport.Load(model, 32L * 1024 * 1024) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
        var initializers = graph.Initializers.Values.OfType<Tensor<float>>().Select(t => (Tensor: t, Hash: Hash(t.ToArray()))).ToArray();
        var nodes = graph.Nodes.Where(n => n.Op == OpType.LSTM).ToArray();
        Require(nodes.Length == 4 && nodes.Select(n => n.Name).Distinct().Count() == 4, "Four recurrent nodes");
        var expectedNames = new[] { "/lstm/LSTM", "/lstm/LSTM_1", "/lstm/LSTM_2", "/lstm/LSTM_3" };
        Require(nodes.Select(n => n.Name).SequenceEqual(expectedNames), "Original recurrent order");
        var calls = new List<CallRecord>(); var checks = new List<object>(); var held = new List<(Tensor<float> Value, string Hash)>();
        void CheckHeld() { foreach (var h in held) Require(Hash(h.Value.ToArray()) == h.Hash, "Held graph output changed"); }
        foreach (var c in spec.GetProperty("cases").EnumerateArray())
        {
            string name = c.GetProperty("name").GetString()!, path = c.GetProperty("input").GetString()!;
            Require(Sha(path) == c.GetProperty("input_sha256").GetString(), "Input identity");
            var input = new DenseTensor<float>(Read(path), new[] { 1, 1, 160000 }); string inputHash = Hash(input.ToArray());
            string expected = c.GetProperty("selected_sha256").GetString()!;
            foreach (bool capture in new[] { false, true })
            {
                var options = ExecutionOptions.Memory;
                var execution = graph.CreateExecution(options);
                for (int repeat = 0; repeat < 2; repeat++)
                {
                    CheckHeld(); execution.Reset();
                    if (capture)
                        foreach (string value in nodes.SelectMany(n => n.Inputs.Concat(n.Outputs)).Where(n => !string.IsNullOrEmpty(n)).Distinct())
                            execution.Outputs.TryAdd(value, null);
                    Require(execution.Execute(new Dictionary<string, ITensor> { ["waveform"] = input }, true, ExecutionProvider.CPU, options), execution.LastErrorMessage ?? "Graph execution");
                    var result = (Tensor<float>)execution.Outputs["scores"]!; string actual = Hash(result.ToArray());
                    Require(actual == expected && Hash(input.ToArray()) == inputHash, "Selected graph bits / readonly input");
                    held.Add((result, actual)); CheckHeld();
                    if (capture)
                    {
                        for (int i = 0; i < nodes.Length; i++)
                        {
                            var n = nodes[i]; Require(n.RequiredInt("hidden_size") == 128 && n.Attr("direction", "forward") == "bidirectional", "LSTM attributes");
                            var inputs = n.Inputs.Select(t => string.IsNullOrEmpty(t) ? null : Save(name + "/" + t, (Tensor<float>)execution.GetInputTensor(t))).ToArray();
                            var outputs = n.Outputs.Select(t => Save(name + "/" + t, (Tensor<float>)execution.GetInputTensor(t))).ToArray();
                            Require(inputs.Length == 7 && inputs[4] is null && outputs.Length == 3, "Complete optional slots");
                            Require(inputs[0]!.shape.SequenceEqual(new[] { 589, 1, i == 0 ? 60 : 256 }), "Actual LSTM shape");
                            Require(outputs[0].shape.SequenceEqual(new[] { 589, 2, 1, 128 }), "Full Y shape");
                            if (repeat == 0) calls.Add(new(name, i, n.Name, n.OpsetVersion, n.Attributes, n.Inputs, n.Outputs, inputs, outputs));
                        }
                        Save(name + "/scores", result);
                    }
                    checks.Add(new { name, capture, repeat, exact = true, input_unchanged = true, sha256 = actual });
                }
                execution.Reset(); CheckHeld();
            }
            Console.WriteLine(name + " captured and reconciled");
        }
        Require(initializers.All(t => Hash(t.Tensor.ToArray()) == t.Hash), "Readonly initializers");
        Require(calls.Count == 12 && checks.Count == 12, "Complete three-crop coverage");
        long bytes = Directory.EnumerateFiles(output, "*.f32").Sum(f => new FileInfo(f).Length); Require(bytes <= 128L * 1024 * 1024, "Frozen fixture bound");
        Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.FileName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Managed capture only");
        File.WriteAllText(Path.Combine(output, "result.json"), JsonSerializer.Serialize(new { passed = true, calls, tensors = stored, checks,
            model_sha256 = Sha(model), core = Sha(typeof(ComputationalGraph).Assembly.Location), data = Sha(typeof(Community1Diarizer).Assembly.Location),
            executable = Sha(typeof(Capture).Assembly.Location), runtime = Environment.Version.ToString(), pid = Environment.ProcessId,
            vector_count = Vector<float>.Count, avx512 = Avx512F.IsSupported, tensor_bytes = bytes, distinct_files = hashes.Count,
            original_graph_unchanged = true, captured_output_bindings_extended = true, capture_policy = "Extra per-execution output bindings retain operands/endpoints; ordinary Memory and prior selected scores exactly reconciled",
            no_performance_measurement = true, held_outputs_unchanged = true }, new JsonSerializerOptions { WriteIndented = true }));
        return 0;
    }
}
