using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

internal record ArrayRecord(string file, int[] shape, int values, long bytes, string sha256);
internal record CallRecord(string name, int step, int index, string node, int opset,
    Dictionary<string, object>? attributes, string[] input_names, string[] output_names,
    ArrayRecord?[] inputs, ArrayRecord[] outputs);

internal static class Capture
{
    static void Require(bool value, string message) { if (!value) throw new InvalidDataException(message); }
    static string Sha(string path) { using var f = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(f)); }
    static string Hash(byte[] values) => Convert.ToHexStringLower(SHA256.HashData(values));
    static byte[] Bytes(ITensor value) => value switch
    {
        Tensor<float> f => MemoryMarshal.AsBytes(f.ToArray().AsSpan()).ToArray(),
        Tensor<int> i => MemoryMarshal.AsBytes(i.ToArray().AsSpan()).ToArray(),
        Tensor<long> l => MemoryMarshal.AsBytes(l.ToArray().AsSpan()).ToArray(),
        _ => throw new InvalidDataException("Unexpected dtype")
    };
    static string Bits(ITensor value) => Hash(Bytes(value));
    static string Text(JsonElement value) => value.GetString() ?? throw new InvalidDataException();
    static int[] Shape(JsonElement value) => value.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
    static object Attribute(object value) => value is ITensor tensor
        ? new { dtype = tensor.ElementType.ToString(), shape = tensor.Dims, sha256 = Bits(tensor) }
        : JsonSerializer.SerializeToElement(value);
    static string Nodes(ComputationalGraph graph) => Hash(JsonSerializer.SerializeToUtf8Bytes(graph.Nodes.Select(n => new
    {
        n.ID, n.Name, op = n.Op.ToString(), n.Inputs, n.Outputs, n.Domain, n.OpTypeName, n.OpsetVersion, n.IsFused,
        attributes = n.Attributes?.OrderBy(p => p.Key, StringComparer.Ordinal).ToDictionary(p => p.Key, p => Attribute(p.Value))
    }).ToArray()));

    static int Main(string[] args)
    {
        Require(args.Length == 2 && OperatingSystem.IsLinux(), "specification new-output-directory; AMD only");
        Require(Environment.ProcessorCount == 1 && Process.GetCurrentProcess().ProcessorAffinity.ToInt64() == 4, "CPU2 before CLR startup");
        Require(Environment.Version.ToString() == "10.0.8", "Runtime identity");
        Require(!Environment.GetEnvironmentVariables().Keys.Cast<string>().Any(k => k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase)
            || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)), "No overrides");
        string specPath = Path.GetFullPath(args[0]), baseDirectory = Path.GetDirectoryName(specPath)!;
        using var document = JsonDocument.Parse(File.ReadAllBytes(specPath)); var spec = document.RootElement;
        Require(Sha(typeof(ComputationalGraph).Assembly.Location) == Text(spec.GetProperty("core")), "Core identity");
        Require(Sha(typeof(ParakeetTranscriber).Assembly.Location) == Text(spec.GetProperty("data")), "Data identity");
        string sourceRoot = Path.GetFullPath(Path.Combine(baseDirectory, Text(spec.GetProperty("selected_arrays"))));
        ITensor Load(JsonElement item)
        {
            string file = Path.GetFullPath(Path.Combine(sourceRoot, Text(item.GetProperty("file"))));
            Require(file.StartsWith(sourceRoot + Path.DirectorySeparatorChar, StringComparison.Ordinal), "Below source root");
            byte[] bytes = File.ReadAllBytes(file); Require(Hash(bytes) == Text(item.GetProperty("sha256")), "Selected array identity");
            int[] shape = Shape(item);
            return Text(item.GetProperty("dtype")) switch
            {
                "Float" => new DenseTensor<float>(MemoryMarshal.Cast<byte, float>(bytes).ToArray(), shape),
                "Int32" => new DenseTensor<int>(MemoryMarshal.Cast<byte, int>(bytes).ToArray(), shape),
                "Int64" => new DenseTensor<long>(MemoryMarshal.Cast<byte, long>(bytes).ToArray(), shape),
                _ => throw new InvalidDataException("Selected dtype")
            };
        }
        Require(!Directory.Exists(args[1]), "Refuse existing output"); Directory.CreateDirectory(args[1]);
        string output = Path.GetFullPath(args[1]); var stored = new Dictionary<string, ArrayRecord>();
        var files = new Dictionary<string, string>(); var held = new Dictionary<ITensor, string>(ReferenceEqualityComparer.Instance);
        void Hold(ITensor tensor) { if (!held.ContainsKey(tensor)) held.Add(tensor, Bits(tensor)); }
        void CheckHeld() { foreach (var pair in held) Require(Bits(pair.Key) == pair.Value, "Held tensor changed after later calls/reset"); }
        ArrayRecord Save(string key, ITensor tensor)
        {
            Require(tensor is Tensor<float>, "Captured LSTM float tensor");
            float[] values = ((Tensor<float>)tensor).ToArray(); Require(values.All(float.IsFinite), "Finite captured tensor");
            byte[] bytes = MemoryMarshal.AsBytes(values.AsSpan()).ToArray(); string hash = Hash(bytes); int[] shape = tensor.Dims;
            Hold(tensor);
            if (stored.TryGetValue(key, out var old))
            {
                Require(old.sha256 == hash && old.shape.SequenceEqual(shape), "Exact capture repeat"); return old;
            }
            if (!files.TryGetValue(hash, out string? file))
            {
                file = files.Count.ToString("D4") + ".f32";
                using var stream = new FileStream(Path.Combine(output, file), FileMode.CreateNew, FileAccess.Write); stream.Write(bytes);
                files.Add(hash, file);
            }
            var record = new ArrayRecord(file, shape, values.Length, bytes.LongLength, hash); stored.Add(key, record); return record;
        }
        string model = Text(spec.GetProperty("model")); Require(Sha(model) == Text(spec.GetProperty("model_sha256")), "Original model identity");
        var graph = OnnxImport.Load(model, 64L * 1024 * 1024) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
        graph.Prepare(); string originalNodes = Nodes(graph);
        var originalOutputs = graph.Outputs.Keys.Order(StringComparer.Ordinal).ToArray();
        var initializers = graph.Initializers.ToDictionary(p => p.Key, p => (Tensor: p.Value, Hash: Bits(p.Value), Shape: p.Value.Dims));
        var nodes = graph.Nodes.Where(n => n.Op == OpType.LSTM).ToArray();
        Require(nodes.Select(n => n.Name).SequenceEqual(new[] { "/decoder/dec_rnn/lstm/LSTM", "/decoder/dec_rnn/lstm/LSTM_1" }), "Both actual recurrent nodes");
        var calls = new List<CallRecord>(); var checks = new List<object>();
        foreach (var item in spec.GetProperty("cases").EnumerateArray())
        {
            string name = Text(item.GetProperty("name")); var steps = item.GetProperty("steps").EnumerateArray().ToArray();
            if (steps.Length == 0) { Require(name == "silence", "Only silence has no decoder call"); continue; }
            var encoded = (Tensor<float>)Load(item.GetProperty("encoder")); int[] encodedShape = encoded.Dimensions.ToArray();
            int frames = encodedShape[2]; float[] enc = encoded.ToArray();
            Require(encodedShape.SequenceEqual(new[] { 1, 1024, frames }) && enc.All(float.IsFinite), "Selected encoder shape");
            foreach (bool capture in new[] { false, true })
            for (int repeat = 0; repeat < 2; repeat++)
            {
                var execution = graph.CreateExecution(ExecutionOptions.Memory);
                ITensor state1 = new DenseTensor<float>(new[] { 2, 1, 640 }), state2 = new DenseTensor<float>(new[] { 2, 1, 640 });
                for (int step = 0; step < steps.Length; step++)
                {
                    var expected = steps[step]; int frame = expected.GetProperty("frame").GetInt32(); Require(frame >= 0 && frame < frames, "Frame bound");
                    var vector = new float[1024]; for (int i = 0; i < vector.Length; i++) vector[i] = enc[i * frames + frame];
                    var feeds = new Dictionary<string, ITensor>
                    {
                        ["encoder_outputs"] = new DenseTensor<float>(vector, new[] { 1, 1024, 1 }),
                        ["targets"] = new DenseTensor<int>(new[] { expected.GetProperty("target").GetInt32() }, new[] { 1, 1 }),
                        ["target_length"] = new DenseTensor<int>(new[] { 1 }, new[] { 1 }),
                        ["input_states_1"] = state1, ["input_states_2"] = state2
                    };
                    var before = feeds.ToDictionary(p => p.Key, p => Bits(p.Value));
                    execution.Reset();
                    if (capture)
                        foreach (string value in nodes.SelectMany(n => n.Inputs.Concat(n.Outputs)).Where(n => !string.IsNullOrEmpty(n)).Distinct())
                            execution.Outputs.TryAdd(value, null);
                    Require(execution.Execute(feeds, true, ExecutionProvider.CPU, ExecutionOptions.Memory), execution.LastErrorMessage ?? "Decoder execution");
                    Require(Nodes(execution) == originalNodes && Nodes(graph) == originalNodes, "All original node fields and attributes unchanged");
                    var original = originalOutputs.ToDictionary(n => n, n => execution.Outputs[n] ?? throw new InvalidDataException("Missing original output"));
                    foreach (var expectedOutput in expected.GetProperty("outputs").EnumerateObject())
                    {
                        var actual = original[expectedOutput.Name]; var descriptor = expectedOutput.Value;
                        Require(actual.ElementType.ToString() == Text(descriptor.GetProperty("dtype")) && actual.Dims.SequenceEqual(Shape(descriptor)), "Original shape/type");
                        Require(Bits(actual) == Text(descriptor.GetProperty("sha256")), "Original selected output bits"); Hold(actual);
                    }
                    Require(original.Count == 4 && feeds.All(p => Bits(p.Value) == before[p.Key]), "All four original outputs and readonly inputs");
                    float[] logits = ((Tensor<float>)original["outputs"]).ToArray(); int token = 0, duration = 0;
                    for (int i = 1; i < 8193; i++) if (logits[i] > logits[token]) token = i;
                    for (int i = 1; i < 5; i++) if (logits[8193 + i] > logits[8193 + duration]) duration = i;
                    Require(token == expected.GetProperty("token").GetInt32() && duration == expected.GetProperty("duration").GetInt32(), "Full decoder decisions");
                    if (token != 8192) { state1 = original["output_states_1"]; state2 = original["output_states_2"]; }
                    if (capture)
                    {
                        for (int i = 0; i < nodes.Length; i++)
                        {
                            var n = nodes[i]; Require(n.RequiredInt("hidden_size") == 640 && n.Attr("direction", "forward") == "forward", "Actual recurrence attributes");
                            string Key(string value) => graph.Initializers.ContainsKey(value) ? "constant/" + value : name + "/" + step + "/" + value;
                            ArrayRecord Record(string value)
                            {
                                var tensor = execution.GetInputTensor(value);
                                if (graph.Initializers.TryGetValue(value, out var constant) && stored.TryGetValue(Key(value), out var cached))
                                { Require(ReferenceEquals(tensor, constant), "Original constant reference"); Hold(tensor); return cached; }
                                return Save(Key(value), tensor);
                            }
                            var inputs = n.Inputs.Select(t => string.IsNullOrEmpty(t) ? null : Record(t)).ToArray();
                            var outputs = n.Outputs.Select(Record).ToArray();
                            Require(inputs.Length == 7 && inputs[4] is null && outputs.Length == 3, "Complete optional slots");
                            Require(inputs[0]!.shape.SequenceEqual(new[] { 1, 1, 640 }) && outputs[0].shape.SequenceEqual(new[] { 1, 1, 1, 640 }), "Actual one-step geometry");
                            if (repeat == 0) calls.Add(new(name, step, i, n.Name, n.OpsetVersion, n.Attributes, n.Inputs, n.Outputs, inputs, outputs));
                        }
                    }
                    checks.Add(new { name, step, capture, repeat, exact_outputs = original.ToDictionary(p => p.Key, p => Bits(p.Value)), inputs_unchanged = true, token, duration });
                }
                execution.Reset(); CheckHeld();
                Require(Nodes(graph) == originalNodes && graph.Outputs.Keys.Order(StringComparer.Ordinal).SequenceEqual(originalOutputs), "Original graph remains unchanged");
            }
            Console.WriteLine(name + " passed all decoder controls and captures");
        }
        Require(calls.Count == 380 && checks.Count == 760, "All 190 decoder steps, two nodes and four controls");
        Require(initializers.Count == graph.Initializers.Count && initializers.All(p => ReferenceEquals(graph.Initializers[p.Key], p.Value.Tensor)
            && Bits(p.Value.Tensor) == p.Value.Hash && p.Value.Shape.SequenceEqual(p.Value.Tensor.Dims)), "Readonly complete initializer inventory");
        CheckHeld(); long bytes = Directory.EnumerateFiles(output, "*.f32").Sum(f => new FileInfo(f).Length);
        Require(bytes <= 64L * 1024 * 1024, "Frozen unique fixture bound");
        Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.FileName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Managed capture only");
        using var result = new FileStream(Path.Combine(output, "result.json"), FileMode.CreateNew, FileAccess.Write);
        JsonSerializer.Serialize(result, new { passed = true, calls, checks, tensors = stored, original_node_sha256 = originalNodes,
            original_node_count = graph.Nodes.Count, original_outputs = originalOutputs, initializer_count = initializers.Count,
            model_sha256 = Sha(model), core = Sha(typeof(ComputationalGraph).Assembly.Location), data = Sha(typeof(ParakeetTranscriber).Assembly.Location),
            executable = Sha(typeof(Capture).Assembly.Location), runtime = Environment.Version.ToString(), pid = Environment.ProcessId,
            vector_count = Vector<float>.Count, avx512 = Avx512F.IsSupported, tensor_bytes = bytes, distinct_files = files.Count,
            spec_sha256 = Sha(specPath), no_performance_measurement = true, original_graph_unchanged = true, held_outputs_unchanged = true },
            new JsonSerializerOptions { WriteIndented = true });
        return 0;
    }
}
