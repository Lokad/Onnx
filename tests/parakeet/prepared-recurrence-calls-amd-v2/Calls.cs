using System.Collections;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

internal record SavedArray(string file, string dtype, int[] shape, long values, long bytes, string sha256);
internal record Weight(string kind, string name, int[] shape, long bytes, string source_sha256, string prepared_sha256);
internal record Residency(string name, long budget, long bytes, Weight[] weights);

internal static class Calls
{
    const long Budget = 64L * 1024 * 1024;
    static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
    static string Sha(string file) { using var s = File.OpenRead(file); return Convert.ToHexStringLower(SHA256.HashData(s)); }
    static string Hash(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
    static byte[] Bytes(ITensor t) => t switch
    {
        Tensor<float> x => MemoryMarshal.AsBytes(x.ToArray().AsSpan()).ToArray(),
        Tensor<int> x => MemoryMarshal.AsBytes(x.ToArray().AsSpan()).ToArray(),
        Tensor<long> x => MemoryMarshal.AsBytes(x.ToArray().AsSpan()).ToArray(),
        _ => throw new InvalidDataException("Unexpected dtype")
    };
    static string Bits(ITensor t) => Hash(Bytes(t));
    static string Text(JsonElement e) => e.GetString() ?? throw new InvalidDataException();
    static int[] Shape(JsonElement e) => e.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
    static object Property(object value, string name) => value.GetType().GetProperty(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.GetValue(value)!;
    static IDictionary Map(ComputationalGraph graph, string field) => (IDictionary?)typeof(ComputationalGraph)
        .GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(graph) ?? new Hashtable();
    static string Nodes(ComputationalGraph graph) => Hash(JsonSerializer.SerializeToUtf8Bytes(graph.Nodes.Select(n => new
    {
        n.ID, n.Name, op = n.Op.ToString(), n.Inputs, n.Outputs, n.Domain, n.OpTypeName, n.OpsetVersion, n.IsFused,
        attributes = n.Attributes?.OrderBy(p => p.Key, StringComparer.Ordinal).ToDictionary(p => p.Key, p => p.Value is ITensor t
            ? (object)new { dtype = t.ElementType.ToString(), shape = t.Dims, sha256 = Bits(t) } : p.Value)
    }).ToArray()));

    static Residency Inspect(ComputationalGraph graph, string name, int expectedRecurrent, long expectedBytes)
    {
        Require(Map(graph, "PackedConvWeights").Count == 0, "Actual decoder has no prepared convolutions");
        var result = new List<Weight>(); var owned = new HashSet<float[]>(ReferenceEqualityComparer.Instance);
        foreach (DictionaryEntry entry in Map(graph, "PackedWeights"))
        {
            object record = entry.Value!; string sourceName = (string)Property(record, "SourceName");
            var source = (DenseTensor<float>)Property(record, "SourceRef"); var packed = (DenseTensor<float>)Property(record, "Packed");
            Require(MemoryMarshal.TryGetArray(source.Buffer, out ArraySegment<float> original) && ReferenceEquals(original.Array, entry.Key)
                && ReferenceEquals(original.Array, Property(record, "SourceArray")), "Matrix source ownership");
            Require(MemoryMarshal.TryGetArray(packed.Buffer, out ArraySegment<float> prepared) && prepared.Array is not null
                && !ReferenceEquals(original.Array, prepared.Array) && owned.Add(prepared.Array), "Matrix clone ownership");
            Require(ReferenceEquals(graph.Initializers[(string)Property(record, "PackedName")], packed), "Matrix binding ownership");
            result.Add(new("matrix", sourceName, ((ITensor)source).Dims, packed.Length * 4, Bits(source), Bits(packed)));
        }
        var recurrent = Map(graph, "PackedLstmWeights"); Require(recurrent.Count == expectedRecurrent, "Actual recurrent count");
        foreach (DictionaryEntry entry in recurrent)
        {
            object record = entry.Value!; string sourceName = (string)Property(record, "SourceName");
            var source = (DenseTensor<float>)Property(record, "Source"); var values = (float[])Property(record, "Values");
            var shape = (int[])Property(record, "Shape");
            Require(ReferenceEquals(graph.Initializers[sourceName], source) && shape.SequenceEqual(new[] { 1, 2560, 640 })
                && shape.SequenceEqual(((ITensor)source).Dims) && values.Length == 2560 * 640, "Original recurrent source/shape");
            Require(MemoryMarshal.TryGetArray(source.Buffer, out ArraySegment<float> original) && original.Offset == 0
                && ReferenceEquals(original.Array, entry.Key) && ReferenceEquals(original.Array, Property(record, "SourceArray")), "Recurrent source array identity");
            Require(!ReferenceEquals(original.Array, values) && owned.Add(values), "Independent recurrent storage");
            Require(!graph.Inputs.ContainsKey(sourceName) && !graph.Outputs.ContainsKey(sourceName), "Weights remain constant-only");
            foreach (var tensor in graph.Initializers.Values.OfType<DenseTensor<float>>())
                Require(!tensor.Buffer.Span.Overlaps(values), "Prepared recurrent arrays are not mutable initializer bindings");
            for (int k = 0; k < 640; k++)
            for (int o = 0; o < 2560; o++)
                Require(BitConverter.SingleToInt32Bits(values[k * 2560 + o]) == BitConverter.SingleToInt32Bits(source.Buffer.Span[o * 640 + k]), "Every prepared element is the exact transpose");
            result.Add(new("recurrent", sourceName, shape, values.LongLength * 4, Bits(source), Hash(MemoryMarshal.AsBytes(values.AsSpan()).ToArray())));
        }
        Require(result.Sum(r => r.bytes) == graph.RetainedPackedWeightBytes && graph.RetainedPackedWeightBytes == expectedBytes
            && expectedBytes <= Budget && graph.MaximumPackedWeightBytes == Budget, "All-map aggregate accounting");
        var context = graph.CreateExecution(ExecutionOptions.Memory);
        foreach (string field in new[] { "PackedWeights", "PackedConvWeights", "PackedLstmWeights" })
            if (typeof(ComputationalGraph).GetField(field, BindingFlags.Instance | BindingFlags.NonPublic) is not null)
                Require(ReferenceEquals(Map(graph, field), Map(context, field)), "Fresh context shares actual prepared map");
        Require(context.RetainedPackedWeightBytes == expectedBytes, "Context aggregate accounting"); context.Reset();
        return new(name, Budget, expectedBytes, result.OrderBy(r => r.kind).ThenBy(r => r.name, StringComparer.Ordinal).ToArray());
    }

    static int Main(string[] args)
    {
        Require(args.Length == 4 && OperatingSystem.IsLinux(), "base-directory role instruction-mode new-output-directory");
        string root = Path.GetFullPath(args[0]), role = args[1]; bool candidate = role == "candidate", wide = args[2] == "512";
        Require((role is "selected" or "candidate") && (args[2] is "512" or "256"), "Declared role/mode");
        Require(Environment.Version.ToString() == "10.0.8" && Environment.ProcessorCount == 1 && Process.GetCurrentProcess().ProcessorAffinity.ToInt64() == 4, "AMD runtime on CPU2");
        Require(Avx512F.IsSupported == wide && Avx2.IsSupported && Fma.IsSupported, "Actual instruction mode");
        var flags = Environment.GetEnvironmentVariables().Cast<DictionaryEntry>().Where(e => ((string)e.Key).StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase)
            || ((string)e.Key).StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase) || ((string)e.Key).StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase))
            .ToDictionary(e => (string)e.Key, e => (string)e.Value!);
        Require(wide ? flags.Count == 0 : flags.Count == 1 && flags.GetValueOrDefault("DOTNET_EnableAVX512") == "0", "Only declared instruction override");
        using var specDoc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, "spec.json"))); var spec = specDoc.RootElement;
        var identity = spec.GetProperty("identities").GetProperty(role);
        Require(Sha(typeof(ComputationalGraph).Assembly.Location) == Text(identity.GetProperty("Lokad.Onnx.dll").GetProperty("sha256")), "Core identity");
        Require(Sha(typeof(ParakeetTranscriber).Assembly.Location) == Text(identity.GetProperty("Lokad.Onnx.Data.dll").GetProperty("sha256")), "Data identity");
        using var captureDoc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, "fixtures/result.json"))); var capture = captureDoc.RootElement;
        using var decoderDoc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, "decoder-spec.json"))); var decoderSpec = decoderDoc.RootElement;
        using var nativeDoc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, "native/result.json"))); var native = nativeDoc.RootElement;
        var nativeRows = native.GetProperty("rows").EnumerateArray().Where(r => r.GetProperty("repeat").GetInt32() == 0)
            .ToDictionary(r => Text(r.GetProperty("name")) + "/" + r.GetProperty("step").GetInt32() + "/" + r.GetProperty("index").GetInt32() + "/" + r.GetProperty("output").GetInt32());
        var cache = new Dictionary<string, float[]>();
        Tensor<float> Load(string folder, JsonElement item)
        {
            string file = Path.GetFullPath(Path.Combine(root, folder, Text(item.GetProperty("file"))));
            Require(file.StartsWith(Path.Combine(root, folder) + Path.DirectorySeparatorChar, StringComparison.Ordinal), "Fixture remains below input root");
            if (!cache.TryGetValue(file, out var values))
            {
                byte[] bytes = File.ReadAllBytes(file); Require(Hash(bytes) == Text(item.GetProperty("sha256")), "Frozen array bytes");
                values = MemoryMarshal.Cast<byte, float>(bytes).ToArray(); Require(values.All(float.IsFinite), "Finite fixture"); cache.Add(file, values);
            }
            return new DenseTensor<float>(values, Shape(item));
        }
        Require(!Directory.Exists(args[3]), "Exclusive-create output"); Directory.CreateDirectory(args[3]); string output = Path.GetFullPath(args[3]);
        var files = new Dictionary<string, string>(); var held = new Dictionary<ITensor, string>(ReferenceEqualityComparer.Instance);
        void Hold(ITensor value) { if (!held.ContainsKey(value)) held.Add(value, Bits(value)); }
        void CheckHeld() { foreach (var pair in held) Require(Bits(pair.Key) == pair.Value, "Held output survives later calls/reset/invalidation"); }
        SavedArray Save(ITensor value)
        {
            if (value is Tensor<float> f) Require(f.ToArray().All(float.IsFinite), "Finite actual output");
            byte[] bytes = Bytes(value); string sha = Hash(bytes); Hold(value);
            if (!files.TryGetValue(sha, out string? filename))
            {
                filename = files.Count.ToString("D4") + ".bin";
                using var stream = new FileStream(Path.Combine(output, filename), FileMode.CreateNew); stream.Write(bytes); files.Add(sha, filename);
            }
            return new(filename, value.ElementType.ToString(), value.Dims, value.Length, bytes.LongLength, sha);
        }
        Dictionary<string, ITensor> Run(GraphExecution context, Dictionary<string, ITensor> feeds, ExecutionOptions? options = null)
        {
            var before = feeds.ToDictionary(p => p.Key, p => Bits(p.Value)); context.Reset();
            Require(context.Execute(feeds, true, ExecutionProvider.CPU, options ?? ExecutionOptions.Memory), context.LastErrorMessage ?? "Graph execution");
            Require(feeds.All(p => Bits(p.Value) == before[p.Key]), "Inputs remain unchanged");
            var result = context.Outputs.ToDictionary(p => p.Key, p => p.Value ?? throw new InvalidDataException("Missing output"));
            foreach (var value in result.Values) Hold(value); return result;
        }
        var residencies = new List<Residency>(); var routes = new List<object>();
        void Lifecycle(ComputationalGraph graph, string name, int count, long bytes)
        {
            var first = Inspect(graph, name, count, bytes); var arrays = Map(graph, "PackedLstmWeights").Values.Cast<object>().Select(r => (float[])Property(r, "Values")).ToArray();
            graph.RefreshLifetimeAnalysis(); var repeated = Inspect(graph, name, count, bytes);
            Require(JsonSerializer.Serialize(first) == JsonSerializer.Serialize(repeated), "Exact repeat residency");
            Require(arrays.SequenceEqual(Map(graph, "PackedLstmWeights").Values.Cast<object>().Select(r => (float[])Property(r, "Values"))), "No repeated preparation copies");
            graph.InvalidatePreparation(); Require(graph.RetainedPackedWeightBytes == 0 && Map(graph, "PackedWeights").Count == 0 && Map(graph, "PackedLstmWeights").Count == 0, "Invalidation releases every map");
            graph.Prepare(); var rebuilt = Inspect(graph, name, count, bytes);
            Require(JsonSerializer.Serialize(first) == JsonSerializer.Serialize(rebuilt), "Exact rebuilt residency");
            Require(!Map(graph, "PackedLstmWeights").Values.Cast<object>().Select(r => (float[])Property(r, "Values")).Any(v => arrays.Contains(v)), "Rebuilt arrays independently owned");
            residencies.Add(first);CheckHeld();
        }
        void Route(ComputationalGraph graph, string name, Dictionary<string, ITensor> feeds, string[] checkedOutputs, int count, long bytes)
        {
            var original = Run(graph.CreateExecution(ExecutionOptions.Memory), feeds); var expected = original.ToDictionary(p => p.Key, p => Bits(p.Value));
            if (candidate)
            {
                foreach (object record in Map(graph, "PackedLstmWeights").Values) Array.Fill((float[])Property(record, "Values"), float.NaN);
                var poison = Run(graph.CreateExecution(ExecutionOptions.Memory), feeds);
                Require(checkedOutputs.All(n => ((Tensor<float>)poison[n]).ToArray().All(float.IsNaN)), "Original nodes actually read prepared arrays");
                graph.InvalidatePreparation(); graph.Prepare(); Inspect(graph, name, count, bytes);
            }
            var restored = Run(graph.CreateExecution(ExecutionOptions.Memory), feeds);
            Require(restored.All(p => Bits(p.Value) == expected[p.Key]), "Recovery restores original model bits");
            CheckHeld();routes.Add(new { name, prepared_route_proven = candidate, recovery_exact = true, held_outputs_unchanged = true });
        }
        string model = Text(decoderSpec.GetProperty("model"));Require(Sha(model) == Text(decoderSpec.GetProperty("model_sha256")), "Original decoder identity");
        var decoder = OnnxImport.Load(model, Budget) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage); decoder.Prepare();
        string originalNodes = Nodes(decoder); string[] originalOutputs = decoder.Outputs.Keys.Order(StringComparer.Ordinal).ToArray();
        var sources = spec.GetProperty("initializers").EnumerateObject().ToDictionary(p => p.Name, p =>
        {
            ITensor tensor = decoder.Initializers[p.Name];Require(Bits(tensor) == Text(p.Value.GetProperty("sha256")) && tensor.Dims.SequenceEqual(Shape(p.Value)), "Original model initializer bytes");
            return (Tensor: tensor, Hash: Bits(tensor));
        });
        long decoderBytes = candidate ? 51461120 : 25246720;
        Lifecycle(decoder, "decoder", candidate ? 4 : 0, decoderBytes);
        var decoderRows = new List<object>(); var componentRows = new List<object>(); bool routedDecoder = false;
        foreach (var item in decoderSpec.GetProperty("cases").EnumerateArray())
        {
            string name = Text(item.GetProperty("name")); var steps = item.GetProperty("steps").EnumerateArray().ToArray(); if (steps.Length == 0) continue;
            var encoded = Load("selected-arrays", item.GetProperty("encoder")); int frames = encoded.Dimensions[2]; float[] enc = encoded.ToArray();
            for (int repeat = 0; repeat < 2; repeat++)
            {
                var context = decoder.CreateExecution(ExecutionOptions.Memory);
                ITensor state1 = new DenseTensor<float>(new[] { 2, 1, 640 }), state2 = new DenseTensor<float>(new[] { 2, 1, 640 });
                for (int step = 0; step < steps.Length; step++)
                {
                    var expected = steps[step]; int frame = expected.GetProperty("frame").GetInt32(); Require(frame >= 0 && frame < frames, "Recorded frame bound");
                    var vector = new float[1024];for (int i = 0; i < vector.Length; i++) vector[i] = enc[i * frames + frame];
                    var feeds = new Dictionary<string, ITensor> { ["encoder_outputs"] = new DenseTensor<float>(vector, new[] { 1, 1024, 1 }),
                        ["targets"] = new DenseTensor<int>(new[] { expected.GetProperty("target").GetInt32() }, new[] { 1, 1 }),
                        ["target_length"] = new DenseTensor<int>(new[] { 1 }, new[] { 1 }), ["input_states_1"] = state1, ["input_states_2"] = state2 };
                    if (!routedDecoder) { Route(decoder, "decoder", feeds, new[] { "output_states_1", "output_states_2" }, candidate ? 4 : 0, decoderBytes);routedDecoder = true; }
                    var actual = Run(context, feeds);Require(actual.Count == 4, "All original decoder outputs");
                    foreach (var descriptor in expected.GetProperty("outputs").EnumerateObject())
                        Require(Bits(actual[descriptor.Name]) == Text(descriptor.Value.GetProperty("sha256")) && actual[descriptor.Name].Dims.SequenceEqual(Shape(descriptor.Value))
                            && actual[descriptor.Name].ElementType.ToString() == Text(descriptor.Value.GetProperty("dtype")), "Exact selected decoder output");
                    var logits = ((Tensor<float>)actual["outputs"]).ToArray();int token = 0, duration = 0;
                    for (int i = 1; i < 8193; i++) if (logits[i] > logits[token]) token = i;
                    for (int i = 1; i < 5; i++) if (logits[8193 + i] > logits[8193 + duration]) duration = i;
                    Require(token == expected.GetProperty("token").GetInt32() && duration == expected.GetProperty("duration").GetInt32(), "Every decoder decision");
                    if (token != 8192) { state1 = actual["output_states_1"];state2 = actual["output_states_2"]; }
                    decoderRows.Add(new { name, step, repeat, token, duration, outputs = actual.ToDictionary(p => p.Key, p => Save(p.Value)) });
                    Require(Nodes(decoder) == originalNodes && Nodes(context) == originalNodes, "All original decoder nodes unchanged");
                }
                context.Reset();CheckHeld();
            }
            Console.WriteLine(name + " original decoder passed");
        }
        Require(decoder.Outputs.Keys.Order(StringComparer.Ordinal).SequenceEqual(originalOutputs), "Original model output bindings unchanged");
        Inspect(decoder, "decoder", candidate ? 4 : 0, decoderBytes);
        foreach (var source in sources) Require(ReferenceEquals(decoder.Initializers[source.Key], source.Value.Tensor) && Bits(source.Value.Tensor) == source.Value.Hash, "All original initializers remain immutable");
        var calls = capture.GetProperty("calls").EnumerateArray().ToArray();Require(calls.Length == 380, "Every captured complete call");
        for (int index = 0; index < 2; index++)
        {
            var first = calls.First(c => c.GetProperty("index").GetInt32() == index);
            string[] inputNames = first.GetProperty("input_names").EnumerateArray().Select(Text).ToArray(), outputNames = first.GetProperty("output_names").EnumerateArray().Select(Text).ToArray();
            var graph = new ComputationalGraph(Budget);graph.Metadata["Name"] = "actual-complete-lstm-" + index;graph.Opset[""] = 17;
            var original = decoder.Nodes.Single(n => n.Name == Text(first.GetProperty("node")));
            Require(original.Inputs.SequenceEqual(inputNames) && original.Outputs.SequenceEqual(outputNames) && original.RequiredInt("hidden_size") == 640 && original.OpsetVersion == 17, "Exact actual node descriptors");
            original.Inputs = original.Inputs.ToArray();original.Outputs = original.Outputs.ToArray();original.Attributes = new(original.Attributes!);graph.Nodes.Add(original);
            for (int i = 0; i < inputNames.Length; i++)
            {
                if (i is 1 or 2 or 3) graph.Initializers[inputNames[i]] = Load("fixtures", first.GetProperty("inputs")[i]);
                else if (inputNames[i].Length > 0) graph.Inputs[inputNames[i]] = new DenseTensor<float>(Shape(first.GetProperty("inputs")[i]));
            }
            for (int i = 0; i < outputNames.Length; i++) graph.Outputs[outputNames[i]] = new DenseTensor<float>(Shape(first.GetProperty("outputs")[i]));
            graph.Prepare();long bytes = candidate ? 13107200 : 0;Lifecycle(graph, "lstm-" + index, candidate ? 2 : 0, bytes);
            bool routed = false;var constants = graph.Initializers.ToDictionary(p => p.Key, p => Bits(p.Value));
            for (int repeat = 0; repeat < 2; repeat++)
            {
                var context = graph.CreateExecution(ExecutionOptions.Memory);
                foreach (var call in calls.Where(c => c.GetProperty("index").GetInt32() == index))
                {
                    string name = Text(call.GetProperty("name"));int step = call.GetProperty("step").GetInt32();
                    var inputs = call.GetProperty("inputs");var feeds = new Dictionary<string, ITensor>();
                    foreach (int i in new[] { 0, 5, 6 }) feeds[inputNames[i]] = Load("fixtures", inputs[i]);
                    foreach (int i in new[] { 1, 2, 3 }) Require(constants[inputNames[i]] == Text(inputs[i].GetProperty("sha256")), "Same actual constant bytes on every call");
                    if (!routed) { Route(graph, "lstm-" + index, feeds, outputNames, candidate ? 2 : 0, bytes);routed = true; }
                    var actual = Run(context, feeds);var outputs = new List<object>();
                    for (int i = 0; i < 3; i++)
                    {
                        var expected = call.GetProperty("outputs")[i];var tensor = actual[outputNames[i]];var saved = Save(tensor);
                        Require(saved.sha256 == Text(expected.GetProperty("sha256")) && saved.shape.SequenceEqual(Shape(expected)), "Exact selected complete-call bits");
                        var reference = nativeRows[name + "/" + step + "/" + index + "/" + i].GetProperty("array");
                        var nativeValues = Load("native", reference).ToArray();var actualValues = ((Tensor<float>)tensor).ToArray();double error = 0;int worst = 0;
                        for (int k = 0; k < nativeValues.Length; k++)
                        {
                            double difference = Math.Abs((double)actualValues[k] - nativeValues[k]) / Math.Max(1, Math.Abs((double)nativeValues[k]));
                            if (difference > error) { error = difference;worst = k; }
                        }
                        Require(error <= 1e-4, "Native complete-call numerical bound");outputs.Add(new { name = outputNames[i], array = saved, max_error = error, worst_index = worst });
                    }
                    componentRows.Add(new { name, step, index, repeat, outputs });
                }
                context.Reset();CheckHeld();
            }
            Require(constants.All(p => Bits(graph.Initializers[p.Key]) == p.Value), "All complete-call constants immutable");
            Inspect(graph, "lstm-" + index, candidate ? 2 : 0, bytes);Console.WriteLine("All actual calls for LSTM " + index + " passed");
        }
        Require(decoderRows.Count == 380 && componentRows.Count == 760 && residencies.Count == 3 && routes.Count == 3, "Full prospective census");CheckHeld();
        Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.FileName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Managed candidate only");
        using var stream = new FileStream(Path.Combine(output, "result.json"), FileMode.CreateNew);
        JsonSerializer.Serialize(stream, new { passed = true, role, mode = args[2], pid = Environment.ProcessId, affinity = 4, processor_count = Environment.ProcessorCount,
            core = Sha(typeof(ComputationalGraph).Assembly.Location), data = Sha(typeof(ParakeetTranscriber).Assembly.Location), consumer = Sha(Assembly.GetExecutingAssembly().Location),
            runtime = Environment.Version.ToString(), avx512 = Avx512F.IsSupported, flags, model_sha256 = Sha(model), original_node_sha256 = originalNodes,
            spec_sha256 = Sha(Path.Combine(root, "spec.json")), decoder_spec_sha256 = Sha(Path.Combine(root, "decoder-spec.json")),
            capture_sha256 = Sha(Path.Combine(root, "fixtures/result.json")), native_sha256 = Sha(Path.Combine(root, "native/result.json")),
            residencies, routes, decoder = decoderRows, calls = componentRows, distinct_files = files.Count, tensor_bytes = Directory.EnumerateFiles(output, "*.bin").Sum(f => new FileInfo(f).Length),
            no_performance_measurement = true, original_graph_unchanged = true, held_outputs_unchanged = true, inputs_unchanged = true }, new JsonSerializerOptions { WriteIndented = true });
        return 0;
    }
}
