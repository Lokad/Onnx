using System.Collections;
using System.Diagnostics;
using System.Diagnostics.Tracing;
using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

internal static class Driver
{
    const string NodeName = "/joint/joint_net/joint_net.2/MatMul";
    const string WeightName = "onnx::MatMul_230";
    const int Warmups = 256, Observations = 1024;
    static void Require(bool value, string message) { if (!value) throw new InvalidDataException(message); }
    static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
    static string FileHash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
    static string Text(JsonElement value) => value.GetString() ?? throw new InvalidDataException("Missing string");
    static byte[] Bytes(ITensor tensor) => tensor switch
    {
        Tensor<float> value => MemoryMarshal.AsBytes(value.ToArray().AsSpan()).ToArray(),
        Tensor<int> value => MemoryMarshal.AsBytes(value.ToArray().AsSpan()).ToArray(),
        Tensor<long> value => MemoryMarshal.AsBytes(value.ToArray().AsSpan()).ToArray(),
        _ => throw new InvalidDataException("Unexpected tensor type")
    };
    static string Bits(ITensor tensor) => Hash(Bytes(tensor));
    static IDictionary Map(ComputationalGraph graph) => (IDictionary)(typeof(ComputationalGraph)
        .GetField("PackedWeights", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(graph)!);
    static object Property(object value, string name) => value.GetType().GetProperty(name)!.GetValue(value)!;
    static string Nodes(ComputationalGraph graph) => Hash(JsonSerializer.SerializeToUtf8Bytes(graph.Nodes.Select(n => new
    {
        n.ID, n.Name, op = n.Op.ToString(), n.Inputs, n.Outputs, n.Domain, n.OpTypeName, n.OpsetVersion, n.IsFused,
        attributes = n.Attributes?.OrderBy(p => p.Key, StringComparer.Ordinal).ToDictionary(p => p.Key,
            p => p.Value is ITensor t ? (object)new { dtype = t.ElementType.ToString(), shape = t.Dims, sha256 = Bits(t) } : p.Value)
    }).ToArray()));
    static object Describe(ITensor tensor)
    {
        Require(tensor is Tensor<float>, "Projection float tensor");
        var value = (Tensor<float>)tensor;
        return new { type = value.GetType().FullName, dtype = value.ElementType.ToString(), shape = value.Dims,
            strides = value.Strides.ToArray(), reversed = value.IsReversedStride, sha256 = Bits(value) };
    }
    static object Mapping(ComputationalGraph graph, GraphExecution execution)
    {
        var source = (DenseTensor<float>)graph.Initializers[WeightName];
        Require(MemoryMarshal.TryGetArray((ReadOnlyMemory<float>)source.Buffer, out var sourceArray)
            && sourceArray.Array is not null, "Weight storage");
        var map = Map(graph);
        Require(ReferenceEquals(map, Map(execution)), "Execution shares prepared map");
        var matches = map.Values.Cast<object>().Where(v => (string)Property(v, "SourceName") == WeightName).ToArray();
        Require(matches.Length <= 1, "Unambiguous weight mapping");
        object? entry = null;
        if (matches.Length == 1)
        {
            object record = matches[0];
            var packed = (DenseTensor<float>)Property(record, "Packed");
            string packedName = (string)Property(record, "PackedName");
            Require(ReferenceEquals(Property(record, "SourceRef"), source)
                && ReferenceEquals(Property(record, "SourceArray"), sourceArray.Array)
                && ReferenceEquals(map[sourceArray.Array!], record)
                && ReferenceEquals(graph.Initializers[packedName], packed), "Mapping reference identities");
            entry = new { source_name = WeightName, packed_name = packedName,
                source_length = (long)Property(record, "SourceLength"), source_array_key_exact = true,
                source_reference_exact = true, packed_reference_exact = true, packed = Describe(packed) };
        }
        return new { present = matches.Length == 1, entries = map.Count, shared_with_execution = true,
            budget_bytes = graph.MaximumPackedWeightBytes, retained_bytes = graph.RetainedPackedWeightBytes,
            source = Describe(source), source_array_offset = sourceArray.Offset, source_array_count = sourceArray.Count, entry };
    }
    [DllImport("libc")]
    static extern int gettid();
    static void Save(string path, object value)
    {
        using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write);
        JsonSerializer.Serialize(stream, value, new JsonSerializerOptions { WriteIndented = true });
    }

    static void Main(string[] args)
    {
        Require(args.Length == 3 && OperatingSystem.IsLinux(), "specification control|trace empty-output-directory; Linux only");
        string specPath = Path.GetFullPath(args[0]), folder = Path.GetDirectoryName(specPath)!, mode = args[1];
        Require(mode is "control" or "trace", "Fixed diagnostic modes");
        Require(Environment.Version.ToString() == "10.0.8" && Environment.ProcessorCount == 1
            && Process.GetCurrentProcess().ProcessorAffinity == (nint)4, "Runtime and CPU2");
        var flags = Environment.GetEnvironmentVariables().Cast<DictionaryEntry>().Where(e =>
            new[] { "LOKAD_", "DOTNET_", "COMPlus_" }.Any(p => ((string)e.Key).StartsWith(p, StringComparison.OrdinalIgnoreCase)))
            .ToDictionary(e => (string)e.Key, e => (string)e.Value!);
        Require(flags.Count == 0, "Ordinary runtime flags");
        var spec = JsonDocument.Parse(File.ReadAllText(specPath)).RootElement;
        string core = FileHash(typeof(ComputationalGraph).Assembly.Location), model = Text(spec.GetProperty("model"));
        Require(core == Text(spec.GetProperty("core_sha256")) && FileHash(model) == Text(spec.GetProperty("model_sha256")), "Product/model identity");
        Require(Text(spec.GetProperty("case")) == "english-16k" && spec.GetProperty("step").GetInt32() == 0, "One original first decoder step");
        ITensor Load(JsonElement item)
        {
            string path = Path.GetFullPath(Path.Combine(folder, Text(item.GetProperty("file"))));
            Require(path.StartsWith(folder + Path.DirectorySeparatorChar, StringComparison.Ordinal), "Fixture stays below root");
            byte[] bytes = File.ReadAllBytes(path);
            Require(bytes.Length == item.GetProperty("bytes").GetInt32() && Hash(bytes) == Text(item.GetProperty("sha256")), "Fixture identity");
            int[] shape = item.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            return Text(item.GetProperty("dtype")) switch
            {
                "Float" => new DenseTensor<float>(MemoryMarshal.Cast<byte, float>(bytes).ToArray(), shape),
                "Int32" => new DenseTensor<int>(MemoryMarshal.Cast<byte, int>(bytes).ToArray(), shape),
                _ => throw new InvalidDataException("Fixture dtype")
            };
        }
        var feeds = spec.GetProperty("inputs").EnumerateObject().ToDictionary(p => p.Name, p => Load(p.Value));
        var expected = spec.GetProperty("outputs").EnumerateObject().ToDictionary(p => p.Name, p => Load(p.Value));
        Require(feeds.Count == 5 && expected.Count == 4, "Complete original public boundary");
        var expectedBytes = expected.ToDictionary(p => p.Key, p => Bytes(p.Value));
        var inputBits = feeds.ToDictionary(p => p.Key, p => Bits(p.Value));
        var graph = OnnxImport.Load(model, 64L * 1024 * 1024) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
        graph.Prepare();
        string nodes = Nodes(graph);
        string[] outputs = graph.Outputs.Keys.Order(StringComparer.Ordinal).ToArray();
        Require(outputs.SequenceEqual(expected.Keys.Order(StringComparer.Ordinal)), "Original output names");
        var node = graph.Nodes.Single(n => n.Name == NodeName);
        Require(node.Op == OpType.MatMul && node.Inputs.Length == 2 && node.Inputs[1] == WeightName && node.Outputs.Length == 1,
            "Original final projection");
        var initializers = graph.Initializers.ToDictionary(p => p.Key, p => (tensor: p.Value, bits: Bits(p.Value)));
        var execution = graph.CreateExecution(ExecutionOptions.Memory);
        object beforeMap = Mapping(graph, execution);
        var held = new Dictionary<ITensor, string>(ReferenceEqualityComparer.Instance);
        void CheckOutputs(GraphExecution current)
        {
            foreach (var pair in expected)
            {
                var value = current.Outputs[pair.Key] ?? throw new InvalidDataException("Missing output");
                Require(value.ElementType == pair.Value.ElementType && value.Dims.SequenceEqual(pair.Value.Dims)
                    && Bytes(value).AsSpan().SequenceEqual(expectedBytes[pair.Key]), "Complete original output bits");
                if (!held.ContainsKey(value)) held.Add(value, Bits(value));
            }
            Require(feeds.All(p => Bits(p.Value) == inputBits[p.Key]), "Immutable inputs");
            Require(held.All(p => Bits(p.Key) == p.Value), "Held outputs after execution/reset");
        }
        void Execute(GraphExecution current)
        {
            current.Reset();
            Require(current.Execute(feeds, true, ExecutionProvider.CPU, ExecutionOptions.Memory), current.LastErrorMessage ?? "Original decoder execution");
        }
        // The observed loop below has only original outputs. Extra bindings live in a separate context.
        Execute(execution); CheckOutputs(execution);
        var capture = graph.CreateExecution(ExecutionOptions.Memory);
        foreach (string name in node.Inputs.Concat(node.Outputs)) capture.Outputs.TryAdd(name, null);
        Require(capture.Execute(feeds, true, ExecutionProvider.CPU, ExecutionOptions.Memory), capture.LastErrorMessage ?? "Operand capture");
        CheckOutputs(capture);
        var a = capture.GetInputTensor(node.Inputs[0]);
        var b = capture.GetInputTensor(node.Inputs[1]);
        var y = capture.GetInputTensor(node.Outputs[0]);
        Require(a.Dims.SequenceEqual(new[] { 1, 1, 1, 640 }) && b.Dims.SequenceEqual(new[] { 640, 8198 })
            && y.Dims.SequenceEqual(new[] { 1, 1, 1, 8198 }), "Actual projection geometry");
        Require(ReferenceEquals(b, graph.Initializers[WeightName]), "Original initializer reaches captured call");
        Require(Nodes(capture) == nodes && Nodes(graph) == nodes, "Capture preserves original nodes");
        object operands = new { a = Describe(a), b = Describe(b), output = Describe(y), original_weight_reference = true };
        byte[] capturedA = Bytes(a);
        capture.Reset();
        Execute(execution); CheckOutputs(execution);
        Require(execution.Outputs.Keys.Order(StringComparer.Ordinal).SequenceEqual(outputs), "Ordinary context original outputs only");

        string output = Path.GetFullPath(args[2]);
        Require(!Directory.Exists(output) || !Directory.EnumerateFileSystemEntries(output).Any(), "Refuse nonempty output");
        Directory.CreateDirectory(output);
        using (var stream = new FileStream(Path.Combine(output, "projection-a.f32"), FileMode.CreateNew)) stream.Write(capturedA);
        var events = MatrixEvents.Log;
        int nativeThread = gettid();
        if (mode == "trace")
        {
            Save(Path.Combine(output, "ready.json"), new { pid = Environment.ProcessId, native_thread = nativeThread, counter = Stopwatch.GetTimestamp() });
            var waiting = Stopwatch.StartNew();
            while (!events.IsEnabled(EventLevel.Informational, (EventKeywords)1))
            {
                Require(waiting.Elapsed.TotalSeconds < 30, "Collector did not enable markers"); Thread.Sleep(10);
            }
            Save(Path.Combine(output, "collector-enabled.json"), new { pid = Environment.ProcessId, counter = Stopwatch.GetTimestamp() });
        }
        var clocks = new List<object>(Warmups + Observations);
        for (int iteration = 0; iteration < Warmups + Observations; iteration++)
        {
            execution.Reset();
            long marker = Stopwatch.GetTimestamp(); events.Begin(0, iteration, marker);
            long start = Stopwatch.GetTimestamp();
            bool ok = execution.Execute(feeds, true, ExecutionProvider.CPU, ExecutionOptions.Memory);
            long stop = Stopwatch.GetTimestamp(); events.End(0, iteration, stop);
            Require(ok, execution.LastErrorMessage ?? "Observed execution");
            clocks.Add(new { iteration, warmup = iteration < Warmups, marker, start, stop,
                ticks = stop - start, decoder_allocated_bytes = execution.LastAllocatedBytes,
                decoder_copy_bytes = execution.LastCopyBytes, decoder_scratch_bytes = execution.LastScratchBytes,
                decoder_gc_collections = execution.LastGcCollections.ToArray() });
            // Check all values on every call without retaining every returned array.
            foreach (var pair in expected)
            {
                var value = execution.Outputs[pair.Key] ?? throw new InvalidDataException("Missing observed output");
                Require(value.ElementType == pair.Value.ElementType && value.Dims.SequenceEqual(pair.Value.Dims)
                    && Bytes(value).AsSpan().SequenceEqual(expectedBytes[pair.Key]), "Every observed output exact");
            }
        }
        CheckOutputs(execution); execution.Reset();
        Require(held.All(p => Bits(p.Key) == p.Value), "Held outputs after final reset");
        Require(Nodes(graph) == nodes && Nodes(execution) == nodes && graph.Outputs.Keys.Order(StringComparer.Ordinal).SequenceEqual(outputs), "Original graph preserved");
        Require(initializers.Count == graph.Initializers.Count && initializers.All(p =>
            ReferenceEquals(p.Value.tensor, graph.Initializers[p.Key]) && p.Value.bits == Bits(graph.Initializers[p.Key])), "All initializer storage unchanged");
        object afterMap = Mapping(graph, execution);
        Require(JsonSerializer.Serialize(beforeMap) == JsonSerializer.Serialize(afterMap), "Prepared map unchanged");
        Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.FileName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Managed observation only");
        Save(Path.Combine(output, "result.json"), new { passed = true, diagnostic_only = true,
            protocol = "parakeet-decoder-projection-observation-v1", mode, flags, pid = Environment.ProcessId,
            native_thread = nativeThread, runtime = Environment.Version.ToString(), core_sha256 = core,
            isa = new { vector_float_count = Vector<float>.Count, avx2 = Avx2.IsSupported, fma = Fma.IsSupported, avx512 = Avx512F.IsSupported },
            options = new { optimization = execution.Options.Optimization.ToString(), simd = execution.Options.Tensor.UseSimd,
                intrinsics = execution.Options.Tensor.UseIntrinsics, max_degree = execution.Options.Tensor.MaxDegreeOfParallelism,
                buffer_pool_disabled = execution.Options.Tensor.DisableBufferPool },
            consumer_sha256 = FileHash(Assembly.GetExecutingAssembly().Location), spec_sha256 = FileHash(specPath),
            model_sha256 = FileHash(model), original_node_sha256 = nodes, original_outputs = outputs,
            mapping = beforeMap, mapping_after = afterMap, operands, controls = 3,
            calls = Warmups + Observations, warmups = Warmups, observations = Observations,
            frequency = Stopwatch.Frequency, clocks, every_output_exact = true, inputs_unchanged = true,
            held_outputs_unchanged = true, initializers_unchanged = true, ordinary_outputs_only = true,
            counter_scope = "Complete decoder execution; never per-node allocation or copy attribution" });
    }
}

[EventSource(Name = "Lokad-Parakeet-MatMul-Diagnostic")]
internal sealed class MatrixEvents : EventSource
{
    public static readonly MatrixEvents Log = new();
    public static class Keywords { public const EventKeywords Calls = (EventKeywords)1; }
    [Event(1, Level = EventLevel.Informational, Keywords = Keywords.Calls)]
    public void Begin(int fixture, int iteration, long counter) => Emit(1, fixture, iteration, counter);
    [Event(2, Level = EventLevel.Informational, Keywords = Keywords.Calls)]
    public void End(int fixture, int iteration, long counter) => Emit(2, fixture, iteration, counter);
    [NonEvent]
    unsafe void Emit(int id, int fixture, int iteration, long counter)
    {
        if (!IsEnabled(EventLevel.Informational, (EventKeywords)1)) return;
        EventData* data = stackalloc EventData[3];
        data[0] = new EventData { DataPointer = (IntPtr)(&fixture), Size = sizeof(int) };
        data[1] = new EventData { DataPointer = (IntPtr)(&iteration), Size = sizeof(int) };
        data[2] = new EventData { DataPointer = (IntPtr)(&counter), Size = sizeof(long) };
        WriteEventCore(id, 3, data);
    }
}
