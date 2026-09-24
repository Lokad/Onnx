using System.Collections;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

internal record TimedCall(string name, int step, int index, Dictionary<string, ITensor> feeds, string[] outputs, string[] hashes, int[][] shapes);
internal record ClockRow(string name, int step, int index, string phase, int repeat, long ticks, long allocated_bytes, string[] output_sha256);
internal record SetupRow(int index, long ticks, long allocated_bytes, long retained_bytes);

internal static class Timing
{
    const long Budget = 64L * 1024 * 1024;
    static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
    static string Sha(string file) { using var s = File.OpenRead(file); return Convert.ToHexStringLower(SHA256.HashData(s)); }
    static string Bits(ITensor t) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(((Tensor<float>)t).ToArray().AsSpan())));
    static string Text(JsonElement e) => e.GetString() ?? throw new InvalidDataException();
    static int[] Shape(JsonElement e) => e.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();

    static int Main(string[] args)
    {
        Require(args.Length == 5 && OperatingSystem.IsLinux(), "base role mode worker new-output-directory");
        string root = Path.GetFullPath(args[0]), role = args[1], mode = args[2], worker = args[3];
        Require((role is "selected" or "candidate") && (mode is "512" or "256"), "Declared role/mode");
        Require(Environment.Version.ToString() == "10.0.8" && Environment.ProcessorCount == 1 && Process.GetCurrentProcess().ProcessorAffinity.ToInt64() == 4, "AMD runtime CPU2");
        Require(Avx512F.IsSupported == (mode == "512") && Avx2.IsSupported && Fma.IsSupported, "Actual instruction mode");
        var flags = Environment.GetEnvironmentVariables().Cast<DictionaryEntry>().Where(e => ((string)e.Key).StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase)
            || ((string)e.Key).StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase) || ((string)e.Key).StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase))
            .ToDictionary(e => (string)e.Key, e => (string)e.Value!);
        Require(mode == "512" ? flags.Count == 0 : flags.Count == 1 && flags.GetValueOrDefault("DOTNET_EnableAVX512") == "0", "Declared override only");
        using var specDoc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, "spec.json"))); var spec = specDoc.RootElement;
        var identities = spec.GetProperty("identities").GetProperty(role);
        Require(Sha(typeof(ComputationalGraph).Assembly.Location) == Text(identities.GetProperty("Lokad.Onnx.dll").GetProperty("sha256")), "Core identity");
        Require(Sha(typeof(ParakeetTranscriber).Assembly.Location) == Text(identities.GetProperty("Lokad.Onnx.Data.dll").GetProperty("sha256")), "Data identity");
        using var captureDoc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, "fixtures/result.json")));
        var captured = captureDoc.RootElement.GetProperty("calls").EnumerateArray().ToArray();Require(captured.Length == 380, "Every captured call");
        var arrays = new Dictionary<string, float[]>(); var sources = new Dictionary<ITensor, string>(ReferenceEqualityComparer.Instance);
        Tensor<float> Load(JsonElement item)
        {
            string file = Path.GetFullPath(Path.Combine(root, "fixtures", Text(item.GetProperty("file"))));
            Require(file.StartsWith(Path.Combine(root, "fixtures") + Path.DirectorySeparatorChar, StringComparison.Ordinal), "Fixture path");
            if (!arrays.TryGetValue(file, out var values))
            {
                Require(Sha(file) == Text(item.GetProperty("sha256")), "Fixture hash");
                values = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(file)).ToArray();Require(values.All(float.IsFinite), "Finite fixture");arrays.Add(file, values);
            }
            var tensor = new DenseTensor<float>(values, Shape(item));sources.Add(tensor, Bits(tensor));return tensor;
        }
        // Parsing, file IO and construction of caller-owned feeds precede all clocks.
        var calls = captured.Select(c =>
        {
            var names = c.GetProperty("input_names").EnumerateArray().Select(Text).ToArray();var inputs = c.GetProperty("inputs");
            return new TimedCall(Text(c.GetProperty("name")), c.GetProperty("step").GetInt32(), c.GetProperty("index").GetInt32(),
                new[] { 0, 5, 6 }.ToDictionary(i => names[i], i => (ITensor)Load(inputs[i])),
                c.GetProperty("output_names").EnumerateArray().Select(Text).ToArray(),
                c.GetProperty("outputs").EnumerateArray().Select(o => Text(o.GetProperty("sha256"))).ToArray(),
                c.GetProperty("outputs").EnumerateArray().Select(Shape).ToArray());
        }).ToArray();
        var graphs = new ComputationalGraph[2];var contexts = new GraphExecution[2];var setup = new List<SetupRow>();
        for (int index = 0; index < 2; index++)
        {
            var item = captured.First(c => c.GetProperty("index").GetInt32() == index);
            string[] ins = item.GetProperty("input_names").EnumerateArray().Select(Text).ToArray(), outs = item.GetProperty("output_names").EnumerateArray().Select(Text).ToArray();
            var constants = new[] { 1, 2, 3 }.ToDictionary(i => ins[i], i => (ITensor)Load(item.GetProperty("inputs")[i]));
            Require(item.GetProperty("opset").GetInt32() == 17 && item.GetProperty("attributes").EnumerateObject().Count() == 1
                && item.GetProperty("attributes").GetProperty("hidden_size").GetInt32() == 640 && ins[4] == "", "Exact actual node geometry");
            long allocation = GC.GetAllocatedBytesForCurrentThread(), begin = Stopwatch.GetTimestamp();
            var graph = new ComputationalGraph(Budget);graph.Metadata["Name"] = "actual-complete-lstm-" + index;graph.Opset[""] = 17;
            foreach (var constant in constants) graph.Initializers[constant.Key] = constant.Value;
            foreach (int i in new[] { 0, 5, 6 }) graph.Inputs[ins[i]] = new DenseTensor<float>(Shape(item.GetProperty("inputs")[i]));
            for (int i = 0; i < 3; i++) graph.Outputs[outs[i]] = new DenseTensor<float>(Shape(item.GetProperty("outputs")[i]));
            graph.Nodes.Add(new Node { Name = Text(item.GetProperty("node")), Op = OpType.LSTM, OpTypeName = "LSTM", Domain = "", OpsetVersion = 17,
                Inputs = ins, Outputs = outs, Attributes = new() { ["hidden_size"] = 640 } });
            graph.Prepare();contexts[index] = graph.CreateExecution(ExecutionOptions.Memory);
            long ticks = Stopwatch.GetTimestamp() - begin, allocated = GC.GetAllocatedBytesForCurrentThread() - allocation;
            Require(graph.RetainedPackedWeightBytes == (role == "candidate" ? 13107200 : 0) && graph.MaximumPackedWeightBytes == Budget, "Prepared residency");
            graphs[index] = graph;setup.Add(new(index, ticks, allocated, graph.RetainedPackedWeightBytes));
        }
        var rows = new List<ClockRow>();var held = new Dictionary<ITensor, string>(ReferenceEqualityComparer.Instance);
        foreach (string phase in new[] { "warmup", "measured" })
        for (int repeat = 0; repeat < 5; repeat++)
        {
            // Preserve the original case/step/node order. Output validation is outside each timed call.
            var results = new List<(TimedCall Call, ITensor[] Outputs, long Ticks, long Allocated)>();
            foreach (var call in calls)
            {
                var context = contexts[call.index];long allocation = GC.GetAllocatedBytesForCurrentThread(), begin = Stopwatch.GetTimestamp();
                context.Reset();bool passed = context.Execute(call.feeds, true, ExecutionProvider.CPU, ExecutionOptions.Memory);
                long ticks = Stopwatch.GetTimestamp() - begin, allocated = GC.GetAllocatedBytesForCurrentThread() - allocation;
                Require(passed, context.LastErrorMessage ?? "Complete graph execution");Require(ticks > 0 && allocated >= 0, "Raw complete-call clock/allocation");
                results.Add((call, call.outputs.Select(n => context.Outputs[n]!).ToArray(), ticks, allocated));
            }
            foreach (var result in results)
            {
                string[] hashes = result.Outputs.Select(Bits).ToArray();Require(hashes.SequenceEqual(result.Call.hashes), "Every captured output bit remains exact");
                for (int i = 0; i < 3; i++)
                {
                    Require(((Tensor<float>)result.Outputs[i]).ToArray().All(float.IsFinite) && result.Outputs[i].Dims.SequenceEqual(result.Call.shapes[i]), "Finite output with exact shape");
                    if (held.TryGetValue(result.Outputs[i], out string? prior)) Require(prior == hashes[i], "No reused output object changes prior values");
                    else held.Add(result.Outputs[i], hashes[i]);
                }
                rows.Add(new(result.Call.name, result.Call.step, result.Call.index, phase, repeat, result.Ticks, result.Allocated, hashes));
            }
            Console.WriteLine(phase + " " + repeat + " complete");
        }
        foreach (var context in contexts) context.Reset();
        foreach (var pair in held) Require(Bits(pair.Key) == pair.Value, "All held outputs survive later calls and resets");
        foreach (var pair in sources) Require(Bits(pair.Key) == pair.Value, "Every original input/constant remains immutable");
        Require(rows.Count == 3800 && held.Count > 0, "All warm and measured clocks retained");
        Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.FileName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Managed complete calls only");
        Require(!Directory.Exists(args[4]), "Exclusive-create output");Directory.CreateDirectory(args[4]);
        using var output = new FileStream(Path.Combine(args[4], "result.json"), FileMode.CreateNew);
        JsonSerializer.Serialize(output, new { passed = true, role, mode, worker, pid = Environment.ProcessId, affinity = 4, processor_count = Environment.ProcessorCount,
            core = Sha(typeof(ComputationalGraph).Assembly.Location), data = Sha(typeof(ParakeetTranscriber).Assembly.Location), consumer = Sha(Assembly.GetExecutingAssembly().Location),
            runtime = Environment.Version.ToString(), avx512 = Avx512F.IsSupported, flags, frequency = Stopwatch.Frequency,
            spec_sha256 = Sha(Path.Combine(root, "spec.json")), capture_sha256 = Sha(Path.Combine(root, "fixtures/result.json")), setup, rows,
            inputs_unchanged = true, held_outputs_unchanged = true, exact_selected_outputs = true, complete_calls = true,
            setup_includes_graph_preparation_and_context = true, per_call_includes_reset_validation_allocation_and_execution = true }, new JsonSerializerOptions { WriteIndented = true });
        return 0;
    }
}
