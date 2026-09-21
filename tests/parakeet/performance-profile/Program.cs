using System.Diagnostics;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

if (args.Length != 3) throw new ArgumentException("root manifest new-output");
if (!OperatingSystem.IsWindows()) throw new PlatformNotSupportedException("Local Windows attribution");
string root = Path.GetFullPath(args[0]), manifestPath = Path.GetFullPath(args[1]), output = Path.GetFullPath(args[2]);
Require(!Directory.Exists(output) && !File.Exists(output), "Output already exists");
using var process = Process.GetCurrentProcess();
Require(process.ProcessorAffinity.ToInt64() == 4 && Environment.ProcessorCount == 1, "CPU2 required before CLR startup");
Require(Sha(typeof(ComputationalGraph).Assembly.Location) == "d1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4", "Production core differs");
Require(Sha(typeof(ParakeetTranscriber).Assembly.Location) == "e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb", "Production Data differs");
var json = new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };
using var document = JsonDocument.Parse(File.ReadAllBytes(manifestPath)); var manifest = document.RootElement;
Require(manifest.GetProperty("schema").GetInt32() == 1 && manifest.GetProperty("family").GetString() == "parakeet", "Manifest scope");
string PathOf(JsonElement spec)
{
    string path = Path.GetFullPath(Path.Combine(root, spec.GetProperty("path").GetString()!));
    Require(path.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase), "Input escapes root");
    Require(new FileInfo(path).Length == spec.GetProperty("bytes").GetInt64() && Sha(path) == spec.GetProperty("sha256").GetString(), "Input identity: " + path);
    return path;
}
var models = manifest.GetProperty("models").EnumerateObject().ToDictionary(p => p.Name, p => PathOf(p.Value));
PathOf(manifest.GetProperty("reference"));
var cases = manifest.GetProperty("cases").EnumerateArray().Select(c =>
{
    var (pcm, shape) = NpySupport.ReadFloat32(PathOf(c.GetProperty("pcm")));
    Require(shape.SequenceEqual(new[] { c.GetProperty("samples").GetInt32() }) && pcm.All(float.IsFinite), "PCM layout");
    return new Case(c.GetProperty("name").GetString()!, pcm, Hash(MemoryMarshal.AsBytes(pcm.AsSpan())), c.GetProperty("expected").Clone());
}).ToArray();
Require(cases.Length == 20 && cases.Sum(c => c.Pcm.Length) == 3412240
    && cases.Sum(c => c.Expected.GetProperty("decoder_calls").GetInt32()) == 1200, "Corpus coverage");
Directory.CreateDirectory(output); Directory.CreateDirectory(Path.Combine(output, "arrays"));
void Write(string name, object value)
{
    using var stream = new FileStream(Path.Combine(output, name), FileMode.CreateNew, FileAccess.Write);
    JsonSerializer.Serialize(stream, value, json);
}
var files = new List<string>(); var applications = new List<object>(); var traced = new List<object>();
var held = new List<(ITensor Tensor, string Hash)>();
var heldPublic = new List<(ParakeetTranscription Value, string Json)>();
var baseline = new Dictionary<string, Dictionary<string, TensorRecord>>();
bool passed = false; string? error = null;
void CheckHeld()
{
    foreach (var item in held) Require(Hash(Bytes(item.Tensor)) == item.Hash, "Held graph output changed");
    foreach (var item in heldPublic) Require(JsonSerializer.Serialize(item.Value) == item.Json, "Held public output changed");
    foreach (var c in cases) Require(Hash(MemoryMarshal.AsBytes(c.Pcm.AsSpan())) == c.Hash, "Caller PCM changed");
}
try
{
    NoNative(); var model = new ParakeetTranscriber(Path.GetDirectoryName(models["nemo128.onnx"])!);
    var graphs = new[] { "frontend", "encoder", "decoder" }.ToDictionary(n => n, n => (ComputationalGraph)Private(model, n));
    object generation = Private(model, "generation");
    var decode = generation.GetType().GetMethod("Decode", BindingFlags.Instance | BindingFlags.NonPublic) ?? throw new InvalidDataException("Missing managed Decode");
    Write("graphs.json", graphs.ToDictionary(p => p.Key, p => p.Value.Nodes.Select(n => new {
        id = n.ID, name = n.Name, op = n.Op.ToString(), inputs = n.Inputs, outputs = n.Outputs,
        initializers = n.Inputs.Where(i => p.Value.Initializers.ContainsKey(i)).Distinct().ToDictionary(i => i,
            i => new { shape = p.Value.Initializers[i].Dims.ToArray(), dtype = p.Value.Initializers[i].ElementType.ToString() })
    }).ToArray()));
    for (int pass = 0; pass < 2; pass++) foreach (var c in cases)
    {
        CheckHeld(); var executions = graphs.ToDictionary(p => p.Key, p => p.Value.CreateExecution(ExecutionOptions.Memory));
        int step = 0;
        IReadOnlyDictionary<string, ITensor> Execute(string graphName, Dictionary<string, ITensor> feeds)
        {
            string key = c.Name + "/" + graphName + "/" + (graphName == "decoder" ? step : 0);
            var context = executions[graphName]; var graph = graphs[graphName];
            string stem = files.Count.ToString("D4"); int array = 0;
            TensorRecord Save(ITensor tensor)
            {
                byte[] bytes = Bytes(tensor); string file = "arrays/" + stem + "-" + (array++).ToString("D2") + ".bin";
                File.WriteAllBytes(Path.Combine(output, file), bytes);
                return new TensorRecord(file, Dtype(tensor), tensor.Dims.ToArray(), tensor.Length, Hash(bytes));
            }
            var input = feeds.ToDictionary(p => p.Key, p => Save(p.Value));
            long resetStart = Stopwatch.GetTimestamp(); context.Reset(); long resetEnd = Stopwatch.GetTimestamp();
            long allocation = GC.GetTotalAllocatedBytes(); int[] gcBefore = Enumerable.Range(0, 3).Select(GC.CollectionCount).ToArray();
            using var profile = pass == 1 ? Profiler.BeginWallExecution() : Profiler.BeginExecution(false);
            long start = Stopwatch.GetTimestamp();
            Require(context.Execute(feeds, true, ExecutionProvider.CPU, ExecutionOptions.Memory), context.LastErrorMessage ?? "Graph failed");
            long end = Stopwatch.GetTimestamp();
            long allocated = GC.GetTotalAllocatedBytes() - allocation; int[] gcAfter = Enumerable.Range(0, 3).Select(GC.CollectionCount).ToArray();
            var outputs = context.Outputs.ToDictionary(p => p.Key, p => p.Value ?? throw new InvalidDataException("Missing output"));
            var nodes = (context.LastWallProfile ?? Array.Empty<WallNode>()).Select(n => new {
                id = n.NodeId, op = n.Op.ToString(), start_ticks = n.StartTicks, end_ticks = n.EndTicks
            }).ToArray();
            Require(pass == 0 ? nodes.Length == 0 : nodes.Length == graph.Nodes.Count, "Node coverage");
            long last = start;
            foreach (var node in nodes)
            {
                Require(last <= node.start_ticks && node.start_ticks <= node.end_ticks && node.end_ticks <= end, "Node clocks"); last = node.end_ticks;
            }
            foreach (var pair in feeds) Require(Hash(Bytes(pair.Value)) == input[pair.Key].Sha256, "Graph changed its input");
            var saved = outputs.ToDictionary(p => p.Key, p => Save(p.Value));
            var all = input.ToDictionary(p => "input/" + p.Key, p => p.Value);
            foreach (var pair in saved) all.Add("output/" + pair.Key, pair.Value);
            if (pass == 0) baseline.Add(key, all);
            else
            {
                var prior = baseline[key]; Require(prior.Count == all.Count, "Capture coverage");
                foreach (var pair in all)
                    Require(prior[pair.Key].Sha256 == pair.Value.Sha256 && prior[pair.Key].Dtype == pair.Value.Dtype
                        && prior[pair.Key].Shape.SequenceEqual(pair.Value.Shape), "Profiling changed tensor bits: " + key + "/" + pair.Key);
            }
            foreach (var pair in outputs) held.Add((pair.Value, saved[pair.Key].Sha256));
            string file = stem + ".json";
            Write(file, new { name = c.Name, pass, phase = pass == 0 ? "unprofiled" : "wall", graph = graphName,
                step = graphName == "decoder" ? step : -1, start_ticks = start, end_ticks = end,
                reset_start_ticks = resetStart, reset_end_ticks = resetEnd, frequency = Stopwatch.Frequency,
                allocated_bytes = allocated, gc_before = gcBefore, gc_after = gcAfter, inputs = input, outputs = saved, nodes });
            files.Add(file); return outputs;
        }
        try
        {
            var features = Execute("frontend", new() { ["waveforms"] = new DenseTensor<float>(c.Pcm, new[] { 1, c.Pcm.Length }),
                ["waveforms_lens"] = new DenseTensor<long>(new[] { (long)c.Pcm.Length }, new[] { 1 }) });
            var encoded = Execute("encoder", new() { ["audio_signal"] = features["features"], ["length"] = features["features_lens"] });
            int frames = checked((int)((Tensor<long>)encoded["encoded_lengths"]).ToArray()[0]);
            Func<Dictionary<string, ITensor>, IReadOnlyDictionary<string, ITensor>> callback = feeds =>
            {
                Require(step < c.Expected.GetProperty("decoder_calls").GetInt32(), "Decoder exceeded complete expected trajectory");
                var values = Execute("decoder", feeds); step++; return values;
            };
            ParakeetTranscription result;
            try { result = (ParakeetTranscription)decode.Invoke(generation, new object[] { encoded["outputs"], frames, ParakeetTranscriptionOptions.Default, callback, CancellationToken.None })!; }
            catch (TargetInvocationException ex) when (ex.InnerException is not null) { ExceptionDispatchInfo.Capture(ex.InnerException).Throw(); throw; }
            var normalized = Normalize(result); Require(JsonNode.DeepEquals(JsonNode.Parse(normalized.GetRawText()), JsonNode.Parse(c.Expected.GetRawText())), "Trace/native decisions differ");
            Require(step == result.DecoderCalls, "Incomplete decoder trajectory");
            heldPublic.Add((result, JsonSerializer.Serialize(result))); traced.Add(new { name = c.Name, pass, result = normalized });
        }
        finally { foreach (var context in executions.Values) context.Reset(); }
        CheckHeld(); Console.WriteLine(c.Name + " " + (pass == 0 ? "unprofiled" : "wall") + " complete; decoder calls=" + step);
    }
    foreach (var c in cases)
    {
        CheckHeld(); long allocation = GC.GetTotalAllocatedBytes(); long start = Stopwatch.GetTimestamp();
        var result = model.Transcribe(c.Pcm, 16000, ParakeetTranscriptionOptions.Default, CancellationToken.None);
        long end = Stopwatch.GetTimestamp(); long allocated = GC.GetTotalAllocatedBytes() - allocation;
        var normalized = Normalize(result); Require(JsonNode.DeepEquals(JsonNode.Parse(normalized.GetRawText()), JsonNode.Parse(c.Expected.GetRawText())), "Public/native decisions differ");
        heldPublic.Add((result, JsonSerializer.Serialize(result))); CheckHeld();
        applications.Add(new { name = c.Name, result = normalized, start_ticks = start, end_ticks = end,
            frequency = Stopwatch.Frequency, allocated_bytes = allocated, input_sha256 = c.Hash });
        Console.WriteLine(c.Name + " public control complete");
    }
    Require(files.Count == 2480 && held.Count == 9760 && traced.Count == 40 && applications.Count == 20, "Complete scope");
    CheckHeld(); NoNative(); passed = true;
}
catch (Exception ex) { error = ex.ToString(); Console.Error.WriteLine(error); }
Write("result.json", new { passed, error, call_files = files, traced, applications,
    inputs_and_held_outputs_unchanged = passed, manifest_sha256 = Sha(manifestPath),
    core_sha256 = Sha(typeof(ComputationalGraph).Assembly.Location), data_sha256 = Sha(typeof(ParakeetTranscriber).Assembly.Location),
    runner_sha256 = Sha(Assembly.GetExecutingAssembly().Location), runtime = RuntimeInformation.FrameworkDescription,
    affinity = process.ProcessorAffinity.ToInt64(), processor_count = Environment.ProcessorCount,
    flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k => k.StartsWith("LOKAD_") || k.StartsWith("DOTNET_") || k.StartsWith("COMPlus_")).ToDictionary(k => k, Environment.GetEnvironmentVariable),
    scope = "Local full-corpus graph attribution and separate public controls; no matched native timing or new tensor-native verdict" });
return passed ? 0 : 1;

static byte[] Bytes(ITensor tensor) => tensor switch {
    Tensor<float> f => FloatBytes(f), Tensor<int> i => MemoryMarshal.AsBytes(i.ToArray().AsSpan()).ToArray(),
    Tensor<long> l => MemoryMarshal.AsBytes(l.ToArray().AsSpan()).ToArray(), _ => throw new InvalidDataException("Unsupported tensor dtype") };
static byte[] FloatBytes(Tensor<float> tensor) { var values = tensor.ToArray(); Require(values.All(float.IsFinite), "Nonfinite tensor"); return MemoryMarshal.AsBytes(values.AsSpan()).ToArray(); }
static string Dtype(ITensor tensor) => tensor is Tensor<float> ? "<f4" : tensor is Tensor<int> ? "<i4" : tensor is Tensor<long> ? "<i8" : throw new InvalidDataException();
static string Sha(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
static object Private(object value, string name) => value.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(value) ?? throw new InvalidDataException("Missing private field: " + name);
static JsonElement Normalize(ParakeetTranscription p) => JsonSerializer.SerializeToElement(new { text = p.Text, token_ids = p.TokenIds, frame_indices = p.FrameIndices,
    duration_frames = p.DurationFrames, stop_reason = p.StopReason.ToString(), encoded_frames = p.EncodedFrames, decoder_calls = p.DecoderCalls });
static void NoNative() { using var process = Process.GetCurrentProcess(); Require(!process.Modules.Cast<ProcessModule>().Any(m => m.FileName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Native ORT loaded"); }
record Case(string Name, float[] Pcm, string Hash, JsonElement Expected);
record TensorRecord(string File, string Dtype, int[] Shape, long Values, string Sha256);
