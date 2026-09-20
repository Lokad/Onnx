using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

Require(args.Length == 3, "root manifest output-directory");
string root = Path.GetFullPath(args[0]), manifestPath = Path.GetFullPath(args[1]), output = Path.GetFullPath(args[2]);
Require(!Directory.Exists(output), "Output exists"); Directory.CreateDirectory(output);
Require(OperatingSystem.IsWindows() && Environment.Version.ToString() == "10.0.12" && Environment.ProcessorCount == 1 && Affinity() == 4, "Qualified local runtime/affinity");
Require(Flags().Count == 0, "Unexpected runtime overrides"); NoNative();
using var document = JsonDocument.Parse(File.ReadAllBytes(manifestPath)); var spec = document.RootElement;
foreach (var p in spec.GetProperty("files").EnumerateObject())
{
    string path = Path.Combine(root, p.Name);
    Require(new FileInfo(path).Length == p.Value.GetProperty("bytes").GetInt64() && Sha(path) == p.Value.GetProperty("sha256").GetString(), "Changed file " + p.Name);
}
string core = Sha(typeof(ComputationalGraph).Assembly.Location);
Require(core == "7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9", "Exact original core required");
var graph = OnnxImport.Load(Path.Combine(root, spec.GetProperty("model").GetString()!), 256L * 1024 * 1024) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
Require(graph.Inputs.Keys.SequenceEqual(new[] { "input_features" }), "Encoder input");
Require(spec.GetProperty("protocol").GetString() == "whisper-input-cross-reused-context-v2", "Declared context lifecycle");
var context = graph.CreateExecution(ExecutionOptions.Memory);
var records = new List<object>(); var held = new Dictionary<string, (Tensor<float> Value, string Hash)>();
bool complete = false; Save();
var requests = spec.GetProperty("requests").EnumerateArray().ToArray(); Require(requests.Length == 21, "Fixed request scope");
for (int request = 0; request < requests.Length; request++)
{
    var item = requests[request]; string name = item.GetProperty("name").GetString()!;
    foreach (string kind in new[] { "MM", "MN" })
    {
        var input = Load(item.GetProperty(kind == "MM" ? "managed_features" : "native_features"));
        Require(input.Dimensions.SequenceEqual(new[] { 1,128,3000 }), "Feature shape");
        string inputHash = Bits(input); var memoryBefore = Memory();
        Tensor<float>? value = null;
        try
        {
            context.Reset(); Require(context.Execute(new Dictionary<string, ITensor> { ["input_features"] = input }, true, ExecutionProvider.CPU, ExecutionOptions.Memory), context.LastErrorMessage ?? "Encoder failed");
            Require(context.Outputs.Keys.SequenceEqual(new[] { "last_hidden_state" }), "Encoder output"); value = (Tensor<float>)context.Outputs["last_hidden_state"]!;
            Require(value.Dimensions.SequenceEqual(new[] { 1,1500,1280 }), "Hidden shape");
            Require(Bits(input) == inputHash, "Input changed");
            float[] actual = value.ToArray(), expected = Load(item.GetProperty("native_hidden")).ToArray();
            double maximum = 0; long failed = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                Require(float.IsFinite(actual[i]) && float.IsFinite(expected[i]), "Nonfinite output");
                double error = Math.Abs((double)actual[i] - expected[i]) / Math.Max(1, Math.Abs((double)expected[i])); maximum = Math.Max(maximum, error); if (error > 1e-4) failed++;
            }
            string file = $"{request:D2}-{name}-{kind}.f32", hash = Bits(value);
            using (var stream = new FileStream(Path.Combine(output, file), FileMode.CreateNew)) stream.Write(MemoryMarshal.AsBytes(actual.AsSpan()));
            bool? baseline = kind == "MM" ? hash == item.GetProperty("managed_hidden").GetProperty("raw_sha256").GetString() : null;
            if (request == 0) held.Add(kind, (value, hash));
            foreach (var h in held.Values) Require(Bits(h.Value) == h.Hash, "Held output changed");
            if (request == 20) Require(hash == held[kind].Hash, "Repeated output changed");
            records.Add(new { request, name, kind, file, sha256 = Sha(Path.Combine(output, file)), shape = new[] { 1,1500,1280 },
                input_sha256 = inputHash, baseline_matches = baseline, inputs_unchanged = true, held_outputs_unchanged = true,
                values = actual.Length, failed_values = failed, max_scaled = maximum, numerical_passed = failed == 0,
                memory_before = memoryBefore, memory_after = Memory(), pool_allocated_bytes = context.LastPoolAllocatedNewBytes,
                pool_reused_bytes = context.LastPoolReusedBytes });
            Save(); Require(baseline != false, "Original managed baseline bridge failed"); NoNative();
            Console.WriteLine($"{request:D2} {name} {kind} complete; numerical={failed == 0}, baseline={baseline}");
        }
        finally { context.Reset(); }
        foreach (var h in held.Values) Require(Bits(h.Value) == h.Hash, "Reset changed held output");
    }
}
complete = true; Save(); return 0;

DenseTensor<float> Load(JsonElement record)
{
    string path = Path.Combine(root, record.GetProperty("file").GetString()!); float[] values; int[] shape;
    if (record.GetProperty("format").GetString() == "npy") (values, shape) = NpySupport.ReadFloat32(path);
    else { values = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(path)).ToArray(); shape = record.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray(); }
    Require(values.All(float.IsFinite) && values.Length == shape.Aggregate(1, (a,b) => checked(a*b)), "Input/reference geometry");
    var tensor = new DenseTensor<float>(values, shape); Require(Bits(tensor) == record.GetProperty("raw_sha256").GetString(), "Raw input/reference hash"); return tensor;
}
void Save()
{
    string temporary = Path.Combine(output, "result.tmp");
    File.WriteAllText(temporary, JsonSerializer.Serialize(new { schema = 1, engine = "managed", context_lifecycle = "one-reused-context", complete, records,
        manifest_sha256 = Sha(manifestPath), core_sha256 = core, probe_sha256 = Sha(Assembly.GetExecutingAssembly().Location),
        runtime = Environment.Version.ToString(), affinity = Affinity(), processor_count = Environment.ProcessorCount, flags = Flags(),
        held_outputs = held.ToDictionary(p => p.Key, p => p.Value.Hash), native_loaded = false, packed_weight_bytes = 256L * 1024 * 1024,
        scope = "Encoder-only crossed-input diagnostic with one reused context on original qualified core; no public pipeline or timing qualification" }, new JsonSerializerOptions { WriteIndented = true }));
    File.Move(temporary, Path.Combine(output, "result.json"), true);
}
static Dictionary<string,string?> Flags() => Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k => k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)).ToDictionary(k => k, Environment.GetEnvironmentVariable);
static string Sha(string path) { using var f = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(f)); }
static string Bits(Tensor<float> tensor) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(tensor.ToArray().AsSpan())));
static long Affinity() { if (OperatingSystem.IsWindows() || OperatingSystem.IsLinux()) return Process.GetCurrentProcess().ProcessorAffinity.ToInt64(); throw new PlatformNotSupportedException(); }
static void NoNative() => Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Native ORT loaded");
static void Require(bool ok, string why) { if (!ok) throw new InvalidDataException(why); }

static object Memory()
{
    var info = GC.GetGCMemoryInfo();
    return new { allocated_total = GC.GetTotalAllocatedBytes(true), managed_estimate = GC.GetTotalMemory(false),
        collections = Enumerable.Range(0, 3).Select(GC.CollectionCount).ToArray(), gc_index = info.Index,
        last_gc_heap = info.HeapSizeBytes, last_gc_fragmented = info.FragmentedBytes, last_gc_committed = info.TotalCommittedBytes };
}
