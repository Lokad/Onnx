using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

Require(args.Length == 4, "root manifest output request-index");
string root = Path.GetFullPath(args[0]), manifestPath = Path.GetFullPath(args[1]), output = Path.GetFullPath(args[2]);
Require(!Directory.Exists(output), "Output exists"); Directory.CreateDirectory(output);
Require(OperatingSystem.IsWindows() && Environment.Version.ToString() == "10.0.12" && Environment.ProcessorCount == 1 && Affinity() == 4, "Qualified local runtime/CPU2");
Require(Flags().Count == 0, "Runtime overrides"); NoNative();
using var document = JsonDocument.Parse(File.ReadAllBytes(manifestPath)); var spec = document.RootElement;
Require(spec.GetProperty("protocol").GetString() == "whisper-selected-natural-trace-v1", "Protocol");
foreach (var p in spec.GetProperty("files").EnumerateObject())
{
    string path = Path.Combine(root, p.Name);
    Require(new FileInfo(path).Length == p.Value.GetProperty("bytes").GetInt64() && Sha(path) == p.Value.GetProperty("sha256").GetString(), "Changed file " + p.Name);
}
string core = Sha(typeof(ComputationalGraph).Assembly.Location);
Require(core == "7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9", "Original core");
int selected = int.Parse(args[3]); Require(selected is >= 0 and < 4, "Selected request");
var requests = spec.GetProperty("requests").EnumerateArray().ToArray(); Require(requests.Length == 4, "Four fixed requests");
var item = requests[selected]; var outputSpecs = spec.GetProperty("outputs").EnumerateArray().ToArray(); Require(outputSpecs.Length == 41, "Full trace coverage");
var graph = OnnxImport.Load(Path.Combine(root, spec.GetProperty("model").GetString()!), 256L * 1024 * 1024) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
Require(graph.Inputs.Keys.SequenceEqual(new[] { "input_features" }), "Trace input");
var records = new List<object>(); var held = new Dictionary<string, (DenseTensor<float> Value, string Hash)>();
bool complete = false; Save();
foreach (string kind in new[] { "MM", "MN" })
{
    var input = Load(item.GetProperty(kind == "MM" ? "managed_features" : "native_features"));
    Require(input.Dimensions.SequenceEqual(new[] { 1, 128, 3000 }), "Feature shape");
    string before = Bits(input); var context = graph.CreateExecution(ExecutionOptions.Memory);
    try
    {
        context.Reset(); Require(context.Execute(new Dictionary<string, ITensor> { ["input_features"] = input }, true, ExecutionProvider.CPU, ExecutionOptions.Memory), context.LastErrorMessage ?? "Trace execution failed");
        Require(context.Outputs.Count == 41, "Trace outputs");
        var outputs = new List<object>();
        for (int index = 0; index < outputSpecs.Length; index++)
        {
            var description = outputSpecs[index]; string name = description.GetProperty("name").GetString()!;
            var tensor = context.Outputs[name] as DenseTensor<float> ?? throw new InvalidDataException("Dense trace output required");
            int[] shape = description.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            Require(tensor.Dimensions.SequenceEqual(shape), "Trace output shape " + name); Canonical(tensor);
            foreach (float value in tensor.Buffer.Span) Require(float.IsFinite(value), "Nonfinite trace output");
            string file = $"{kind}-{index:D2}.f32", hash = Bits(tensor);
            using (var stream = new FileStream(Path.Combine(output, file), FileMode.CreateNew)) stream.Write(MemoryMarshal.AsBytes(tensor.Buffer.Span));
            held.Add(kind + ":" + index, (tensor, hash));
            outputs.Add(new { index, name, shape, file, sha256 = hash, values = tensor.Length });
        }
        Require(Bits(input) == before, "Input mutation"); CheckHeld();
        var final = held[kind + ":40"].Value; var reference = Load(item.GetProperty("baselines").GetProperty(kind));
        Require(reference.Dimensions.SequenceEqual(new[] { 1, 1500, 1280 }), "Unmodified baseline shape");
        double maximum = 0; long failed = 0;
        for (int i = 0; i < final.Buffer.Length; i++)
        {
            double error = Math.Abs((double)final.Buffer.Span[i] - reference.Buffer.Span[i]) / Math.Max(1, Math.Abs((double)reference.Buffer.Span[i]));
            maximum = Math.Max(maximum, error); if (error > 1e-4) failed++;
        }
        records.Add(new { kind, request = selected, name = item.GetProperty("name").GetString(), input_sha256 = before,
            inputs_unchanged = true, held_outputs_unchanged = true, outputs,
            instrumentation = new { baseline_sha256 = Bits(reference), final_sha256 = Bits(final), bitwise = Bits(reference) == Bits(final), max_scaled = maximum, failed_values = failed } });
    }
    finally { context.Reset(); }
    CheckHeld(); Save(); NoNative(); Console.WriteLine($"Completed trace request {selected} {kind}: all 41 outputs retained");
}
complete = true; Save(); return 0;

void CheckHeld() { foreach (var entry in held.Values) Require(Bits(entry.Value) == entry.Hash, "Held trace output changed"); }
DenseTensor<float> Load(JsonElement record)
{
    string path = Path.Combine(root, record.GetProperty("file").GetString()!); float[] values; int[] shape;
    if (record.GetProperty("format").GetString() == "npy") (values, shape) = NpySupport.ReadFloat32(path);
    else { values = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(path)).ToArray(); shape = record.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray(); }
    Require(values.All(float.IsFinite) && values.LongLength == shape.Aggregate(1L, (a,b) => checked(a*b)), "Input/reference shape");
    var tensor = new DenseTensor<float>(values, shape); Require(Bits(tensor) == record.GetProperty("raw_sha256").GetString(), "Input/reference bytes"); return tensor;
}
void Save()
{
    string temporary = Path.Combine(output, "result.tmp");
    File.WriteAllText(temporary, JsonSerializer.Serialize(new { schema = 1, engine = "managed", request_index = selected, complete, records,
        manifest_sha256 = Sha(manifestPath), core_sha256 = core, probe_sha256 = Sha(Assembly.GetExecutingAssembly().Location), runtime = Environment.Version.ToString(),
        affinity = Affinity(), processor_count = Environment.ProcessorCount, flags = Flags(), held_outputs = held.ToDictionary(p => p.Key, p => p.Value.Hash),
        native_loaded = false, packed_weight_bytes = 256L * 1024 * 1024, optimized_nodes = graph.Nodes.Count,
        node_census = graph.Nodes.GroupBy(n => n.Op.ToString()).ToDictionary(g => g.Key, g => g.Count()),
        scope = "Selected natural-case trace; original-output instrumentation differences retained; no product qualification" }, new JsonSerializerOptions { WriteIndented = true }));
    File.Move(temporary, Path.Combine(output, "result.json"), true);
}
static void Canonical(DenseTensor<float> tensor)
{
    Require(!tensor.IsReversedStride && tensor.Buffer.Length == tensor.Length, "Canonical dense storage");
    int stride = 1;
    for (int i = tensor.Rank - 1; i >= 0; i--) { Require(tensor.Strides[i] == stride, "Canonical trace strides"); stride = checked(stride * tensor.Dimensions[i]); }
}
static string Bits(DenseTensor<float> tensor) { Canonical(tensor); return Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(tensor.Buffer.Span))); }
static string Sha(string path) { using var f = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(f)); }
static Dictionary<string,string?> Flags() => Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k => k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)).ToDictionary(k => k, Environment.GetEnvironmentVariable);
static long Affinity() { if (OperatingSystem.IsWindows() || OperatingSystem.IsLinux()) return Process.GetCurrentProcess().ProcessorAffinity.ToInt64(); throw new PlatformNotSupportedException(); }
static void NoNative() => Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Native ORT loaded");
static void Require(bool ok, string why) { if (!ok) throw new InvalidDataException(why); }
