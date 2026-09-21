using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

Require(args.Length == 4, "root manifest output job-id");
string root = Path.GetFullPath(args[0]), manifestPath = Path.GetFullPath(args[1]), output = Path.GetFullPath(args[2]);
Require(!Directory.Exists(output), "Output exists"); Directory.CreateDirectory(output);
Require(OperatingSystem.IsWindows() && Environment.Version.ToString() == "10.0.12" && Environment.ProcessorCount == 1 && Affinity() == 4, "Qualified runtime/CPU2");
Require(Flags().Count == 0, "Runtime overrides"); NoNative();
using var document = JsonDocument.Parse(File.ReadAllBytes(manifestPath)); var spec = document.RootElement;
Require(spec.GetProperty("protocol").GetString() == "whisper-managed-gelu-v1", "Protocol");
var job = spec.GetProperty("jobs").EnumerateArray().Single(x => x.GetProperty("id").GetString() == args[3]);
int selected = job.GetProperty("request").GetInt32(); string mode = job.GetProperty("mode").GetString()!;
var item = spec.GetProperty("requests")[selected]; var model = spec.GetProperty("models").GetProperty(mode);
foreach (var p in spec.GetProperty("worker_files").EnumerateObject())
{
    string path = Path.Combine(root, p.Name);
    Require(new FileInfo(path).Length == p.Value.GetProperty("bytes").GetInt64() && Sha(path) == p.Value.GetProperty("sha256").GetString(), "Changed file " + p.Name);
}
string core = Sha(typeof(ComputationalGraph).Assembly.Location);
Require(core == "d1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4", "Qualified current core");
var outputSpecs = model.GetProperty("outputs").EnumerateArray().ToArray();
Require(outputSpecs.Length == (mode == "baseline" ? 41 : 75), "Output coverage");
var graph = OnnxImport.Load(Path.Combine(root, model.GetProperty("file").GetString()!), 256L * 1024 * 1024) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
Require(graph.Inputs.Keys.SequenceEqual(new[] { "input_features" }), "Trace input");
int erf = graph.Nodes.Count(n => n.Op.ToString() == "Erf");
int gelu = graph.Nodes.Count(n => n.Op.ToString() is "Gelu" or "BiasGelu" or "GemmGelu");
Require(mode == "baseline" ? erf == 0 && gelu == 34 : erf == 34 && gelu == 0, "Actual fusion intervention");
var nodes = graph.Nodes.Select(n => new { name = n.Name, op = n.Op.ToString(), inputs = n.Inputs, outputs = n.Outputs,
    domain = n.Domain, opset = n.OpsetVersion, fused = n.IsFused,
    attributes = n.Attributes?.ToDictionary(p => p.Key, p => p.Value is ITensor t
        ? (object)new { type = t.ElementType.ToString(), shape = t.Dims, values = t.ToArray() } : p.Value) }).ToArray();
File.WriteAllText(Path.Combine(output, "nodes.json"), JsonSerializer.Serialize(nodes, new JsonSerializerOptions { WriteIndented = true }));
var held = new List<(DenseTensor<float> Value, string Hash)>(); var outputs = new List<object>();
bool complete = false; string before = ""; Save();
var input = Load(item.GetProperty("input")); Require(input.Dimensions.SequenceEqual(new[] { 1, 128, 3000 }), "Feature shape");
before = Bits(input); var context = graph.CreateExecution(ExecutionOptions.Memory);
try
{
    context.Reset(); Require(context.Execute(new Dictionary<string, ITensor> { ["input_features"] = input }, true, ExecutionProvider.CPU, ExecutionOptions.Memory), context.LastErrorMessage ?? "Execution failed");
    Require(context.Outputs.Count == outputSpecs.Length, "Actual outputs");
    for (int index = 0; index < outputSpecs.Length; index++)
    {
        var description = outputSpecs[index]; string name = description.GetProperty("name").GetString()!;
        var tensor = context.Outputs[name] as DenseTensor<float> ?? throw new InvalidDataException("Dense float output required");
        int[] shape = description.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        Require(tensor.Dimensions.SequenceEqual(shape), "Output shape " + name); Canonical(tensor);
        foreach (float value in tensor.Buffer.Span) Require(float.IsFinite(value), "Nonfinite output");
        string file = $"{index:D2}.f32", hash = Bits(tensor);
        using (var stream = new FileStream(Path.Combine(output, file), FileMode.CreateNew)) stream.Write(MemoryMarshal.AsBytes(tensor.Buffer.Span));
        held.Add((tensor, hash)); outputs.Add(new { index, name, shape, file, sha256 = hash, values = tensor.Length });
    }
    Require(Bits(input) == before, "Input mutation"); CheckHeld();
}
finally { context.Reset(); }
CheckHeld(); Require(Bits(input) == before, "Input changed after reset"); NoNative(); complete = true; Save();
Console.WriteLine($"Completed {args[3]}: {outputs.Count} arrays, {graph.Nodes.Count} optimized nodes");
return 0;

void CheckHeld() { foreach (var entry in held) Require(Bits(entry.Value) == entry.Hash, "Held output changed"); }
DenseTensor<float> Load(JsonElement record)
{
    string path = Path.Combine(root, record.GetProperty("file").GetString()!); float[] values; int[] shape;
    if (record.GetProperty("format").GetString() == "npy") (values, shape) = NpySupport.ReadFloat32(path);
    else { values = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(path)).ToArray(); shape = record.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray(); }
    Require(values.All(float.IsFinite) && values.LongLength == shape.Aggregate(1L, (a,b) => checked(a*b)), "Input shape");
    var tensor = new DenseTensor<float>(values, shape); Require(Bits(tensor) == record.GetProperty("raw_sha256").GetString(), "Input bytes"); return tensor;
}
void Save()
{
    string temporary = Path.Combine(output, "result.tmp");
    File.WriteAllText(temporary, JsonSerializer.Serialize(new { schema = 1, job, complete, outputs, input_sha256 = before,
        inputs_unchanged = complete, held_outputs_unchanged_after_reset = complete,
        manifest_sha256 = Sha(manifestPath), core_sha256 = core, probe_sha256 = Sha(Assembly.GetExecutingAssembly().Location),
        runtime = Environment.Version.ToString(), affinity = Affinity(), processor_count = Environment.ProcessorCount, flags = Flags(),
        native_loaded = false, packed_weight_bytes = 256L * 1024 * 1024, optimized_nodes = graph.Nodes.Count,
        nodes_sha256 = Sha(Path.Combine(output, "nodes.json")),
        node_census = graph.Nodes.GroupBy(n => n.Op.ToString()).ToDictionary(g => g.Key, g => g.Count()) }, new JsonSerializerOptions { WriteIndented = true }));
    File.Move(temporary, Path.Combine(output, "result.json"), true);
}
static void Canonical(DenseTensor<float> tensor)
{
    Require(!tensor.IsReversedStride && tensor.Buffer.Length == tensor.Length, "Canonical storage"); int stride = 1;
    for (int i = tensor.Rank - 1; i >= 0; i--) { Require(tensor.Strides[i] == stride, "Canonical strides"); stride = checked(stride * tensor.Dimensions[i]); }
}
static string Bits(DenseTensor<float> tensor) { Canonical(tensor); return Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(tensor.Buffer.Span))); }
static string Sha(string path) { using var f = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(f)); }
static Dictionary<string,string?> Flags() => Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k => k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)).ToDictionary(k => k, Environment.GetEnvironmentVariable);
static long Affinity() { if (OperatingSystem.IsWindows() || OperatingSystem.IsLinux()) return Process.GetCurrentProcess().ProcessorAffinity.ToInt64(); throw new PlatformNotSupportedException(); }
static void NoNative() => Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Native ORT loaded");
static void Require(bool ok, string why) { if (!ok) throw new InvalidDataException(why); }
