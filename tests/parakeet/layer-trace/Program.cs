using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

Require(args.Length == 3, "root manifest job");
string root = Path.GetFullPath(args[0]), manifestPath = Path.GetFullPath(args[1]);
using var document = JsonDocument.Parse(File.ReadAllBytes(manifestPath)); var spec = document.RootElement;
Require(spec.GetProperty("protocol").GetString() == "parakeet-layer-trace-v1", "Protocol");
var job = spec.GetProperty("jobs").EnumerateArray().Single(j => j.GetProperty("id").GetString() == args[2]);
Require(job.GetProperty("engine").GetString() == "managed", "Engine");
Require(OperatingSystem.IsWindows() && Environment.Version.ToString() == "10.0.12"
    && Environment.ProcessorCount == 1 && Process.GetCurrentProcess().ProcessorAffinity.ToInt64() == 4, "Runtime/affinity");
Require(!Environment.GetEnvironmentVariables().Keys.Cast<string>().Any(k => k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)
    || k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)), "Runtime overrides");
NoNative();
string core = Sha(typeof(ComputationalGraph).Assembly.Location);
Require(core == "d1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4", "Qualified core");
foreach (var p in spec.GetProperty("worker_files").EnumerateObject())
{
    string file = Path.Combine(root, p.Name);
    Require(new FileInfo(file).Length == p.Value.GetProperty("bytes").GetInt64() && Sha(file) == p.Value.GetProperty("sha256").GetString(), "Changed input " + p.Name);
}
string output = Path.Combine(root, spec.GetProperty("base").GetString()!, "outputs", args[2]);
Require(!Directory.Exists(output), "Output exists"); Directory.CreateDirectory(output);
string mode = job.GetProperty("mode").GetString()!, inputKind = job.GetProperty("input").GetString()!;
var model = spec.GetProperty("models").GetProperty(mode);
var graph = OnnxImport.Load(Path.Combine(root, model.GetString()!), 256L * 1024 * 1024)
    ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
var nodes = graph.Nodes.Select(n => new { name = n.Name, op = n.Op.ToString(), inputs = n.Inputs, outputs = n.Outputs,
    domain = n.Domain, opset = n.OpsetVersion, fused = n.IsFused }).ToArray();
File.WriteAllText(Path.Combine(output, "nodes.json"), JsonSerializer.Serialize(nodes));
var inputSpec = spec.GetProperty("inputs").GetProperty(inputKind);
var features = NpySupport.ReadTensor(Path.Combine(root, inputSpec.GetProperty("features").GetString()!));
var length = NpySupport.ReadTensor(Path.Combine(root, inputSpec.GetProperty("length").GetString()!));
Require(features.Dims.SequenceEqual(new[] { 1, 128, 586 }), "Feature shape");
var feeds = new Dictionary<string, ITensor> { ["audio_signal"] = features, ["length"] = length };
var original = feeds.ToDictionary(p => p.Key, p => Bits(p.Value));
var context = graph.CreateExecution(ExecutionOptions.Memory); var held = new List<(ITensor Value, string Hash)>();
var outputs = new List<object>();
try
{
    Require(context.Execute(feeds, true, ExecutionProvider.CPU, ExecutionOptions.Memory), context.LastErrorMessage ?? "Execute failed");
    var names = spec.GetProperty("outputs").GetProperty(mode).EnumerateArray().Select(v => v.GetString()!).ToArray();
    Require(context.Outputs.Keys.ToHashSet().SetEquals(names), "Output coverage");
    for (int i = 0; i < names.Length; i++)
    {
        ITensor tensor = context.Outputs[names[i]]; byte[] bytes = Bytes(tensor);
        if (tensor is Tensor<float> f) Require(f.ToArray().All(float.IsFinite), "Nonfinite output");
        string file = $"{i:D2}.bin"; using (var stream = new FileStream(Path.Combine(output, file), FileMode.CreateNew)) stream.Write(bytes);
        string hash = Convert.ToHexStringLower(SHA256.HashData(bytes)); held.Add((tensor, hash));
        outputs.Add(new { name = names[i], file, shape = tensor.Dims, dtype = tensor is Tensor<float> ? "float32" : tensor is Tensor<long> ? "int64" : "int32", bytes = bytes.Length, sha256 = hash });
    }
}
finally { context.Reset(); }
Require(feeds.All(p => Bits(p.Value) == original[p.Key]), "Input mutation");
Require(held.All(p => Bits(p.Value) == p.Hash), "Held output mutation after Reset"); NoNative();
File.WriteAllText(Path.Combine(output, "result.json"), JsonSerializer.Serialize(new { complete = true, job, outputs,
    manifest_sha256 = Sha(manifestPath), core_sha256 = core, runner_sha256 = Sha(Assembly.GetExecutingAssembly().Location),
    runtime = Environment.Version.ToString(), affinity = new[] { 2 }, processor_count = Environment.ProcessorCount,
    inputs_unchanged = true, held_outputs_unchanged = true, native_loaded = false,
    nodes_sha256 = Sha(Path.Combine(output, "nodes.json")), optimized_nodes = nodes.Length }, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"{args[2]}: {outputs.Count} arrays, {nodes.Length} optimized nodes");

static byte[] Bytes(ITensor tensor) => tensor switch
{
    Tensor<float> f => MemoryMarshal.AsBytes(f.ToArray().AsSpan()).ToArray(),
    Tensor<long> l => MemoryMarshal.AsBytes(l.ToArray().AsSpan()).ToArray(),
    Tensor<int> i => MemoryMarshal.AsBytes(i.ToArray().AsSpan()).ToArray(),
    _ => throw new InvalidDataException("Unexpected dtype")
};
static string Bits(ITensor tensor) => Convert.ToHexStringLower(SHA256.HashData(Bytes(tensor)));
static string Sha(string path) { using var f = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(f)); }
static void NoNative() => Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Native loaded");
static void Require(bool ok, string why) { if (!ok) throw new InvalidDataException(why); }
