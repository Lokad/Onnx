using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

if (args.Length != 2) throw new ArgumentException("model-directory output-json");
if (Console.ReadLine() != "GO") throw new InvalidOperationException("Bounded supervisor handshake required");
using var process = Process.GetCurrentProcess();
if (process.ProcessorAffinity.ToInt64() != 4) throw new InvalidOperationException("CPU2 required");
if (File.Exists(args[1])) throw new IOException("Existing result");
var first = OnnxImport.Load(Path.Combine(args[0], "decoder_model.onnx"), 64L * 1024 * 1024) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
var past = OnnxImport.Load(Path.Combine(args[0], "decoder_with_past_model.onnx"), 64L * 1024 * 1024) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
var beforeFirst = Snapshot(first);var beforePast = Snapshot(past);
var beforePhysical = Physical(first, past);
long logicalShared = (long)typeof(WhisperTranscriber).Assembly.GetType("Lokad.Onnx.WhisperDecoderWeights", true)!
    .GetMethod("Share", BindingFlags.Static | BindingFlags.NonPublic)!.Invoke(null, new object[] { first, past })!;
past.Prepare();
var afterFirst = Snapshot(first);var afterPast = Snapshot(past);var afterPhysical = Physical(first, past);
Require(JsonSerializer.Serialize(beforeFirst) == JsonSerializer.Serialize(afterFirst), "First graph values/plans changed");
Require(JsonSerializer.Serialize(beforePast) == JsonSerializer.Serialize(afterPast), "Past graph values/plans changed");
Require(logicalShared >= 600_000_000, "Sharing below the independently identified payload opportunity");
Require(beforePhysical.Bytes - afterPhysical.Bytes >= 600_000_000, "Insufficient reduction in unique rooted initializer payload");
Require(!process.Modules.Cast<ProcessModule>().Any(m => m.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "ORT loaded");
File.WriteAllText(args[1], JsonSerializer.Serialize(new { passed = true, inference = false, logical_shared_bytes = logicalShared,
    before = new { first = beforeFirst, past = beforePast, unique_arrays = beforePhysical.Count, unique_payload_bytes = beforePhysical.Bytes },
    after = new { first = afterFirst, past = afterPast, unique_arrays = afterPhysical.Count, unique_payload_bytes = afterPhysical.Bytes },
    runtime = RuntimeInformation.FrameworkDescription, affinity = process.ProcessorAffinity.ToInt64(), processor_count = Environment.ProcessorCount,
    core_sha256 = Sha(typeof(ComputationalGraph).Assembly.Location), data_sha256 = Sha(typeof(WhisperTranscriber).Assembly.Location), runner_sha256 = Sha(Assembly.GetExecutingAssembly().Location),
    flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k => k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)
        || k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)).ToDictionary(k => k, Environment.GetEnvironmentVariable)
}, new JsonSerializerOptions { WriteIndented = true }));
GC.KeepAlive(first);GC.KeepAlive(past);
return 0;

static string Sha(string path) { using var stream = File.OpenRead(path);return Convert.ToHexStringLower(SHA256.HashData(stream)); }
static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
static object Snapshot(ComputationalGraph graph) => new {
    initializers = graph.Initializers.OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => new { name = p.Key, tensor_name = p.Value.Name, type = p.Value.ElementType.ToString(),
        shape = p.Value.Dims, bytes = Payload(p.Value), sha256 = Digest(p.Value) }).ToArray(),
    nodes = graph.Nodes.Select(n => new { n.Name, op = n.Op.ToString(), n.Inputs, n.Outputs }).ToArray(),
    packed_bytes = graph.RetainedPackedWeightBytes
};
static long Payload(ITensor value) => value switch {
    DenseTensor<float> t => t.Buffer.Length * 4L,
    DenseTensor<long> t => t.Buffer.Length * 8L,
    DenseTensor<int> t => t.Buffer.Length * 4L,
    _ => throw new NotSupportedException(value.GetType().ToString())
};
static string Digest(ITensor value) => value switch {
    DenseTensor<float> t => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(t.Buffer.Span))),
    DenseTensor<long> t => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(t.Buffer.Span))),
    DenseTensor<int> t => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(t.Buffer.Span))),
    _ => throw new NotSupportedException(value.GetType().ToString())
};
static Array RootArray(ITensor value)
{
    if (value is DenseTensor<float> f) { Require(MemoryMarshal.TryGetArray<float>(f.Buffer, out var a) && a.Offset == 0 && a.Array!.Length == a.Count, "Float storage");return a.Array!; }
    if (value is DenseTensor<long> l) { Require(MemoryMarshal.TryGetArray<long>(l.Buffer, out var a) && a.Offset == 0 && a.Array!.Length == a.Count, "Long storage");return a.Array!; }
    if (value is DenseTensor<int> i) { Require(MemoryMarshal.TryGetArray<int>(i.Buffer, out var a) && a.Offset == 0 && a.Array!.Length == a.Count, "Int storage");return a.Array!; }
    throw new NotSupportedException(value.GetType().ToString());
}
static (int Count, long Bytes) Physical(params ComputationalGraph[] graphs)
{
    var arrays = new Dictionary<Array, long>(ReferenceEqualityComparer.Instance);
    foreach (var graph in graphs) foreach (var tensor in graph.Initializers.Values)
    {
        var array = RootArray(tensor);long bytes = Payload(tensor);
        if (arrays.TryGetValue(array, out long prior)) Require(prior == bytes, "Shared array size");else arrays.Add(array, bytes);
    }
    return (arrays.Count, arrays.Values.Sum());
}
