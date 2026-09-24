using System.Collections;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

if (args.Length != 6) throw new ArgumentException("model.onnx budget-bytes core-sha data-sha avx512 new-result.json");
using var process = Process.GetCurrentProcess();
Require(process.ProcessorAffinity.ToInt64() == 4 && Environment.ProcessorCount == 1, "CPU2 before CLR startup");
Require(RuntimeInformation.FrameworkDescription == ".NET 10.0.8", "Runtime identity");
Require(Sha(typeof(ComputationalGraph).Assembly.Location) == args[2], "Core identity");
Require(Sha(typeof(ParakeetTranscriber).Assembly.Location) == args[3], "Data identity");
Require(Avx512F.IsSupported == (args[4] == "1") && Avx2.IsSupported && Fma.IsSupported, "Instruction mode");
long budget = long.Parse(args[1], System.Globalization.CultureInfo.InvariantCulture);
Require(budget is 268435456 or 67108864, "Unchanged encoder or decoder budget");
var graph = OnnxImport.Load(args[0], budget) ?? throw new InvalidOperationException(OnnxImport.LastErrorMessage);
var held = new List<(DenseTensor<float> Source, string Hash)>();
var first = Capture(graph, budget, held);
var context = graph.CreateExecution(ExecutionOptions.Memory);
Require(context.MaximumPackedWeightBytes == budget && context.RetainedPackedWeightBytes == graph.RetainedPackedWeightBytes, "Context shares accounted preparation");
context.Reset();
long retained = graph.RetainedPackedWeightBytes;
graph.InvalidatePreparation();
Require(graph.RetainedPackedWeightBytes == 0 && Mapping(graph).Count == 0, "Invalidation releases accounting");
Require(!graph.Initializers.Keys.Any(n => n.StartsWith("packed:", StringComparison.Ordinal)), "Invalidation drops every packed initializer");
foreach (var item in held) Require(TensorSha(item.Source) == item.Hash, "Invalidation preserves source bytes");
graph.RefreshLifetimeAnalysis();
var rebuilt = Capture(graph, budget, new());
Require(JsonSerializer.Serialize(first) == JsonSerializer.Serialize(rebuilt), "Rebuilt mappings and bytes match");
Require(graph.RetainedPackedWeightBytes == retained, "Rebuilt retained-byte accounting");
foreach (var item in held) Require(TensorSha(item.Source) == item.Hash, "Repacking preserves held source bytes");
Require(!process.Modules.Cast<ProcessModule>().Any(m => m.FileName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Native runtime loaded");
using var stream = new FileStream(args[5], FileMode.CreateNew, FileAccess.Write);
JsonSerializer.Serialize(stream, new { passed = true, pid = process.Id, model = args[0], budget,
    retained_bytes = retained, weights = first, rebuilt_weights = rebuilt, invalidation_rebuild_passed = true,
    core_sha256 = args[2], data_sha256 = args[3], runner_sha256 = Sha(Assembly.GetExecutingAssembly().Location),
    runtime = RuntimeInformation.FrameworkDescription, affinity = process.ProcessorAffinity.ToInt64(),
    processor_count = Environment.ProcessorCount, avx512 = Avx512F.IsSupported, avx2 = Avx2.IsSupported, fma = Fma.IsSupported,
    scope = "Prepared graph residency, source ownership and invalidation only; no inference or timing claim" }, new JsonSerializerOptions { WriteIndented = true });
return 0;

static IDictionary Mapping(ComputationalGraph graph) =>
    (IDictionary)typeof(ComputationalGraph).GetField("PackedWeights", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(graph)!;

static List<Weight> Capture(ComputationalGraph graph, long budget, List<(DenseTensor<float> Source, string Hash)> held)
{
    var records = new List<Weight>(); var names = new HashSet<string>(); long bytes = 0;
    foreach (DictionaryEntry entry in Mapping(graph))
    {
        object rec = entry.Value!;
        object Property(string name) => rec.GetType().GetProperty(name)!.GetValue(rec)!;
        string name = (string)Property("SourceName"), packedName = (string)Property("PackedName");
        var original = (DenseTensor<float>)Property("SourceRef"); var packed = (DenseTensor<float>)Property("Packed");
        var sourceKeys = graph.Initializers.Where(p => ReferenceEquals(p.Value, original)).Select(p => p.Key).Order().ToArray();
        Require(names.Add(name) && sourceKeys.Length > 0 && ReferenceEquals(graph.Initializers[packedName], packed), "Mapping ownership");
        Require(original.Dims.SequenceEqual(packed.Dims) && original.Length == packed.Length, "Packed geometry");
        Require(MemoryMarshal.TryGetArray(original.Buffer, out ArraySegment<float> sourceStorage)
            && sourceStorage.Array is not null && ReferenceEquals(sourceStorage.Array, Property("SourceArray"))
            && ReferenceEquals(sourceStorage.Array, entry.Key), "Source array ownership");
        Require(MemoryMarshal.TryGetArray(packed.Buffer, out ArraySegment<float> packedStorage)
            && packedStorage.Array is not null && !ReferenceEquals(sourceStorage.Array, packedStorage.Array), "Independent clone storage");
        bytes = checked(bytes + packed.Length * sizeof(float));
        string sourceHash = TensorSha(original); held.Add((original, sourceHash));
        records.Add(new(name, packedName, sourceKeys, original.Dims, packed.Length * sizeof(float), sourceHash, TensorSha(packed)));
    }
    Require(bytes == graph.RetainedPackedWeightBytes && bytes <= budget && graph.MaximumPackedWeightBytes == budget, "Aggregate accounting");
    Require(graph.Initializers.Keys.Count(n => n.StartsWith("packed:", StringComparison.Ordinal)) == records.Count, "No unaccounted clones");
    return records.OrderBy(r => r.name, StringComparer.Ordinal).ToList();
}

static string TensorSha(DenseTensor<float> tensor) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(tensor.Buffer.Span)));
static string Sha(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
record Weight(string name, string packed_name, string[] source_initializer_keys, int[] shape, long bytes, string source_sha256, string packed_sha256);
