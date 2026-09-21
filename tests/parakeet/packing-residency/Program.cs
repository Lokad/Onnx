using System.Collections;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

if (args.Length != 5) throw new ArgumentException("encoder.onnx budget-bytes core-sha data-sha new-result.json");
using var process = Process.GetCurrentProcess();
Require(process.ProcessorAffinity.ToInt64() == 4 && Environment.ProcessorCount == 1, "CPU2 before CLR startup");
Require(Sha(typeof(ComputationalGraph).Assembly.Location) == args[2], "Core identity");
Require(Sha(typeof(ParakeetTranscriber).Assembly.Location) == args[3], "Data identity");
long budget = long.Parse(args[1], System.Globalization.CultureInfo.InvariantCulture);
Require(budget is 268435456 or 536870912 or 2130706432, "Declared bounded encoder budget");
var graph = OnnxImport.Load(args[0], budget) ?? throw new InvalidOperationException(OnnxImport.LastErrorMessage);
var context = graph.CreateExecution(ExecutionOptions.Memory);
var map = (IDictionary)typeof(ComputationalGraph).GetField("PackedWeights", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(graph)!;
var records = new List<object>(); var names = new HashSet<string>(); long bytes = 0;
foreach (DictionaryEntry entry in map)
{
    object rec = entry.Value!;
    object Property(string name) => rec.GetType().GetProperty(name)!.GetValue(rec)!;
    string name = (string)Property("SourceName"), packedName = (string)Property("PackedName");
    var original = (ITensor)Property("SourceRef"); var packed = (ITensor)Property("Packed");
    Require(names.Add(name) && ReferenceEquals(graph.Initializers[name], original)
        && ReferenceEquals(graph.Initializers[packedName], packed), "Mapping ownership");
    Require(original.Dims.SequenceEqual(packed.Dims) && original.Length == packed.Length, "Packed geometry");
    bytes = checked(bytes + packed.Length * 4);
    records.Add(new { name, packed_name = packedName, shape = original.Dims, bytes = packed.Length * 4 });
}
Require(bytes == graph.RetainedPackedWeightBytes && bytes <= budget && graph.MaximumPackedWeightBytes == budget, "Aggregate accounting");
Require(graph.Initializers.Keys.Count(n => n.StartsWith("packed:", StringComparison.Ordinal)) == records.Count, "No unaccounted clones");
context.Reset();
Require(!process.Modules.Cast<ProcessModule>().Any(m => m.FileName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Native runtime loaded");
using var stream = new FileStream(args[4], FileMode.CreateNew, FileAccess.Write);
JsonSerializer.Serialize(stream, new { passed = true, budget, retained_bytes = bytes, weights = records,
    core_sha256 = args[2], data_sha256 = args[3], runner_sha256 = Sha(Assembly.GetExecutingAssembly().Location),
    runtime = RuntimeInformation.FrameworkDescription, affinity = process.ProcessorAffinity.ToInt64(), processor_count = Environment.ProcessorCount,
    scope = "Prepared encoder residency only; no inference, peak-application-memory or timing claim" }, new JsonSerializerOptions { WriteIndented = true });
return 0;

static string Sha(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
