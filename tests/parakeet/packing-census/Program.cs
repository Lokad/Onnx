using System.Collections;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

if (args.Length != 2) throw new ArgumentException("model-directory new-output-json");
using var process = Process.GetCurrentProcess();
Require(process.ProcessorAffinity.ToInt64() == 4 && Environment.ProcessorCount == 1, "CPU2 before CLR startup");
Require(Sha(typeof(ComputationalGraph).Assembly.Location) == "d1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4", "Core identity");
Require(Sha(typeof(ParakeetTranscriber).Assembly.Location) == "e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb", "Data identity");
var model = new ParakeetTranscriber(args[0]);
var graphs = new Dictionary<string, object>();
foreach (string name in new[] { "frontend", "encoder", "decoder" })
{
    var graph = (ComputationalGraph)typeof(ParakeetTranscriber).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
    var context = graph.CreateExecution(ExecutionOptions.Memory);
    var packed = (IDictionary)typeof(ComputationalGraph).GetField("PackedWeights", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(graph)!;
    var entries = new List<object>(); var sources = new HashSet<string>(); long bytes = 0;
    foreach (DictionaryEntry pair in packed)
    {
        object value = pair.Value!;
        object Property(string p) => value.GetType().GetProperty(p)!.GetValue(value)!;
        string source = (string)Property("SourceName"), target = (string)Property("PackedName");
        var original = (ITensor)Property("SourceRef"); var clone = (ITensor)Property("Packed");
        Require(sources.Add(source) && ReferenceEquals(graph.Initializers[source], original)
            && ReferenceEquals(graph.Initializers[target], clone), "Mapping ownership");
        Require(original.Dims.SequenceEqual(clone.Dims) && original.Length == clone.Length, "Packed geometry");
        long size = checked(clone.Length * 4); bytes += size;
        entries.Add(new { source, target, shape = original.Dims.ToArray(), bytes = size });
    }
    Require(bytes == graph.RetainedPackedWeightBytes && bytes <= graph.MaximumPackedWeightBytes, "Packed residency");
    var nodes = graph.Nodes.Select(n => new { id = n.ID, name = n.Name, op = n.Op.ToString(),
        inputs = n.Inputs, outputs = n.Outputs,
        initializers = n.Inputs.Where(i => graph.Initializers.ContainsKey(i)).Distinct().ToDictionary(i => i,
            i => new { shape = graph.Initializers[i].Dims.ToArray(), dtype = graph.Initializers[i].ElementType.ToString() }) }).ToArray();
    graphs.Add(name, new { maximum_packed_bytes = graph.MaximumPackedWeightBytes, retained_packed_bytes = bytes,
        packed_weights = entries, nodes });
    context.Reset();
}
Require(!process.Modules.Cast<ProcessModule>().Any(m => m.FileName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Native runtime loaded");
var result = new { passed = true, graphs, runtime = RuntimeInformation.FrameworkDescription,
    affinity = process.ProcessorAffinity.ToInt64(), processor_count = Environment.ProcessorCount,
    core_sha256 = Sha(typeof(ComputationalGraph).Assembly.Location), data_sha256 = Sha(typeof(ParakeetTranscriber).Assembly.Location),
    runner_sha256 = Sha(Assembly.GetExecutingAssembly().Location),
    scope = "Prepared graph mapping census only; no inference, kernel dispatch trace or native timing" };
using var stream = new FileStream(args[1], FileMode.CreateNew, FileAccess.Write);
JsonSerializer.Serialize(stream, result, new JsonSerializerOptions { WriteIndented = true });
return 0;

static string Sha(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
