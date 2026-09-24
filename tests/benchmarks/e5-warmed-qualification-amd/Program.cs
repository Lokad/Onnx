using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

static void Require(bool value, string message) { if (!value) throw new InvalidDataException(message); }
static string HashFile(string path) { using var f = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(f)); }
static string Hash(float[] values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
static string HashInput(ITensor tensor) => tensor switch
{
    Tensor<float> f => Hash(f.ToArray()),
    Tensor<long> l => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(l.ToArray().AsSpan()))),
    _ => throw new InvalidDataException("Unexpected input type")
};
static string Text(JsonElement e, string name) => e.GetProperty(name).GetString()!;
static int[] Shape(JsonElement e) => e.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();

Require(args.Length == 4, "manifest, case, output, mode required");
string manifest = Path.GetFullPath(args[0]), key = args[1], output = Path.GetFullPath(args[2]), mode = args[3];
Require(mode is "verify" or "timing", "Unknown mode");
Require(!Directory.Exists(output) && Environment.ProcessorCount == 1, "Existing output or CPU confinement");
Directory.CreateDirectory(output);
using var document = JsonDocument.Parse(File.ReadAllBytes(manifest));
var spec = document.RootElement;
var item = spec.GetProperty("cases").EnumerateArray().Single(c => Text(c, "key") == key);
string core = HashFile(typeof(ComputationalGraph).Assembly.Location);
Require(core == Text(spec, "core"), "Wrong actual product DLL");
var flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k => k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)).ToDictionary(k => k, Environment.GetEnvironmentVariable);
Require(flags.Count == 0, "Runtime override");
string root = Path.GetDirectoryName(manifest)!;
var feed = new Dictionary<string, ITensor>();
foreach (var input in item.GetProperty("inputs").EnumerateArray())
{
    ITensor tensor = input.TryGetProperty("values", out var values)
        ? new DenseTensor<long>(values.EnumerateArray().Select(x => x.GetInt64()).ToArray(), Shape(input))
        : NpySupport.ReadTensor(Path.Combine(root, Text(input, "file")));
    Require(tensor.Dims.SequenceEqual(Shape(input)), "Input shape");
    feed.Add(Text(input, "name"), tensor);
}
var inputHashes = feed.ToDictionary(p => p.Key, p => HashInput(p.Value));
var expected = item.GetProperty("outputs").EnumerateArray().ToArray();
var references = expected.Select(e => Text(e, "file").EndsWith(".npy") ? NpySupport.ReadFloat32(Path.Combine(root, Text(e, "file"))).Values : MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(root, Text(e, "file")))).ToArray()).ToArray();
var setup = Stopwatch.StartNew();
var graph = OnnxImport.Load(Text(item, "model")) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
graph.Prepare(); setup.Stop();
float[][]? first = null;
string[]? hashes = null;
double[] errors = new double[expected.Length];
var clocks = new List<object>();
int calls = mode == "verify" ? 3 : 1380;
for (int index = 0; index < calls; index++)
{
    long start = Stopwatch.GetTimestamp();
    graph.Reset();
    Require(graph.Execute(feed, false), graph.LastErrorMessage ?? "Graph failed");
    var outputs = expected.Select(e => ((Tensor<float>)graph.Outputs[Text(e, "name")]).ToArray()).ToArray();
    long end = Stopwatch.GetTimestamp();
    Require(graph.Outputs.Keys.Order().SequenceEqual(expected.Select(e => Text(e, "name")).Order()), "Output names");
    for (int j = 0; j < expected.Length; j++)
    {
        Require(graph.Outputs[Text(expected[j], "name")].Dims.SequenceEqual(Shape(expected[j])), "Output shape");
        var actual = outputs[j]; var reference = references[j];
        Require(actual.Length == reference.Length, "Output extent");
        for (int k = 0; k < actual.Length; k++)
        {
            double error = Math.Abs((double)actual[k] - reference[k]) / Math.Max(1, Math.Abs((double)reference[k]));
            Require(float.IsFinite(actual[k]) && float.IsFinite(reference[k]) && error <= 1e-4, "Native reference bound");
            errors[j] = Math.Max(errors[j], error);
        }
        if (hashes != null) Require(Hash(actual) == hashes[j] && Hash(first![j]) == hashes[j], "Nondeterministic or overwritten result");
    }
    foreach (var p in feed) Require(HashInput(p.Value) == inputHashes[p.Key], "Input mutation");
    if (first == null) { first = outputs; hashes = outputs.Select(Hash).ToArray(); }
    clocks.Add(new { index, warmup = mode == "verify" || index < 1200, ticks = end - start, frequency = Stopwatch.Frequency });
}
var arrays = new List<object>();
for (int j = 0; j < expected.Length; j++)
{
    string file = j + ".f32";
    using (var f = File.Create(Path.Combine(output, file))) f.Write(MemoryMarshal.AsBytes(first![j].AsSpan()));
    arrays.Add(new { name = Text(expected[j], "name"), shape = Shape(expected[j]), file, sha256 = hashes![j], values = first![j].Length, max_scaled_error = errors[j] });
}
Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "ORT loaded into managed worker");
File.WriteAllText(Path.Combine(output, "result.json"), JsonSerializer.Serialize(new { passed = true, role = "current", key, mode, pid = Environment.ProcessId, runtime = Environment.Version.ToString(), core, consumer = HashFile(Assembly.GetExecutingAssembly().Location), flags, setup_seconds = setup.Elapsed.TotalSeconds, calls, clocks, arrays, inputs_unchanged = true, held_outputs_unchanged = true }, new JsonSerializerOptions { WriteIndented = true }));
