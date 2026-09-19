using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
static string HashFile(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
static string HashTensor(ITensor tensor) => tensor switch
{
    Tensor<float> f => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(f.ToArray().AsSpan()))),
    Tensor<long> l => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(l.ToArray().AsSpan()))),
    _ => throw new InvalidDataException("Unexpected tensor type")
};
static string Field(JsonElement value, string name) => value.GetProperty(name).GetString() ?? throw new InvalidDataException(name);
string root = Path.GetFullPath(args[0]), reference = Path.GetFullPath(args[1]), output = Path.GetFullPath(args[2]);
Require(!File.Exists(output), "Output already exists");
string core = HashFile(typeof(ComputationalGraph).Assembly.Location);
Require(core == "187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4", "Core identity differs");
using var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(reference, "manifest.json")));
var rows = new List<object>(); bool passed = true;
foreach (var model in doc.RootElement.GetProperty("models").EnumerateArray())
{
    foreach (var asset in model.GetProperty("assets").EnumerateArray())
    {
        string path = Path.Combine(root, Field(asset, "file"));
        Require(new FileInfo(path).Length == asset.GetProperty("bytes").GetInt64() && HashFile(path) == Field(asset, "sha256"), "Asset identity differs");
    }
    var graph = OnnxImport.Load(Path.Combine(root, Field(model, "model"))) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage, OnnxImport.LastErrorCause);
    var held = new List<(ITensor Tensor, string Hash)>();
    foreach (var scenario in model.GetProperty("scenarios").EnumerateArray())
    {
        var previous = new Dictionary<string, ITensor>();
        int stepIndex = 0;
        foreach (var step in scenario.GetProperty("steps").EnumerateArray())
        {
            var feed = new Dictionary<string, ITensor>();
            foreach (var item in step.GetProperty("inputs").EnumerateArray())
            {
                string path = Path.Combine(reference, Field(item, "file"));
                Require(HashFile(path) == Field(item, "sha256"), "Input identity differs");
                feed.Add(Field(item, "name"), NpySupport.ReadTensor(path));
            }
            foreach (var carry in step.GetProperty("carry").EnumerateObject())
                feed.Add(carry.Name, previous[carry.Value.GetString() ?? throw new InvalidDataException("Missing state source")]);
            var inputHashes = feed.ToDictionary(p => p.Key, p => HashTensor(p.Value));
            graph.Reset();
            Require(graph.Execute(feed, false), graph.LastErrorMessage ?? "Execution failed");
            previous = graph.Outputs.ToDictionary(p => p.Key, p => p.Value ?? throw new InvalidDataException("Null output"));
            Require(previous.Keys.Order().SequenceEqual(step.GetProperty("outputs").EnumerateArray().Select(o => Field(o, "name")).Order()), "Output names differ");
            foreach (var expected in step.GetProperty("outputs").EnumerateArray())
            {
                string name = Field(expected, "name"), file = Path.Combine(reference, Field(expected, "file"));
                Require(HashFile(file) == Field(expected, "sha256"), "Oracle identity differs");
                var (want, shape) = NpySupport.ReadFloat32(file);
                var tensor = previous[name] as Tensor<float> ?? throw new InvalidDataException("Output dtype differs");
                Require(tensor.Dimensions.SequenceEqual(shape), "Output shape differs");
                var actual = tensor.ToArray(); double maximum = 0; int failed = 0;
                for (int i = 0; i < actual.Length; i++)
                {
                    double error = Math.Abs((double)actual[i] - want[i]) / Math.Max(1, Math.Abs((double)want[i]));
                    maximum = Math.Max(maximum, error);
                    if (!float.IsFinite(actual[i]) || !float.IsFinite(want[i]) || error > 1e-4) failed++;
                }
                passed &= failed == 0;
                string resultPath = output + "." + rows.Count + ".f32";
                File.WriteAllBytes(resultPath, MemoryMarshal.AsBytes(actual.AsSpan()).ToArray());
                rows.Add(new { model = Field(model, "key"), scenario = Field(scenario, "name"), step = stepIndex, name,
                    values = actual.Length, max_scaled_error = maximum, failed_values = failed, file = Path.GetFileName(resultPath), sha256 = HashFile(resultPath) });
                held.Add((tensor, HashTensor(tensor)));
            }
            foreach (var item in feed) Require(HashTensor(item.Value) == inputHashes[item.Key], "Input/state changed");
            foreach (var item in held) Require(HashTensor(item.Tensor) == item.Hash, "Retained output changed");
            stepIndex++;
        }
        graph.Reset();
        Require(!graph.Execute(new Dictionary<string, ITensor>(), false), "Missing input unexpectedly succeeded");
        foreach (var item in held) Require(HashTensor(item.Tensor) == item.Hash, "Failure/reset changed a retained output");
    }
    Console.WriteLine(Field(model, "key") + " completed");
}
Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Native ORT loaded in managed process");
File.WriteAllText(output, JsonSerializer.Serialize(new { passed, core_sha256 = core, reference_sha256 = HashFile(Path.Combine(reference, "manifest.json")),
    runtime = Environment.Version.ToString(), flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k => k.StartsWith("LOKAD_") || k.StartsWith("DOTNET_")).ToDictionary(k => k, Environment.GetEnvironmentVariable), rows }, new JsonSerializerOptions { WriteIndented = true }));
return passed ? 0 : 2;
