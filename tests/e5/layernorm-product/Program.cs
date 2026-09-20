using System.Diagnostics;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
static string HashFile(string path)
{
    using var stream = File.OpenRead(path);
    return Convert.ToHexStringLower(SHA256.HashData(stream));
}
static string HashTensor(ITensor tensor) => tensor switch
{
    Tensor<float> f => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(f.ToArray().AsSpan()))),
    Tensor<long> l => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(l.ToArray().AsSpan()))),
    _ => throw new InvalidDataException("Unexpected tensor type")
};
static string Field(JsonElement value, string name) => value.GetProperty(name).GetString() ?? throw new InvalidDataException(name);

string mode = args[0], root = Path.GetFullPath(args[1]), reference = Path.GetFullPath(args[2]), output = Path.GetFullPath(args[3]);
Require(args.Length == 5 && !Directory.Exists(output), "Arguments or existing output");
string core = HashFile(typeof(ComputationalGraph).Assembly.Location);
Require(core == args[4], "Core identity differs");
Require(Environment.ProcessorCount == 1, "One CPU required before CLR startup");
bool enabled = Environment.GetEnvironmentVariable("LOKAD_ONNX_LAYERNORM_WIDE_OUTPUT") == "1";
var flags = Environment.GetEnvironmentVariables().Keys.Cast<string>()
    .Where(k => k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase))
    .ToDictionary(k => k, Environment.GetEnvironmentVariable);
Require(flags.Count == (mode == "code" ? 3 : 1) && flags.ContainsKey("LOKAD_ONNX_LAYERNORM_WIDE_OUTPUT"), "Unexpected runtime overrides");
var actualSwitch = (bool)typeof(ComputationalGraph).Assembly.GetType("Lokad.Onnx.AblationSwitches")!
    .GetField("EnableLayerNormWideOutput", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!;
Require(actualSwitch == enabled, "Actual product switch differs");
var hardware = new { vector_float = Vector<float>.Count, vector_accelerated = Vector.IsHardwareAccelerated,
    vector512 = Vector512.IsHardwareAccelerated, avx512f = Avx512F.IsSupported };
Directory.CreateDirectory(output);
if (mode == "code")
{
    Require(enabled && hardware.vector512 && hardware.avx512f && hardware.vector_float == 8, "Qualified AMD hardware required");
    CodeProbe.Run(output, core, hardware, flags);
    return 0;
}
var graphs = new List<object>();
var rows = new List<object>();
var held = new List<(ITensor Tensor, string Hash)>();

void CheckGraph(ComputationalGraph graph, string name)
{
    graphs.Add(new { name, nodes = graph.Nodes.Count, layernorm = graph.Nodes.Count(n => n.Op == OpType.LayerNormalization) });
}

void CheckHeld()
{
    foreach (var item in held) Require(HashTensor(item.Tensor) == item.Hash, "Retained output changed");
}

void CheckOutput(ITensor value, float[] want, int[] shape, string model, string scenario, int step, string name, string referenceFile)
{
    var tensor = value as Tensor<float> ?? throw new InvalidDataException("Output dtype differs");
    Require(tensor.Dimensions.SequenceEqual(shape), "Output shape differs");
    var actual = tensor.ToArray(); Require(actual.Length == want.Length, "Output length differs");
    double maximum = 0; int failed = 0;
    for (int i = 0; i < actual.Length; i++)
    {
        double error = Math.Abs((double)actual[i] - want[i]) / Math.Max(1, Math.Abs((double)want[i]));
        maximum = Math.Max(maximum, error);
        if (!float.IsFinite(actual[i]) || !float.IsFinite(want[i]) || error > 1e-4) failed++;
    }
    string file = rows.Count + ".f32";
    using (var stream = new FileStream(Path.Combine(output, file), FileMode.CreateNew))
        stream.Write(MemoryMarshal.AsBytes(actual.AsSpan()));
    rows.Add(new { model, scenario, step, name, shape, values = actual.Length, max_scaled_error = maximum,
        failed_values = failed, file, sha256 = HashFile(Path.Combine(output, file)), reference_file = referenceFile,
        reference_sha256 = HashFile(Path.Combine(reference, referenceFile)) });
    held.Add((tensor, HashTensor(tensor)));
    Require(failed == 0, "Native numerical gate");
}

if (mode == "e5")
{
    foreach (string name in new[] { "e5-8tok", "e5-30tok", "e5-30pad128", "e5-128tok", "e5-512tok" })
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(reference, name + ".json")));
        var fixture = doc.RootElement;
        string model = Path.Combine(root, "models/multilingual-e5-small/model.onnx");
        Require(HashFile(model) == Field(fixture, "model_sha256"), "e5 model identity");
        string file = Field(fixture, "reference_file");
        Require(HashFile(Path.Combine(reference, file)) == Field(fixture, "reference_sha256"), "e5 reference identity");
        var want = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(reference, file))).ToArray();
        var shape = fixture.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var feed = fixture.GetProperty("inputs").EnumerateObject().ToDictionary(p => p.Name, p =>
            (ITensor)new DenseTensor<long>(p.Value.EnumerateArray().Select(v => v.GetInt64()).ToArray(), new[] { 1, shape[1] }));
        var inputHashes = feed.ToDictionary(p => p.Key, p => HashTensor(p.Value));
        foreach (bool memory in new[] { false, true })
        foreach (bool explicitContext in new[] { false, true })
        {
            var options = memory ? ExecutionOptions.Memory : ExecutionOptions.Default;
            var owner = OnnxImport.Load(model) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
            owner.Prepare();
            var graph = explicitContext ? owner.CreateExecution(options) : owner;
            string scenario = (memory ? "memory" : "default") + (explicitContext ? "-context" : "-facade");
            CheckGraph(graph, name + "/" + scenario);
            for (int step = 0; step < 3; step++)
            {
                graph.Reset(); Require(graph.Execute(feed, false, ExecutionProvider.CPU, options), graph.LastErrorMessage ?? "e5 execution failed");
                Require(graph.Outputs.Keys.SequenceEqual(new[] { "last_hidden_state" }), "e5 output names");
                CheckGraph(graph, name + "/" + scenario + "/" + step);
                CheckOutput(graph.Outputs["last_hidden_state"], want, shape, name, scenario, step, "last_hidden_state", file);
                foreach (var item in feed) Require(HashTensor(item.Value) == inputHashes[item.Key], "Input changed");
                CheckHeld();
            }
            graph.Reset(); Require(!graph.Execute(new Dictionary<string, ITensor>(), false, ExecutionProvider.CPU, options), "Missing input succeeded");
            CheckHeld();
        }
        Console.WriteLine(name + " completed");
    }
}
else if (mode == "shared")
{
    using var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(reference, "manifest.json")));
    foreach (var model in doc.RootElement.GetProperty("models").EnumerateArray())
    {
        foreach (var asset in model.GetProperty("assets").EnumerateArray())
        {
            string path = Path.Combine(root, Field(asset, "file"));
            Require(new FileInfo(path).Length == asset.GetProperty("bytes").GetInt64() && HashFile(path) == Field(asset, "sha256"), "Shared asset identity");
        }
        var graph = OnnxImport.Load(Path.Combine(root, Field(model, "model"))) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
        graph.Prepare(); CheckGraph(graph, Field(model, "key"));
        foreach (var scenario in model.GetProperty("scenarios").EnumerateArray())
        {
            var previous = new Dictionary<string, ITensor>(); int stepIndex = 0;
            foreach (var step in scenario.GetProperty("steps").EnumerateArray())
            {
                var feed = new Dictionary<string, ITensor>();
                foreach (var item in step.GetProperty("inputs").EnumerateArray())
                {
                    string path = Path.Combine(reference, Field(item, "file"));
                    Require(HashFile(path) == Field(item, "sha256"), "Shared input identity");
                    feed.Add(Field(item, "name"), NpySupport.ReadTensor(path));
                }
                foreach (var carry in step.GetProperty("carry").EnumerateObject()) feed.Add(carry.Name, previous[carry.Value.GetString()!]);
                var inputHashes = feed.ToDictionary(p => p.Key, p => HashTensor(p.Value));
                graph.Reset(); Require(graph.Execute(feed, false), graph.LastErrorMessage ?? "Shared execution failed");
                previous = graph.Outputs.ToDictionary(p => p.Key, p => p.Value ?? throw new InvalidDataException("Null output"));
                Require(previous.Keys.Order().SequenceEqual(step.GetProperty("outputs").EnumerateArray().Select(o => Field(o, "name")).Order()), "Shared output names");
                foreach (var expected in step.GetProperty("outputs").EnumerateArray())
                {
                    string file = Field(expected, "file");
                    Require(HashFile(Path.Combine(reference, file)) == Field(expected, "sha256"), "Shared reference identity");
                    var (want, shape) = NpySupport.ReadFloat32(Path.Combine(reference, file));
                    CheckOutput(previous[Field(expected, "name")], want, shape, Field(model, "key"), Field(scenario, "name"), stepIndex, Field(expected, "name"), file);
                }
                foreach (var item in feed) Require(HashTensor(item.Value) == inputHashes[item.Key], "Input/state changed");
                CheckHeld(); CheckGraph(graph, Field(model, "key")); stepIndex++;
            }
            graph.Reset(); Require(!graph.Execute(new Dictionary<string, ITensor>(), false), "Missing input succeeded"); CheckHeld();
        }
        Console.WriteLine(Field(model, "key") + " completed");
    }
}
else throw new InvalidDataException("Unknown replay mode");
Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Native ORT loaded");
File.WriteAllText(Path.Combine(output, "result.json"), JsonSerializer.Serialize(new { passed = true, mode, enabled, core_sha256 = core,
    probe_sha256 = HashFile(Assembly.GetExecutingAssembly().Location), runtime = Environment.Version.ToString(), flags, actual_switch = actualSwitch, hardware, graphs, rows,
    inputs_unchanged = true, held_outputs_unchanged = true }, new JsonSerializerOptions { WriteIndented = true }));
return 0;
