using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

if (args.Length != 3)
    throw new ArgumentException("Usage: DecoderReplay <model-directory> <fixture-manifest> <result.json>");
string modelRoot = Path.GetFullPath(args[0]);
string manifestPath = Path.GetFullPath(args[1]);
string fixtureRoot = Path.GetDirectoryName(manifestPath)!;
string destination = Path.GetFullPath(args[2]);
if (File.Exists(destination)) throw new IOException("Result already exists: " + destination);
using var document = JsonDocument.Parse(File.ReadAllText(manifestPath));
var manifest = document.RootElement;
using var assetStream = Assembly.GetExecutingAssembly().GetManifestResourceStream("decoder-assets.json")
    ?? throw new InvalidDataException("Missing pinned asset manifest");
using var pinnedAssets = JsonDocument.Parse(assetStream);
if (manifest.GetProperty("schema").GetInt32() != 1 || manifest.GetProperty("scaled_absolute_tolerance").GetDouble() != 1e-4)
    throw new InvalidDataException("Unknown fixture schema or changed tolerance");
if (manifest.GetProperty("revision").GetString() != "360ebcde2559d60bb474678be3c1de9ef347d01a")
    throw new InvalidDataException("Unexpected model revision");
var reports = new List<object>();
bool passed = true;
var graphs = new Dictionary<string, ComputationalGraph>();
var modelReports = new List<object>();
string Sha(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
string Below(string root, string relative)
{
    string path = Path.GetFullPath(Path.Combine(root, relative));
    if (!path.StartsWith(root.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
        throw new InvalidDataException("Fixture path escapes its root: " + relative);
    return path;
}
ITensor Read(string relative)
{
    string path = Below(fixtureRoot, relative);
    var record = manifest.GetProperty("files").GetProperty(relative);
    if (new FileInfo(path).Length != record.GetProperty("bytes").GetInt64()
        || Sha(path) != record.GetProperty("sha256").GetString()) throw new InvalidDataException("Fixture digest differs: " + relative);
    ITensor tensor = NpySupport.ReadTensor(path);
    string dtype = tensor.ElementType switch { TensorElementType.Float => "float32", TensorElementType.Int64 => "int64", _ => "unsupported" };
    if (record.GetProperty("dtype").GetString() != dtype) throw new InvalidDataException("Fixture dtype differs: " + relative);
    if (!tensor.Dims.SequenceEqual(record.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32())))
        throw new InvalidDataException("Fixture dimensions differ: " + relative);
    return tensor;
}
string Bits(ITensor tensor) => tensor switch
{
    Tensor<float> value => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(value.ToArray().AsSpan()))),
    Tensor<long> value => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(value.ToArray().AsSpan()))),
    _ => throw new InvalidDataException("Unexpected fixture dtype " + tensor.ElementType)
};
void NoNativeRuntime()
{
    foreach (ProcessModule module in Process.GetCurrentProcess().Modules)
        if (module.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Native ORT must not load in the managed replay");
}
NoNativeRuntime();
try
{
    if (!manifest.GetProperty("models").EnumerateObject().Select(p => p.Name).Order()
        .SequenceEqual(pinnedAssets.RootElement.EnumerateObject().Select(p => p.Name).Order()))
        throw new InvalidDataException("Model set differs from the pinned split export");
    foreach (var item in manifest.GetProperty("models").EnumerateObject())
    {
        var asset = item.Value;
        var pin = pinnedAssets.RootElement.GetProperty(item.Name);
        if (asset.GetProperty("path").GetString() != pin.GetProperty("path").GetString()
            || asset.GetProperty("sha256").GetString() != pin.GetProperty("sha256").GetString()
            || asset.GetProperty("bytes").GetInt64() != pin.GetProperty("bytes").GetInt64())
            throw new InvalidDataException("Fixture model differs from the embedded pin: " + item.Name);
        string path = Below(modelRoot, asset.GetProperty("path").GetString()!);
        if (new FileInfo(path).Length != asset.GetProperty("bytes").GetInt64() || Sha(path) != asset.GetProperty("sha256").GetString())
            throw new InvalidDataException("Model digest differs: " + item.Name);
        var watch = Stopwatch.StartNew();
        var graph = OnnxImport.Load(path) ?? throw new InvalidOperationException(OnnxImport.LastErrorMessage, OnnxImport.LastErrorCause);
        watch.Stop();
        var unsupported = graph.Nodes.Where(n => !OperatorSchemas.TryResolve(n, out _, out _)).Select(n => n.DescribeOperator()).Distinct().ToArray();
        if (unsupported.Length != 0) throw new NotSupportedException(string.Join(", ", unsupported));
        graphs.Add(item.Name, graph);
        modelReports.Add(new { name = item.Name, sha256 = asset.GetProperty("sha256").GetString(), nodes = graph.Nodes.Count, load_seconds = watch.Elapsed.TotalSeconds });
        Console.WriteLine($"Loaded {item.Name}: {graph.Nodes.Count} nodes in {watch.Elapsed.TotalSeconds:F3}s");
    }
    // Both scenarios share the loaded plans. Each call uses a fresh context;
    // outputs retained from all previous calls must stay readable unchanged.
    var held = new List<(ITensor Value, string Hash)>();
    foreach (var scenario in manifest.GetProperty("scenarios").EnumerateArray())
    {
        string scenarioName = scenario.GetProperty("name").GetString()!;
        var history = new List<Dictionary<string, ITensor>>();
        int index = 0;
        foreach (var step in scenario.GetProperty("steps").EnumerateArray())
        {
            var inputs = new Dictionary<string, ITensor>();
            foreach (var item in step.GetProperty("inputs").EnumerateObject())
            {
                var binding = item.Value;
                if (item.Name.StartsWith("past_key_values.", StringComparison.Ordinal))
                {
                    int expectedStep = item.Name.Contains(".encoder.", StringComparison.Ordinal) ? 0 : index - 1;
                    string expectedName = "present." + item.Name["past_key_values.".Length..];
                    if (index == 0 || binding.TryGetProperty("file", out _) || binding.GetProperty("step").GetInt32() != expectedStep
                        || binding.GetProperty("output").GetString() != expectedName)
                        throw new InvalidDataException("Past input must advance managed cache outputs: " + item.Name);
                }
                else if (!binding.TryGetProperty("file", out _))
                    throw new InvalidDataException("Non-cache input must be a frozen fixture: " + item.Name);
                inputs[item.Name] = binding.TryGetProperty("file", out var file) ? Read(file.GetString()!)
                    : history[binding.GetProperty("step").GetInt32()][binding.GetProperty("output").GetString()!];
            }
            var inputBefore = inputs.ToDictionary(p => p.Key, p => Bits(p.Value));
            var graph = graphs[step.GetProperty("model").GetString()!];
            if (!inputs.Keys.Order().SequenceEqual(graph.Inputs.Keys.Order())) throw new InvalidDataException("Input name set differs");
            graph.Reset();
            var watch = Stopwatch.StartNew();
            bool success = graph.Execute(inputs, true, ExecutionProvider.CPU, ExecutionOptions.Default);
            watch.Stop();
            if (!success) throw new InvalidOperationException($"{scenarioName} step {index}: {graph.LastErrorMessage}", graph.LastErrorCause);
            if (!graph.Outputs.Keys.Order().SequenceEqual(step.GetProperty("outputs").EnumerateObject().Select(p => p.Name).Order()))
                throw new InvalidDataException("Output name set differs");
            var outputs = new Dictionary<string, ITensor>();
            var comparisons = new List<object>();
            foreach (var item in step.GetProperty("outputs").EnumerateObject())
            {
                var actual = graph.Outputs[item.Name] ?? throw new InvalidDataException("Null output");
                var expected = (Tensor<float>)Read(item.Value.GetString()!);
                if (actual is not Tensor<float> got || !actual.Dims.SequenceEqual(expected.Dimensions.ToArray()))
                    throw new InvalidDataException("Output type/dimensions differ: " + item.Name);
                var want = expected.ToArray(); var values = got.ToArray();
                double maxScaled = 0, maxAbsolute = 0;
                int worst = -1, bad = 0;
                for (int i = 0; i < want.Length; i++)
                {
                    double absolute = Math.Abs((double)values[i] - want[i]);
                    double scaled = absolute / Math.Max(1, Math.Abs((double)want[i]));
                    if (!float.IsFinite(values[i]) || !float.IsFinite(want[i])) { bad++; continue; }
                    if (scaled > maxScaled) { maxScaled = scaled; worst = i; }
                    maxAbsolute = Math.Max(maxAbsolute, absolute);
                    if (scaled > 1e-4) bad++;
                }
                if (bad != 0) passed = false;
                comparisons.Add(new { name = item.Name, shape = actual.Dims, max_scaled_error = maxScaled,
                    max_absolute_error = maxAbsolute, worst_index = worst, failed_values = bad });
                outputs.Add(item.Name, actual);
                held.Add((actual, Bits(actual)));
                Console.WriteLine($"{scenarioName} step {index} {item.Name}: max scaled {maxScaled:G6}, failures {bad}");
            }
            foreach (var item in inputs)
                if (Bits(item.Value) != inputBefore[item.Key]) throw new InvalidDataException("Input/cache was mutated: " + item.Key);
            foreach (var item in held)
                if (Bits(item.Value) != item.Hash) throw new InvalidDataException("Earlier output/cache was overwritten");
            NoNativeRuntime();
            history.Add(outputs);
            reports.Add(new { scenario = scenarioName, step = index++, execute_seconds = watch.Elapsed.TotalSeconds,
                allocated_bytes = graph.LastAllocatedBytes, pool_new_bytes = graph.LastPoolAllocatedNewBytes,
                pool_peak_bytes = graph.LastPoolPeakOutstandingBytes, peak_live_bytes = graph.LastPeakLiveBytes, outputs = comparisons });
        }
    }
}
catch (Exception exception)
{
    passed = false;
    reports.Add(new { error = exception.ToString() });
    Console.Error.WriteLine(exception);
}
NoNativeRuntime();
File.WriteAllText(destination, JsonSerializer.Serialize(new { passed, manifest_sha256 = Sha(manifestPath),
    core_sha256 = Sha(typeof(ComputationalGraph).Assembly.Location), runner_sha256 = Sha(Assembly.GetExecutingAssembly().Location),
    runtime = RuntimeInformation.FrameworkDescription, os = RuntimeInformation.OSDescription,
    switches = new[] { "PACKED_AVX512_ROWS", "DEFERRED_RELEASE_CACHE", "RELEASED_BUFFER_CACHE", "FUSED_TEMP_RELEASE", "SOFTMAX_EXP_PRUNE", "BIAS_GELU_INLINE" }
        .ToDictionary(name => "LOKAD_ONNX_" + name, name => Environment.GetEnvironmentVariable("LOKAD_ONNX_" + name)),
    tiered_compilation = Environment.GetEnvironmentVariable("DOTNET_TieredCompilation"),
    boundary = "Diagnostic component execution; no ASR or performance qualification", models = modelReports, steps = reports }, new JsonSerializerOptions { WriteIndented = true }));
return passed ? 0 : 2;
