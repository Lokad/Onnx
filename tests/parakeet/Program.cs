using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

if (args.Length != 3) throw new ArgumentException("Usage: ParakeetReplay <model-directory> <fixture-manifest> <new-result.json>");
string models = Path.GetFullPath(args[0]), manifestPath = Path.GetFullPath(args[1]), destination = Path.GetFullPath(args[2]);
string tensorRoot = destination + ".tensors";
string fixtureRoot = Path.GetDirectoryName(manifestPath) ?? throw new InvalidDataException("Fixture directory missing");
if (File.Exists(destination)) throw new IOException("Result exists: " + destination);
if (Directory.Exists(tensorRoot)) throw new IOException("Output tensors exist: " + tensorRoot);
var reports = new List<object>();
var modelReports = new List<object>();
var held = new List<(ITensor Tensor, string Hash)>();
var graphs = new Dictionary<string, ComputationalGraph>();
var fixtures = new Dictionary<string, ITensor>();
bool passed = true;
int comparisons = 0, rejections = 0;
string Sha(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
string SourceSha(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(Encoding.UTF8.GetString(bytes).Replace("\r\n", "\n", StringComparison.Ordinal))));
string Text(JsonElement value) => value.GetString() ?? throw new InvalidDataException("Required string is null");
void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
string Below(string root, string relative)
{
    string path = Path.GetFullPath(Path.Combine(root, relative));
    Require(path.StartsWith(root.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar,
        OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal), "Path escapes its root");
    return path;
}
byte[] Bytes(ITensor tensor) => tensor switch
{
    Tensor<float> value => MemoryMarshal.AsBytes(value.ToArray().AsSpan()).ToArray(),
    Tensor<int> value => MemoryMarshal.AsBytes(value.ToArray().AsSpan()).ToArray(),
    Tensor<long> value => MemoryMarshal.AsBytes(value.ToArray().AsSpan()).ToArray(),
    _ => throw new InvalidDataException("Unexpected tensor dtype")
};
string Bits(ITensor tensor) => Convert.ToHexStringLower(SHA256.HashData(Bytes(tensor)));
void CheckHeld()
{
    foreach (var item in held) Require(Bits(item.Tensor) == item.Hash, "An input or retained output was changed by execution/reset");
}
void NoNativeRuntime()
{
    foreach (ProcessModule module in Process.GetCurrentProcess().Modules)
        Require(!module.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase), "Native ORT loaded in the managed replay");
}
try
{
    NoNativeRuntime();
    using var document = JsonDocument.Parse(File.ReadAllText(manifestPath));
    var manifest = document.RootElement;
    using var resource = Assembly.GetExecutingAssembly().GetManifestResourceStream("parakeet-assets.json")
        ?? throw new InvalidDataException("Missing embedded asset manifest");
    using var memory = new MemoryStream();
    resource.CopyTo(memory);
    byte[] assetBytes = memory.ToArray();
    using var assetDocument = JsonDocument.Parse(assetBytes);
    var assets = assetDocument.RootElement;
    Require(manifest.GetProperty("schema").GetInt32() == 1 && Text(manifest.GetProperty("scope")) == "parakeet-components"
        && manifest.GetProperty("scaled_absolute_tolerance").GetDouble() == 1e-4, "Unknown schema or changed tolerance");
    Require(Text(manifest.GetProperty("onnxruntime")) == "1.29.0" && Text(manifest.GetProperty("numpy")) == "2.2.4"
        && Text(manifest.GetProperty("onnx")) == "1.22.0", "Native oracle versions differ");
    Require(JsonNode.DeepEquals(JsonNode.Parse(assets.GetRawText()), JsonNode.Parse(manifest.GetProperty("assets").GetRawText())), "Fixture asset pins differ");
    Require(Text(manifest.GetProperty("assets_lf_sha256")) == SourceSha(assetBytes), "Asset manifest digest differs");
    using var generator = Assembly.GetExecutingAssembly().GetManifestResourceStream("parakeet-generator.py")
        ?? throw new InvalidDataException("Missing embedded generator");
    using var generatorBytes = new MemoryStream();
    generator.CopyTo(generatorBytes);
    Require(Text(manifest.GetProperty("generator_lf_sha256")) == SourceSha(generatorBytes.ToArray()), "Native generator source differs");
    var settings = manifest.GetProperty("native_settings");
    Require(Text(settings.GetProperty("provider")) == "CPUExecutionProvider" && settings.GetProperty("threads").GetInt32() == 1
        && Text(settings.GetProperty("execution")) == "sequential" && Text(settings.GetProperty("optimization")) == "all"
        && !settings.GetProperty("spinning").GetBoolean(), "Native execution settings differ");
    foreach (var file in assets.GetProperty("files").EnumerateObject())
    {
        string path = Below(models, file.Name);
        Require(new FileInfo(path).Length == file.Value.GetProperty("bytes").GetInt64()
            && Sha(path) == Text(file.Value.GetProperty("sha256")), "Model asset differs: " + file.Name);
    }
    foreach (var file in manifest.GetProperty("files").EnumerateObject())
    {
        string path = Below(fixtureRoot, file.Name);
        Require(new FileInfo(path).Length == file.Value.GetProperty("bytes").GetInt64()
            && Sha(path) == Text(file.Value.GetProperty("sha256")), "Fixture digest differs: " + file.Name);
        var tensor = NpySupport.ReadTensor(path);
        string dtype = tensor.ElementType switch { TensorElementType.Float => "float32", TensorElementType.Int32 => "int32", TensorElementType.Int64 => "int64", _ => "unsupported" };
        Require(dtype == Text(file.Value.GetProperty("dtype"))
            && tensor.Dims.SequenceEqual(file.Value.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32())), "Fixture dtype/shape differs");
        if (tensor is Tensor<float> values) Require(values.ToArray().All(float.IsFinite), "Fixture is nonfinite");
        fixtures.Add(file.Name, tensor);
    }
    string[] scenarioNames = ["encoder-lengths", "decoder-single", "decoder-multiple", "decoder-batch-two"];
    int[] stepCounts = [7, 7, 4, 4];
    var scenarios = manifest.GetProperty("scenarios").EnumerateArray().ToArray();
    Require(scenarios.Select(s => Text(s.GetProperty("name"))).SequenceEqual(scenarioNames), "Scenario coverage differs");
    // Verify state bindings before loading gigabyte weights. Carried states must come
    // from this engine's previous output, never from saved native state arrays.
    for (int s = 0; s < scenarios.Length; s++)
    {
        var steps = scenarios[s].GetProperty("steps").EnumerateArray().ToArray();
        Require(steps.Length == stepCounts[s], "Scenario step coverage differs");
        for (int index = 0; index < steps.Length; index++)
        {
            var step = steps[index];
            string model = Text(step.GetProperty("model"));
            Require(model == (s == 0 ? "encoder" : "decoder"), "Scenario model differs");
            bool failure = index == steps.Length - 2, repeat = index == steps.Length - 1;
            Require(step.TryGetProperty("expected_failure", out var failed) == failure, "Failure coverage differs");
            if (failure) Require(Text(failed) == "invalid-input" && Text(step.GetProperty("native_error")).Contains("INVALID_ARGUMENT"), "Native rejection missing");
            Require(step.TryGetProperty("repeat_of", out var repeated) == repeat && (!repeat || repeated.GetInt32() == 0), "Recovery repeat missing");
            foreach (var input in step.GetProperty("inputs").EnumerateObject())
            {
                bool carried = s != 0 && index > 0 && !failure && !repeat && input.Name.StartsWith("input_states_", StringComparison.Ordinal);
                if (carried)
                {
                    Require(!input.Value.TryGetProperty("file", out _) && input.Value.GetProperty("step").GetInt32() == index - 1
                        && Text(input.Value.GetProperty("output")) == input.Name.Replace("input_", "output_", StringComparison.Ordinal), "State must advance from managed output");
                }
                else Require(input.Value.TryGetProperty("file", out var file) && fixtures.ContainsKey(Text(file)), "Input fixture missing");
            }
        }
    }
    Directory.CreateDirectory(tensorRoot);
    foreach (var item in assets.GetProperty("graphs").EnumerateObject())
    {
        string path = Below(models, Text(item.Value));
        long budget = (item.Name == "encoder" ? 256L : 64L) * 1024 * 1024;
        var watch = Stopwatch.StartNew();
        var graph = OnnxImport.Load(path, budget) ?? throw new InvalidOperationException(OnnxImport.LastErrorMessage, OnnxImport.LastErrorCause);
        watch.Stop();
        var unsupported = graph.Nodes.Where(n => !OperatorSchemas.TryResolve(n, out _, out _)).Select(n => n.DescribeOperator()).Distinct().ToArray();
        Require(unsupported.Length == 0, "Unsupported operators: " + string.Join(", ", unsupported));
        graphs.Add(item.Name, graph);
        modelReports.Add(new { name = item.Name, sha256 = Sha(path), nodes = graph.Nodes.Count, load_seconds = watch.Elapsed.TotalSeconds, maximum_packed_weight_bytes = budget });
        Console.WriteLine($"Loaded {item.Name}: {graph.Nodes.Count} nodes");
    }
    foreach (var scenario in scenarios)
    {
        string name = Text(scenario.GetProperty("name"));
        var history = new List<Dictionary<string, ITensor>>();
        var steps = scenario.GetProperty("steps").EnumerateArray().ToArray();
        var context = graphs[Text(steps[0].GetProperty("model"))].CreateExecution(ExecutionOptions.Memory);
        for (int index = 0; index < steps.Length; index++)
        {
            var step = steps[index];
            var inputs = new Dictionary<string, ITensor>();
            foreach (var input in step.GetProperty("inputs").EnumerateObject())
                inputs.Add(input.Name, input.Value.TryGetProperty("file", out var file) ? fixtures[Text(file)]
                    : history[input.Value.GetProperty("step").GetInt32()][Text(input.Value.GetProperty("output"))]);
            Require(inputs.Keys.Order().SequenceEqual(context.Inputs.Keys.Order()), "Model input names differ");
            foreach (var input in inputs.Values) held.Add((input, Bits(input)));
            context.Reset();
            CheckHeld();
            var watch = Stopwatch.StartNew();
            bool success = context.Execute(inputs, true, ExecutionProvider.CPU, ExecutionOptions.Memory);
            watch.Stop();
            CheckHeld();
            NoNativeRuntime();
            if (step.TryGetProperty("expected_failure", out _))
            {
                Require(!success && !string.IsNullOrWhiteSpace(context.LastErrorMessage), "Invalid input was not rejected");
                rejections++;
                reports.Add(new { scenario = name, step = index, expected_failure = "invalid-input", error = context.LastErrorMessage });
                history.Add(new Dictionary<string, ITensor>());
                continue;
            }
            if (!success) throw new InvalidOperationException(name + " step " + index + ": " + context.LastErrorMessage, context.LastErrorCause);
            Require(context.Outputs.Keys.Order().SequenceEqual(step.GetProperty("outputs").EnumerateObject().Select(p => p.Name).Order()), "Output name set differs");
            var outputs = new Dictionary<string, ITensor>();
            var results = new List<object>();
            foreach (var output in step.GetProperty("outputs").EnumerateObject())
            {
                ITensor actual = context.Outputs[output.Name] ?? throw new InvalidDataException("Null model output");
                ITensor expected = fixtures[Text(output.Value)];
                Require(actual.ElementType == expected.ElementType && actual.Dims.SequenceEqual(expected.Dims), "Output type/shape differs: " + output.Name);
                int bad = 0, worst = -1;
                double maxScaled = 0, maxAbsolute = 0;
                if (actual is Tensor<float> floats && expected is Tensor<float> reference)
                {
                    var got = floats.ToArray(); var want = reference.ToArray();
                    for (int i = 0; i < got.Length; i++)
                    {
                        if (!float.IsFinite(got[i])) { bad++; continue; }
                        double absolute = Math.Abs((double)got[i] - want[i]);
                        double scaled = absolute / Math.Max(1, Math.Abs((double)want[i]));
                        if (scaled > maxScaled) { maxScaled = scaled; worst = i; }
                        maxAbsolute = Math.Max(maxAbsolute, absolute);
                        if (scaled > 1e-4) bad++;
                    }
                }
                else if (Bits(actual) != Bits(expected)) bad = 1;
                if (bad > 0) passed = false;
                if (step.TryGetProperty("repeat_of", out var repeat))
                    Require(Bits(actual) == Bits(history[repeat.GetInt32()][output.Name]), "Managed fresh-request repeat differs");
                outputs.Add(output.Name, actual);
                held.Add((actual, Bits(actual)));
                comparisons++;
                string rawFile = name + "-" + index + "-" + output.Name + ".bin";
                File.WriteAllBytes(Below(tensorRoot, rawFile), Bytes(actual));
                results.Add(new { name = output.Name, dtype = actual.ElementType.ToString(), shape = actual.Dims,
                    max_scaled_error = maxScaled, max_absolute_error = maxAbsolute, worst_index = worst, failed_values = bad,
                    raw_file = rawFile, sha256 = Bits(actual) });
                Console.WriteLine($"{name} {index} {output.Name}: scaled {maxScaled:G6}, failures {bad}");
            }
            history.Add(outputs);
            reports.Add(new { scenario = name, step = index, execute_seconds = watch.Elapsed.TotalSeconds,
                allocated_bytes = context.LastAllocatedBytes, peak_live_bytes = context.LastPeakLiveBytes, outputs = results });
        }
        context.Reset();
        CheckHeld();
    }
    Require(comparisons == 60 && rejections == 4, "Full component coverage incomplete");
}
catch (Exception exception)
{
    passed = false;
    reports.Add(new { error = exception.ToString() });
    Console.Error.WriteLine(exception);
}
NoNativeRuntime();
using var process = Process.GetCurrentProcess();
process.Refresh();
File.WriteAllText(destination, JsonSerializer.Serialize(new { passed, comparisons, rejections,
    manifest_sha256 = Sha(manifestPath), core_sha256 = Sha(typeof(ComputationalGraph).Assembly.Location),
    runner_sha256 = Sha(Assembly.GetExecutingAssembly().Location), core_version = typeof(ComputationalGraph).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion,
    runtime = RuntimeInformation.FrameworkDescription, os = RuntimeInformation.OSDescription, execution_options = "Memory",
    environment = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(name => name.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)
        || name.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || name.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)).Order()
        .ToDictionary(name => name, Environment.GetEnvironmentVariable), peak_working_set_bytes = process.PeakWorkingSet64,
    boundary = "Complete tensor component replay; preprocessing, TDT transcription and benchmark qualification remain separate",
    models = modelReports, steps = reports }, new JsonSerializerOptions { WriteIndented = true }));
return passed ? 0 : 2;
