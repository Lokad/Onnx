using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

if (args.Length != 3) throw new ArgumentException("Usage: FrontendReplay <nemo128.onnx> <manifest.json> <new-result.json>");
string model = Path.GetFullPath(args[0]), manifestPath = Path.GetFullPath(args[1]), destination = Path.GetFullPath(args[2]);
string fixtureRoot = Path.GetDirectoryName(manifestPath) ?? throw new InvalidDataException();
string tensorRoot = destination + ".tensors";
if (File.Exists(destination) || Directory.Exists(tensorRoot)) throw new IOException("Output already exists");
Directory.CreateDirectory(tensorRoot);
string Sha(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
string Text(JsonElement value) => value.GetString() ?? throw new InvalidDataException("Missing string");
void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
string Below(string root, string relative)
{
    string path = Path.GetFullPath(Path.Combine(root, relative));
    Require(path.StartsWith(root.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase), "Fixture path escapes its directory");
    return path;
}
byte[] Bytes(ITensor tensor) => tensor switch {
    Tensor<float> f => MemoryMarshal.AsBytes(f.ToArray().AsSpan()).ToArray(),
    Tensor<long> l => MemoryMarshal.AsBytes(l.ToArray().AsSpan()).ToArray(),
    _ => throw new InvalidDataException("Unexpected tensor type") };
string Bits(ITensor tensor) => Convert.ToHexStringLower(SHA256.HashData(Bytes(tensor)));
void NoNative()
{
    using var process = Process.GetCurrentProcess();
    foreach (ProcessModule module in process.Modules)
        Require(!module.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase), "Native ORT loaded");
}
var rows = new List<object>();
var held = new List<(ITensor Tensor, string Hash)>();
var fixtures = new Dictionary<string, ITensor>();
bool passed = true;
int comparisons = 0, rejections = 0;
void CheckHeld() { foreach (var item in held) Require(Bits(item.Tensor) == item.Hash, "Input or retained output changed"); }
try
{
    using var document = JsonDocument.Parse(File.ReadAllText(manifestPath));
    var manifest = document.RootElement;
    const string ModelSha = "a9fde1486ebfcc08f328d75ad4610c67835fea58c73ba57e3209a6f6cf019e9f";
    Require(Sha(model) == ModelSha && Text(manifest.GetProperty("model_sha256")) == ModelSha, "Pinned frontend differs");
    Require(manifest.GetProperty("schema").GetInt32() == 1 && Text(manifest.GetProperty("scope")) == "parakeet-frontend", "Manifest scope differs");
    Require(Text(manifest.GetProperty("repository")) == "istupakov/parakeet-tdt-0.6b-v3-onnx"
        && Text(manifest.GetProperty("revision")) == "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce", "Model revision differs");
    Require(Text(manifest.GetProperty("numpy")) == "2.2.4" && Text(manifest.GetProperty("onnx")) == "1.22.0"
        && Text(manifest.GetProperty("onnxruntime")) == "1.29.0", "Native versions differ");
    var settings = manifest.GetProperty("native_settings");
    Require(Text(settings.GetProperty("provider")) == "CPUExecutionProvider" && settings.GetProperty("threads").GetInt32() == 1
        && Text(settings.GetProperty("execution")) == "sequential" && Text(settings.GetProperty("optimization")) == "all"
        && !settings.GetProperty("spinning").GetBoolean(), "Native settings differ");
    using var generator = Assembly.GetExecutingAssembly().GetManifestResourceStream("frontend-generator.py") ?? throw new InvalidDataException();
    using var buffer = new MemoryStream(); generator.CopyTo(buffer);
    string source = Encoding.UTF8.GetString(buffer.ToArray()).Replace("\r\n", "\n", StringComparison.Ordinal);
    Require(Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(source))) == Text(manifest.GetProperty("generator_lf_sha256")), "Generator differs");
    foreach (var file in manifest.GetProperty("files").EnumerateObject())
    {
        string path = Below(fixtureRoot, file.Name);
        Require(new FileInfo(path).Length == file.Value.GetProperty("bytes").GetInt64() && Sha(path) == Text(file.Value.GetProperty("sha256")), "Fixture digest differs");
        var tensor = NpySupport.ReadTensor(path);
        Require(tensor.Dims.SequenceEqual(file.Value.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32())), "Fixture shape differs");
        string dtype = tensor is Tensor<float> ? "float32" : tensor is Tensor<long> ? "int64" : "unsupported";
        Require(dtype == Text(file.Value.GetProperty("dtype")), "Fixture type differs");
        if (tensor is Tensor<float> floats) Require(floats.ToArray().All(float.IsFinite), "Nonfinite fixture");
        fixtures.Add(file.Name, tensor);
    }
    string[] coverage = ["noise-257", "noise-511", "noise-512", "noise-16000", "noise-32001", "silence", "tones", "impulses",
        "batch-masked", "english-16k", "french-44k-stereo", "jfk-48k-stereo", "invalid-rank", "recovery"];
    var cases = manifest.GetProperty("cases").EnumerateArray().ToArray();
    Require(cases.Select(c => Text(c.GetProperty("name"))).SequenceEqual(coverage), "Case coverage differs");
    foreach (var item in cases)
    {
        Require(item.GetProperty("inputs").EnumerateObject().Select(p => p.Name).Order().SequenceEqual(new[] { "waveforms", "waveforms_lens" }), "Input names differ");
        foreach (var input in item.GetProperty("inputs").EnumerateObject()) Require(fixtures.ContainsKey(Text(input.Value)), "Missing input fixture");
        bool failure = Text(item.GetProperty("name")) == "invalid-rank";
        Require(item.TryGetProperty("expected_failure", out var failed) == failure, "Failure coverage differs");
        if (failure) Require(Text(failed) == "invalid-input" && Text(item.GetProperty("native_error")).Contains("INVALID_ARGUMENT"), "Native failure missing");
        else
        {
            Require(item.GetProperty("outputs").EnumerateObject().Select(p => p.Name).Order().SequenceEqual(new[] { "features", "features_lens" }), "Output names differ");
            foreach (var output in item.GetProperty("outputs").EnumerateObject()) Require(fixtures.ContainsKey(Text(output.Value)), "Missing output fixture");
        }
    }
    var graph = OnnxImport.Load(model, 0) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage, OnnxImport.LastErrorCause);
    Require(graph.Nodes.All(n => OperatorSchemas.TryResolve(n, out _, out _)), "Unsupported frontend operator");
    var context = graph.CreateExecution(ExecutionOptions.Memory);
    var first = new Dictionary<string, string>();
    foreach (var item in cases)
    {
        string name = Text(item.GetProperty("name"));
        var inputs = item.GetProperty("inputs").EnumerateObject().ToDictionary(p => p.Name, p => fixtures[Text(p.Value)]);
        foreach (var tensor in inputs.Values) held.Add((tensor, Bits(tensor)));
        context.Reset(); CheckHeld();
        var watch = Stopwatch.StartNew();
        bool success = context.Execute(inputs, true, ExecutionProvider.CPU, ExecutionOptions.Memory);
        watch.Stop(); CheckHeld(); NoNative();
        if (name == "invalid-rank")
        {
            Require(!success && !string.IsNullOrWhiteSpace(context.LastErrorMessage), "Invalid input accepted");
            rejections++; rows.Add(new { name, expected_failure = true, error = context.LastErrorMessage }); continue;
        }
        Require(success, name + ": " + context.LastErrorMessage);
        var outputs = new List<object>();
        foreach (var binding in item.GetProperty("outputs").EnumerateObject())
        {
            ITensor actual = context.Outputs[binding.Name], reference = fixtures[Text(binding.Value)];
            Require(actual.ElementType == reference.ElementType && actual.Dims.SequenceEqual(reference.Dims), "Output type/shape differs");
            double maximum = 0; int bad = 0;
            if (actual is Tensor<float> values && reference is Tensor<float> expected)
            {
                float[] got = values.ToArray(), want = expected.ToArray();
                for (int i = 0; i < got.Length; i++)
                {
                    double error = Math.Abs((double)got[i] - want[i]) / Math.Max(1, Math.Abs((double)want[i]));
                    if (!float.IsFinite(got[i]) || error > 1e-4) bad++;
                    maximum = Math.Max(maximum, error);
                }
            }
            else if (Bits(actual) != Bits(reference)) bad++;
            string bits = Bits(actual);
            if (name == "noise-257") first[binding.Name] = bits;
            if (name == "recovery") Require(first[binding.Name] == bits, "Recovery output differs");
            held.Add((actual, bits));
            string raw = name + "-" + binding.Name + ".bin";
            File.WriteAllBytes(Below(tensorRoot, raw), Bytes(actual));
            outputs.Add(new { name = binding.Name, shape = actual.Dims, dtype = actual.ElementType.ToString(), max_scaled_error = maximum, failed_values = bad, raw_file = raw, sha256 = bits });
            comparisons++; if (bad != 0) passed = false;
            Console.WriteLine($"{name} {binding.Name}: scaled {maximum:G6}, failures {bad}");
        }
        rows.Add(new { name, execute_seconds = watch.Elapsed.TotalSeconds, outputs });
    }
    context.Reset(); CheckHeld();
    Require(comparisons == 26 && rejections == 1, "Incomplete coverage");
}
catch (Exception exception) { passed = false; rows.Add(new { error = exception.ToString() }); Console.Error.WriteLine(exception); }
NoNative();
using var current = Process.GetCurrentProcess(); current.Refresh();
File.WriteAllText(destination, JsonSerializer.Serialize(new { passed, comparisons, rejections, steps = rows,
    model_sha256 = Sha(model), manifest_sha256 = Sha(manifestPath), core_sha256 = Sha(typeof(ComputationalGraph).Assembly.Location),
    runner_sha256 = Sha(Assembly.GetExecutingAssembly().Location), runtime = RuntimeInformation.FrameworkDescription,
    execution_options = "Memory", peak_working_set_bytes = current.PeakWorkingSet64,
    environment = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(n => n.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)
        || n.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || n.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)).Order().ToDictionary(n => n, Environment.GetEnvironmentVariable),
    boundary = "Complete frontend features and lengths; TDT decoding and full ASR remain separate" }, new JsonSerializerOptions { WriteIndented = true }));
return passed ? 0 : 2;
