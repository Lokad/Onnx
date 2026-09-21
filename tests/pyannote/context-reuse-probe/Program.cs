using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

if (args.Length != 4) throw new ArgumentException("root manifest new-output forward|reverse");
string root = Path.GetFullPath(args[0]), manifest = Path.GetFullPath(args[1]), output = Path.GetFullPath(args[2]);
Require(OperatingSystem.IsWindows() && Environment.ProcessorCount == 1
    && Process.GetCurrentProcess().ProcessorAffinity.ToInt64() == 4, "Windows CPU2 before CLR");
Require(!Environment.GetEnvironmentVariables().Keys.Cast<string>().Any(k => k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)
    || k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)), "Runtime overrides");
Require(!Directory.Exists(output), "Existing output");
Require(Sha(typeof(ComputationalGraph).Assembly.Location) == "469cb2d6a4558d917266434bd1f968c8b9f2762795b963800b2d945083852edd", "Qualified core");
Require(Sha(typeof(Community1Diarizer).Assembly.Location) == "e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb", "Qualified Data");
using var document = JsonDocument.Parse(File.ReadAllBytes(manifest)); var spec = document.RootElement;
string PathOf(JsonElement item)
{
    string path = Path.Combine(root, item.GetProperty("path").GetString()!);
    Require(Sha(path) == item.GetProperty("sha256").GetString() && new FileInfo(path).Length == item.GetProperty("bytes").GetInt64(), "Asset identity");
    return path;
}
var graphs = spec.GetProperty("models").EnumerateObject().ToDictionary(p => p.Name,
    p => OnnxImport.Load(PathOf(p.Value), (p.Name == "segmentation" ? 32L : 64L) * 1024 * 1024)
        ?? throw new InvalidDataException(OnnxImport.LastErrorMessage));
var cases = spec.GetProperty("cases").EnumerateArray().Select(c =>
{
    var input = c.GetProperty("input"); var expected = c.GetProperty("expected");
    byte[] bytes = File.ReadAllBytes(PathOf(input));
    var values = MemoryMarshal.Cast<byte, float>(bytes.AsSpan()).ToArray();
    Require(values.All(float.IsFinite), "Finite input");
    return new Case(c.GetProperty("name").GetString()!, c.GetProperty("graph").GetString()!,
        new DenseTensor<float>(values, input.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray()),
        input.GetProperty("sha256").GetString()!, expected.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray(),
        Sha(PathOf(expected)), c.GetProperty("input_name").GetString()!, c.GetProperty("output_name").GetString()!);
}).ToArray();
Require(cases.Length == 6 && graphs.Count == 2, "Coverage");
string[] modes = args[3] == "forward" ? ["fresh", "reuse", "no-cache"]
    : args[3] == "reverse" ? ["no-cache", "reuse", "fresh"] : throw new ArgumentException("Order");
Directory.CreateDirectory(output);
var records = new List<object>(); var held = new List<(Tensor<float> Tensor, string Hash)>();
void CheckOwnership()
{
    foreach (var item in cases) Require(Hash(Bytes(item.Input)) == item.InputHash, "Input mutated");
    foreach (var item in held) Require(Hash(Bytes(item.Tensor)) == item.Hash, "Held output mutated");
}
foreach (string mode in modes)
{
    var contexts = new Dictionary<string, GraphExecution>();
    for (int pass = 0; pass < 3; pass++) foreach (var c in cases)
    {
        CheckOwnership();
        var feeds = new Dictionary<string, ITensor> { [c.InputName] = c.Input };
        int[] before = Enumerable.Range(0, 3).Select(GC.CollectionCount).ToArray();
        long allocatedBefore = GC.GetTotalAllocatedBytes(true), pauseBefore = GC.GetTotalPauseDuration().Ticks;
        long start = Stopwatch.GetTimestamp();
        bool created = mode == "fresh" || !contexts.TryGetValue(c.Graph, out _);
        GraphExecution execution;
        if (created)
        {
            execution = mode == "no-cache" ? graphs[c.Graph].CreateExecution(ExecutionOptions.Memory, 0)
                : graphs[c.Graph].CreateExecution(ExecutionOptions.Memory);
            if (mode != "fresh") contexts.Add(c.Graph, execution);
        }
        else execution = contexts[c.Graph];
        execution.Reset();
        Require(execution.Execute(feeds, true, ExecutionProvider.CPU, ExecutionOptions.Memory), execution.LastErrorMessage ?? "Graph execution");
        long end = Stopwatch.GetTimestamp(), pauseAfter = GC.GetTotalPauseDuration().Ticks;
        long allocatedAfter = GC.GetTotalAllocatedBytes(true);
        int[] after = Enumerable.Range(0, 3).Select(GC.CollectionCount).ToArray();
        var actual = execution.Outputs[c.OutputName] as Tensor<float> ?? throw new InvalidDataException("Output type");
        Require(actual.Dimensions.SequenceEqual(c.Shape) && Hash(Bytes(actual)) == c.ExpectedHash, "Reference bits/shape");
        held.Add((actual, c.ExpectedHash)); execution.Reset(); CheckOwnership();
        string stem = records.Count.ToString("D3"), arrayFile = stem + ".f32";
        using (var stream = new FileStream(Path.Combine(output, arrayFile), FileMode.CreateNew)) stream.Write(Bytes(actual));
        var row = new { name = c.Name, graph = c.Graph, mode, pass, phase = pass == 0 ? "first" : "repeat", context_created = created,
            start_ticks = start, end_ticks = end, frequency = Stopwatch.Frequency,
            allocated_before = allocatedBefore, allocated_after = allocatedAfter, allocated_bytes = allocatedAfter - allocatedBefore,
            gc_before = before, gc_after = after, pause_before_ticks = pauseBefore, pause_after_ticks = pauseAfter,
            pause_ticks = pauseAfter - pauseBefore, pause_frequency = TimeSpan.TicksPerSecond,
            pool_new_bytes = execution.LastPoolAllocatedNewBytes, pool_reused_bytes = execution.LastPoolReusedBytes,
            pool_peak_outstanding_bytes = execution.LastPoolPeakOutstandingBytes,
            output = new { file = arrayFile, shape = c.Shape, values = actual.Length, sha256 = c.ExpectedHash },
            input_sha256 = c.InputHash, ownership = true };
        records.Add(row); Write(stem + ".json", row);
    }
    foreach (var context in contexts.Values) context.Reset();
    CheckOwnership(); Console.WriteLine(mode + " complete");
}
Require(records.Count == 54 && !Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
    .Any(m => m.FileName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Coverage/native module");
Write("result.json", new { passed = true, records, order = args[3], runtime = RuntimeInformation.FrameworkDescription,
    affinity = Process.GetCurrentProcess().ProcessorAffinity.ToInt64(), processor_count = Environment.ProcessorCount,
    core_sha256 = Sha(typeof(ComputationalGraph).Assembly.Location), data_sha256 = Sha(typeof(Community1Diarizer).Assembly.Location),
    runner_sha256 = Sha(typeof(Case).Assembly.Location), manifest_sha256 = Sha(manifest), inputs_and_held_outputs_unchanged = true,
    scope = "Diagnostic context/reset/graph interval; precise allocation and cumulative pause counters; no public application or ORT timing." });
void Write(string name, object value) => File.WriteAllText(Path.Combine(output, name), JsonSerializer.Serialize(value, new JsonSerializerOptions { WriteIndented = true }));
static ReadOnlySpan<byte> Bytes(Tensor<float> tensor) => MemoryMarshal.AsBytes(tensor is DenseTensor<float> dense ? dense.Buffer.Span : tensor.ToArray().AsSpan());
static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
static string Sha(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
sealed record Case(string Name, string Graph, DenseTensor<float> Input, string InputHash, int[] Shape, string ExpectedHash, string InputName, string OutputName);
