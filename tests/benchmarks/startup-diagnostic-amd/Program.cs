using System.Diagnostics;
using System.Diagnostics.Tracing;
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
Require(mode == "diagnostic" && key == "gpt2", "Diagnostic GPT-2 only");
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
var process = Process.GetCurrentProcess();
var events = MatrixEvents.Log;
int workerThread = NativeThread.gettid();
File.WriteAllText(Path.Combine(Path.GetDirectoryName(output)!,"ready.json"),JsonSerializer.Serialize(new { pid = Environment.ProcessId, native_thread = workerThread, counter = Stopwatch.GetTimestamp() }));
var waiting = Stopwatch.StartNew();
while (!events.IsEnabled(EventLevel.Informational,(EventKeywords)1))
{
    Require(waiting.Elapsed.TotalSeconds < 30,"collector did not enable markers");
    Thread.Sleep(10);
}
File.WriteAllText(Path.Combine(Path.GetDirectoryName(output)!,"collector-enabled.json"),JsonSerializer.Serialize(new { pid = Environment.ProcessId, counter = Stopwatch.GetTimestamp() }));
var setup = Stopwatch.StartNew();
var graph = OnnxImport.Load(Text(item, "model")) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
graph.Prepare(); setup.Stop();
float[][]? first = null;
string[]? hashes = null;
double[] errors = new double[expected.Length];
var clocks = new List<object>();
int calls = 1200;
for (int index = 0; index < calls; index++)
{
    int gc0 = GC.CollectionCount(0), gc1 = GC.CollectionCount(1), gc2 = GC.CollectionCount(2);
    long allocated = GC.GetTotalAllocatedBytes(false);
    long cpuBefore = process.TotalProcessorTime.Ticks;
    long marker = Stopwatch.GetTimestamp();
    events.Begin(0,index,marker);
    long start = Stopwatch.GetTimestamp();
    graph.Reset();
    Require(graph.Execute(feed, false), graph.LastErrorMessage ?? "Graph failed");
    var outputs = expected.Select(e => ((Tensor<float>)graph.Outputs[Text(e, "name")]).ToArray()).ToArray();
    long end = Stopwatch.GetTimestamp();
    events.End(0,index,end);
    long cpuAfter = process.TotalProcessorTime.Ticks;
    long allocatedAfter = GC.GetTotalAllocatedBytes(false);
    int after0 = GC.CollectionCount(0), after1 = GC.CollectionCount(1), after2 = GC.CollectionCount(2);
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
    clocks.Add(new { index, warmup = index < 60, marker, start, end, ticks = end - start, frequency = Stopwatch.Frequency, cpuBefore, cpuAfter, allocated, allocatedAfter, gc0, gc1, gc2, after0, after1, after2 });
}
var arrays = new List<object>();
for (int j = 0; j < expected.Length; j++)
{
    string file = j + ".f32";
    using (var f = File.Create(Path.Combine(output, file))) f.Write(MemoryMarshal.AsBytes(first![j].AsSpan()));
    arrays.Add(new { name = Text(expected[j], "name"), shape = Shape(expected[j]), file, sha256 = hashes![j], values = first![j].Length, max_scaled_error = errors[j] });
}
Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "ORT loaded into managed worker");
File.WriteAllText(Path.Combine(output, "result.json"), JsonSerializer.Serialize(new { passed = true, diagnosticOnly = true, nativeThread = workerThread, role = "current", key, mode, pid = Environment.ProcessId, runtime = Environment.Version.ToString(), core, consumer = HashFile(Assembly.GetExecutingAssembly().Location), flags, setup_seconds = setup.Elapsed.TotalSeconds, calls, clocks, arrays, inputs_unchanged = true, held_outputs_unchanged = true }, new JsonSerializerOptions { WriteIndented = true }));

static class NativeThread
{
    [DllImport("libc")]
    public static extern int gettid();
}

[EventSource(Name = "Lokad-Parakeet-MatMul-Diagnostic")]
sealed class MatrixEvents : EventSource
{
    public static readonly MatrixEvents Log = new();
    public static class Keywords { public const EventKeywords Calls = (EventKeywords)1; }
    [Event(1, Level = EventLevel.Informational, Keywords = Keywords.Calls)]
    public void Begin(int fixture, int iteration, long counter) => Emit(1,fixture,iteration,counter);
    [Event(2, Level = EventLevel.Informational, Keywords = Keywords.Calls)]
    public void End(int fixture, int iteration, long counter) => Emit(2,fixture,iteration,counter);
    [NonEvent]
    unsafe void Emit(int id, int fixture, int iteration, long counter)
    {
        if (!IsEnabled(EventLevel.Informational,(EventKeywords)1)) return;
        EventData* data = stackalloc EventData[3];
        data[0] = new EventData { DataPointer = (IntPtr)(&fixture), Size = sizeof(int) };
        data[1] = new EventData { DataPointer = (IntPtr)(&iteration), Size = sizeof(int) };
        data[2] = new EventData { DataPointer = (IntPtr)(&counter), Size = sizeof(long) };
        WriteEventCore(id,3,data);
    }
}
