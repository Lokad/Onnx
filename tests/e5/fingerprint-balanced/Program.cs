using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Lokad.Onnx;

if (args.Length != 6) throw new ArgumentException("model case.json output-directory visit case-index smoke|aa|compare");
string model = Path.GetFullPath(args[0]), caseFile = Path.GetFullPath(args[1]), output = Path.GetFullPath(args[2]), phase = args[5];
int visit = int.Parse(args[3]), caseIndex = int.Parse(args[4]); bool smoke = phase == "smoke";
Require(phase is "smoke" or "aa" or "compare", "Phase");
Require(visit >= 0 && visit < 4 && caseIndex >= 0 && caseIndex < 5, "Schedule position");
Require(!Directory.Exists(output), "Output exists");
Require(Environment.ProcessorCount == 1 && Affinity() == 4, "CPU2 required before CLR startup");
if (!smoke) Require(OperatingSystem.IsLinux() && Avx512F.IsSupported && Environment.Version.ToString() == "10.0.8", "AMD runtime/ISA");
var flags = Environment.GetEnvironmentVariables().Keys.Cast<string>()
    .Where(k => k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase))
    .ToDictionary(k => k, Environment.GetEnvironmentVariable);
Require(flags.Count == 1 && flags.GetValueOrDefault("LOKAD_ONNX_FINGERPRINT_STRINGS") == "1", "Unexpected runtime settings");
string core = HashFile(typeof(ComputationalGraph).Assembly.Location);
Require(core == "48ca1d62ee2586d81072b8e347a671013d00abf8fe637c89eff65314e13cc710", "Qualified core identity");
NoOrt(); Directory.CreateDirectory(output);
using var doc = JsonDocument.Parse(File.ReadAllBytes(caseFile)); var fixture = doc.RootElement;
string[] cases = ["e5-8tok", "e5-30tok", "e5-30pad128", "e5-128tok", "e5-512tok"];
Require(fixture.GetProperty("name").GetString() == cases[caseIndex], "Case identity");
Require(HashFile(model) == fixture.GetProperty("model_sha256").GetString(), "Model identity");
string reference = Path.Combine(Path.GetDirectoryName(caseFile)!, fixture.GetProperty("reference_file").GetString()!);
Require(HashFile(reference) == fixture.GetProperty("reference_sha256").GetString(), "Reference identity");
int[] shape = fixture.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
float[] want = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(reference)).ToArray();
Require(shape.Length == 3 && shape[0] == 1 && shape[2] == 384 && want.Length == shape.Aggregate(1, (a,b) => checked(a*b)) && want.All(float.IsFinite), "Reference geometry");
var inputValues = fixture.GetProperty("inputs").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.EnumerateArray().Select(v => v.GetInt64()).ToArray());
Require(inputValues.Keys.Order().SequenceEqual(new[] { "attention_mask", "input_ids", "token_type_ids" }), "Input names");
Require(inputValues.Values.All(v => v.Length == shape[1]), "Input geometry");
var inputs = inputValues.ToDictionary(p => p.Key, p => (ITensor)new DenseTensor<long>(p.Value.ToArray(), new[] { 1, p.Value.Length }));
var options = ExecutionOptions.Memory;
string inputHash = HashInputs(inputs); Require(inputHash == fixture.GetProperty("input_sha256").GetString(), "Canonical input hash");
long start = Stopwatch.GetTimestamp();
var graph = OnnxImport.Load(model) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
long loadTicks = Stopwatch.GetTimestamp() - start;
var flag = typeof(ComputationalGraph).GetProperty("CacheFingerprintStrings", BindingFlags.Instance | BindingFlags.NonPublic)!;
var setCache = flag.GetSetMethod(true)!.CreateDelegate<Action<ComputationalGraph, bool>>();
var cacheField = typeof(ComputationalGraph).GetField("FingerprintStrings", BindingFlags.Instance | BindingFlags.NonPublic)!;
var fingerprint = typeof(ComputationalGraph).GetMethod("ComputeStructureFingerprint", BindingFlags.Instance | BindingFlags.NonPublic, null, Type.EmptyTypes, null)!.CreateDelegate<Func<long>>(graph);
Require((bool)flag.GetValue(graph)!, "Cache not enabled at construction");
long allocation = GC.GetTotalAllocatedBytes(true); start = Stopwatch.GetTimestamp(); graph.Prepare();
long preparationTicks = Stopwatch.GetTimestamp() - start, preparationBytes = GC.GetTotalAllocatedBytes(true) - allocation;
int cycles = smoke ? 12 : 48, calls = smoke ? 2 : new[] { 32, 16, 4, 4, 2 }[caseIndex];
double seconds = smoke ? .1 : 15;
int[][] permutations = [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]];
int[] order = Enumerable.Range(0, cycles).Select(i => i % 6).ToArray();
uint random = checked((uint)(20260920 + 100 * visit + caseIndex));
for (int block = 0; block < cycles; block += 6)
for (int i = 5; i > 0; i--)
{
    random = unchecked(random * 1664525 + 1013904223); int j = (int)(random % (uint)(i + 1));
    (order[block + i], order[block + j]) = (order[block + j], order[block + i]);
}
var schedule = new { protocol = "single-graph-fingerprint-local6-v2", phase, visit, case_index = caseIndex, cycles, calls,
    conditioning_seconds_per_setting = seconds, order, permutations };
Write(Path.Combine(output, "schedule.json"), schedule);
var settling = new Row[4]; var conditioning = new Row[15000]; int count = 0;
var measured = new Row[cycles * 3 * calls]; int cursor = 0;
Tensor<float> held = null!; byte[] heldBits = null!;
for (int i = 0; i < settling.Length; i++)
{
    setCache(graph, true); settling[i] = Take(2, "settling", i, 0, 0, true);
    if (i == 0) { held = (Tensor<float>)graph.Outputs["last_hidden_state"]; heldBits = Bits(held); }
}
var snapshot = cacheField.GetValue(graph) as Array ?? throw new InvalidDataException("Prepared cache absent");
Require(snapshot.Length > 0, "Empty cache"); string snapshotHash = HashCache(snapshot);
double beforeError = CheckOutput("before"); long preparedFingerprint = CheckFingerprint();
double[] conditioned = [0,0]; int round = 0;
while (conditioned.Any(v => v < seconds))
{
    for (int position = 0; position < 2; position++)
    {
        int setting = (round + position + visit) % 2;
        if (conditioned[setting] >= seconds) continue;
        Require(count < conditioning.Length, "Conditioning capacity");
        setCache(graph, setting != 0);
        var row = Take(setting, "conditioning", round, position, 0, setting != 0);
        conditioning[count++] = row; conditioned[setting] += (double)row.execute / Stopwatch.Frequency;
    }
    round++;
}
Require(ReferenceEquals(cacheField.GetValue(graph), snapshot), "Preparation changed during conditioning");
Console.WriteLine("Conditioned " + cases[caseIndex] + " " + phase);
var before = Resources();
for (int cycle = 0; cycle < cycles; cycle++)
for (int position = 0; position < 3; position++)
{
    int role = permutations[order[cycle]][position]; bool enabled = phase != "aa" && role == 2;
    setCache(graph, enabled);
    Require((bool)flag.GetValue(graph)! == enabled && ReferenceEquals(cacheField.GetValue(graph), snapshot), "Role/cache state");
    for (int call = 0; call < calls; call++) measured[cursor++] = Take(role, "measured", cycle, position, call, enabled);
    Require(ReferenceEquals(cacheField.GetValue(graph), snapshot), "Cache replaced during measurement");
}
var after = Resources(); Require(cursor == measured.Length, "Measured coverage");
Require(HashCache(snapshot) == snapshotHash && ReferenceEquals(cacheField.GetValue(graph), snapshot), "Prepared cache changed");
Require(HashInputs(inputs) == inputHash && Bits(held).AsSpan().SequenceEqual(heldBits), "Input/held-output mutation");
double afterError = CheckOutput("after"); Require(CheckFingerprint() == preparedFingerprint, "Graph fingerprint changed");
NoOrt(); Require(AppDomain.CurrentDomain.GetAssemblies().Count(a => a.GetName().Name == "Lokad.Onnx") == 1, "Multiple product cores");
Write(Path.Combine(output, "result.json"), new { passed = true, schedule, core_sha256 = core,
    probe_sha256 = HashFile(Assembly.GetExecutingAssembly().Location), case_sha256 = HashFile(caseFile), model_sha256 = HashFile(model),
    input_sha256 = inputHash, reference_sha256 = HashFile(reference), shape, frequency = Stopwatch.Frequency,
    runtime = Environment.Version.ToString(), affinity = Affinity(), processor_count = Environment.ProcessorCount, avx2 = Avx2.IsSupported, avx512 = Avx512F.IsSupported,
    flags, load_ticks = loadTicks, preparation_ticks = preparationTicks, preparation_bytes = preparationBytes,
    cache_entries = snapshot.Length, cache_sha256 = snapshotHash, fingerprint = preparedFingerprint,
    unchanged_cache = true, unchanged_inputs = true, unchanged_held_output = true, single_graph = true,
    before_error = beforeError, after_error = afterError, output_sha256 = HashFile(Path.Combine(output, "after.f32")),
    settling, conditioning = conditioning.Take(count).ToArray(), conditioned, measured, before, after });
Console.WriteLine("Completed " + phase + " " + cases[caseIndex] + " visit " + visit);

Row Take(int role, string stage, int cycle, int position, int call, bool enabled)
{
    int g0 = GC.CollectionCount(0), g1 = GC.CollectionCount(1), g2 = GC.CollectionCount(2);
    long allocated = GC.GetTotalAllocatedBytes(true), request = Stopwatch.GetTimestamp();
    graph.Reset(); long execute = Stopwatch.GetTimestamp();
    bool ok = graph.Execute(inputs, true, ExecutionProvider.CPU, options);
    long end = Stopwatch.GetTimestamp(); Require(ok, graph.LastErrorMessage ?? "Execution failed");
    return new Row(role, stage, cycle, position, call, enabled, end - execute, end - request,
        GC.GetTotalAllocatedBytes(true) - allocated, GC.CollectionCount(0) - g0, GC.CollectionCount(1) - g1, GC.CollectionCount(2) - g2);
}
long CheckFingerprint()
{
    bool saved = (bool)flag.GetValue(graph)!;
    try { setCache(graph, false); long a = fingerprint(); setCache(graph, true); long b = fingerprint(); Require(a == b, "Cached/original hash mismatch"); return a; }
    finally { setCache(graph, saved); }
}
double CheckOutput(string name)
{
    Require(graph.Outputs.Keys.SequenceEqual(new[] { "last_hidden_state" }), "Output names");
    var tensor = graph.Outputs["last_hidden_state"] as Tensor<float> ?? throw new InvalidDataException("Output dtype");
    Require(tensor.Dimensions.SequenceEqual(shape), "Output shape"); var actual = tensor.ToArray();
    double maximum = 0;
    for (int i = 0; i < want.Length; i++) { Require(float.IsFinite(actual[i]), "Nonfinite output"); maximum = Math.Max(maximum, Math.Abs((double)actual[i] - want[i]) / Math.Max(1, Math.Abs((double)want[i]))); }
    Require(maximum <= 1e-4, "Native numerical gate");
    byte[] bits = MemoryMarshal.AsBytes(actual.AsSpan()).ToArray(); Require(bits.AsSpan().SequenceEqual(heldBits), "Output bits changed");
    using var file = new FileStream(Path.Combine(output, name + ".f32"), FileMode.CreateNew); file.Write(bits); return maximum;
}
static string HashInputs(Dictionary<string, ITensor> inputs)
{
    using var memory = new MemoryStream();
    using (var writer = new BinaryWriter(memory, Encoding.UTF8, true))
    {
        writer.Write(Encoding.ASCII.GetBytes("LOKAD-CAMPAIGN-INPUTS-1\0")); writer.Write(inputs.Count);
        foreach (string key in inputs.Keys.Order(StringComparer.Ordinal))
        {
            byte[] bytes = Encoding.UTF8.GetBytes(key); writer.Write(bytes.Length); writer.Write(bytes);
            var tensor = (Tensor<long>)inputs[key]; writer.Write(7); writer.Write(tensor.Dimensions.Length);
            foreach (int dimension in tensor.Dimensions) writer.Write(dimension);
            writer.Write((long)tensor.Length); foreach (long value in tensor.ToArray()) writer.Write(value);
        }
    }
    return Convert.ToHexStringLower(SHA256.HashData(memory.ToArray()));
}
static byte[] Bits(Tensor<float> tensor) => MemoryMarshal.AsBytes(tensor.ToArray().AsSpan()).ToArray();
static string HashCache(Array cache) => Convert.ToHexStringLower(SHA256.HashData(JsonSerializer.SerializeToUtf8Bytes(cache, cache.GetType())));
static void Require(bool ok, string message) { if (!ok) throw new InvalidDataException(message); }
static string HashFile(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
static void Write(string path, object value) { using var stream = new FileStream(path, FileMode.CreateNew); JsonSerializer.Serialize(stream, value, new JsonSerializerOptions { WriteIndented = true }); }
static long Affinity()
{
    if (OperatingSystem.IsWindows() || OperatingSystem.IsLinux()) return Process.GetCurrentProcess().ProcessorAffinity.ToInt64();
    throw new PlatformNotSupportedException();
}
static void NoOrt() => Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Native ORT loaded");
static object Resources() { using var process = Process.GetCurrentProcess(); var memory = GC.GetGCMemoryInfo(); return new { rss = process.WorkingSet64, peak = process.PeakWorkingSet64, heap = memory.HeapSizeBytes, committed = memory.TotalCommittedBytes, gc = new[] { GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2) } }; }
readonly record struct Row(int role, string stage, int cycle, int position, int call, bool enabled, long execute, long request, long bytes, int g0, int g1, int g2);
