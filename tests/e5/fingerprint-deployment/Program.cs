using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Lokad.Onnx;
using Microsoft.ML.OnnxRuntime;

if (args.Length != 10) throw new ArgumentException("model fixture output policy role phase visit case-index native-library smoke|full");
string model = Path.GetFullPath(args[0]), fixturePath = Path.GetFullPath(args[1]), output = Path.GetFullPath(args[2]);
string policy = args[3], role = args[4], phase = args[5], nativePath = Path.GetFullPath(args[8]);
int visit = int.Parse(args[6]), caseIndex = int.Parse(args[7]); bool smoke = args[9] == "smoke";
Require(args[9] is "smoke" or "full", "Measurement scope");
Require(policy is "default" or "memory" && role is "A" or "B" or "C" or "N" && phase is "aa" or "compare", "Configuration");
Require(visit >= 0 && visit < 4 && caseIndex >= 0 && caseIndex < 5, "Schedule identity");
Require(!Directory.Exists(output), "Output exists");
Require(Environment.ProcessorCount == 1 && Affinity() == 4, "CPU2 inherited before CLR startup");
if (!smoke) Require(OperatingSystem.IsLinux() && Environment.Version.ToString() == "10.0.8" && Avx512F.IsSupported, "AMD runtime/ISA");
bool native = role == "N", enabled = phase == "compare" && role == "C";
var flags = Environment.GetEnvironmentVariables().Keys.Cast<string>()
    .Where(k => k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase))
    .ToDictionary(k => k, Environment.GetEnvironmentVariable);
Require(enabled ? flags.Count == 1 && flags.GetValueOrDefault("LOKAD_ONNX_FINGERPRINT_STRINGS") == "1" : flags.Count == 0, "Runtime settings");
string coreHash = HashFile(typeof(ComputationalGraph).Assembly.Location);
Require(coreHash == "48ca1d62ee2586d81072b8e347a671013d00abf8fe637c89eff65314e13cc710", "Qualified core identity");
NoOrt(); Directory.CreateDirectory(output);
using var document = JsonDocument.Parse(File.ReadAllBytes(fixturePath)); var fixture = document.RootElement;
string[] cases = ["e5-8tok", "e5-30tok", "e5-30pad128", "e5-128tok", "e5-512tok"];
Require(fixture.GetProperty("name").GetString() == cases[caseIndex], "Case identity");
Require(HashFile(model) == fixture.GetProperty("model_sha256").GetString(), "Model identity");
string referencePath = Path.Combine(Path.GetDirectoryName(fixturePath)!, fixture.GetProperty("reference_file").GetString()!);
Require(HashFile(referencePath) == fixture.GetProperty("reference_sha256").GetString(), "Native reference identity");
int[] shape = fixture.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
float[] reference = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(referencePath)).ToArray();
Require(shape.Length == 3 && shape[0] == 1 && shape[2] == 384 && reference.Length == shape.Aggregate(1, (a,b) => checked(a*b)) && reference.All(float.IsFinite), "Reference geometry");
var inputValues = fixture.GetProperty("inputs").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.EnumerateArray().Select(v => v.GetInt64()).ToArray());
Require(inputValues.Keys.Order().SequenceEqual(new[] { "attention_mask", "input_ids", "token_type_ids" }) && inputValues.Values.All(v => v.Length == shape[1]), "Input geometry");
var inputs = inputValues.ToDictionary(p => p.Key, p => (ITensor)new DenseTensor<long>(p.Value.ToArray(), new[] { 1, p.Value.Length }));
string inputHash = HashInputs(inputs); Require(inputHash == fixture.GetProperty("input_sha256").GetString(), "Canonical input identity");
var options = policy == "memory" ? ExecutionOptions.Memory : ExecutionOptions.Default;
Require(options.Tensor.MaxDegreeOfParallelism == 1, "Public execution parallelism");
var cacheField = typeof(ComputationalGraph).GetField("FingerprintStrings", BindingFlags.Instance | BindingFlags.NonPublic)!;
var cacheFlag = typeof(ComputationalGraph).GetProperty("CacheFingerprintStrings", BindingFlags.Instance | BindingFlags.NonPublic)!;
ComputationalGraph? graph = null; InferenceSession? session = null; SessionOptions? sessionOptions = null; RunOptions? runOptions = null;
var nativeInputs = new Dictionary<string, OrtValue>(StringComparer.Ordinal);
string[] outputNames = ["last_hidden_state"];
IDisposableReadOnlyCollection<OrtValue>? nativeResult = null, heldNative = null;
long loadStart = Stopwatch.GetTimestamp();
if (native)
{
    Require(File.Exists(nativePath), "Native library absent");
    if (!smoke) Require(HashFile(nativePath) == "13ab8084954fa4a47c777880180b90810d6020f021441395712b48a75b74c68b", "Pinned native library");
    NativeLibrary.SetDllImportResolver(typeof(OrtEnv).Assembly, (name, assembly, search) => name == "onnxruntime" ? NativeLibrary.Load(nativePath) : IntPtr.Zero);
    sessionOptions = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL, IntraOpNumThreads = 1, InterOpNumThreads = 1, ExecutionMode = ExecutionMode.ORT_SEQUENTIAL };
    sessionOptions.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
    sessionOptions.AddSessionConfigEntry("session.inter_op.allow_spinning", "0");
    session = new InferenceSession(model, sessionOptions); runOptions = new RunOptions();
    foreach (var p in inputValues) nativeInputs.Add(p.Key, OrtValue.CreateTensorValueFromMemory(p.Value.ToArray(), new long[] { 1, p.Value.Length }));
    Require(session.OutputMetadata.Keys.SequenceEqual(new[] { "last_hidden_state" }), "Native output names");
}
else
{
    graph = OnnxImport.Load(model) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
    Require((bool)cacheFlag.GetValue(graph)! == enabled, "Graph construction flag");
    if (!enabled) Require(cacheField.GetValue(graph) is null, "Disabled graph constructed cache");
}
long loadTicks = Stopwatch.GetTimestamp() - loadStart;
object? nativeIdentity = NativeIdentity();
int blocks = smoke ? 2 : 48, calls = smoke ? 2 : new[] { 32,16,4,4,2 }[caseIndex];
var conditioning = new Row[20000]; var measured = new Row[blocks * calls];
Row first = Take(-1, 0); Tensor<float>? held = graph?.Outputs["last_hidden_state"] as Tensor<float>;
if (native) { heldNative = nativeResult; nativeResult = null; }
byte[] heldBits = native ? MemoryMarshal.AsBytes(heldNative!.Single().GetTensorDataAsSpan<float>()).ToArray() : Bits(held!);
Array? snapshot = graph is null ? null : cacheField.GetValue(graph) as Array;
Require(enabled ? snapshot?.Length == 2330 : snapshot is null, "Cache presence after first request");
string? cacheHash = snapshot is null ? null : HashCache(snapshot);
Func<long>? fingerprint = graph is null ? null : typeof(ComputationalGraph).GetMethod("ComputeStructureFingerprint", BindingFlags.Instance | BindingFlags.NonPublic, null, Type.EmptyTypes, null)!.CreateDelegate<Func<long>>(graph);
long? beforeFingerprint = fingerprint?.Invoke();
double beforeError = Validate("before", true); var afterFirst = Resources();
double target = smoke ? .1 : 30, conditioned = 0; int count = 0;
long conditioningStart = Stopwatch.GetTimestamp();
while (conditioned < target)
{
    Require(count < conditioning.Length && (double)(Stopwatch.GetTimestamp() - conditioningStart) / Stopwatch.Frequency < 90, "Conditioning cap");
    Row row = Take(-2, count); conditioning[count++] = row; conditioned += (double)row.execute / Stopwatch.Frequency;
}
long conditioningWall = Stopwatch.GetTimestamp() - conditioningStart;
CheckCache(); var before = Resources();
for (int block = 0; block < blocks; block++)
for (int call = 0; call < calls; call++) measured[block * calls + call] = Take(block, call);
var after = Resources(); double afterError = Validate("after", false); CheckCache();
Require(fingerprint?.Invoke() == beforeFingerprint, "Graph fingerprint changed");
Require(HashInputs(inputs) == inputHash, "Managed input bytes changed");
foreach (var p in nativeInputs) Require(p.Value.GetTensorDataAsSpan<long>().SequenceEqual(inputValues[p.Key]), "Actual native input bytes changed");
byte[] currentHeld = native ? MemoryMarshal.AsBytes(heldNative!.Single().GetTensorDataAsSpan<float>()).ToArray() : Bits(held!);
Require(currentHeld.AsSpan().SequenceEqual(heldBits), "Held output changed");
Require(JsonSerializer.Serialize(NativeIdentity()) == JsonSerializer.Serialize(nativeIdentity), "Native identity changed");
Require(AppDomain.CurrentDomain.GetAssemblies().Count(a => a.GetName().Name == "Lokad.Onnx") == 1, "Multiple product cores");
var specification = new { protocol = "fingerprint-isolated-deployment-v1", phase, policy, role, visit, case_index = caseIndex, blocks, calls, conditioning_seconds = target, smoke };
Write(Path.Combine(output, "result.json"), new { passed = true, specification, core_sha256 = coreHash,
    probe_sha256 = HashFile(Assembly.GetExecutingAssembly().Location), ort_managed_sha256 = HashFile(typeof(OrtEnv).Assembly.Location),
    fixture_sha256 = HashFile(fixturePath), model_sha256 = HashFile(model), input_sha256 = inputHash, reference_sha256 = HashFile(referencePath), shape,
    native_identity = nativeIdentity, runtime = Environment.Version.ToString(), affinity = Affinity(), processor_count = Environment.ProcessorCount,
    avx2 = Avx2.IsSupported, avx512 = Avx512F.IsSupported, flags, enabled, optimization = options.Optimization.ToString(),
    frequency = Stopwatch.Frequency, load_ticks = loadTicks, first, after_first = afterFirst, conditioning = conditioning.Take(count).ToArray(), conditioned, conditioning_wall_ticks = conditioningWall,
    measured, before, after, before_error = beforeError, after_error = afterError, output_sha256 = HashFile(Path.Combine(output, "after.f32")),
    fingerprint = beforeFingerprint, cache_entries = snapshot?.Length ?? 0, cache_sha256 = cacheHash,
    unchanged_cache = true, unchanged_inputs = true, unchanged_held_output = true });
Reset(); heldNative?.Dispose(); foreach (var item in nativeInputs.Values) item.Dispose(); runOptions?.Dispose(); session?.Dispose(); sessionOptions?.Dispose();
Console.WriteLine($"Completed {phase} {policy} {cases[caseIndex]} {role} visit {visit}, {measured.Length} measured calls");

void Reset() { graph?.Reset(); nativeResult?.Dispose(); nativeResult = null; }
Row Take(int block, int call)
{
    int g0 = GC.CollectionCount(0), g1 = GC.CollectionCount(1), g2 = GC.CollectionCount(2); long allocated = GC.GetTotalAllocatedBytes(true);
    long request = Stopwatch.GetTimestamp(); Reset(); long execute = Stopwatch.GetTimestamp(); bool ok = true;
    if (graph is not null) ok = graph.Execute(inputs, true, ExecutionProvider.CPU, options);
    else nativeResult = session!.Run(runOptions!, nativeInputs, outputNames);
    long end = Stopwatch.GetTimestamp(); Require(ok, graph?.LastErrorMessage ?? "Execution failed");
    return new Row(block, call, end - execute, end - request, GC.GetTotalAllocatedBytes(true) - allocated,
        GC.CollectionCount(0) - g0, GC.CollectionCount(1) - g1, GC.CollectionCount(2) - g2);
}
void CheckCache()
{
    if (graph is null) return;
    Require((bool)cacheFlag.GetValue(graph)! == enabled && ReferenceEquals(cacheField.GetValue(graph), snapshot), "Cache setting/identity changed");
    if (snapshot is not null) Require(HashCache(snapshot) == cacheHash, "Cache contents changed");
}
double Validate(string stage, bool firstOutput)
{
    float[] values; int[] dimensions;
    if (graph is not null)
    {
        var tensor = (Tensor<float>)graph.Outputs["last_hidden_state"]; values = tensor.ToArray(); dimensions = tensor.Dimensions.ToArray();
    }
    else
    {
        var tensor = (firstOutput ? heldNative : nativeResult)!.Single(); var info = tensor.GetTensorTypeAndShape();
        Require(info.ElementDataType == Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float, "Native output dtype");
        values = tensor.GetTensorDataAsSpan<float>().ToArray(); dimensions = info.Shape.Select(v => checked((int)v)).ToArray();
    }
    Require(dimensions.SequenceEqual(shape) && values.Length == reference.Length, "Output geometry"); double error = 0;
    for (int i = 0; i < values.Length; i++) { Require(float.IsFinite(values[i]), "Nonfinite output"); error = Math.Max(error, Math.Abs((double)values[i] - reference[i]) / Math.Max(1, Math.Abs((double)reference[i]))); }
    Require(error <= 1e-4, "Native numerical gate"); byte[] bytes = MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    Require(bytes.AsSpan().SequenceEqual(heldBits), "Repeated output bits changed");
    using var stream = new FileStream(Path.Combine(output, stage + ".f32"), FileMode.CreateNew); stream.Write(bytes); return error;
}
object? NativeIdentity()
{
    var modules = Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Where(m => Path.GetFileName(m.FileName).StartsWith("libonnxruntime", StringComparison.OrdinalIgnoreCase) || Path.GetFileName(m.FileName).Equals("onnxruntime.dll", StringComparison.OrdinalIgnoreCase)).ToArray();
    if (!native) { Require(modules.Length == 0, "Native ORT in managed worker"); return null; }
    Require(modules.Length == 1 && HashFile(modules[0].FileName) == HashFile(nativePath), "Actual native module identity");
    Require(OrtEnv.Instance().GetVersionString() == "1.23.2", "Native version");
    return new { path = modules[0].FileName, sha256 = HashFile(modules[0].FileName), version = OrtEnv.Instance().GetVersionString(), threads = 1, sequential = true, all_optimizations = true, spinning = false };
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
static void Require(bool ok, string why) { if (!ok) throw new InvalidDataException(why); }
static string HashFile(string path) { using var f = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(f)); }
static void Write(string path, object value) { using var f = new FileStream(path, FileMode.CreateNew); JsonSerializer.Serialize(f, value, new JsonSerializerOptions { WriteIndented = true }); }
static long Affinity() { if (OperatingSystem.IsWindows() || OperatingSystem.IsLinux()) return Process.GetCurrentProcess().ProcessorAffinity.ToInt64(); throw new PlatformNotSupportedException(); }
static void NoOrt() => Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => Path.GetFileName(m.FileName).StartsWith("libonnxruntime", StringComparison.OrdinalIgnoreCase) || Path.GetFileName(m.FileName).Equals("onnxruntime.dll", StringComparison.OrdinalIgnoreCase)), "Native library already loaded");
static object Resources() { using var p = Process.GetCurrentProcess(); var m = GC.GetGCMemoryInfo(); return new { rss = p.WorkingSet64, peak = p.PeakWorkingSet64, heap = m.HeapSizeBytes, committed = m.TotalCommittedBytes, gc = new[] { GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2) } }; }
readonly record struct Row(int block, int call, long execute, long request, long bytes, int g0, int g1, int g2);
