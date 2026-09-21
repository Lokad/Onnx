using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

if (args.Length != 3) throw new ArgumentException("root manifest new-output");
if (!OperatingSystem.IsWindows()) throw new PlatformNotSupportedException("This local attribution uses Windows CPU2.");
string root = Path.GetFullPath(args[0]), manifest = Path.GetFullPath(args[1]), output = Path.GetFullPath(args[2]);
Require(!Directory.Exists(output), "Existing output");
Require(Environment.ProcessorCount == 1 && Process.GetCurrentProcess().ProcessorAffinity.ToInt64() == 4, "CPU2 before CLR");
Require(!Environment.GetEnvironmentVariables().Keys.Cast<string>().Any(k => k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)
    || k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)), "Overrides");
Require(Sha(typeof(ComputationalGraph).Assembly.Location) == "d1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4", "Qualified core");
Require(Sha(typeof(Community1Diarizer).Assembly.Location) == "e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb", "Qualified data");
using var document = JsonDocument.Parse(File.ReadAllBytes(manifest)); var spec = document.RootElement;
string FileOf(JsonElement item)
{
    string path = Path.Combine(root, item.GetProperty("path").GetString()!);
    Require(Sha(path) == item.GetProperty("sha256").GetString() && new FileInfo(path).Length == item.GetProperty("bytes").GetInt64(), "Asset identity");
    return path;
}
var models = spec.GetProperty("models").EnumerateObject().ToDictionary(p => p.Name, p => FileOf(p.Value));
var cases = spec.GetProperty("cases").EnumerateArray().Select(c => (Name: c.GetProperty("name").GetString()!,
    Pcm: NpySupport.ReadFloat32(FileOf(c.GetProperty("pcm"))).Values, Expected: c.GetProperty("expected").Clone())).ToArray();
Require(cases.Length == 4 && cases.Count(c => c.Pcm.Length == 160000) == 3, "Case coverage");
Directory.CreateDirectory(output);
var segmentation = OnnxImport.Load(models["segmentation"], 32L * 1024 * 1024) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
var encoder = OnnxImport.Load(models["encoder"], 64L * 1024 * 1024) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
var rows = new List<object>(); var applications = new List<object>(); var frontend = new List<object>();
var held = new List<(Tensor<float> Tensor, string Hash)>();
object Save(string name, float[] data, int[] shape)
{
    string file = name + ".f32"; File.WriteAllBytes(Path.Combine(output, file), MemoryMarshal.AsBytes(data.AsSpan()).ToArray());
    return new { file, shape, sha256 = Hash(data), values = data.Length };
}
void CheckHeld() { foreach (var old in held) Require(Hash(old.Tensor.ToArray()) == old.Hash, "Held graph output changed"); }
void Profile(string name, string model, ComputationalGraph graph, Tensor<float> input, string inputName, string outputName)
{
    string inputHash = Hash(input.ToArray());
    var inputRecord = Save(name + "-" + model + "-input", input.ToArray(), input.Dimensions.ToArray());
    string? first = null; var execution = graph.CreateExecution(ExecutionOptions.Memory);
    for (int pass = 0; pass < 3; pass++)
    {
        CheckHeld(); execution.Reset();
        Require(Hash(input.ToArray()) == inputHash, "Input changed");
        using var scope = pass == 2 ? Profiler.BeginWallExecution() : Profiler.BeginExecution(false);
        long started = Stopwatch.GetTimestamp();
        Require(execution.Execute(new Dictionary<string, ITensor> { [inputName] = input }, true, ExecutionProvider.CPU, ExecutionOptions.Memory), execution.LastErrorMessage ?? "Execution");
        long ended = Stopwatch.GetTimestamp();
        var result = (Tensor<float>)execution.Outputs[outputName]!; float[] values = result.ToArray();
        Require(values.All(float.IsFinite), "Nonfinite output"); string hash = Hash(values);
        if (first is null) first = hash; else Require(hash == first, "Profile/repeat changed bits");
        var nodes = execution.LastWallProfile!.Select(n => new { id = n.NodeId,
            name = graph.Nodes.Single(v => v.ID == n.NodeId).Name, op = n.Op.ToString(),
            start_ticks = n.StartTicks, end_ticks = n.EndTicks, seconds = (n.EndTicks - n.StartTicks) / (double)Stopwatch.Frequency }).ToArray();
        Require(pass != 2 || (nodes.Length == graph.Nodes.Count && nodes.All(n => n.end_ticks >= n.start_ticks && n.start_ticks >= started && n.end_ticks <= ended)), "Wall coverage");
        Require(pass == 2 || nodes.Length == 0, "Unexpected profiling");
        Require(Hash(input.ToArray()) == inputHash, "Input mutation");
        held.Add((result, hash)); CheckHeld();
        rows.Add(new { name, model, pass, phase = pass == 0 ? "warmup" : pass == 1 ? "unprofiled" : "wall",
            seconds = (ended - started) / (double)Stopwatch.Frequency, start_ticks = started, end_ticks = ended,
            frequency = Stopwatch.Frequency, nodes, input = inputRecord,
            output = Save(name + "-" + model + "-" + pass, values, result.Dimensions.ToArray()) });
    }
    execution.Reset(); CheckHeld(); Console.WriteLine(name + " " + model + " complete");
}
foreach (var c in cases.Where(c => c.Pcm.Length == 160000))
{
    var waveform = new DenseTensor<float>(c.Pcm, new[] { 1, 1, 160000 });
    Profile(c.Name, "segmentation", segmentation, waveform, "waveform", "scores");
    long start = Stopwatch.GetTimestamp(); var features = WeSpeakerAudio.LogMelFilterbank(c.Pcm, 16000, CancellationToken.None);
    frontend.Add(new { name = c.Name, seconds = Stopwatch.GetElapsedTime(start).TotalSeconds });
    Profile(c.Name, "embedding", encoder, features, "fbank_features", "/resnet/pool/Reshape_output_0");
}
var diarizer = new Community1Diarizer(models["segmentation"], models["encoder"], models["projection"], models["plda"]);
var heldApplications = new List<(Community1Diarization Value, string Snapshot)>();
foreach (var c in cases)
{
    string before = Hash(c.Pcm); long start = Stopwatch.GetTimestamp();
    var actual = diarizer.Diarize(c.Pcm, 16000, CancellationToken.None);
    double seconds = Stopwatch.GetElapsedTime(start).TotalSeconds;
    var e = c.Expected;
    Require(actual.Status.ToString() == e.GetProperty("status").GetString() && actual.Windows == e.GetProperty("windows").GetInt32()
        && actual.AudioDuration == e.GetProperty("audio_seconds").GetDouble(), "Public result");
    foreach (var pair in new[] { ("intervals", actual.Intervals), ("exclusive_intervals", actual.ExclusiveIntervals) })
    {
        var expected = e.GetProperty(pair.Item1).EnumerateArray().ToArray(); Require(expected.Length == pair.Item2.Count, "Intervals");
        for (int i = 0; i < expected.Length; i++) Require(Math.Abs(expected[i][0].GetDouble() - pair.Item2[i].Start) <= 1e-12
            && Math.Abs(expected[i][1].GetDouble() - pair.Item2[i].End) <= 1e-12 && expected[i][2].GetInt32() == pair.Item2[i].Speaker, "Interval value");
    }
    var speakers = e.GetProperty("speakers").EnumerateArray().ToArray(); Require(speakers.Length == actual.Speakers.Count, "Speakers");
    double maximum = 0;
    for (int s = 0; s < speakers.Length; s++)
    {
        Require(speakers[s].GetProperty("speaker").GetInt32() == actual.Speakers[s].Speaker
            && speakers[s].GetProperty("has_embedding").GetBoolean() == actual.Speakers[s].HasEmbedding, "Speaker identity");
        var values = speakers[s].GetProperty("centroid").EnumerateArray().Select(v => v.GetDouble()).ToArray();
        Require(values.Length == actual.Speakers[s].Centroid.Count, "Centroid shape");
        for (int j = 0; j < values.Length; j++) maximum = Math.Max(maximum, Math.Abs(values[j] - actual.Speakers[s].Centroid[j]) / Math.Max(1, Math.Abs(values[j])));
    }
    Require(double.IsFinite(maximum) && maximum <= 1e-4 && Hash(c.Pcm) == before, "Centroid/input contract");
    heldApplications.Add((actual, JsonSerializer.Serialize(actual)));
    Require(heldApplications.All(h => JsonSerializer.Serialize(h.Value) == h.Snapshot), "Held application changed");
    applications.Add(new { name = c.Name, seconds, maximum_centroid_error = maximum, result = actual });
    Console.WriteLine(c.Name + " public contract complete");
}
CheckHeld(); Require(rows.Count == 18 && applications.Count == 4, "Complete coverage");
Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.FileName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Native loaded");
File.WriteAllText(Path.Combine(output, "result.json"), JsonSerializer.Serialize(new { passed = true, runtime = Environment.Version.ToString(),
    core_sha256 = Sha(typeof(ComputationalGraph).Assembly.Location), data_sha256 = Sha(typeof(Community1Diarizer).Assembly.Location),
    manifest_sha256 = Sha(manifest), rows, applications, frontend, inputs_and_held_outputs_unchanged = true }, new JsonSerializerOptions { WriteIndented = true }));
static string Sha(string p) { using var s = File.OpenRead(p); return Convert.ToHexStringLower(SHA256.HashData(s)); }
static string Hash(float[] a) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(a.AsSpan())));
static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
