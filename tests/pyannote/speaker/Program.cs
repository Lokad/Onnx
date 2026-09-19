using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

if (args.Length != 3) throw new ArgumentException("Usage: SpeakerReplay REFERENCE ENCODER.onnx NEW_RESULT.json");
string root = Path.GetFullPath(args[0]), encoder = Path.GetFullPath(args[1]), resultPath = Path.GetFullPath(args[2]);
if (File.Exists(resultPath) || Directory.Exists(resultPath + ".arrays")) throw new IOException("Choose a fresh result path.");
string Sha(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
string FileSha(string path) => Sha(File.ReadAllBytes(path));
string Bits(float[] values) => Sha(MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
string Resource(string name)
{
    using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(name) ?? throw new InvalidDataException(name);
    using var reader = new StreamReader(stream); return reader.ReadToEnd().Replace("\r\n", "\n", StringComparison.Ordinal);
}
using var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, "manifest.json")));
var manifest = doc.RootElement;
using var pinDocument = JsonDocument.Parse(Resource("speaker-pins.json")); var pins = pinDocument.RootElement;
if (!JsonElement.DeepEquals(pins, manifest.GetProperty("pins"))
    || manifest.GetProperty("pins_lf_sha256").GetString() != Sha(Encoding.UTF8.GetBytes(Resource("speaker-pins.json")))
    || manifest.GetProperty("generator_lf_sha256").GetString() != Sha(Encoding.UTF8.GetBytes(Resource("speaker-generator.py")))
    || manifest.GetProperty("prepare_lf_sha256").GetString() != Sha(Encoding.UTF8.GetBytes(Resource("speaker-prepare.py"))))
    throw new InvalidDataException("Reference implementation identity mismatch.");
if (FileSha(encoder) != pins.GetProperty("encoder_sha256").GetString()
    || FileSha(Path.Combine(root, "projection.onnx")) != manifest.GetProperty("projection").GetProperty("output_sha256").GetString())
    throw new InvalidDataException("Model digest mismatch.");
var arrays = new Dictionary<string, float[]>(); var shapes = new Dictionary<string, int[]>();
foreach (var file in manifest.GetProperty("files").EnumerateObject())
{
    string path = Path.GetFullPath(Path.Combine(root, file.Name));
    if (Path.GetDirectoryName(path) != root || !file.Name.EndsWith(".npy", StringComparison.Ordinal)
        || new FileInfo(path).Length != file.Value.GetProperty("bytes").GetInt64() || FileSha(path) != file.Value.GetProperty("sha256").GetString()
        || file.Value.GetProperty("dtype").GetString() != "float32") throw new InvalidDataException("Fixture identity mismatch.");
    var data = NpySupport.ReadFloat32(path);
    if (!data.Shape.SequenceEqual(file.Value.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()))
        || data.Values.Any(v => !float.IsFinite(v))) throw new InvalidDataException("Fixture shape/values mismatch.");
    arrays.Add(file.Name, data.Values); shapes.Add(file.Name, data.Shape);
}
string Name(JsonElement value, string key) => value.GetProperty(key).GetString() ?? throw new InvalidDataException("Null fixture name.");
float[] Get(JsonElement value, string key) => arrays[Name(value, key)];
var expectedCases = (from recording in pins.GetProperty("recordings").EnumerateArray()
    from mask in pins.GetProperty("masks").EnumerateArray() select recording.GetString() + "-" + mask.GetString()).ToHashSet(StringComparer.Ordinal);
var cases = manifest.GetProperty("cases").EnumerateArray().ToArray(); var referenced = new HashSet<string>(StringComparer.Ordinal);
foreach (var c in cases)
{
    string name = Name(c, "name");
    if (!expectedCases.Remove(name) || name != Name(c, "recording") + "-" + Name(c, "mask_kind")) throw new InvalidDataException("Case coverage mismatch.");
    var pcm = Get(c, "pcm"); referenced.Add(Name(c, "pcm"));
    if (!shapes[Name(c, "pcm")].SequenceEqual(new[] { pcm.Length }) || pcm.Length < 400 || pcm.Length > 480000 || pcm.Any(v => Math.Abs(v) > 1))
        throw new InvalidDataException("PCM contract mismatch.");
    int frames = ((1 + (pcm.Length - 400) / 160) + 7) / 8;
    string? maskFile = c.GetProperty("mask").GetString(); float[] mask = maskFile is null ? Array.Empty<float>() : arrays[maskFile];
    if ((Name(c, "mask_kind") == "none") != (maskFile is null)) throw new InvalidDataException("Mask presence mismatch.");
    if (maskFile is not null)
    {
        referenced.Add(maskFile);
        if (mask.Length < 1 || mask.Length > 480000 || !shapes[maskFile].SequenceEqual(new[] { mask.Length }) || mask.Any(v => v < 0 || v > 1))
            throw new InvalidDataException("Mask contract mismatch.");
    }
    int positive = Enumerable.Range(0, frames).Count(t => maskFile is null || mask[(int)((long)t * mask.Length / frames)] > 0);
    if (frames != c.GetProperty("frames").GetInt32() || positive != c.GetProperty("positive").GetInt32()
        || Name(c, "status") != (positive >= 2 ? "Completed" : "InsufficientFrames")) throw new InvalidDataException("Mask policy mismatch.");
    string? vector = c.GetProperty("vector").GetString();
    if ((positive >= 2) != (vector is not null)) throw new InvalidDataException("Vector presence mismatch.");
    if (vector is not null) { referenced.Add(vector); if (!shapes[vector].SequenceEqual(new[] { 1, 256 })) throw new InvalidDataException("Vector shape mismatch."); }
}
if (expectedCases.Count != 0 || !referenced.SetEquals(arrays.Keys)) throw new InvalidDataException("Incomplete fixture coverage.");
Directory.CreateDirectory(resultPath + ".arrays");
var inputHashes = arrays.ToDictionary(p => p.Key, p => Bits(p.Value));
var embedder = new WeSpeakerEmbedder(encoder, Path.Combine(root, "projection.onnx"));
var reports = new List<object>(); var held = new List<(WeSpeakerEmbedding Value, string Hash)>();
var first = new Dictionary<string, string>(); bool passed = true; long count = 0; double maximum = 0;
void CheckHeld()
{
    foreach (var h in held) if (Bits(h.Value.Values.ToArray()) != h.Hash) throw new InvalidDataException("Held vector changed.");
    foreach (var input in arrays) if (Bits(input.Value) != inputHashes[input.Key]) throw new InvalidDataException("Input changed.");
}
WeSpeakerEmbedding Extract(JsonElement c) => embedder.Extract(Get(c, "pcm"), 16000,
    c.GetProperty("mask").GetString() is string file ? arrays[file] : Array.Empty<float>(), CancellationToken.None);
for (int repeat = 0; repeat < 2; repeat++) foreach (var c in cases)
{
    CheckHeld(); string name = Name(c, "name"); var result = Extract(c);
    if (result.Status.ToString() != Name(c, "status") || result.EncoderFrames != c.GetProperty("frames").GetInt32()
        || result.PositiveFrames != c.GetProperty("positive").GetInt32()) throw new InvalidDataException("Result policy mismatch.");
    float[] values = result.Values.ToArray(); string bits = Bits(values);
    if (repeat == 0) first.Add(name, bits); else if (first[name] != bits) throw new InvalidDataException("Repeated vector changed.");
    if (result.Values is not IList<float> view || !view.IsReadOnly) throw new InvalidDataException("Mutable public result.");
    double error = 0; int bad = 0; string? fileName = null;
    if (result.Status == WeSpeakerEmbeddingStatus.Completed)
    {
        var expected = Get(c, "vector"); if (values.Length != 256) throw new InvalidDataException("Vector width.");
        for (int i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i])) throw new InvalidDataException("Nonfinite vector.");
            double e = Math.Abs((double)values[i] - expected[i]) / Math.Max(1, Math.Abs((double)expected[i])); error = Math.Max(error, e); if (e > 1e-4) bad++;
        }
        fileName = name + "-" + repeat + ".f32";
        File.WriteAllBytes(Path.Combine(resultPath + ".arrays", fileName), MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
    }
    else if (values.Length != 0) throw new InvalidDataException("Insufficient data produced a vector.");
    held.Add((result, bits)); count += values.Length; maximum = Math.Max(maximum, error); passed &= bad == 0;
    reports.Add(new { name, repeat, status = result.Status.ToString(), frames = result.EncoderFrames, positive = result.PositiveFrames,
        values = values.Length, sha256 = bits, file = fileName, error, bad });
    Console.WriteLine(name + " repeat " + repeat + " " + result.Status);
}
var recovery = cases.Single(c => Name(c, "name") == "short-1680-none"); var valid = Get(recovery, "pcm"); var refusals = new List<string>();
void Refuse<T>(string name, Action action) where T : Exception
{
    try { action(); throw new InvalidOperationException("Accepted " + name); }
    catch (T) { refusals.Add(name); }
    if (Bits(Extract(recovery).Values.ToArray()) != first[Name(recovery, "name")]) throw new InvalidDataException("Recovery changed vector.");
    CheckHeld();
}
Refuse<OperationCanceledException>("cancelled", () => embedder.Extract(valid, 16000, Array.Empty<float>(), new CancellationToken(true)));
Refuse<ArgumentOutOfRangeException>("rate", () => embedder.Extract(valid, 8000, Array.Empty<float>(), CancellationToken.None));
Refuse<ArgumentOutOfRangeException>("short", () => embedder.Extract(new float[399], 16000, Array.Empty<float>(), CancellationToken.None));
Refuse<ArgumentOutOfRangeException>("long", () => embedder.Extract(new float[480001], 16000, Array.Empty<float>(), CancellationToken.None));
Refuse<ArgumentException>("pcm", () => embedder.Extract(new float[] { float.NaN }.Concat(valid).ToArray(), 16000, Array.Empty<float>(), CancellationToken.None));
Refuse<ArgumentException>("mask", () => embedder.Extract(valid, 16000, new float[] { 0, float.NaN }, CancellationToken.None));
Parallel.For(0, 3, _ => { if (Bits(Extract(recovery).Values.ToArray()) != first[Name(recovery, "name")]) throw new InvalidDataException("Concurrent vector changed."); });
CheckHeld(); using var process = Process.GetCurrentProcess();
foreach (ProcessModule module in process.Modules) if (module.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)) throw new InvalidDataException("Native ORT loaded.");
var report = new { scope = Name(manifest, "scope"), passed, values = count, maximum, reports, refusals, concurrent_requests = 3,
    manifest_sha256 = FileSha(Path.Combine(root, "manifest.json")), core_sha256 = FileSha(typeof(Tensor<float>).Assembly.Location),
    data_sha256 = FileSha(typeof(WeSpeakerEmbedder).Assembly.Location), runner_sha256 = FileSha(Assembly.GetExecutingAssembly().Location),
    runtime = Environment.Version.ToString(), peak = process.PeakWorkingSet64 };
File.WriteAllText(resultPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Passed={passed} values={count} maximum={maximum:R}"); return passed ? 0 : 1;
