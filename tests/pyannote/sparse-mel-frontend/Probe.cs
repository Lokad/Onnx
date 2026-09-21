using System.Collections;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

if (args.Length != 4) throw new ArgumentException("root, manifest, new output, order");
string root = Path.GetFullPath(args[0]), manifestPath = Path.GetFullPath(args[1]), output = Path.GetFullPath(args[2]);
if (Directory.Exists(output) || args[3] is not ("dense-first" or "sparse-first")) throw new InvalidDataException();
Directory.CreateDirectory(output);
using var document = JsonDocument.Parse(File.ReadAllText(manifestPath));
var manifest = document.RootElement;
static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
static byte[] Bytes(float[] values) => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
string core = Hash(File.ReadAllBytes(typeof(Tensor<float>).Assembly.Location));
string data = Hash(File.ReadAllBytes(typeof(WeSpeakerAudio).Assembly.Location));
if (core != manifest.GetProperty("core").GetString() || data != manifest.GetProperty("data").GetString()) throw new InvalidDataException("Runtime identity");
string baselinePath = Path.Combine(root, manifest.GetProperty("baseline_data_path").GetString()!);
string baselineHash = Hash(File.ReadAllBytes(baselinePath));
if (baselineHash != "1d34666456a5da749dc3b40ee25621af0bed806c9167f3ad12e96d0736dca662") throw new InvalidDataException("Dense identity");
var context = new AssemblyLoadContext("dense-frontend", isCollectible: false);
context.Resolving += (_, name) => name.Name == "Lokad.Onnx" ? typeof(Tensor<float>).Assembly : null;
var denseType = context.LoadFromAssemblyPath(baselinePath).GetType("Lokad.Onnx.WeSpeakerAudio")!;
var dense = denseType.GetMethod("LogMelFilterbank", BindingFlags.Public | BindingFlags.Static)!.CreateDelegate<Frontend>();
const BindingFlags hidden = BindingFlags.Static | BindingFlags.NonPublic;
var weights = (float[])typeof(WeSpeakerAudio).GetField("MelWeights", hidden)!.GetValue(null)!;
var originalWeights = (float[])denseType.GetField("MelWeights", hidden)!.GetValue(null)!;
var support = ((int Start, int End)[])typeof(WeSpeakerAudio).GetField("MelSupport", hidden)!.GetValue(null)!;
if (!Bytes(weights).AsSpan().SequenceEqual(Bytes(originalWeights)) || weights.Length != 20480 || support.Length != 80) throw new InvalidDataException("Coefficients");
var ranges = new List<object>(); int retained = 0, nonzero = 0;
for (int band = 0; band < 80; band++)
{
    var (first, last) = support[band];
    if (first < 0 || last < first || last > 256) throw new InvalidDataException("Support bounds");
    int count = 0;
    for (int bin = 0; bin < 256; bin++)
    {
        float weight = weights[band * 256 + bin];
        if (!float.IsFinite(weight) || weight < 0 || weight > 1) throw new InvalidDataException("Weight bounds");
        if (weight != 0) { count++; if (bin < first || bin >= last) throw new InvalidDataException("Omitted nonzero"); }
    }
    retained += last - first; nonzero += count;
    ranges.Add(new { band, first, last, nonzero = count });
}
var cache = new Dictionary<string, (float[] values, string hash)>();
var held = new List<(DenseTensor<float> tensor, byte[] bytes)>();
var records = new List<object>(); int total = 0, ordinal = 0;
foreach (var fixture in manifest.GetProperty("cases").EnumerateArray())
{
    string path = Path.Combine(root, fixture.GetProperty("path").GetString()!);
    if (!cache.TryGetValue(path, out var input))
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length != fixture.GetProperty("bytes").GetInt32() || Hash(bytes) != fixture.GetProperty("sha256").GetString()) throw new InvalidDataException("Input identity");
        input = (MemoryMarshal.Cast<byte, float>(bytes).ToArray(), Hash(bytes)); cache.Add(path, input);
    }
    int offset = fixture.GetProperty("offset").GetInt32(), length = fixture.GetProperty("samples").GetInt32();
    var results = new Dictionary<string, DenseTensor<float>>();
    foreach (string role in args[3] == "dense-first" ? new[] { "dense", "sparse" } : new[] { "sparse", "dense" })
    {
        var slice = input.values.AsSpan(offset, length);
        var result = role == "dense" ? dense(slice, 16000, CancellationToken.None)
            : WeSpeakerAudio.LogMelFilterbank(slice, 16000, CancellationToken.None);
        results.Add(role, result); held.Add((result, Bytes(result.ToArray())));
    }
    byte[] expected = Bytes(results["dense"].ToArray()), actual = Bytes(results["sparse"].ToArray());
    if (!actual.AsSpan().SequenceEqual(expected) || Hash(Bytes(input.values)) != input.hash) throw new InvalidDataException("Feature or input bits");
    int[] shape = { 1, 1 + (length - 400) / 160, 80 };
    if (!results["sparse"].Dimensions.ToArray().SequenceEqual(shape) || !results["dense"].Dimensions.ToArray().SequenceEqual(shape)) throw new InvalidDataException("Feature shape");
    File.WriteAllBytes(Path.Combine(output, ordinal + ".dense.f32"), expected);
    File.WriteAllBytes(Path.Combine(output, ordinal + ".sparse.f32"), actual);
    records.Add(new { ordinal, name = fixture.GetProperty("name").GetString(), shape, values = actual.Length / 4, sha256 = Hash(actual), inputs_unchanged = true });
    total += actual.Length / 4; ordinal++;
}
foreach (var item in held)
    if (!Bytes(item.tensor.ToArray()).AsSpan().SequenceEqual(item.bytes)) throw new InvalidDataException("Held output changed");
foreach (var input in cache.Values)
    if (Hash(Bytes(input.values)) != input.hash) throw new InvalidDataException("Retained input changed");
string[] flags = Environment.GetEnvironmentVariables().Cast<DictionaryEntry>().Select(e => (string)e.Key)
    .Where(k => k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)).ToArray();
if (flags.Length != 0 || Environment.ProcessorCount != 1) throw new InvalidDataException("Runtime policy");
File.WriteAllText(Path.Combine(output, "result.json"), JsonSerializer.Serialize(new {
    passed = true, pid = Environment.ProcessId, runtime = RuntimeInformation.FrameworkDescription,
    processor_count = Environment.ProcessorCount, flags, core, data, baseline_data = baselineHash,
    executable = Hash(File.ReadAllBytes(Assembly.GetExecutingAssembly().Location)), manifest = Hash(File.ReadAllBytes(manifestPath)),
    order = args[3], calls = ordinal * 2, values = total, records, coefficient_sha256 = Hash(Bytes(weights)),
    dense_terms_per_frame = 20480, retained_terms_per_frame = retained, nonzero_terms_per_frame = nonzero, ranges,
    inputs_and_held_outputs_unchanged = true, scope = "Exact frontend feature and coefficient support proof; no latency conclusion."
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"{ordinal} feature pairs and {total} values pass; retained {retained}/20480 terms per frame.");

delegate DenseTensor<float> Frontend(ReadOnlySpan<float> input, int rate, CancellationToken cancellation);
