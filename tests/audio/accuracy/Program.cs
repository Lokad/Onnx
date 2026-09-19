using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

if (args.Length != 5 || args[0] is not ("parakeet" or "whisper"))
    throw new ArgumentException("Usage: AccuracyReplay <parakeet|whisper> <models> <audio.json> <native-manifest.json> <new-result-directory>");
string family = args[0], models = Path.GetFullPath(args[1]), audioPath = Path.GetFullPath(args[2]),
    nativePath = Path.GetFullPath(args[3]), destination = Path.GetFullPath(args[4]);
if (Directory.Exists(destination) || File.Exists(destination)) throw new IOException("Result already exists");
static string Sha(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
static void NoOrt()
{
    foreach (ProcessModule module in Process.GetCurrentProcess().Modules)
        if (module.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)) throw new InvalidOperationException("Native ORT loaded");
}
static bool SameIds(IReadOnlyList<int> actual, JsonElement expected) => actual.SequenceEqual(expected.EnumerateArray().Select(x => x.GetInt32()));
var json = new JsonSerializerOptions { WriteIndented = true, Converters = { new JsonStringEnumConverter() } };
using var audio = JsonDocument.Parse(File.ReadAllBytes(audioPath));
using var native = JsonDocument.Parse(File.ReadAllBytes(nativePath));
using var pinStream = Assembly.GetExecutingAssembly().GetManifestResourceStream(family + "-assets.json") ?? throw new InvalidDataException("Asset pin missing");
using var pins = JsonDocument.Parse(pinStream);
if (audio.RootElement.GetProperty("schema").GetInt32() != 1 || audio.RootElement.GetProperty("sample_rate").GetInt32() != 16000
    || native.RootElement.GetProperty("schema").GetInt32() != 1 || native.RootElement.GetProperty("audio_manifest_sha256").GetString() != Sha(audioPath))
    throw new InvalidDataException("Manifest identity/schema differs");
var cases = audio.RootElement.GetProperty("cases").EnumerateArray().ToArray();
var references = native.RootElement.GetProperty("cases").EnumerateArray().ToArray();
if (cases.Length != 20 || cases.Select(x => x.GetProperty("name").GetString()).Distinct().Count() != 20
    || references.Length != (family == "parakeet" ? 21 : 20)
    || !cases.Select(x => x.GetProperty("name").GetString()).SequenceEqual(references.Take(20).Select(x => x.GetProperty("name").GetString())))
    throw new InvalidDataException("Case coverage differs");
var expectedFiles = pins.RootElement.GetProperty("files");
var nativeFiles = native.RootElement.GetProperty("assets").GetProperty("files");
if (!expectedFiles.EnumerateObject().Select(x => x.Name).Order().SequenceEqual(nativeFiles.EnumerateObject().Select(x => x.Name).Order()))
    throw new InvalidDataException("Native asset set differs");
foreach (var file in expectedFiles.EnumerateObject())
{
    var supplied = nativeFiles.GetProperty(file.Name);
    string path = Path.Combine(models, file.Name);
    if (supplied.GetProperty("sha256").GetString() != file.Value.GetProperty("sha256").GetString()
        || supplied.GetProperty("bytes").GetInt64() != file.Value.GetProperty("bytes").GetInt64()
        || new FileInfo(path).Length != file.Value.GetProperty("bytes").GetInt64() || Sha(path) != file.Value.GetProperty("sha256").GetString())
        throw new InvalidDataException("Model asset differs: " + file.Name);
}
Directory.CreateDirectory(destination);
NoOrt();
var watch = Stopwatch.StartNew();
ParakeetTranscriber? parakeet = family == "parakeet" ? new(models) : null;
WhisperTranscriber? whisper = family == "whisper" ? new(models) : null;
double loadSeconds = watch.Elapsed.TotalSeconds;
var rows = new List<object>();
var held = new List<(object Result, string Json, float[] Pcm, byte[] Original)>();
bool passed = true;
for (int index = 0; index <= cases.Length; index++)
{
    int caseIndex = index % cases.Length;
    var item = cases[caseIndex]; var reference = references[caseIndex];
    string name = item.GetProperty("name").GetString()!;
    string pcmPath = Path.Combine(Path.GetDirectoryName(audioPath)!, item.GetProperty("pcm").GetString()!);
    if (Sha(pcmPath) != item.GetProperty("pcm_sha256").GetString()
        || item.GetProperty("pcm_sha256").GetString() != reference.GetProperty("pcm_sha256").GetString())
        throw new InvalidDataException("PCM identity differs");
    var (pcm, shape) = NpySupport.ReadFloat32(pcmPath);
    if (!shape.SequenceEqual(new[] { item.GetProperty("samples").GetInt32() }) || pcm.Any(x => !float.IsFinite(x)) || !pcm.Any(x => x != 0))
        throw new InvalidDataException("PCM contract differs");
    var original = MemoryMarshal.AsBytes(pcm.AsSpan()).ToArray();
    object result; bool same;
    watch.Restart();
    if (parakeet is not null)
    {
        var actual = parakeet.Transcribe(pcm, 16000, ParakeetTranscriptionOptions.Default, CancellationToken.None);
        watch.Stop();result = actual;
        var expected = reference.GetProperty("expected");
        same = actual.Text == expected.GetProperty("text").GetString() && SameIds(actual.TokenIds, expected.GetProperty("token_ids"))
            && SameIds(actual.FrameIndices, expected.GetProperty("frame_indices")) && SameIds(actual.DurationFrames, expected.GetProperty("duration_frames"))
            && actual.StopReason.ToString() == expected.GetProperty("stop_reason").GetString()
            && actual.EncodedFrames == expected.GetProperty("encoded_frames").GetInt32() && actual.DecoderCalls == expected.GetProperty("decoder_calls").GetInt32();
    }
    else
    {
        if (item.GetProperty("language").GetString() != "en" || item.GetProperty("max_new_tokens").GetInt32() != 444)
            throw new InvalidDataException("Whisper policy differs");
        var actual = whisper!.Transcribe(pcm, 16000, WhisperTranscriptionOptions.ForLanguage("en"), CancellationToken.None);
        watch.Stop();result = actual;
        same = actual.Text == reference.GetProperty("text").GetString() && SameIds(actual.TokenIds, reference.GetProperty("tokens"))
            && actual.StopReason.ToString() == reference.GetProperty("stop_reason").GetString()
            && actual.SkippedAsNoSpeech == reference.GetProperty("skipped_as_no_speech").GetBoolean();
    }
    held.Add((result, JsonSerializer.Serialize(result, json), pcm, original));
    foreach (var prior in held)
        if (!MemoryMarshal.AsBytes(prior.Pcm.AsSpan()).SequenceEqual(prior.Original) || JsonSerializer.Serialize(prior.Result, json) != prior.Json)
            throw new InvalidDataException("An actual input or held result changed across requests");
    NoOrt();passed &= same;
    var row = new { name, repeat = index == cases.Length, matches = same, result, seconds = watch.Elapsed.TotalSeconds,
        pcm_sha256 = Sha(pcmPath), input_and_held_results_unchanged = true };
    rows.Add(row);
    File.WriteAllText(Path.Combine(destination, $"{index:D2}-{name}.json"), JsonSerializer.Serialize(row, json));
    Console.WriteLine($"{index:D2} {name} native_match={same} seconds={watch.Elapsed.TotalSeconds:F3}");
}
File.WriteAllText(Path.Combine(destination, "result.json"), JsonSerializer.Serialize(new {
    schema = 1, family, passed, cases = rows, load_seconds = loadSeconds,
    core_sha256 = Sha(typeof(ComputationalGraph).Assembly.Location), data_sha256 = Sha(typeof(WhisperTranscriber).Assembly.Location),
    runner_sha256 = Sha(Assembly.GetExecutingAssembly().Location), audio_manifest_sha256 = Sha(audioPath), native_manifest_sha256 = Sha(nativePath),
    runtime = RuntimeInformation.FrameworkDescription, os = RuntimeInformation.OSDescription,
    peak_working_set = Process.GetCurrentProcess().PeakWorkingSet64,
    flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k => k.StartsWith("LOKAD_") || k.StartsWith("COMPlus_") || k.StartsWith("DOTNET_")).ToDictionary(k => k, Environment.GetEnvironmentVariable),
    boundary = "Application decisions and labeled accuracy only; full numerical acceptance and long audio remain separate."
}, json));
return passed ? 0 : 2;
