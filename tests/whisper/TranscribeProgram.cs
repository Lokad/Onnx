using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

if (args.Length != 3) throw new ArgumentException("Usage: TranscribeReplay <model-directory> <fixture-manifest> <result.json>");
string models = Path.GetFullPath(args[0]), manifestPath = Path.GetFullPath(args[1]), destination = Path.GetFullPath(args[2]);
string fixtures = Path.GetDirectoryName(manifestPath)!;
if (File.Exists(destination)) throw new IOException("Result already exists");
static string Sha(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
static void NoOrt()
{
    foreach (ProcessModule module in Process.GetCurrentProcess().Modules)
        if (module.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)) throw new InvalidOperationException("Native ORT loaded into managed replay");
}
using var manifest = JsonDocument.Parse(File.ReadAllText(manifestPath));
using var assetStream = Assembly.GetExecutingAssembly().GetManifestResourceStream("transcription-assets.json") ?? throw new InvalidDataException("Missing asset pin");
using var pinned = JsonDocument.Parse(assetStream);
if (manifest.RootElement.GetProperty("schema").GetInt32() != 1) throw new InvalidDataException("Unknown manifest schema");
var expectedFiles = pinned.RootElement.GetProperty("files");
var fixtureFiles = manifest.RootElement.GetProperty("assets").GetProperty("files");
if (!fixtureFiles.EnumerateObject().Select(x => x.Name).Order().SequenceEqual(expectedFiles.EnumerateObject().Select(x => x.Name).Order()))
    throw new InvalidDataException("Fixture asset set differs");
foreach (var item in expectedFiles.EnumerateObject())
{
    string path = Path.Combine(models, item.Name);
    var supplied = fixtureFiles.GetProperty(item.Name);
    if (supplied.GetProperty("sha256").GetString() != item.Value.GetProperty("sha256").GetString()
        || supplied.GetProperty("bytes").GetInt64() != item.Value.GetProperty("bytes").GetInt64()
        || new FileInfo(path).Length != item.Value.GetProperty("bytes").GetInt64() || Sha(path) != item.Value.GetProperty("sha256").GetString())
        throw new InvalidDataException("Pinned model asset differs: " + item.Name);
}
NoOrt();var watch = Stopwatch.StartNew();
var transcriber = new WhisperTranscriber(models);
double loadSeconds = watch.Elapsed.TotalSeconds;
var rows = new List<object>();bool passed = true;
var cases = manifest.RootElement.GetProperty("cases").EnumerateArray().ToArray();
if (cases.Length == 0) throw new InvalidDataException("No transcription cases");
// Repeat the first request after all other cases and intentional input failures.
foreach (var item in cases.Concat(cases.Take(1)))
{
    string name = item.GetProperty("name").GetString()!;
    string pcmPath = Path.Combine(fixtures, item.GetProperty("pcm").GetString()!);
    if (Sha(pcmPath) != item.GetProperty("pcm_sha256").GetString()) throw new InvalidDataException("PCM digest");
    var (pcm, shape) = NpySupport.ReadFloat32(pcmPath);
    if (!shape.SequenceEqual(new[] { item.GetProperty("samples").GetInt32() })) throw new InvalidDataException("PCM shape");
    var original = MemoryMarshal.AsBytes(pcm.AsSpan()).ToArray();
    var options = WhisperTranscriptionOptions.ForLanguage(item.GetProperty("language").GetString()!) with { MaxNewTokens = item.GetProperty("max_new_tokens").GetInt32() };
    watch.Restart();var result = transcriber.Transcribe(pcm, 16000, options, CancellationToken.None);watch.Stop();
    if (!MemoryMarshal.AsBytes(pcm.AsSpan()).SequenceEqual(original)) throw new InvalidDataException("PCM input changed");
    bool same = result.TokenIds.SequenceEqual(item.GetProperty("tokens").EnumerateArray().Select(x => x.GetInt32()))
        && result.Text == item.GetProperty("text").GetString()
        && result.SkippedAsNoSpeech == item.GetProperty("skipped_as_no_speech").GetBoolean()
        && result.StopReason.ToString() == item.GetProperty("stop_reason").GetString();
    passed &= same;
    rows.Add(new { name, matches = same, result, seconds = watch.Elapsed.TotalSeconds,
        native_no_speech = item.GetProperty("no_speech_probability"), native_average_logprob = item.GetProperty("average_log_probability") });
    Console.WriteLine(JsonSerializer.Serialize(rows[^1]));
    NoOrt();
    try { transcriber.Transcribe(pcm, 16000, options with { MaxNewTokens = 445 }, CancellationToken.None); throw new Exception("Invalid limit accepted"); }
    catch (ArgumentOutOfRangeException) { }
    try { transcriber.Transcribe(new float[480001], 16000, options, CancellationToken.None); throw new Exception("Long request silently truncated"); }
    catch (ArgumentOutOfRangeException) { }
    try { transcriber.Transcribe(new[] { float.NaN }, 16000, options, CancellationToken.None); throw new Exception("Nonfinite PCM accepted"); }
    catch (ArgumentException) { }
    try { transcriber.Transcribe(pcm, 16000, options, new CancellationToken(true)); throw new Exception("Canceled request executed"); }
    catch (OperationCanceledException) { }
}
File.WriteAllText(destination, JsonSerializer.Serialize(new { passed, load_seconds = loadSeconds, cases = rows,
    core_sha256 = Sha(typeof(ComputationalGraph).Assembly.Location), data_sha256 = Sha(typeof(WhisperTranscriber).Assembly.Location),
    runner_sha256 = Sha(Assembly.GetExecutingAssembly().Location), manifest_sha256 = Sha(manifestPath),
    peak_working_set = Process.GetCurrentProcess().PeakWorkingSet64,
    flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k => k.StartsWith("LOKAD_")).ToDictionary(k => k, Environment.GetEnvironmentVariable),
    boundary = "Application token/text/stop agreement only. Existing encoder/logit numerical failures remain separate and unchanged." }, new JsonSerializerOptions { WriteIndented = true }));
return passed ? 0 : 2;
