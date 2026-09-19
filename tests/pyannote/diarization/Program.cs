using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;
using System.Text.Json;
using System.Diagnostics;
using System.Security.Cryptography;
var evidence = new Evidence(args); string source = evidence.Reference, destination = evidence.Output;
var model = new Community1Diarizer(evidence.Models["segmentation"],
    evidence.Models["encoder"],
    evidence.Models["projection"],
    evidence.Models["plda"]);
using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(source, "manifest.json")));
var reports = new List<object>(); bool passed = true; double maximum = 0;
var held = new List<(Community1Diarization Value, string Snapshot)>();
var first = new Dictionary<string, string>();
void CheckHeld()
{
    foreach (var item in held) if (JsonSerializer.Serialize(item.Value) != item.Snapshot) throw new InvalidDataException("Held output changed");
}
void Replay(JsonElement c, int repeat)
{
    CheckHeld();
    string name = c.GetProperty("name").GetString() ?? ""; string filename = c.GetProperty("pcm").GetString() ?? "";
    string path = Path.Combine(source, filename);
    if (Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path))) != doc.RootElement.GetProperty("files").GetProperty(filename).GetProperty("sha256").GetString()) throw new Exception("PCM hash");
    var samples = NpySupport.ReadFloat32(path).Values; var before = samples.ToArray();
    var watch = Stopwatch.StartNew(); var result = model.Diarize(samples, 16000, CancellationToken.None); watch.Stop();
    bool same = result.Status.ToString() == c.GetProperty("status").GetString()
        && result.AudioDuration == c.GetProperty("seconds").GetDouble() && result.Windows == c.GetProperty("windows").GetArrayLength();
    foreach (var pair in new[] { ("intervals", result.Intervals), ("exclusive_intervals", result.ExclusiveIntervals) })
    {
        var expected = c.GetProperty(pair.Item1).EnumerateArray().ToArray(); same &= expected.Length == pair.Item2.Count;
        for (int i = 0; i < Math.Min(expected.Length, pair.Item2.Count); i++)
        { var actual = pair.Item2[i]; same &= Math.Abs(actual.Start - expected[i][0].GetDouble()) <= 1e-12 && Math.Abs(actual.End - expected[i][1].GetDouble()) <= 1e-12 && actual.Speaker == expected[i][2].GetInt32(); }
    }
    double error = 0;
    if (c.TryGetProperty("speakers", out var speakers))
    {
        same &= speakers.GetArrayLength() == result.Speakers.Count;
        for (int s = 0; s < Math.Min(speakers.GetArrayLength(), result.Speakers.Count); s++)
        {
            var expected = speakers[s]; var actual = result.Speakers[s]; same &= actual.Speaker == expected.GetProperty("speaker").GetInt32() && actual.HasEmbedding == expected.GetProperty("has_embedding").GetBoolean();
            var e = expected.GetProperty("centroid").EnumerateArray().Select(v => v.GetDouble()).ToArray();
            if (actual.Centroid.Count != 256 || actual.Centroid.Any(v => !double.IsFinite(v))) throw new InvalidDataException("Centroid contract");
            for (int d = 0; d < 256; d++) error = Math.Max(error, Math.Abs(actual.Centroid[d] - e[d]) / Math.Max(1, Math.Abs(e[d])));
        }
    }
    else same &= result.Speakers.Count == 0;
    maximum = Math.Max(maximum, error); bool ok = same && error <= 1e-4; passed &= ok;
    if (!samples.SequenceEqual(before)) throw new Exception("Input mutation");
    string snapshot = JsonSerializer.Serialize(result);
    if (repeat == 0) first.Add(name, snapshot); else if (first[name] != snapshot) throw new InvalidDataException("Repeated result changed");
    held.Add((result, snapshot)); CheckHeld();
    reports.Add(new { name, repeat, passed = ok, exact_timeline = same, error, elapsed_ms = watch.Elapsed.TotalMilliseconds, result });
    Console.WriteLine($"{name} {ok} timeline={same} error={error:R} intervals={result.Intervals.Count} speakers={result.Speakers.Count} ms={watch.Elapsed.TotalMilliseconds:F0}");
}
var cases = doc.RootElement.GetProperty("cases").EnumerateArray().ToArray();
for (int repeat = 0; repeat < 2; repeat++) foreach (var c in cases) Replay(c, repeat);
var refusals = new List<string>();
void Refuse<T>(string name, Action action) where T : Exception
{
    try { action(); } catch (T) { refusals.Add(name); CheckHeld(); return; }
    throw new InvalidDataException("Expected rejection: " + name);
}
Refuse<ArgumentOutOfRangeException>("sample-rate", () => model.Diarize(new float[1], 8000, CancellationToken.None));
Refuse<ArgumentException>("nonfinite", () => model.Diarize(new[] { float.NaN }, 16000, CancellationToken.None));
Refuse<ArgumentException>("range", () => model.Diarize(new[] { 1.01f }, 16000, CancellationToken.None));
using var cancellation = new CancellationTokenSource(); cancellation.Cancel();
Refuse<OperationCanceledException>("canceled", () => model.Diarize(new float[1], 16000, cancellation.Token));
var empty = model.Diarize(Array.Empty<float>(), 16000, CancellationToken.None);
if (empty.Status != Community1DiarizationStatus.NoSpeech || empty.Windows != 0 || empty.AudioDuration != 0 || empty.Speakers.Count != 0 || empty.Intervals.Count != 0 || empty.ExclusiveIntervals.Count != 0) throw new InvalidDataException("Empty PCM");
Replay(cases[0], 2); // recovery on the same instance after every refusal
var concurrent = Enumerable.Range(0, 2).Select(_ => Task.Run(() => model.Diarize(
    NpySupport.ReadFloat32(Path.Combine(source, cases[0].GetProperty("pcm").GetString() ?? "")).Values, 16000, CancellationToken.None))).ToArray();
Task.WaitAll(concurrent);
foreach (var task in concurrent) if (JsonSerializer.Serialize(task.Result) != first[cases[0].GetProperty("name").GetString() ?? ""]) throw new InvalidDataException("Concurrent result changed");
CheckHeld();
using var process = Process.GetCurrentProcess();
File.WriteAllText(destination, JsonSerializer.Serialize(new { reference_sha256 = evidence.ManifestSha, assemblies = Evidence.Assemblies(), passed, maximum, reports,
    refusals, empty, concurrent = concurrent.Select(t => t.Result), held_outputs_unchanged = true, peak = process.PeakWorkingSet64 }, new JsonSerializerOptions { WriteIndented = true }));
return passed ? 0 : 1;
