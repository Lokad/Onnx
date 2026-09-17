namespace Lokad.Onnx.Bench;

using System;
using System.Collections.Generic;
using System.Text.Json;

// W0 versioned manifest (I4): machine-readable identity plus every timed
// sample of one case repetition. Numbers serialize invariant-culture by
// construction (JSON numbers); times are milliseconds, ticks are monotonic
// Stopwatch.GetTimestamp values for order correlation, never wall time.
sealed class BenchmarkManifest
{
    public int SchemaVersion { get; set; } = 1;

    public string Case { get; set; } = "";

    public string LokDesc { get; set; } = "";

    public string OrtDesc { get; set; } = "";

    public int Iters { get; set; }

    public int Warmup { get; set; }

    // Round-trip UTC ("o") instant the manifest was written.
    public string StartedUtc { get; set; } = "";

    public List<PairedSample> Samples { get; set; } = new List<PairedSample>();

    public static string ToJson(BenchmarkManifest manifest) =>
        JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });

    public static BenchmarkManifest? ReadJson(string json) =>
        JsonSerializer.Deserialize<BenchmarkManifest>(json);
}
