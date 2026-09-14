namespace Lokad.Onnx.Bench;

using System.Security.Cryptography;
using System.Text.Json;

/// <summary>
/// Provenance checks for voice benchmark assets (benchmark-only, ships nowhere).
/// Replay input bytes are verified against the SHA-256 hashes recorded in
/// models/voice-fixtures/replay/replay.json before timing; model and
/// external-weight identities print in every repetition header under their
/// actual file names.
/// </summary>
static class VoiceProvenance
{
    static Dictionary<string, (long Bytes, string Sha256)>? replayCache;

    internal static void VerifyReplayFile(string path)
    {
        var table = ReplayTable(path);
        string file = Path.GetFileName(path);
        if (file is null || !table.TryGetValue(file, out var expected))
            throw new InvalidOperationException("voice replay asset without manifest record: " + path + ".");
        long actualBytes = new FileInfo(path).Length;
        if (actualBytes != expected.Bytes)
            throw new InvalidOperationException("voice replay asset byte mismatch: " + path + " (expected " + expected.Bytes + ", found " + actualBytes + ").");
        string actual = FileHash(path);
        if (!actual.Equals(expected.Sha256, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("voice replay asset hash mismatch: " + path + ".");
    }

    internal static void VerifyModelFile(string name, string model)
    {
        if (!File.Exists(model))
            throw new InvalidOperationException(name + ": model asset missing: " + model + " (git-ignored local asset).");
    }

    internal static string SidecarInfo(string model)
    {
        var parts = new List<string>();
        foreach (string candidate in new[] { model + ".data", Path.ChangeExtension(model, ".onnx_data") })
        {
            if (candidate is null || !File.Exists(candidate)) continue;
            parts.Add("sidecar=" + Path.GetFileName(candidate) + " bytes=" + new FileInfo(candidate).Length + " sha12=" + ShortHash(candidate));
        }
        return parts.Count == 0 ? string.Empty : " " + string.Join(" ", parts);
    }

    internal static string ModelLabel(string model) =>
        Path.GetFileName(Path.GetDirectoryName(model)) + "/" + Path.GetFileName(model);

    internal static string ShortHash(string file)
    {
        using var sha = SHA256.Create();
        using var s = File.OpenRead(file);
        return Convert.ToHexString(sha.ComputeHash(s)).Substring(0, 12);
    }

    static string FileHash(string path)
    {
        using var sha = SHA256.Create();
        using var s = File.OpenRead(path);
        return Convert.ToHexString(sha.ComputeHash(s));
    }

    static Dictionary<string, (long Bytes, string Sha256)> ReplayTable(string path)
    {
        if (replayCache is not null) return replayCache;
        string dir = Path.GetDirectoryName(path) ?? ".";
        string manifest = Path.Combine(dir, "replay.json");
        if (!File.Exists(manifest))
            throw new InvalidOperationException("voice replay manifest missing: " + manifest + ".");
        var table = new Dictionary<string, (long, string)>(StringComparer.Ordinal);
        using var doc = JsonDocument.Parse(File.ReadAllBytes(manifest));
        if (!doc.RootElement.TryGetProperty("cases", out var cases))
            throw new InvalidDataException("voice replay manifest lacks cases: " + manifest + ".");
        foreach (var kase in cases.EnumerateObject())
        {
            if (!kase.Value.TryGetProperty("inputs", out var inputs)) continue;
            foreach (var input in inputs.EnumerateObject())
            {
                var entry = input.Value;
                if (entry.TryGetProperty("ref", out _)) continue;
                if (!entry.TryGetProperty("file", out var fileProp)
                    || !entry.TryGetProperty("bytes", out var bytesProp)
                    || !entry.TryGetProperty("sha256", out var shaProp))
                    throw new InvalidDataException("voice replay manifest entry lacks file/bytes/sha256: " + manifest + ".");
                string key = Path.GetFileName(fileProp.GetString() ?? string.Empty);
                if (key.Length > 0 && !table.ContainsKey(key))
                    table[key] = (bytesProp.GetInt64(), shaProp.GetString() ?? string.Empty);
            }
        }
        replayCache = table;
        return table;
    }
}
