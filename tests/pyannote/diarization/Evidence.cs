using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Lokad.Onnx;
using Lokad.Onnx.Tests.Support;

internal sealed class Evidence
{
    internal string Reference { get; }
    internal string Output { get; }
    internal string ManifestSha { get; }
    internal Dictionary<string, string> Models { get; } = new();
    internal static string Sha(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
    static void Require(bool value, string message) { if (!value) throw new InvalidDataException(message); }
    static string Resource(string name)
    {
        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(name) ?? throw new InvalidDataException(name);
        using var reader = new StreamReader(stream); return reader.ReadToEnd().Replace("\r\n", "\n", StringComparison.Ordinal);
    }
    internal static object Assemblies() => new[] { typeof(Community1Diarizer).Assembly, typeof(Tensor<float>).Assembly, Assembly.GetExecutingAssembly() }
        .ToDictionary(a => a.GetName().Name ?? "unknown", a => new { sha256 = Sha(a.Location), version = a.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion });
    internal Evidence(string[] args)
    {
        if (args.Length != 6) throw new ArgumentException("Usage: REFERENCE SEGMENTATION.onnx ENCODER.onnx PROJECTION.onnx PLDA.json NEW_OUTPUT");
        Reference = Path.GetFullPath(args[0]); Output = Path.GetFullPath(args[5]);
        if (File.Exists(Output) || Directory.Exists(Output)) throw new IOException("Choose a fresh output.");
        using var pinDoc = JsonDocument.Parse(Resource("pins.json")); var pins = pinDoc.RootElement;
        string path = Path.Combine(Reference, "manifest.json"); ManifestSha = Sha(path);
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path)); var m = doc.RootElement;
        Require(JsonElement.DeepEquals(pins, m.GetProperty("pins")), "Reference pin identity");
        foreach (string name in new[] { "pins.json", "generate_reference.py", "native_rules.py", "evidence.py" })
            Require(m.GetProperty("recipes").GetProperty(name).GetString() == Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(Resource(name)))), "Reference recipe identity: " + name);
        var keys = new[] { "segmentation", "encoder", "projection", "plda" };
        for (int i = 0; i < keys.Length; i++)
        {
            Models.Add(keys[i], Path.GetFullPath(args[i + 1]));
            Require(Sha(Models[keys[i]]) == pins.GetProperty("models").GetProperty(keys[i]).GetString(), "Model digest: " + keys[i]);
        }
        Require(JsonElement.DeepEquals(m.GetProperty("models"), pins.GetProperty("models")), "Reference models");
        var layout = pins.GetProperty("layout"); var files = m.GetProperty("files");
        Require(files.EnumerateObject().Select(p => p.Name).ToHashSet().SetEquals(layout.EnumerateObject().Select(p => p.Name)), "Reference file coverage");
        foreach (var file in files.EnumerateObject())
        {
            Require(file.Name == Path.GetFileName(file.Name) && !file.Name.Contains('/') && !file.Name.Contains('\\'), "Unsafe reference path");
            path = Path.Combine(Reference, file.Name); var meta = file.Value; var expected = layout.GetProperty(file.Name);
            Require(new FileInfo(path).Length == meta.GetProperty("bytes").GetInt64() && Sha(path) == meta.GetProperty("sha256").GetString(), "Reference digest");
            Require(meta.GetProperty("dtype").GetString() == expected.GetProperty("dtype").GetString()
                && JsonElement.DeepEquals(meta.GetProperty("shape"), expected.GetProperty("shape")), "Reference layout");
        }
        var cases = m.GetProperty("cases").EnumerateArray().ToArray(); var expectedCases = pins.GetProperty("cases").EnumerateArray().ToArray();
        Require(cases.Length == expectedCases.Length, "Case coverage");
        var used = new HashSet<string>();
        for (int i = 0; i < cases.Length; i++)
        {
            var c = cases[i]; var e = expectedCases[i]; string name = e.GetProperty("name").GetString() ?? "";
            Require(c.GetProperty("name").GetString() == name, "Case order"); string pcm = name + "-pcm.npy";
            Require(c.GetProperty("pcm").GetString() == pcm && Sha(Path.Combine(Reference, pcm)) == e.GetProperty("pcm_sha256").GetString(), "PCM identity"); used.Add(pcm);
            var data = NpySupport.ReadFloat32(Path.Combine(Reference, pcm)); int samples = e.GetProperty("samples").GetInt32();
            Require(data.Shape.SequenceEqual(new[] { samples }) && data.Values.All(v => float.IsFinite(v) && Math.Abs(v) <= 1), "PCM contract");
            var windows = c.GetProperty("windows").EnumerateArray().ToArray();
            Require(c.GetProperty("seconds").GetDouble() == samples / 16000d && windows.Length == e.GetProperty("windows").GetInt32(), "Recording geometry");
            for (int w = 0; w < windows.Length; w++)
            {
                var stages = name == "silence" ? new[] { "scores", "activity" } : new[] { "scores", "activity", "features", "encoded", "masks", "pooled", "vectors" };
                Require(windows[w].EnumerateObject().Select(p => p.Name).ToHashSet().SetEquals(stages), "Stage coverage");
                foreach (string stage in stages)
                {
                    string file = $"{name}-{w}-{stage}.npy";
                    Require(windows[w].GetProperty(stage).GetString() == file, "Stage reference"); used.Add(file);
                }
            }
            foreach (string stage in new[] { "count", "centroids" }.Concat(name == "silence" ? Array.Empty<string>() : new[] { "labels", "original_labels", "ordinary_native_frames", "ordinary_frames", "exclusive_native_frames", "exclusive_frames" }))
            {
                string file = name + "-" + stage.Replace('_', '-') + ".npy";
                Require(c.GetProperty(stage).GetString() == file, "Case reference"); used.Add(file);
            }
        }
        Require(used.SetEquals(layout.EnumerateObject().Select(p => p.Name)), "Unreferenced fixture");
    }
}
