using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

sealed class Evidence
{
    public string Reference { get; }
    public string Output { get; }
    public string ManifestSha { get; }
    public Dictionary<string, string> Models { get; } = new();

    public Evidence(string[] args)
    {
        if (args.Length != 6) throw new ArgumentException("Usage: LimitProbe <reference> <segmentation> <encoder> <projection> <plda> <new-output.json>");
        Reference = Path.GetFullPath(args[0]);
        Output = Path.GetFullPath(args[5]);
        if (File.Exists(Output) || Directory.Exists(Output)) throw new IOException("Existing output");
        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("reference-pins.json")
            ?? throw new InvalidDataException("Missing reference pins");
        using var pins = JsonDocument.Parse(stream);
        var root = pins.RootElement;
        ManifestSha = Sha(Path.Combine(Reference, "manifest.json"));
        Require(ManifestSha == root.GetProperty("manifest_sha256").GetString(), "Short manifest identity differs");
        var pcm = root.GetProperty("pcm");
        string name = pcm.GetProperty("name").GetString()!;
        Require(Path.GetFileName(name) == name, "Unsafe PCM name");
        Require(Sha(Path.Combine(Reference, name)) == pcm.GetProperty("sha256").GetString(), "Short PCM identity differs");
        string[] keys = ["segmentation", "encoder", "projection", "plda"];
        for (int i = 0; i < keys.Length; i++)
        {
            string path = Path.GetFullPath(args[i + 1]);
            Require(Sha(path) == root.GetProperty("models").GetProperty(keys[i]).GetString(), "Model identity differs: " + keys[i]);
            Models.Add(keys[i], path);
        }
        Require(Sha(typeof(ComputationalGraph).Assembly.Location) == root.GetProperty("core_sha256").GetString(), "Core differs");
        Require(Sha(typeof(Community1Diarizer).Assembly.Location) == root.GetProperty("data_sha256").GetString(), "Data differs");
        NoOrt();
    }

    static string Sha(string path)
    {
        using var input = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(input));
    }

    static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidDataException(message);
    }

    static void NoOrt()
    {
        foreach (ProcessModule module in Process.GetCurrentProcess().Modules)
            Require(!module.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase), "Native ORT loaded");
    }

    public static object Assemblies()
    {
        NoOrt();
        return new[] { typeof(ComputationalGraph).Assembly, typeof(Community1Diarizer).Assembly, Assembly.GetExecutingAssembly() }
            .ToDictionary(a => a.GetName().Name!, a => new
            {
                sha256 = Sha(a.Location),
                version = a.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
            });
    }
}
