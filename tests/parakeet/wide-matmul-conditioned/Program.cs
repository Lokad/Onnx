using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

if (args.Length < 4 || args[0] is not ("capture" or "probe")) throw new ArgumentException("capture|probe root manifest new-output [ordinal]");
using var process = Process.GetCurrentProcess();
Support.Require(process.ProcessorAffinity.ToInt64() == 4 && Environment.ProcessorCount == 1, "CPU2 before CLR startup");
Support.Require(Support.Sha(typeof(ComputationalGraph).Assembly.Location) == "d1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4", "Frozen core");
Support.Require(!Directory.Exists(args[3]), "Output exists"); Directory.CreateDirectory(args[3]);
if (args[0] == "capture") Capture.Run(args[1], args[2], args[3]);
else Probe.Run(args[1], args[2], args[3], int.Parse(args[4]));
return 0;

static class Support
{
    public static readonly JsonSerializerOptions Json = new() { WriteIndented = true };
    public static string Sha(string p) { using var f = File.OpenRead(p); return Convert.ToHexStringLower(SHA256.HashData(f)); }
    public static string Hash(ReadOnlySpan<byte> p) => Convert.ToHexStringLower(SHA256.HashData(p));
    public static void Require(bool b, string s) { if (!b) throw new InvalidDataException(s); }
    public static JsonElement Read(string p) { using var d = JsonDocument.Parse(File.ReadAllBytes(p)); return d.RootElement.Clone(); }
    public static void Write(string p, object v) { using var f = new FileStream(p, FileMode.CreateNew); JsonSerializer.Serialize(f, v, Json); }
    public static byte[] Bytes(ITensor t) => t switch {
        Tensor<float> f => MemoryMarshal.AsBytes(f.ToArray().AsSpan()).ToArray(),
        Tensor<long> l => MemoryMarshal.AsBytes(l.ToArray().AsSpan()).ToArray(), _ => throw new InvalidDataException("dtype") };
    public static int[] Shape(JsonElement v) => v.GetProperty("shape").EnumerateArray().Select(d => d.GetInt32()).ToArray();
    public static string PathOf(string root, JsonElement v)
    {
        string p = Path.GetFullPath(Path.Combine(root, v.GetProperty("path").GetString()!));
        Require(p.StartsWith(Path.GetFullPath(root)+Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase), "Path escapes root");
        Require(new FileInfo(p).Length == v.GetProperty("bytes").GetInt64() && Sha(p) == v.GetProperty("sha256").GetString(), "File identity"); return p;
    }
    public static object Identity() => new { core_sha256 = Sha(typeof(ComputationalGraph).Assembly.Location),
        runner_sha256 = Sha(typeof(Support).Assembly.Location), runtime = RuntimeInformation.FrameworkDescription,
        affinity = Process.GetCurrentProcess().ProcessorAffinity.ToInt64(), processor_count = Environment.ProcessorCount,
        flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k => k.StartsWith("LOKAD_") || k.StartsWith("DOTNET_") || k.StartsWith("COMPlus_")).ToDictionary(k => k, Environment.GetEnvironmentVariable) };
}
