using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

namespace LayerNormOutput;

internal delegate void Kernel(DenseTensor<float> x, DenseTensor<float> scale, DenseTensor<float>? bias, DenseTensor<float> output, int block, int outer, float epsilon);

internal static partial class Program
{
    internal const string Core = "48ca1d62ee2586d81072b8e347a671013d00abf8fe637c89eff65314e13cc710";
    internal const string Filter = "*LayerNormFloatInto* *WideOutput*";
    internal static readonly string[] Cases = ["e5-8tok", "e5-30tok", "e5-30pad128", "e5-128tok", "e5-512tok"];
    internal static readonly Kernel Product = typeof(Tensor<float>).GetMethod("LayerNormFloatInto", BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<Kernel>();
    static string mode = "";

    public static void Main(string[] args)
    {
        Require(args.Length == 1 && args[0] == "identity" || args.Length == 3 && args[0] is "proof" or "code", "identity | proof/code capture output");
        mode = args[0];
        Require(OperatingSystem.IsWindows() || OperatingSystem.IsLinux(), "Supported host");
        Require(Environment.ProcessorCount == 1 && Affinity() == 4, "CPU2 inherited before runtime startup");
        Require(Hash(typeof(ComputationalGraph).Assembly.Location) == Core, "Exact qualified core");
        Require(Vector.IsHardwareAccelerated && Vector<float>.Count == 8, "Eight-float product vector width");
        NoNative();
        var flags = Settings();
        if (mode == "identity")
        {
            Require(flags.Count == 0, "Clean identity settings");
            Console.WriteLine(JsonSerializer.Serialize(Identity())); return;
        }
        Require(OperatingSystem.IsLinux() && Environment.Version.ToString() == "10.0.8" && Vector512.IsHardwareAccelerated && Avx512F.IsSupported, "Actual qualified AMD runtime and AVX512 required");
        string capture = Path.GetFullPath(args[1]), output = Path.GetFullPath(args[2]);
        if (mode == "code")
        {
            Require(flags.Count == 2 && flags.GetValueOrDefault("COMPlus_JitDisasm") == Filter
                && flags.GetValueOrDefault("COMPlus_JitStdOutFile") == Path.Combine(Path.GetDirectoryName(output)!, "jit.txt"), "Exactly the declared disassembly settings");
        }
        else Require(flags.Count == 0, "Clean proof settings");
        Proof(capture, output);
        if (mode == "code") WarmCode(capture, Path.Combine(Path.GetDirectoryName(output)!, "warmup.json"));
        NoNative();
    }

    static void WarmCode(string capture, string path)
    {
        string folder = Path.Combine(capture, "e5-30tok");
        using var metadata = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(folder, "capture.json")));
        var node = metadata.RootElement.GetProperty("nodes")[0];
        int block = node.GetProperty("block").GetInt32(), outer = node.GetProperty("outer").GetInt32();
        float epsilon = node.GetProperty("epsilon").GetSingle();
        var x = new DenseTensor<float>(ReadFloats(Path.Combine(folder, "00-x.f32")), new[] { outer, block });
        var scale = new DenseTensor<float>(ReadFloats(Path.Combine(folder, "00-scale.f32")), new[] { block });
        var bias = node.GetProperty("has_bias").GetBoolean() ? new DenseTensor<float>(ReadFloats(Path.Combine(folder, "00-bias.f32")), new[] { block }) : null;
        var a = new DenseTensor<float>(new[] { outer, block }); var b = new DenseTensor<float>(new[] { outer, block });
        string xHash = Hash(x.ToArray()), sHash = Hash(scale.ToArray()); string? bHash = bias is null ? null : Hash(bias.ToArray());
        long pairs = 0; var watch = Stopwatch.StartNew();
        do { Product(x, scale, bias, a, block, outer, epsilon); Kernels.WideOutput(x, scale, bias, b, block, outer, epsilon); pairs++; } while (watch.Elapsed.TotalSeconds < 3);
        watch.Stop(); string expected = Hash(Path.Combine(folder, "00-y.f32"));
        Require(Hash(a.ToArray()) == expected && Hash(b.ToArray()) == expected, "Code warmup output bits");
        Require(Hash(x.ToArray()) == xHash && Hash(scale.ToArray()) == sHash && (bias is null ? null : Hash(bias.ToArray())) == bHash, "Code warmup inputs changed");
        Write(path, new { passed = true, identity = Identity(), pairs, seconds = watch.Elapsed.TotalSeconds, block, outer, epsilon,
            input_sha256 = xHash, scale_sha256 = sHash, bias_sha256 = bHash, output_sha256 = expected, scope = "Tiered code observation only; no latency comparison" });
    }

    internal static Dictionary<string, string?> Settings() => Environment.GetEnvironmentVariables().Keys.Cast<string>()
        .Where(k => k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase))
        .ToDictionary(k => k, Environment.GetEnvironmentVariable);
    internal static void Require(bool ok, string why) { if (!ok) throw new InvalidDataException(why); }
    internal static string Hash(string path) { using var f = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(f)); }
    internal static string Hash(float[] values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
    internal static void Save(string path, float[] values) { using var f = new FileStream(path, FileMode.CreateNew); f.Write(MemoryMarshal.AsBytes(values.AsSpan())); }
    internal static float[] ReadFloats(string path) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(path)).ToArray();
    internal static void Write(string path, object value) { using var f = new FileStream(path, FileMode.CreateNew); JsonSerializer.Serialize(f, value, new JsonSerializerOptions { WriteIndented = true }); }
    internal static long Affinity()
    {
        if (OperatingSystem.IsWindows() || OperatingSystem.IsLinux()) return Process.GetCurrentProcess().ProcessorAffinity.ToInt64();
        throw new PlatformNotSupportedException();
    }
    internal static object Identity() => new { mode, core_sha256 = Core, probe_sha256 = Hash(typeof(Program).Assembly.Location), runtime = Environment.Version.ToString(),
        affinity = Affinity(), vector_width = Vector<float>.Count, vector512_hardware = Vector512.IsHardwareAccelerated,
        avx512 = Avx512F.IsSupported, settings = Settings() };
    internal static Dictionary<string, object> Inventory(string folder) => Directory.GetFiles(folder, "*", SearchOption.AllDirectories)
        .ToDictionary(p => Path.GetRelativePath(folder, p).Replace('\\', '/'), p => (object)new { bytes = new FileInfo(p).Length, sha256 = Hash(p) });
    static void NoNative() => Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Native ORT loaded");
}
