using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

namespace LayerNormOutput;

internal delegate void Kernel(DenseTensor<float> x, DenseTensor<float> scale, DenseTensor<float>? bias, DenseTensor<float> output, int block, int outer, float epsilon);

internal static class Program
{
    const string Core = "48ca1d62ee2586d81072b8e347a671013d00abf8fe637c89eff65314e13cc710";
    static readonly string[] Variants = ["Product", "CopyA", "CopyB", "Wide"];
    static readonly Kernel Product = (typeof(Tensor<float>).GetMethod("LayerNormFloatInto", BindingFlags.Static | BindingFlags.NonPublic)
        ?? throw new MissingMethodException("LayerNormFloatInto")).CreateDelegate<Kernel>();
    static readonly Kernel[] Methods = [Product, Kernels.Copy, Kernels.Copy, Kernels.WideOutput];
    sealed record Node(DenseTensor<float> X, DenseTensor<float> Scale, DenseTensor<float>? Bias, int Width, int Rows, float Epsilon);
    sealed record Bank(string Name, int Repeats, bool Diagnostic, Node[] Nodes, DenseTensor<float>[] Outputs, string? Expected);
    sealed record Sample(int cycle, int position, string variant, int repeats, long ticks, long allocated, int[] gc, string output_sha256);
    static string mode = "";

    public static void Main(string[] args)
    {
        Require(args.Length == 1 && args[0] == "identity" || args.Length == 3 && args[0] == "inspect" || args.Length == 5 && args[0] == "run", "identity | inspect origin banks.json | run origin banks.json output visit");
        mode = args[0];
        Require(Environment.ProcessorCount == 1 && Affinity() == 4 && Vector<float>.Count == 8 && Vector.IsHardwareAccelerated, "CPU2 and eight-float vectors");
        Require(Hash(typeof(ComputationalGraph).Assembly.Location) == Core && Settings().Count == 0, "Exact core and clean runtime");
        NoNative();
        if (mode == "identity") { Console.WriteLine(JsonSerializer.Serialize(Identity())); return; }
        if (mode == "inspect")
        {
            using var inspection = JsonDocument.Parse(File.ReadAllBytes(args[2]));
            var banks = inspection.RootElement.EnumerateArray().Select(d => Load(args[1], d)).Select(b => new {
                name = b.Name, repeats = b.Repeats, diagnostic = b.Diagnostic, width = b.Nodes[0].Width, rows = b.Nodes[0].Rows,
                input_sha256 = Hash(b.Nodes.Select(n => n.X)), scale_sha256 = Hash(b.Nodes.Select(n => n.Scale)),
                bias_sha256 = Hash(b.Nodes.Where(n => n.Bias is not null).Select(n => n.Bias!)), output_sha256 = b.Expected }).ToArray();
            Console.WriteLine(JsonSerializer.Serialize(new { identity = Identity(), banks })); return;
        }
        Require(OperatingSystem.IsLinux() && Environment.Version.ToString() == "10.0.8" && Avx512F.IsSupported && Vector512.IsHardwareAccelerated, "Qualified AMD runtime");
        int visit = int.Parse(args[4]); Require(visit >= 0 && visit < 4 && !Directory.Exists(args[3]), "Fresh output and fixed visit");
        Directory.CreateDirectory(args[3]);
        using var metadata = JsonDocument.Parse(File.ReadAllBytes(args[2]));
        var definitions = metadata.RootElement.EnumerateArray().Select(x => x.Clone()).ToArray();
        Require(definitions.Length == 9, "Nine fixed banks");
        int[] order = Enumerable.Range(0, 9).Select(i => (i + visit) % 9).ToArray();
        if (visit % 2 == 1) Array.Reverse(order);
        var results = new object[9];
        foreach (int index in order)
        {
            var bank = Load(args[1], definitions[index]);
            string input = Hash(bank.Nodes.Select(x => x.X)), scale = Hash(bank.Nodes.Select(x => x.Scale));
            string bias = Hash(bank.Nodes.Where(x => x.Bias is not null).Select(x => x.Bias!));
            var first = new Sample[4];
            first[0] = Measure(Product, bank, 1, -1, 0, "Product");
            string expected = bank.Expected ?? first[0].output_sha256;
            Require(first[0].output_sha256 == expected, "First actual product matches original capture");
            var held = bank.Outputs;
            if (bank.Diagnostic) Save(Path.Combine(args[3], bank.Name + "-reference.f32"), held);
            bank = bank with { Outputs = bank.Nodes.Select(n => new DenseTensor<float>(new[] { n.Rows, n.Width })).ToArray() };
            for (int variant = 1; variant < 4; variant++)
            {
                first[variant] = Measure(Methods[variant], bank, 1, -1, variant, Variants[variant]);
                Require(first[variant].output_sha256 == expected, "First variant output bits");
            }
            var warmup = new Sample[64]; var measured = new Sample[192];
            for (int phase = 0; phase < 2; phase++)
            {
                int cycles = phase == 0 ? 16 : 48;
                for (int cycle = 0; cycle < cycles; cycle++) for (int position = 0; position < 4; position++)
                {
                    int variant = (visit + cycle + position) % 4;
                    var sample = Measure(Methods[variant], bank, bank.Repeats, cycle, position, Variants[variant]);
                    Require(sample.output_sha256 == expected, "Complete timed bank differs: " + bank.Name + "/" + Variants[variant]);
                    (phase == 0 ? warmup : measured)[cycle * 4 + position] = sample;
                }
            }
            Require(Hash(held) == expected && Hash(bank.Nodes.Select(x => x.X)) == input && Hash(bank.Nodes.Select(x => x.Scale)) == scale
                && Hash(bank.Nodes.Where(x => x.Bias is not null).Select(x => x.Bias!)) == bias, "Held output/input/parameters changed");
            results[index] = new { name = bank.Name, repeats = bank.Repeats, diagnostic = bank.Diagnostic, nodes = 25,
                width = bank.Nodes[0].Width, rows = bank.Nodes[0].Rows, values = bank.Nodes.Sum(n => n.Width * n.Rows),
                input_sha256 = input, scale_sha256 = scale, bias_sha256 = bias, output_sha256 = expected, held_unchanged = true,
                first, warmup, measured };
            Console.WriteLine("Completed complete bank " + bank.Name);
        }
        NoNative();
        Write(Path.Combine(args[3], "timing.json"), new { schema = 1, protocol = "layernorm-complete-bank-v1", visit,
            frequency = Stopwatch.Frequency, identity = Identity(), case_order = order.Select(i => definitions[i].GetProperty("name").GetString()).ToArray(), results });
    }

    static Bank Load(string origin, JsonElement definition)
    {
        string name = definition.GetProperty("name").GetString() ?? throw new InvalidDataException("Name");
        string source = definition.GetProperty("source").GetString() ?? throw new InvalidDataException("Source");
        int width = definition.GetProperty("width").GetInt32(); bool diagnostic = definition.GetProperty("diagnostic").GetBoolean();
        bool withBias = definition.GetProperty("bias").GetBoolean();
        string folder = Path.Combine(origin, "capture", source);
        using var metadata = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(folder, "capture.json")));
        var nodes = new Node[25]; var outputs = new DenseTensor<float>[25];
        using var expected = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Require(metadata.RootElement.GetProperty("nodes").GetArrayLength() == 25, "Complete captured node bank");
        for (int i = 0; i < 25; i++)
        {
            var node = metadata.RootElement.GetProperty("nodes")[i];
            Require(node.GetProperty("index").GetInt32() == i && node.GetProperty("block").GetInt32() == 384
                && node.GetProperty("has_bias").GetBoolean(), "Captured order/width/bias");
            int rows = node.GetProperty("outer").GetInt32();
            string prefix = Path.Combine(folder, i.ToString("D2"));
            float[] x = Map(Read(prefix + "-x.f32"), rows, width), s = Map(Read(prefix + "-scale.f32"), 1, width);
            float[]? b = withBias ? Map(Read(prefix + "-bias.f32"), 1, width) : null;
            nodes[i] = new(new DenseTensor<float>(x, new[] { rows, width }), new DenseTensor<float>(s, new[] { width }),
                b is null ? null : new DenseTensor<float>(b, new[] { width }), width, rows, node.GetProperty("epsilon").GetSingle());
            outputs[i] = new(new[] { rows, width });
            if (!diagnostic) expected.AppendData(File.ReadAllBytes(prefix + "-y.f32"));
        }
        return new(name, definition.GetProperty("repeats").GetInt32(), diagnostic, nodes, outputs,
            diagnostic ? null : Convert.ToHexStringLower(expected.GetHashAndReset()));
    }

    static float[] Map(float[] source, int rows, int width)
    {
        Require(source.Length == rows * 384, "Source dimensions");
        if (width == 384) return source;
        var result = new float[rows * width];
        for (int row = 0; row < rows; row++) for (int col = 0; col < width; col++) result[row * width + col] = source[row * 384 + col % 384];
        return result;
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    static Sample Measure(Kernel kernel, Bank bank, int repeats, int cycle, int position, string variant)
    {
        int g0 = GC.CollectionCount(0), g1 = GC.CollectionCount(1), g2 = GC.CollectionCount(2);
        long allocated = GC.GetAllocatedBytesForCurrentThread(), start = Stopwatch.GetTimestamp();
        for (int repeat = 0; repeat < repeats; repeat++) for (int i = 0; i < bank.Nodes.Length; i++)
        {
            var node = bank.Nodes[i]; kernel(node.X, node.Scale, node.Bias, bank.Outputs[i], node.Width, node.Rows, node.Epsilon);
        }
        long ticks = Stopwatch.GetTimestamp() - start, allocation = GC.GetAllocatedBytesForCurrentThread() - allocated;
        int[] gc = [GC.CollectionCount(0) - g0, GC.CollectionCount(1) - g1, GC.CollectionCount(2) - g2];
        return new(cycle, position, variant, repeats, ticks, allocation, gc, Hash(bank.Outputs));
    }
    static string Hash(IEnumerable<DenseTensor<float>> values)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (var value in values) hash.AppendData(MemoryMarshal.AsBytes(value.Buffer.Span));
        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }
    static string Hash(string path) { using var f = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(f)); }
    static float[] Read(string path) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(path)).ToArray();
    static void Save(string path, IEnumerable<DenseTensor<float>> values) { using var f = new FileStream(path, FileMode.CreateNew); foreach (var v in values) f.Write(MemoryMarshal.AsBytes(v.Buffer.Span)); }
    static void Write(string path, object value) { using var f = new FileStream(path, FileMode.CreateNew); JsonSerializer.Serialize(f, value, new JsonSerializerOptions { WriteIndented = true }); }
    static void Require(bool value, string message) { if (!value) throw new InvalidDataException(message); }
    static long Affinity()
    {
        if (OperatingSystem.IsWindows() || OperatingSystem.IsLinux()) return Process.GetCurrentProcess().ProcessorAffinity.ToInt64();
        throw new PlatformNotSupportedException();
    }
    static Dictionary<string, string?> Settings() => Environment.GetEnvironmentVariables().Keys.Cast<string>()
        .Where(k => k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase))
        .ToDictionary(k => k, Environment.GetEnvironmentVariable);
    static object Identity() => new { mode, core_sha256 = Core, probe_sha256 = Hash(typeof(Program).Assembly.Location), runtime = Environment.Version.ToString(),
        affinity = Affinity(), vector_width = Vector<float>.Count, vector512_hardware = Vector512.IsHardwareAccelerated, avx512 = Avx512F.IsSupported, settings = Settings() };
    static void NoNative() => Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Native ORT loaded");
}
