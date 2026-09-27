using System.Collections;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static class Screen
{
    static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
    static string Hash(ReadOnlySpan<float> values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values)));
    static string FileHash(string path) { using var f = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(f)); }
    record Clock(int iteration, bool warmup, long start, long ticks, long allocated, long copies, long scratch);
    sealed class Fixture
    {
        public required string Name, Kind, AHash, BHash, Expected;
        public required int Index, Batch;
        public required int[] Shape;
        public required DenseTensor<float> A, B;
        public required TensorExecutionOptions Options;
        public required CopyAccountant Copy;
        public required ScratchAccountant Scratch;
        public required Tensor<float>[] Returned;
        public Tensor<float>? Held, Last;
        public long SetupTicks;
        public readonly List<Clock> Clocks = new(780);
    }
    static void Verify(Fixture f, Tensor<float> value)
    {
        Require(value.Dimensions.SequenceEqual(f.Shape) && Hash(value.ToArray()) == f.Expected, "Exact output " + f.Name);
    }
    static void Main(string[] args)
    {
        Require(OperatingSystem.IsLinux() && args.Length == 4, "Linux: base role sequence output");
        string folder = Path.GetFullPath(args[0]), role = args[1]; int sequence = int.Parse(args[2]);
        Require(new[] { "current", "candidate", "candidate", "current" }[sequence] == role, "Order");
        Require(Environment.Version.ToString() == "10.0.8" && Environment.ProcessorCount == 1
            && Process.GetCurrentProcess().ProcessorAffinity == (nint)4 && Fma.IsSupported && Avx512F.IsSupported, "Runtime/CPU2/hardware");
        var flags = Environment.GetEnvironmentVariables().Cast<DictionaryEntry>()
            .Where(e => new[] { "LOKAD_", "DOTNET_", "COMPlus_" }.Any(p => ((string)e.Key).StartsWith(p, StringComparison.OrdinalIgnoreCase)))
            .ToDictionary(e => (string)e.Key, e => (string)e.Value!);
        Require(flags.Count == 0, "Ordinary runtime");
        using var specDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(folder, "spec.json")));
        using var fixtureDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(folder, "fixture.json")));
        using var censusDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(folder, "census.json")));
        var spec = specDoc.RootElement; var capture = fixtureDoc.RootElement;
        string core = FileHash(typeof(Tensor<float>).Assembly.Location);
        Require(core == spec.GetProperty("products").GetProperty(role).GetProperty("sha256").GetString(), "Core identity");
        string model = capture.GetProperty("model").GetString()!;
        Require(FileHash(model) == capture.GetProperty("model_sha256").GetString(), "Original model");
        var graph = OnnxImport.Load(model, 64L * 1024 * 1024) ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
        graph.Prepare();
        var map = (IDictionary)graph.GetType().GetField("PackedWeights", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(graph)!;
        Require(map.Count == 3, "Original prepared map");
        var packed = map.Values.Cast<object>().ToDictionary(v => (string)v.GetType().GetProperty("SourceName")!.GetValue(v)!,
            v => (DenseTensor<float>)v.GetType().GetProperty("Packed")!.GetValue(v)!);
        var packedHashes = packed.ToDictionary(p => p.Key, p => Hash(p.Value.Buffer.Span));
        Require(packedHashes["onnx::MatMul_230"] == capture.GetProperty("packed_sha256").GetString(), "Prepared B identity");
        string aPath = Path.Combine(folder, "projection-a.f32");
        Require(FileHash(aPath) == capture.GetProperty("a_sha256").GetString(), "Captured A identity");
        var capturedA = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(aPath)).ToArray();
        var optionProperty = typeof(TensorExecutionOptions).GetProperty("PackedMatMulWeights", BindingFlags.Instance | BindingFlags.NonPublic)!;
        var fixtures = new List<Fixture>();
        foreach (var item in censusDoc.RootElement.GetProperty("cases").EnumerateArray())
        {
            long start = Stopwatch.GetTimestamp();
            int index = item.GetProperty("index").GetInt32(), n = item.GetProperty("reduction").GetInt32();
            string kind = item.GetProperty("kind").GetString()!, weight = item.GetProperty("weight").GetString()!;
            var b = (DenseTensor<float>)graph.Initializers[weight];
            int[] aShape = item.GetProperty("a_shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            int[] shape = item.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            Require(b.Dimensions.SequenceEqual(new[] { n, shape[^1] }), "Weight shape");
            // Narrow controls use deterministic synthetic activations at the two actual projection geometries.
            float[] values = kind == "narrow" ? Enumerable.Range(0, n).Select(i => ((i * 37 + 17) % 101 - 50) * .03125f).ToArray() : (float[])capturedA.Clone();
            var a = new DenseTensor<float>(values, aShape);
            var mode = kind == "scalar" ? TensorExecutionOptions.Scalar : kind == "simd" ? TensorExecutionOptions.Simd : TensorExecutionOptions.Auto;
            string expected = Hash(Tensor<float>.MatMul(a, b, mode).ToArray());
            if (kind is "target" or "unmapped") Require(expected == capture.GetProperty("output_sha256").GetString(), "Captured output");
            if (weight == "onnx::MatMul_230") Require(Hash(b.Buffer.Span) == capture.GetProperty("b_sha256").GetString(), "Original B identity");
            var copy = new CopyAccountant(); var scratch = new ScratchAccountant();
            object boxed = mode with { CopyReporter = copy, ScratchReporter = scratch };
            if (kind != "unmapped") optionProperty.SetValue(boxed, map);
            Require(ReferenceEquals(optionProperty.GetValue(boxed), kind == "unmapped" ? null : map), "Map identity");
            int batch = item.GetProperty("batch").GetInt32();
            fixtures.Add(new Fixture { Index = index, Name = item.GetProperty("name").GetString()!, Kind = kind,
                A = a, B = b, Shape = shape, Batch = batch, AHash = Hash(a.Buffer.Span), BHash = Hash(b.Buffer.Span), Expected = expected,
                Options = (TensorExecutionOptions)boxed, Copy = copy, Scratch = scratch, Returned = new Tensor<float>[batch],
                SetupTicks = Stopwatch.GetTimestamp() - start });
        }
        Require(fixtures.Count == 6, "Six fixed cases");
        for (int iteration = 0; iteration < 780; iteration++)
        foreach (var f in fixtures)
        {
            long allocation = GC.GetAllocatedBytesForCurrentThread(), copies = f.Copy.TotalCopyBytes, scratch = f.Scratch.TotalScratchBytes;
            long start = Stopwatch.GetTimestamp();
            for (int call = 0; call < f.Batch; call++) f.Returned[call] = Tensor<float>.MatMul(f.A, f.B, f.Options);
            long ticks = Stopwatch.GetTimestamp() - start;
            allocation = GC.GetAllocatedBytesForCurrentThread() - allocation;
            Require(ticks > 0, "Clock");
            f.Clocks.Add(new(iteration, iteration < 600, start, ticks, allocation, f.Copy.TotalCopyBytes - copies, f.Scratch.TotalScratchBytes - scratch));
            foreach (var result in f.Returned) Require(result is DenseTensor<float> && result.Length == f.Shape[^1], "Public result");
            f.Last = f.Returned[^1];
            if (iteration == 0) f.Held = f.Returned[0];
            if (iteration is 0 or 599 or 779) foreach (var result in f.Returned) Verify(f, result);
        }
        Require(packed.All(p => Hash(p.Value.Buffer.Span) == packedHashes[p.Key]), "Immutable packed weights");
        var rows = new List<object>();
        foreach (var f in fixtures)
        {
            Require(Hash(f.A.Buffer.Span) == f.AHash && Hash(f.B.Buffer.Span) == f.BHash, "Immutable inputs");
            Require(f.Held != null && f.Last != null && !ReferenceEquals(f.Held, f.Last), "Owned result");
            Verify(f, f.Held!); f.A.Buffer.Span.Fill(91); ((DenseTensor<float>)f.Last!).Buffer.Span.Fill(-7); Verify(f, f.Held!);
            rows.Add(new { index = f.Index, name = f.Name, kind = f.Kind, shape = f.Shape, batch = f.Batch,
                setup_ticks = f.SetupTicks, input_sha256 = f.AHash, weight_sha256 = f.BHash, output_sha256 = f.Expected,
                exact = true, inputs = true, ownership = true, clocks = f.Clocks });
        }
        long batches = fixtures.Sum(f => (long)f.Batch);
        using var output = new FileStream(args[3], FileMode.CreateNew);
        JsonSerializer.Serialize(output, new { passed = true, protocol = "decoder-packed-row-public-600-180-v1", role, sequence,
            runtime = Environment.Version.ToString(), pid = Environment.ProcessId, flags, core_sha256 = core,
            assembly = FileHash(Assembly.GetExecutingAssembly().Location), census_sha256 = FileHash(Path.Combine(folder, "census.json")),
            frequency = Stopwatch.Frequency, samples = 780 * 6, measured_samples = 180 * 6, warmup_samples = 600 * 6,
            calls = 780 * batches, warmups = 600 * batches, measured = 180 * batches, rows }, new JsonSerializerOptions { WriteIndented = true });
        GC.KeepAlive(graph);
    }
}
