using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static unsafe class Program
{
    static void Require(bool value, string message) { if (!value) throw new InvalidOperationException(message); }
    static string Hash(float[] value) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(value.AsSpan())));
    static string FileHash(string path) { using var f = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(f)); }
    static float[] Read(string path) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(path)).ToArray();
    static void Write(string path, object value)
    {
        using var f = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(f, value, new JsonSerializerOptions { WriteIndented = true });
    }
    static object Output(string folder, string name, float[] values)
    {
        Require(values.All(float.IsFinite), "Nonfinite output");
        string path = Path.Combine(folder, name + ".bin");
        using (var f = new FileStream(path, FileMode.CreateNew)) f.Write(MemoryMarshal.AsBytes(values.AsSpan()));
        return new { file = Path.GetFileName(path), bytes = values.Length * 4, sha256 = Hash(values) };
    }
    static int SelfTest()
    {
        var rng = new Random(67231); int tests = 0;
        foreach (int m in new[] { 2, 4 })
        foreach (int k in new[] { 1, 127, 128, 129, 255, 256, 257, 511, 512, 513 })
        foreach (int n in new[] { 32, 64 })
        {
            float[] a = Enumerable.Range(0, m*k).Select(_ => (float)(rng.NextDouble() - .5)).ToArray();
            float[] b = Enumerable.Range(0, k*n).Select(_ => (float)(rng.NextDouble() - .5)).ToArray();
            float[] packed = new float[b.Length]; string ah = Hash(a), bh = Hash(b);
            fixed (float* ap = a, bp = b, pp = packed)
            {
                MathOps.PackPanelsB(k, n, bp, pp); string ph = Hash(packed);
                foreach (int block in new[] { 128, 256, 512, 4096 })
                {
                    float[] expected = new float[m*n];
                    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++)
                    {
                        float sum = 0;
                        for (int begin = 0; begin < k; begin += block)
                        {
                            float partial = 0;
                            for (int p = begin; p < Math.Min(k, begin + block); p++)
                                partial = MathF.FusedMultiplyAdd(b[p*n+j], a[i*k+p], partial);
                            sum = begin == 0 ? partial : sum + partial;
                        }
                        expected[i*n+j] = sum;
                    }
                    float[] guarded = Enumerable.Repeat(-98765.5f, m*n+32).ToArray();
                    guarded.AsSpan(16, m*n).Fill(float.NaN);
                    fixed (float* cp = guarded) Partial.Run(m, k, n, ap, pp, cp+16, block);
                    Require(MemoryMarshal.AsBytes(guarded.AsSpan(16, m*n)).SequenceEqual(MemoryMarshal.AsBytes(expected.AsSpan())), "Scalar arithmetic mismatch");
                    Require(guarded.AsSpan(0, 16).IndexOfAnyExcept(-98765.5f) < 0 && guarded.AsSpan(m*n+16, 16).IndexOfAnyExcept(-98765.5f) < 0, "Canary changed");
                    Require(Hash(packed) == ph && Hash(a) == ah && Hash(b) == bh, "Input changed"); tests++;
                }
            }
        }
        return tests;
    }
    static void Main(string[] args)
    {
        Require(args.Length == 2 && Fma.IsSupported, "Expected prepared JSON, output directory and FMA hardware");
        Require(Environment.Version.ToString() == "10.0.12" && Environment.ProcessorCount == 1, "Runtime/CPU count");
        Require((long)Process.GetCurrentProcess().ProcessorAffinity == 4, "Affinity");
        Require(!Environment.GetEnvironmentVariables().Keys.Cast<string>().Any(k => k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)), "Runtime override");
        var spec = JsonDocument.Parse(File.ReadAllText(args[0])).RootElement;
        string root = spec.GetProperty("root").GetString()!, output = args[1]; Directory.CreateDirectory(output);
        string core = typeof(MathOps).Assembly.Location;
        Require(FileHash(core) == "d1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4", "Frozen core mismatch");
        int tests = SelfTest(); float[] b = Read(Path.Combine(root, spec.GetProperty("weight").GetString()!));
        float[] bias = Read(Path.Combine(root, spec.GetProperty("bias").GetString()!));
        Require(b.Length == 4096*1024 && bias.Length == 1024, "Weight geometry");
        string bh = Hash(b), biasHash = Hash(bias); float[] packed = new float[b.Length];
        var records = new List<object>(); var held = new List<(float[] Value, string Hash)>();
        fixed (float* bp = b, pp = packed)
        {
            MathOps.PackPanelsB(4096, 1024, bp, pp); string ph = Hash(packed);
            foreach (var route in spec.GetProperty("routes").EnumerateArray())
            {
                string id = route.GetProperty("id").GetString()!; string folder = Path.Combine(output, id); Directory.CreateDirectory(folder);
                float[] a = Read(Path.Combine(root, route.GetProperty("input").GetString()!));
                Require(a.Length == 74*4096, "Input geometry"); string ah = Hash(a);
                float[] baseline = new float[74*1024];
                fixed (float* ap = a, cp = baseline) MathOps.mm_unsafe_vectorized_intrinsics_2x4packed_bump(74, 4096, 1024, ap, pp, cp);
                records.Add(new { route = id, block = 0, projection = Output(folder, "baseline-projection", baseline), stem = Output(folder, "baseline-stem", AddBias(baseline, bias)) });
                foreach (int block in new[] { 128, 256, 512, 4096 })
                {
                    float[] actual = new float[74*1024], repeat = new float[74*1024];
                    actual.AsSpan().Fill(float.NaN); repeat.AsSpan().Fill(6789.5f);
                    fixed (float* ap = a, cp = actual, rp = repeat)
                    {
                        Partial.Run(74, 4096, 1024, ap, pp, cp, block);
                        Partial.Run(74, 4096, 1024, ap, pp, rp, block);
                    }
                    Require(Hash(actual) == Hash(repeat), "Repeat mismatch");
                    if (block == 4096) Require(Hash(actual) == Hash(baseline), "One-block baseline mismatch");
                    var stem = AddBias(actual, bias); held.Add((actual, Hash(actual))); held.Add((stem, Hash(stem)));
                    records.Add(new { route = id, block, projection = Output(folder, block+"-projection", actual), stem = Output(folder, block+"-stem", stem) });
                }
                Require(Hash(a) == ah, "Input mutation"); Console.WriteLine(id + " complete");
            }
            Require(Hash(packed) == ph, "Packed weight mutation");
        }
        Require(Hash(b) == bh && Hash(bias) == biasHash && held.All(v => Hash(v.Value) == v.Hash), "Ownership mutation");
        Write(Path.Combine(output, "result.json"), new { passed = true, prepared_sha256 = FileHash(args[0]), core_sha256 = FileHash(core),
            runner_sha256 = FileHash(typeof(Program).Assembly.Location), runtime = Environment.Version.ToString(), affinity = 4, processor_count = 1,
            fma = Fma.IsSupported, avx512 = Avx512F.IsSupported, selftests = tests, records, ownership = true, repeats = 16,
            scope = "Finite projection arithmetic only; no model execution, timing or generic kernel promotion" });
    }
    static float[] AddBias(float[] values, float[] bias)
    {
        float[] result = new float[values.Length];
        for (int i = 0; i < values.Length; i++) result[i] = values[i] + bias[i % bias.Length];
        return result;
    }
}
