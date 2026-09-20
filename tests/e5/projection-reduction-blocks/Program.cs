using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using ReductionProbe;

static string Hash(float[] values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
static void Require(bool condition, string message) { if (!condition) throw new InvalidOperationException(message); }
static unsafe bool Invoke(int m, int n, int k, float* a, float* p, float* c, int block) =>
    block == 0 ? Original.TryPackedAvx512Rows(m, n, k, a, p, c) : Blocked.Run(m, n, k, a, p, c, block);

Require(args.Length == 1, "Expected --host or new output directory.");
if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux()) throw new PlatformNotSupportedException("Affinity proof requires Windows or Linux.");
int refusals = 0;
unsafe
{
    foreach (var (m, n, k) in new[] { (7, 384, 384), (8, 0, 384), (8, 384, 31), (8, 384, 0), (14, -1, 32), (14, 5, -32), (0, 5, 32), (14, 5, 33) })
        foreach (int block in new[] { 0, 128, 256 })
        { Require(!Invoke(m, n, k, null, null, null, block), "Invalid geometry accepted"); refusals++; }
    foreach (int block in new[] { -1, 0, 1, 127, 129, 255, 257 })
    { Require(!Blocked.Run(12, 384, 384, null, null, null, block), "Invalid block accepted"); refusals++; }
}
bool supported = Avx512F.IsSupported && Fma.IsSupported;
if (args[0] == "--host")
{
    if (!supported) unsafe
    {
        foreach (int block in new[] { 0, 128, 256 })
        { Require(!Invoke(128, 384, 384, null, null, null, block), "Unsupported host accepted"); refusals++; }
    }
    Console.WriteLine(JsonSerializer.Serialize(new { supported, refusals, runtime = Environment.Version.ToString() }));
    return;
}
Require(supported, "AVX-512/FMA target required; local refusal is not numerical proof.");
Require(Environment.ProcessorCount == 1, "Affinity before runtime startup required.");
string output = Path.GetFullPath(args[0]); Require(!Directory.Exists(output), "Output exists."); Directory.CreateDirectory(output);
var checks = new List<object>(); int scalarValues = 0;
unsafe void Check(int m, int n, int k, bool exceptional)
{
    const int guard = 5; const float sentinel = -12345.5f;
    var random = new Random(17 + m * 123 + n + k);
    var a = Enumerable.Repeat(sentinel, m * n + 2 * guard).ToArray(); var b = new float[n * k];
    var p = Enumerable.Repeat(sentinel, n * k + 2 * guard).ToArray(); var initial = Enumerable.Repeat(sentinel, m * k + 2 * guard).ToArray();
    for (int i = 0; i < m * n; i++) a[guard + i] = random.NextSingle() * 2 - 1;
    for (int i = 0; i < b.Length; i++) b[i] = random.NextSingle() * 2 - 1;
    for (int i = 0; i < m * k; i++) initial[guard + i] = random.NextSingle() * 2 - 1;
    if (exceptional)
    {
        a[guard] = -0f; a[guard + n + 1] = BitConverter.Int32BitsToSingle(0x7fa12345);
        a[guard + 2 * n + 127] = float.Epsilon; b[17] = float.PositiveInfinity; b[^1] = float.NegativeInfinity;
        initial[guard + 1] = -0f;
    }
    string ah = Hash(a), bh = Hash(b); var baseline = initial.ToArray();
    fixed (float* ap = a, bp = b, pp = p, cp = baseline)
    {
        Lokad.Onnx.MathOps.PackPanelsB(n, k, bp, pp + guard); string ph = Hash(p);
        Require(p.Take(guard).Concat(p.Skip(guard + n * k)).All(v => v == sentinel), "Pack guards");
        Require(Invoke(m, n, k, ap + guard, pp + guard, cp + guard, 0), "Original declined");
        Require(baseline.Take(guard).Concat(baseline.Skip(guard + m * k)).All(v => v == sentinel), "Output guards");
        var twice = baseline.ToArray(); fixed (float* target = twice) Invoke(m, n, k, ap + guard, pp + guard, target + guard, 0);
        foreach (int block in new[] { 128, 256 })
        {
            var actual = initial.ToArray(); fixed (float* target = actual) Require(Invoke(m, n, k, ap + guard, pp + guard, target + guard, block), "Candidate declined");
            Require(Hash(actual) == Hash(baseline), $"Bits {m}/{n}/{k} block{block} exceptional{exceptional}");
            for (int row = 0; row < m; row++) foreach (int col in new[] { 0, 15, 16, 31, k - 1 }.Distinct())
            {
                float expected = initial[guard + row * k + col];
                for (int j = 0; j < n; j++) expected = MathF.FusedMultiplyAdd(a[guard + row * n + j], b[j * k + col], expected);
                float value = actual[guard + row * k + col];
                Require(float.IsNaN(expected) ? float.IsNaN(value) : BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(value), "Scalar FMA mismatch"); scalarValues++;
            }
            fixed (float* target = actual) Invoke(m, n, k, ap + guard, pp + guard, target + guard, block);
            Require(Hash(actual) == Hash(twice), "Second accumulation changed");
        }
        Require(Hash(p) == ph && Hash(a) == ah && Hash(b) == bh, "Inputs/packed weights changed");
        string name = $"{checks.Count:0000}.f32";
        using (var stream = new FileStream(Path.Combine(output, name), FileMode.CreateNew)) stream.Write(MemoryMarshal.AsBytes(baseline.AsSpan()));
        checks.Add(new { m, n, k, exceptional, input = ah, weights = bh, packed = ph, file = name, output = Hash(baseline), twice = Hash(twice), values = m * k });
    }
}
foreach (int m in Enumerable.Range(8, 38).Concat(new[] { 64, 128, 512 }))
    foreach (int n in new[] { 1, 127, 128, 129, 255, 256, 257, 383, 384, 385, 513 }) Check(m, n, 96, false);
foreach (int m in new[] { 8, 30, 128, 512 })
    foreach (var (n, k) in new[] { (384, 384), (384, 1536), (1536, 384) }) Check(m, n, k, false);
foreach (int m in new[] { 8, 13, 14, 15, 27, 28, 29, 30, 42, 128 }) Check(m, 385, 192, true);
using var process = Process.GetCurrentProcess();
var result = new { complete = true, checks, scalar_values = scalarValues, refusals, supported,
    runtime = Environment.Version.ToString(), pid = process.Id, processor_count = Environment.ProcessorCount,
    affinity = OperatingSystem.IsLinux() || OperatingSystem.IsWindows() ? process.ProcessorAffinity.ToInt64() : throw new PlatformNotSupportedException(),
    core = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(Lokad.Onnx.MathOps).Assembly.Location))),
    probe = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(Original).Assembly.Location))) };
using (var stream = new FileStream(Path.Combine(output, "result.json"), FileMode.CreateNew)) JsonSerializer.Serialize(stream, result, new JsonSerializerOptions { WriteIndented = true });
Console.WriteLine(JsonSerializer.Serialize(new { passed = true, cases = checks.Count, scalar_values = scalarValues, refusals }));
