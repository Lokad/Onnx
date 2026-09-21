using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static class Probe
{
    const string Core = "0d098ba5fd3fd8799bb1dd018148123f296802c769db5f6148a1a5fa9d80118e";
    const float Guard = -12345.25f;
    static void Require(bool value, string message) { if (!value) throw new InvalidDataException(message); }
    static string Hash(float[] values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
    static string FileHash(string file) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(file)));
    static float[] Data(int count, uint seed)
    {
        var result = new float[count + 6]; Array.Fill(result, Guard);
        for (int i = 0; i < count; i++)
        {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            result[i + 3] = i % 23 == 0 ? -0f : ((seed >> 8) / 8388608f - 1f) * (i % 17 == 0 ? 1e-10f : .2f);
        }
        return result;
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    static float Multiply(float a, float b) => a * b;

    // Every output retains the original j-ascending recurrence. Both raw
    // kernels remain in the exact qualified assembly; only row assignment changes.
    static unsafe void Consume(bool candidate, int m, int n, int k, float* a, float* p, float* c)
    {
        if (!candidate || n < 64) { MathOps.mm_unsafe_vectorized_intrinsics_2x4packed_bump(m, n, k, a, p, c); return; }
        int rows = m / 6 * 6;
        if (rows != 0) MathOps.mm_unsafe_vectorized_intrinsics_3x4packed(rows, n, k, a, p, c);
        if (rows != m) MathOps.mm_unsafe_vectorized_intrinsics_2x4packed_bump(m - rows, n, k, a + rows * n, p, c + rows * k);
    }
    static unsafe object Validate(int m, int n, int k, bool nonzero)
    {
        var a = Data(m * n, 1234567); var b = Data(n * k, 9876543);
        var p = Data(n * k, 33); var seed = Data(m * k, 444);
        if (!nonzero) seed.AsSpan(3, m * k).Clear();
        var baseline = (float[])seed.Clone(); var candidate = (float[])seed.Clone();
        var expected = (float[])seed.Clone();
        string ah = Hash(a), bh = Hash(b);
        fixed (float* aa = a, bb = b, pp = p, cc = baseline, dd = candidate)
        {
            MathOps.PackPanelsB(n, k, bb + 3, pp + 3);
            string ph = Hash(p);
            Consume(false, m, n, k, aa + 3, pp + 3, cc + 3);
            Consume(true, m, n, k, aa + 3, pp + 3, dd + 3);
            Require(ph == Hash(p), "Packed input changed");
        }
        for (int i = 0; i < m; i++) for (int col = 0; col < k; col++)
        {
            float sum = expected[3 + i * k + col];
            for (int j = 0; j < n; j++)
                sum = col < k - k % 8 ? MathF.FusedMultiplyAdd(b[3 + j * k + col], a[3 + i * n + j], sum)
                    : sum + Multiply(a[3 + i * n + j], b[3 + j * k + col]);
            expected[3 + i * k + col] = sum;
        }
        Require(ah == Hash(a) && bh == Hash(b), "Original input changed");
        for (int i = 0; i < expected.Length; i++)
            Require(BitConverter.SingleToInt32Bits(baseline[i]) == BitConverter.SingleToInt32Bits(candidate[i])
                && BitConverter.SingleToInt32Bits(candidate[i]) == BitConverter.SingleToInt32Bits(expected[i]),
                $"Bits/guard m={m} n={n} k={k} seed={nonzero} index={i}");
        for (int i = 0; i < 3; i++) Require(p[i] == Guard && p[p.Length - 1 - i] == Guard, "Packing guard");
        return new { m, n, k, nonzero, values = m * k, digest = Hash(candidate), passed = true };
    }
    static unsafe void Iterations(bool candidate, int count, int m, int n, int k, float[] a, float[] b, float[] p, float[] c)
    {
        fixed (float* aa = a, bb = b, pp = p, cc = c)
        for (int i = 0; i < count; i++)
        {
            c.AsSpan(3, m * k).Clear();
            MathOps.PackPanelsB(n, k, bb + 3, pp + 3);
            Consume(candidate, m, n, k, aa + 3, pp + 3, cc + 3);
        }
    }
    static int Main(string[] args)
    {
        Require(args.Length == 3, "shapes mode output");
        Require(Fma.IsSupported, "FMA required");
        Require(FileHash(typeof(MathOps).Assembly.Location) == Core, "Qualified Core mismatch");
        var flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k =>
            k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)
            || k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)).ToArray();
        Require(flags.Length == 0 && Environment.ProcessorCount == 1, "Runtime environment");
        Require(args[1] is "validate" or "baseline" or "candidate", "Mode");
        using var shapesDoc = JsonDocument.Parse(File.ReadAllText(args[0]));
        var shapes = shapesDoc.RootElement.GetProperty("shapes").EnumerateArray().ToArray();
        var records = new List<object>();
        if (args[1] == "validate")
        {
            foreach (int m in new[] { 0, 2, 4, 6, 8, 10, 16, 32, 64 })
            foreach (int n in new[] { 0, 1, 9, 31, 64 })
            foreach (int k in new[] { 0, 1, 7, 8, 15, 31, 32, 33, 63, 64, 65 })
            foreach (bool nonzero in new[] { false, true }) records.Add(Validate(m, n, k, nonzero));
            foreach (var shape in shapes)
                foreach (bool nonzero in new[] { false, true }) records.Add(Validate(shape.GetProperty("m").GetInt32(), shape.GetProperty("n").GetInt32(), shape.GetProperty("k").GetInt32(), nonzero));
        }
        else foreach (var shape in shapes)
        {
            int m = shape.GetProperty("m").GetInt32(), n = shape.GetProperty("n").GetInt32(), k = shape.GetProperty("k").GetInt32();
            int count = shape.GetProperty("iterations").GetInt32();
            bool candidate = args[1] == "candidate";
            var a = Data(m * n, 1234567); var b = Data(n * k, 9876543); var p = Data(n * k, 33); var c = Data(m * k, 444);
            string ah = Hash(a), bh = Hash(b);
            var warm = Stopwatch.StartNew(); int warmCalls = 0;
            do { Iterations(candidate, 16, m, n, k, a, b, p, c); warmCalls += 16; } while (warm.Elapsed.TotalSeconds < 1);
            double warmSeconds = warm.Elapsed.TotalSeconds;
            using var process = Process.GetCurrentProcess();
            for (int block = 0; block < 6; block++)
            {
                process.Refresh(); var cpu = process.TotalProcessorTime; long start = Stopwatch.GetTimestamp();
                Iterations(candidate, count, m, n, k, a, b, p, c);
                double seconds = Stopwatch.GetElapsedTime(start).TotalSeconds;
                process.Refresh(); double cpuSeconds = (process.TotalProcessorTime - cpu).TotalSeconds;
                records.Add(new { m, n, k, block, iterations = count, seconds, cpu_seconds = cpuSeconds,
                    warm_calls = warmCalls, warm_seconds = warmSeconds, digest = Hash(c) });
            }
            Require(Hash(a) == ah && Hash(b) == bh, "Timing input changed");
            var reference = Data(m * k, 444);
            Iterations(false, 1, m, n, k, a, b, p, reference);
            Require(Hash(reference) == Hash(c), "Timing result/guards differ");
        }
        Require(!File.Exists(args[2]), "Refuse overwrite");
        File.WriteAllText(args[2], JsonSerializer.Serialize(new { passed = true, mode = args[1], records,
            runtime = Environment.Version.ToString(), flags, processor_count = Environment.ProcessorCount,
            core = Core, executable = FileHash(typeof(Probe).Assembly.Location), shapes = FileHash(args[0]),
            pid = Environment.ProcessId, fma = Fma.IsSupported, avx512 = Avx512F.IsSupported }, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"{args[1]} passed, records={records.Count}");
        return 0;
    }
}
