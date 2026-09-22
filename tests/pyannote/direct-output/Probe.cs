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
    const string Core = "e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838";
    const float Guard = -12345.25f;
    static void Require(bool ok, string message) { if (!ok) throw new InvalidDataException(message); }
    static string Hash(float[] values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
    static string FileHash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
    static float[] Data(int count, uint seed, string pattern)
    {
        float[] specials = [0f, -0f, float.PositiveInfinity, float.NegativeInfinity,
            BitConverter.Int32BitsToSingle(0x7fc12345), float.Epsilon, -float.Epsilon, float.MaxValue, -float.MaxValue, .125f, -.5f];
        var values = new float[count + 6]; Array.Fill(values, Guard);
        for (int i = 0; i < count; i++)
        {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            values[i + 3] = pattern == "special" ? specials[(i + (int)(seed % 11)) % 11]
                : pattern == "zero" ? (i % 2 == 0 ? 0f : -0f)
                : i % 23 == 0 ? -0f : ((seed >> 8) / 8388608f - 1f) * (i % 17 == 0 ? 1e-10f : .2f);
        }
        return values;
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    static float Multiply(float a, float b) => a * b;
    static bool Portable(int m, int n, int k) => m >= 32 && (m & 1) == 0 && m % 3 != 0 && n >= 64 && k > 0
        && (long)n * k <= 67108864 && (m >= 64 || (long)n * k <= 65536);

    static unsafe void Epilogue(int m, int k, int stride, bool hasBias, float* temp, float* output, float* bias)
    {
        for (int i = 0; i < m; i++)
        {
            float bi = hasBias ? bias[i] : 0f;
            for (int j = 0; j < k; j++) output[i * stride + j] = hasBias ? temp[i * k + j] + bi : temp[i * k + j];
        }
    }
    static unsafe void Work(bool candidate, int m, int n, int k, int stride, bool hasBias,
        float* a, float* b, float* packed, float* temp, float* output, float* bias)
    {
        bool portable = Portable(m, n, k);
        int grouped = portable ? m / 6 * 6 : 0;
        new Span<float>(temp, (candidate && portable ? m - grouped : m) * k).Clear();
        MathOps.PackPanelsB(n, k, b, packed);
        if (candidate && portable)
        {
            DirectOutput.Multiply(grouped, n, k, a, packed, output, stride, bias, hasBias);
            int remainder = m - grouped;
            MathOps.mm_unsafe_vectorized_intrinsics_2x4packed_bump(remainder, n, k, a + grouped * n, packed, temp);
            Epilogue(remainder, k, stride, hasBias, temp, output + grouped * stride, bias + grouped);
        }
        else
        {
            if (grouped != 0) MathOps.mm_unsafe_vectorized_intrinsics_3x4packed(grouped, n, k, a, packed, temp);
            MathOps.mm_unsafe_vectorized_intrinsics_2x4packed_bump(m - grouped, n, k, a + grouped * n, packed, temp + grouped * k);
            Epilogue(m, k, stride, hasBias, temp, output, bias);
        }
    }
    static void Guards(float[] a) { for (int i = 0; i < 3; i++) Require(a[i] == Guard && a[a.Length - 1 - i] == Guard, "Outer guard"); }
    static object Validate(JsonElement shape)
    {
        int m = shape.GetProperty("m").GetInt32(), n = shape.GetProperty("n").GetInt32(), k = shape.GetProperty("k").GetInt32();
        int stride = shape.GetProperty("stride").GetInt32(), start = shape.GetProperty("start").GetInt32();
        bool biasPresent = shape.GetProperty("bias").GetBoolean(); string pattern = shape.GetProperty("pattern").GetString()!;
        Require(start >= 0 && start + k <= stride, "Output geometry");
        var a = Data(m * n, 1234567, pattern); var b = Data(n * k, 9876543, pattern); var bias = Data(m, 777, pattern);
        var p = Data(n * k, 33, "finite"); var temp = Data(m * k, 444, "finite");
        var seed = Data(m * stride, 999, "finite"); var baseline = (float[])seed.Clone(); var candidate = (float[])seed.Clone();
        var expected = (float[])seed.Clone(); string ah = Hash(a), bh = Hash(b), bih = Hash(bias);
        unsafe
        {
            fixed (float* aa = a, bb = b, pp = p, tt = temp, rr = baseline, cc = candidate, bi = bias)
            {
                Work(false, m, n, k, stride, biasPresent, aa + 3, bb + 3, pp + 3, tt + 3, rr + 3 + start, bi + 3);
                string ph = Hash(p); Guards(temp);
                Work(true, m, n, k, stride, biasPresent, aa + 3, bb + 3, pp + 3, tt + 3, cc + 3 + start, bi + 3);
                Require(Hash(p) == ph, "Packed input changed"); Guards(temp); Guards(p);
            }
        }
        for (int i = 0; i < m; i++) for (int col = 0; col < k; col++)
        {
            float sum = 0f;
            for (int j = 0; j < n; j++) sum = col < k - k % 8
                ? MathF.FusedMultiplyAdd(b[3 + j * k + col], a[3 + i * n + j], sum)
                : sum + Multiply(a[3 + i * n + j], b[3 + j * k + col]);
            expected[3 + start + i * stride + col] = biasPresent ? sum + bias[3 + i] : sum;
        }
        int nanValues = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Require(BitConverter.SingleToInt32Bits(candidate[i]) == BitConverter.SingleToInt32Bits(baseline[i]),
                $"Candidate bits m={m} n={n} k={k} stride={stride} start={start} bias={biasPresent} pattern={pattern} index={i} baseline={BitConverter.SingleToInt32Bits(baseline[i]):x8} candidate={BitConverter.SingleToInt32Bits(candidate[i]):x8}");
            if (float.IsNaN(expected[i])) { Require(float.IsNaN(candidate[i]), "Oracle NaN class"); nanValues++; }
            else Require(BitConverter.SingleToInt32Bits(candidate[i]) == BitConverter.SingleToInt32Bits(expected[i]), "Oracle finite/guard bits");
        }
        Require(Hash(a) == ah && Hash(b) == bh && Hash(bias) == bih, "Input/bias changed");
        return new { m, n, k, stride, start, bias = biasPresent, pattern, values = m * k, checked_buffer_values = candidate.Length,
            nan_values = nanValues, digest = Hash(candidate), passed = true };
    }
    static unsafe void Iterations(bool candidate, int count, int m, int n, int k, int stride, int start,
        float[] a, float[] b, float[] p, float[] temp, float[] output, float[] bias)
    {
        fixed (float* aa = a, bb = b, pp = p, tt = temp, cc = output, bi = bias)
        for (int i = 0; i < count; i++) Work(candidate, m, n, k, stride, true, aa + 3, bb + 3, pp + 3, tt + 3, cc + 3 + start, bi + 3);
    }
    static int Main(string[] args)
    {
        Require(args.Length == 3 && Fma.IsSupported, "shapes mode output, FMA required");
        string mode = args[1]; Require(mode is "validate" or "validate-no-avx2" or "baseline" or "candidate", "Mode");
        Require(FileHash(typeof(MathOps).Assembly.Location) == Core, "Selected Core mismatch");
        var flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k => k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase)
            || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)).ToArray();
        Require(Environment.ProcessorCount == 1, "Inherited affinity");
        if (mode == "validate-no-avx2") Require(flags.SequenceEqual(new[] { "DOTNET_EnableAVX2" }) && Environment.GetEnvironmentVariable("DOTNET_EnableAVX2") == "0" && !Avx2.IsSupported, "Scalar-tail flags");
        else Require(flags.Length == 0 && Avx2.IsSupported, "Normal runtime flags");
        using var doc = JsonDocument.Parse(File.ReadAllText(args[0])); var conditioning = new List<object>(); var records = new List<object>();
        if (mode.StartsWith("validate", StringComparison.Ordinal))
            foreach (var c in doc.RootElement.GetProperty("cases").EnumerateArray()) records.Add(Validate(c));
        else
        {
            bool candidate = mode == "candidate";
            foreach (var s in doc.RootElement.GetProperty("shapes").EnumerateArray())
            {
                int m = s.GetProperty("m").GetInt32(), n = s.GetProperty("n").GetInt32(), k = s.GetProperty("k").GetInt32();
                int stride = s.GetProperty("stride").GetInt32(), start = s.GetProperty("timing_start").GetInt32();
                var a = Data(m * n, 1234567, "finite"); var b = Data(n * k, 9876543, "finite"); var p = Data(n * k, 33, "finite");
                var temp = Data(m * k, 444, "finite"); var output = Data(m * stride, 999, "finite"); var bias = Data(m, 777, "finite");
                var timer = Stopwatch.StartNew(); int calls = 0;
                do { Iterations(candidate, 16, m, n, k, stride, start, a, b, p, temp, output, bias); calls += 16; } while (timer.Elapsed.TotalSeconds < 1);
                conditioning.Add(new { m, n, k, stride, start, calls, seconds = timer.Elapsed.TotalSeconds });
            }
            foreach (var s in doc.RootElement.GetProperty("shapes").EnumerateArray())
            {
                int m = s.GetProperty("m").GetInt32(), n = s.GetProperty("n").GetInt32(), k = s.GetProperty("k").GetInt32();
                int stride = s.GetProperty("stride").GetInt32(), start = s.GetProperty("timing_start").GetInt32(), count = s.GetProperty("iterations").GetInt32();
                var a = Data(m * n, 1234567, "finite"); var b = Data(n * k, 9876543, "finite"); var p = Data(n * k, 33, "finite");
                var temp = Data(m * k, 444, "finite"); var output = Data(m * stride, 999, "finite"); var bias = Data(m, 777, "finite");
                string ah = Hash(a), bh = Hash(b), bih = Hash(bias);
                var warm = Stopwatch.StartNew(); int warmCalls = 0;
                do { Iterations(candidate, 16, m, n, k, stride, start, a, b, p, temp, output, bias); warmCalls += 16; } while (warm.Elapsed.TotalSeconds < 1);
                double warmSeconds = warm.Elapsed.TotalSeconds;
                using var process = Process.GetCurrentProcess();
                for (int block = 0; block < 6; block++)
                {
                    process.Refresh(); var cpu = process.TotalProcessorTime; long begin = Stopwatch.GetTimestamp();
                    Iterations(candidate, count, m, n, k, stride, start, a, b, p, temp, output, bias);
                    double seconds = Stopwatch.GetElapsedTime(begin).TotalSeconds;
                    process.Refresh(); double cpuSeconds = (process.TotalProcessorTime - cpu).TotalSeconds;
                    records.Add(new { m, n, k, stride, start, block, iterations = count, seconds, cpu_seconds = cpuSeconds,
                        warm_calls = warmCalls, warm_seconds = warmSeconds, digest = Hash(output) });
                }
                Require(Hash(a) == ah && Hash(b) == bh && Hash(bias) == bih, "Timing input changed"); Guards(temp); Guards(p);
                var reference = Data(m * stride, 999, "finite");
                Iterations(false, 1, m, n, k, stride, start, a, b, p, temp, reference, bias);
                Require(Hash(reference) == Hash(output), "Complete timing output/guards differ");
            }
        }
        Require(!File.Exists(args[2]), "Refuse overwrite");
        File.WriteAllText(args[2], JsonSerializer.Serialize(new { passed = true, mode, records, conditioning, runtime = Environment.Version.ToString(),
            flags, processor_count = Environment.ProcessorCount, core = Core, executable = FileHash(typeof(Probe).Assembly.Location),
            shapes = FileHash(args[0]), pid = Environment.ProcessId, fma = Fma.IsSupported, avx2 = Avx2.IsSupported, avx512 = Avx512F.IsSupported }, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"{mode} passed, records={records.Count}");
        return 0;
    }
}
