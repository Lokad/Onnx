using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static class Driver
{
    const string Filter = "mm_unsafe_vectorized_intrinsics_3x4packed DirectOutput:*";
    delegate void Iterate(bool candidate, int count, int m, int n, int k, int stride, int start,
        float[] a, float[] b, float[] p, float[] temp, float[] output, float[] bias);
    static void Require(bool ok, string message) { if (!ok) throw new InvalidDataException(message); }
    static string FileHash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
    static string Hash(float[] a) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(a.AsSpan())));
    static T Bind<T>(Type type, string name) where T : Delegate
        => type.GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static)!.CreateDelegate<T>();

    static int Main(string[] args)
    {
        Require(args.Length == 5, "probe shapes baseline|candidate expected-probe-hash output");
        Require(args[2] is "baseline" or "candidate", "Role");
        Require(!File.Exists(args[4]), "Refuse overwrite");
        Require(Environment.ProcessorCount == 1 && Fma.IsSupported && Avx2.IsSupported, "Hardware and affinity");
        var flags = Environment.GetEnvironmentVariables().Keys.Cast<string>()
            .Where(k => k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase)
                || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)
                || k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)).Order().ToArray();
        Require(flags.SequenceEqual(new[] { "DOTNET_JitDisasm" })
            && Environment.GetEnvironmentVariable(flags[0]) == Filter, "Diagnostic flag allowlist");
        string core = FileHash(typeof(MathOps).Assembly.Location);
        Require(core == "e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838", "Core identity");
        Require(FileHash(args[0]) == args[3], "Probe identity");
        var assembly = Assembly.LoadFrom(Path.GetFullPath(args[0]));
        var probe = assembly.GetType("Probe", true)!;
        var data = Bind<Func<int, uint, string, float[]>>(probe, "Data");
        var iterations = Bind<Iterate>(probe, "Iterations");
        var validate = Bind<Func<JsonElement, object>>(probe, "Validate");
        using var doc = JsonDocument.Parse(File.ReadAllText(args[1]));
        bool candidate = args[2] == "candidate";
        var conditioning = new List<object>(); var blocks = new List<object>(); var validation = new List<object>();
        for (int phase = 0; phase < 2; phase++)
        foreach (var shape in doc.RootElement.GetProperty("shapes").EnumerateArray())
        {
            int m = shape.GetProperty("m").GetInt32(), n = shape.GetProperty("n").GetInt32(), k = shape.GetProperty("k").GetInt32();
            int stride = shape.GetProperty("stride").GetInt32(), start = shape.GetProperty("timing_start").GetInt32();
            int count = shape.GetProperty("iterations").GetInt32();
            var a = data(m * n, 1234567, "finite"); var b = data(n * k, 9876543, "finite"); var p = data(n * k, 33, "finite");
            var temp = data(m * k, 444, "finite"); var output = data(m * stride, 999, "finite"); var bias = data(m, 777, "finite");
            string ah = Hash(a), bh = Hash(b), bih = Hash(bias);
            int calls = 0; var warm = Stopwatch.StartNew();
            do { iterations(candidate, 16, m, n, k, stride, start, a, b, p, temp, output, bias); calls += 16; }
            while (warm.Elapsed.TotalSeconds < 1);
            conditioning.Add(new { phase, m, n, k, stride, start, calls, seconds = warm.Elapsed.TotalSeconds });
            if (phase == 0) continue;
            for (int block = 0; block < 6; block++)
            {
                iterations(candidate, count, m, n, k, stride, start, a, b, p, temp, output, bias);
                blocks.Add(new { m, n, k, stride, start, block, iterations = count, digest = Hash(output) });
            }
            Require(ah == Hash(a) && bh == Hash(b) && bih == Hash(bias), "Input changed");
            foreach (var buffer in new[] { temp, p }) for (int i = 0; i < 3; i++)
                Require(buffer[i] == -12345.25f && buffer[buffer.Length - 1 - i] == -12345.25f, "Guard");
            var reference = data(m * stride, 999, "finite");
            iterations(false, 1, m, n, k, stride, start, a, b, p, temp, reference, bias);
            Require(Hash(reference) == Hash(output), "Baseline output bits");
        }
        foreach (var shape in doc.RootElement.GetProperty("cases").EnumerateArray()) validation.Add(validate(shape));
        File.WriteAllText(args[4], JsonSerializer.Serialize(new { passed = true, mode = args[2], core,
            executable = FileHash(typeof(Driver).Assembly.Location), probe = FileHash(assembly.Location),
            shapes = FileHash(args[1]), pid = Environment.ProcessId, runtime = Environment.Version.ToString(),
            processor_count = Environment.ProcessorCount, flags, fma = Fma.IsSupported, avx2 = Avx2.IsSupported,
            avx512 = Avx512F.IsSupported, conditioning, blocks, validation }, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"Diagnostic complete, blocks={blocks.Count}, validation={validation.Count}");
        return 0;
    }
}
