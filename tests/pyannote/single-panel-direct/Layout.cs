using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static class Layout
{
    const string Core = "e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838";
    const int Guard = unchecked((int)0xff812345);
    static void Require(bool ok, string message) { if (!ok) throw new InvalidDataException(message); }
    static string Hash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
    static unsafe int Main(string[] args)
    {
        Require(args.Length == 2 && (args[0] == "normal" || args[0] == "disabled"), "Arguments");
        Require(!File.Exists(args[1]), "Existing output");
        Require(Hash(typeof(MathOps).Assembly.Location) == Core, "Core identity");
        Require(Environment.ProcessorCount == 1 && Fma.IsSupported == (args[0] == "normal"), "Hardware mode");
        var flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k =>
            k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) ||
            k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase) ||
            k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)).Order().ToArray();
        Require(flags.SequenceEqual(args[0] == "normal" ? Array.Empty<string>() : new[] { "DOTNET_EnableHWIntrinsic" }), "Runtime flags");
        int[] reductions = [0, 1, 9, 63, 64, 65, 288, 576, 1152, 2304];
        int[] widths = Enumerable.Range(0, 66).Concat(new[] { 96, 128, 192, 320 }).ToArray();
        int[] specials = [0, unchecked((int)0x80000000), 0x7f800000, unchecked((int)0xff800000),
            0x7fc12345, unchecked((int)0xffc54321), 1, unchecked((int)0x80000001), 0x3f000000];
        int cases = 0, identities = 0, distinctWide = 0; long values = 0;
        foreach (int n in reductions) foreach (int k in widths) foreach (bool special in new[] { false, true })
        {
            int count = checked(n * k);
            var input = new float[count + 6]; var output = new float[count + 6];
            var ib = MemoryMarshal.Cast<float, int>(input.AsSpan());
            var ob = MemoryMarshal.Cast<float, int>(output.AsSpan());
            ib.Fill(Guard); ob.Fill(Guard);
            for (int i = 0; i < count; i++) ib[3 + i] = special ? specials[i % specials.Length] : 0x3f000000 + i;
            int[] before = ib.ToArray(), expected = ob.ToArray();
            int index = 3, blocked = k / 32 * 32;
            for (int column = 0; column < blocked; column += 32)
                for (int row = 0; row < n; row++)
                    for (int lane = 0; lane < 32; lane++) expected[index++] = before[3 + row * k + column + lane];
            for (int row = 0; row < n; row++)
                for (int column = blocked; column < k; column++) expected[index++] = before[3 + row * k + column];
            Require(index == count + 3, "Independent layout count");
            fixed (float* b = input, p = output) MathOps.PackPanelsB(n, k, b + 3, p + 3);
            Require(ib.SequenceEqual(before), "Input mutated");
            Require(ob.SequenceEqual(expected), "Permutation or guard mismatch");
            if (k <= 32) { Require(ob.SequenceEqual(before), "Single-panel layout is not identity"); identities++; }
            else if (n > 1 && !special) { Require(!ob.SequenceEqual(before), "Wide case did not require packing"); distinctWide++; }
            cases++; values += count;
        }
        Require(cases == 1400 && identities == 660 && distinctWide == 296, "Coverage");
        var result = new { passed = true, cases, identities, distinct_wide_cases = distinctWide, values,
            mode = args[0], core = Core, runtime = Environment.Version.ToString(), processor_count = Environment.ProcessorCount,
            flags, fma = Fma.IsSupported, pid = Environment.ProcessId, executable = Hash(typeof(Layout).Assembly.Location) };
        File.WriteAllText(args[1], JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine(JsonSerializer.Serialize(result));
        return 0;
    }
}
