using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static class BaselineNaN
{
    static void Require(bool ok, string message) { if (!ok) throw new InvalidDataException(message); }
    static string Hash(float[] data) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(data.AsSpan())));
    static int Main(string[] args)
    {
        Require(args.Length == 2, "block output");
        int block = int.Parse(args[0]); Require(block is 1 or 7, "Known failing geometry");
        Require(Environment.ProcessorCount == 1 && !Environment.GetEnvironmentVariables().Keys.Cast<string>().Any(k =>
            k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)
            || k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)), "Default runtime/affinity");
        string core = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(MathOps).Assembly.Location)));
        Require(core == "e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838", "Accepted baseline");
        const int rows = 32, reduction = 64, groups = 2;
        int columns = block * 2 + 1, inBatch = groups * reduction * columns, outBatch = rows * groups * columns;
        float[] values = [0f, -0f, float.PositiveInfinity, float.NegativeInfinity, BitConverter.Int32BitsToSingle(0x7fc12345),
            float.Epsilon, -float.Epsilon, float.MaxValue, -float.MaxValue, .125f, -.5f];
        float[] Data(int size, int seed)
        {
            var data = Enumerable.Repeat(-12345.25f, size + 6).ToArray();
            for (int i = 0; i < size; i++) data[3 + i] = values[(i * 7 + seed) % values.Length];
            return data;
        }
        var input = Data(2 * inBatch, 19); var weights = Data(rows * groups * reduction, 37); var bias = Data(rows * groups, 61);
        string ih = Hash(input), wh = Hash(weights), bh = Hash(bias);
        var output = Enumerable.Repeat(-12345.25f, 2 * outBatch + 6).ToArray();
        var scratch = new float[(groups * reduction + rows * groups) * block];
        object[] parameters = [input.AsMemory(3, 2 * inBatch), weights.AsMemory(3, rows * groups * reduction),
            bias.AsMemory(3, rows * groups), true, output.AsMemory(3, 2 * outBatch), scratch,
            1, groups, groups * reduction, 1, columns, rows * groups, 1, 1, 1, 1, 1, 1,
            new MathOps.PadInfo(), 1, columns, inBatch, outBatch, columns, block, TensorExecutionOptions.Auto];
        var method = typeof(Tensor<float>).GetMethod("RunTiledBatchFloat", BindingFlags.Static | BindingFlags.NonPublic)!;
        method.Invoke(null, parameters); var cold = (float[])output.Clone();
        int calls = 0; var warm = Stopwatch.StartNew();
        do { method.Invoke(null, parameters); calls++; } while (warm.Elapsed.TotalSeconds < 2);
        method.Invoke(null, parameters); var hot = (float[])output.Clone();
        var differences = Enumerable.Range(0, cold.Length).Where(i => BitConverter.SingleToInt32Bits(cold[i]) != BitConverter.SingleToInt32Bits(hot[i])).ToArray();
        Require(differences.All(i => float.IsNaN(cold[i]) && float.IsNaN(hot[i])), "Non-NaN baseline instability");
        Require(Hash(input) == ih && Hash(weights) == wh && Hash(bias) == bh, "Input mutation");
        using var stream = new FileStream(args[1], FileMode.CreateNew);
        JsonSerializer.Serialize(stream, new { passed = true, diagnostic_only = true, core, block, calls, seconds = warm.Elapsed.TotalSeconds,
            pid = Environment.ProcessId, runtime = Environment.Version.ToString(), processor_count = Environment.ProcessorCount,
            cold_digest = Hash(cold), hot_digest = Hash(hot), differing_nan_payloads = differences.Length,
            differences = differences.Select(i => new { index = i, cold = BitConverter.SingleToInt32Bits(cold[i]).ToString("x8"),
                hot = BitConverter.SingleToInt32Bits(hot[i]).ToString("x8") }), inputs_unchanged = true,
            all_non_nan_bits_unchanged = true }, new JsonSerializerOptions { WriteIndented = true });
        Console.WriteLine($"Baseline block={block}, changed NaN payloads={differences.Length}");
        return 0;
    }
}
