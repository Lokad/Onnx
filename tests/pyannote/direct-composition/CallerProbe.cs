using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static class CallerProbe
{
    const float Guard = -12345.25f;
    static void Require(bool ok, string message) { if (!ok) throw new InvalidDataException(message); }
    static string Hash(float[] data) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(data.AsSpan())));
    static string FileHash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
    static float[] Data(int size, string pattern, int seed)
    {
        float[] special = [0f, -0f, float.PositiveInfinity, float.NegativeInfinity,
            BitConverter.Int32BitsToSingle(0x7fc12345), float.Epsilon, -float.Epsilon, float.MaxValue, -float.MaxValue, .125f, -.5f];
        var result = Enumerable.Repeat(Guard, size + 6).ToArray();
        for (int i = 0; i < size; i++) result[i + 3] = pattern == "special" ? special[(i * 7 + seed) % special.Length]
            : pattern == "zero" ? (i % 2 == 0 ? 0f : -0f) : ((i * 17 + seed) % 113 - 56) / 997f;
        return result;
    }

    static int Main(string[] args)
    {
        Require(args.Length == 5, "baseline runtime, shape manifest, output, normal|disabled, candidate hash");
        bool disabled = args[3] == "disabled"; Require(disabled || args[3] == "normal", "Mode");
        var flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k => k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase)
            || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)).Order().ToArray();
        Require(flags.SequenceEqual(disabled ? new[] { "DOTNET_EnableHWIntrinsic" } : Array.Empty<string>()), "Flags");
        if (disabled) Require(Environment.GetEnvironmentVariable(flags[0]) == "0", "Disabled flag");
        Require(Environment.ProcessorCount == 1 && Fma.IsSupported != disabled, "Affinity/hardware");
        Require(FileHash(typeof(MathOps).Assembly.Location) == args[4], "Candidate core");
        Require(FileHash(Path.Combine(args[0], "Lokad.Onnx.dll")) == "e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838", "Baseline core");
        var context = new AssemblyLoadContext("baseline", false);
        context.Resolving += (owner, name) => owner.LoadFromAssemblyPath(Path.Combine(args[0], name.Name + ".dll"));
        var baseline = context.LoadFromAssemblyPath(Path.Combine(args[0], "Lokad.Onnx.dll"));
        var oldTensor = baseline.GetType("Lokad.Onnx.Tensor`1")!.MakeGenericType(typeof(float));
        var oldMethod = oldTensor.GetMethod("RunTiledBatchFloat", BindingFlags.Static | BindingFlags.NonPublic)!;
        var newMethod = typeof(Tensor<float>).GetMethod("RunTiledBatchFloat", BindingFlags.Static | BindingFlags.NonPublic)!;
        var oldOptionsType = baseline.GetType("Lokad.Onnx.TensorExecutionOptions")!;
        var oldPad = Activator.CreateInstance(baseline.GetType("Lokad.Onnx.MathOps+PadInfo")!)!;
        var records = new List<object>();

        void Check(int rows, int reduction, int block, int groups, bool hasBias, string pattern, string policy, bool mutable)
        {
            int columns = block * 2 + 1, totalRows = rows * groups, channels = reduction * groups;
            int inBatch = channels * columns, outBatch = totalRows * columns;
            var input = Data(2 * inBatch, pattern, 19); var weights = Data(totalRows * reduction, pattern, 37);
            var bias = Data(totalRows, pattern, 61);
            object oldOptions = oldOptionsType.GetProperty(policy)!.GetValue(null)!;
            var newOptions = (TensorExecutionOptions)typeof(TensorExecutionOptions).GetProperty(policy)!.GetValue(null)!;
            var initial = Enumerable.Repeat(Guard, 2 * outBatch + 6).ToArray();
            for (int pass = 0; pass < (mutable ? 2 : 1); pass++)
            {
                if (pass == 1) { weights[3] = .25f; input[3] = -.125f; bias[3] = .5f; }
                string ih = Hash(input), wh = Hash(weights), bh = Hash(bias);
                var wanted = (float[])initial.Clone(); var actual = (float[])initial.Clone();
                foreach (int batch in new[] { 1, 0 })
                {
                    object[] Parameters(float[] output, object pad, object options) => new object[] {
                        input.AsMemory(3, 2 * inBatch), weights.AsMemory(3, totalRows * reduction),
                        hasBias ? bias.AsMemory(3, totalRows) : default(Memory<float>), hasBias,
                        output.AsMemory(3, 2 * outBatch), Enumerable.Repeat(Guard, (channels + totalRows) * block + 3).ToArray(),
                        batch, groups, channels, 1, columns, totalRows, 1, 1, 1, 1, 1, 1,
                        pad, 1, columns, inBatch, outBatch, columns, block, options };
                    var oldArgs = Parameters(wanted, oldPad, oldOptions);
                    var newArgs = Parameters(actual, new MathOps.PadInfo(), newOptions);
                    oldMethod.Invoke(null, oldArgs); newMethod.Invoke(null, newArgs);
                    for (int i = 0; i < wanted.Length; i++)
                        Require(BitConverter.SingleToInt32Bits(wanted[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                            $"Caller bits rows={rows} reduction={reduction} block={block} groups={groups} bias={hasBias} pattern={pattern} policy={policy} pass={pass} batch={batch} index={i} old={BitConverter.SingleToInt32Bits(wanted[i]):x8} new={BitConverter.SingleToInt32Bits(actual[i]):x8}");
                    if (batch == 1) Require(actual.AsSpan(3, outBatch).IndexOfAnyExcept(Guard) < 0, "Other batch written");
                    foreach (var call in new[] { oldArgs, newArgs })
                        Require(((float[])call[5]).AsSpan((channels + totalRows) * block).IndexOfAnyExcept(Guard) < 0, "Scratch guard");
                }
                Require(Hash(input) == ih && Hash(weights) == wh && Hash(bias) == bh, "Input mutation");
                Require(actual.AsSpan(0, 3).IndexOfAnyExcept(Guard) < 0 && actual.AsSpan(actual.Length - 3).IndexOfAnyExcept(Guard) < 0, "Outer guard");
                records.Add(new { rows, reduction, block, groups, hasBias, pattern, policy, pass,
                    checked_values = actual.Length, nan_values = actual.Count(float.IsNaN), digest = Hash(actual) });
            }
        }

        using var manifest = JsonDocument.Parse(File.ReadAllText(args[1]));
        foreach (var shape in manifest.RootElement.GetProperty("shapes").EnumerateArray())
            Check(shape.GetProperty("m").GetInt32(), shape.GetProperty("n").GetInt32(), shape.GetProperty("k").GetInt32(), 1, true, "finite", "Auto", false);
        foreach (int rows in new[] { 32, 64 }) foreach (int reduction in new[] { 64, 65 })
        foreach (int block in new[] { 1, 2, 7, 8, 31, 32, 33 }) foreach (bool bias in new[] { false, true })
        foreach (string pattern in new[] { "finite", "zero", "special" })
            Check(rows, reduction, block, 2, bias, pattern, "Auto", true);
        foreach (string policy in new[] { "Scalar", "Simd" }) foreach (int block in new[] { 2, 8, 33 })
        foreach (bool bias in new[] { false, true }) foreach (string pattern in new[] { "finite", "zero", "special" })
            Check(32, 64, block, 2, bias, pattern, policy, false);
        foreach (var shape in new[] { (5, 9, 2), (30, 64, 8), (33, 64, 2), (96, 64, 8), (32, 63, 32), (32, 1024, 65) })
            Check(shape.Item1, shape.Item2, shape.Item3, 2, true, "finite", "Auto", false);
        Require(records.Count == 736, "Complete schedule");
        using var stream = new FileStream(args[2], FileMode.CreateNew);
        JsonSerializer.Serialize(stream, new { passed = true, records, mode = args[3], pid = Environment.ProcessId,
            flags, processor_count = Environment.ProcessorCount, fma = Fma.IsSupported, runtime = Environment.Version.ToString(),
            candidate = FileHash(typeof(MathOps).Assembly.Location), baseline = FileHash(Path.Combine(args[0], "Lokad.Onnx.dll")) }, new JsonSerializerOptions { WriteIndented = true });
        Console.WriteLine($"Real caller passed {records.Count} records");
        return 0;
    }
}
