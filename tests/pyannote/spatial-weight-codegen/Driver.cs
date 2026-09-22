using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

internal static class Driver
{
    const BindingFlags Members = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static;
    const string Filter = "Kernel512*";
    static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
    static string FileHash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
    static string Hash(DenseTensor<float> tensor) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(tensor.Buffer.Span)));
    static T Field<T>(object value, string name) => (T)value.GetType().GetField(name, Members)!.GetValue(value)!;

    static int Main(string[] args)
    {
        Require(args.Length == 7, "probe fixtures reference output expected-core expected-probe width");
        Require(args[6] is "256" or "512", "Instruction width");
        int lanes = int.Parse(args[6]) / 32;
        Require(Environment.ProcessorCount == 1 && Avx2.IsSupported && Fma.IsSupported, "Hardware/affinity");
        Require((lanes == 16) == Avx512F.IsSupported, "Actual instruction width");
        Require(!File.Exists(args[3]), "Refuse overwrite");
        var flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k =>
            k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)
            || k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)).Order().ToArray();
        Require(flags.SequenceEqual(new[] { "DOTNET_JitDisasm" }) && Environment.GetEnvironmentVariable(flags[0]) == Filter, "Diagnostic flag allowlist");
        Require(FileHash(typeof(Tensor<float>).Assembly.Location) == args[4] && FileHash(args[0]) == args[5], "Pinned assemblies");
        var assembly = Assembly.LoadFrom(Path.GetFullPath(args[0]));
        var probe = assembly.GetType("ModelProbe", true)!;
        var fixtureType = probe.GetNestedType("Fixtures", Members)!;
        var fixtures = fixtureType.GetConstructor(Members, null, new[] { typeof(string), typeof(int) }, null)!.Invoke(new object[] { args[1], lanes });
        var calls = Field<IEnumerable>(fixtures, "Calls").Cast<object>().ToArray();
        Require(calls.Length == 108, "Complete call census");
        using var reference = JsonDocument.Parse(File.ReadAllBytes(args[2]));
        Require(reference.RootElement.GetProperty("passed").GetBoolean(), "Qualified reference");
        var hashes = reference.RootElement.GetProperty("observations").EnumerateArray().ToDictionary(
            r => (r.GetProperty("name").GetString()!, r.GetProperty("index").GetInt32()),
            r => r.GetProperty("production").GetString()!);
        var rows = new List<object>();
        var runners = calls.Select(call => call.GetType().GetMethod("Candidate", Members)!.CreateDelegate<Func<int, DenseTensor<float>>>(call)).ToArray();
        // Four fixed passes exercise the unchanged ordinary graph callers.
        // No time calibration, tiering override, timing score or kernel rebuild.
        for (int pass = 0; pass < 4; pass++)
        for (int index = 0; index < calls.Length; index++)
        {
            var call = calls[index]; string name = Field<string>(call, "Name"); int ordinal = Field<int>(call, "Index");
            var output = runners[index](lanes); string hash = Hash(output);
            Require(hash == hashes[(name, ordinal)], "Changed diagnostic output");
            rows.Add(new { pass, name, index = ordinal, form = Field<int>(call, "Form"), eligible = Field<bool>(call, "Eligible"),
                channels = Field<int[]>(call, "OutputShape")[1], values = output.Length, sha256 = hash, exact = true });
        }
        fixtureType.GetMethod("Unchanged", Members)!.Invoke(fixtures, null);
        var graphType = assembly.GetType("GraphCalls", true)!;
        var graphRows = (ICollection)graphType.GetField("Rows", Members)!.GetValue(null)!;
        Require(rows.Count == graphRows.Count && rows.Count == 432, "Every graph call recorded");
        File.WriteAllText(args[3], JsonSerializer.Serialize(new { passed = true, core = args[4], probe = args[5],
            executable = FileHash(typeof(Driver).Assembly.Location), pid = Environment.ProcessId, runtime = Environment.Version.ToString(),
            processor_count = Environment.ProcessorCount, lanes, flags, avx512 = Avx512F.IsSupported,
            cases = 108, passes = 4, calls = 432, observations = rows, readonly_operands = true,
            no_performance_measurement = true, tiering_and_isa_unchanged = true }, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine("Diagnostic complete: 432 exact ordinary graph outputs.");
        return 0;
    }
}
