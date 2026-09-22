using System;
using System.Collections;
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

internal static class Driver
{
    const BindingFlags Members = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static;
    static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
    static string FileHash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
    static string Hash(ReadOnlySpan<float> values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values)));
    static T Field<T>(object value, string name) => (T)value.GetType().GetField(name, Members)!.GetValue(value)!;
    sealed record Call(string Name, int Index, string Node, int Form, bool Eligible, long Work, int Iterations,
        long WeightBytes, string PreparedHash, Func<int, DenseTensor<float>> Run, Func<ComputationalGraph> Create);

    static int Main(string[] args)
    {
        Require(args.Length == 9, "validate|benchmark role fixtures reference iterations output expected-core expected-probe width");
        bool benchmark = args[0] == "benchmark";
        Require(benchmark || args[0] == "validate", "Mode");
        Require(args[1] is "production" or "candidate", "Role");
        Require(args[8] is "256" or "512", "Width"); int lanes = int.Parse(args[8]) / 32;
        Require(Environment.ProcessorCount == 1 && Avx2.IsSupported && Fma.IsSupported && Avx512F.IsSupported == (lanes == 16), "Actual hardware and affinity");
        Require(!benchmark || (OperatingSystem.IsLinux() && lanes == 16 && Environment.Version.ToString() == "10.0.8"), "Target-only timing");
        var flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k =>
            k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)
            || k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)).ToArray();
        Require(flags.Length == 0, "No runtime overrides or diagnostic flags");
        string directory = Path.GetDirectoryName(typeof(Driver).Assembly.Location)!;
        string probePath = Path.Combine(directory, "LayerGraphs.dll");
        Require(FileHash(typeof(Tensor<float>).Assembly.Location) == args[6] && FileHash(probePath) == args[7], "Pinned assemblies");
        Require(Directory.Exists(args[5]) && !File.Exists(Path.Combine(args[5], "journal.jsonl")), "Refuse existing output");
        var assembly = Assembly.LoadFrom(probePath); var probe = assembly.GetType("ModelProbe", true)!;
        var fixtureType = probe.GetNestedType("Fixtures", Members)!;
        var fixtures = fixtureType.GetConstructor(Members, null, new[] { typeof(string), typeof(int) }, null)!.Invoke(new object[] { args[2], lanes });
        var originals = Field<IEnumerable>(fixtures, "Calls").Cast<object>().ToArray();
        var graphType = assembly.GetType("GraphCalls", true)!; var create = graphType.GetMethod("Create", Members)!;
        var calls = originals.Select(call =>
        {
            int[] x = Field<int[]>(call, "InputShape"), y = Field<int[]>(call, "OutputShape"), k = Field<int[]>(call, "Kernel");
            long work = checked((long)y[1] * x[1] * k[0] * k[1] * y[2] * y[3]);
            int iterations = checked((int)(((1L << 31) + work - 1) / work)); Require(iterations is >= 1 and <= 128, "Iteration bounds");
            return new Call(Field<string>(call, "Name"), Field<int>(call, "Index"), Field<string>(call, "Node"),
                Field<int>(call, "Form"), Field<bool>(call, "Eligible"), work, iterations,
                Field<float[]>(call, "Weight").LongLength * 4, Hash(Field<float[]>(call, "Prepared")),
                call.GetType().GetMethod("Candidate", Members)!.CreateDelegate<Func<int, DenseTensor<float>>>(call),
                create.CreateDelegate<Func<ComputationalGraph>>(call));
        }).ToArray();
        Require(calls.Length == 108 && calls.Count(c => c.Eligible) == 96 && calls.Select(c => c.Form).Distinct().Count() == 15, "Full census");
        using var reference = JsonDocument.Parse(File.ReadAllBytes(args[3]));
        Require(reference.RootElement.GetProperty("passed").GetBoolean(), "Qualified reference");
        var hashes = reference.RootElement.GetProperty("observations").EnumerateArray().ToDictionary(
            r => (r.GetProperty("name").GetString()!, r.GetProperty("index").GetInt32()), r => r.GetProperty("production").GetString()!);
        using var frozen = JsonDocument.Parse(File.ReadAllBytes(args[4]));
        var schedule = frozen.RootElement.GetProperty("calls").EnumerateArray().ToArray(); Require(schedule.Length == 108, "Frozen iteration census");
        for (int i = 0; i < calls.Length; i++)
            Require(schedule[i].GetProperty("case").GetString() == calls[i].Name && schedule[i].GetProperty("index").GetInt32() == calls[i].Index
                && schedule[i].GetProperty("work").GetInt64() == calls[i].Work && schedule[i].GetProperty("iterations").GetInt32() == calls[i].Iterations, "Frozen geometry counts");
        int perPass = calls.Sum(c => c.Iterations); Require(perPass == 1074 && frozen.RootElement.GetProperty("per_pass").GetInt32() == perPass, "Iteration total");
        var constants = calls.Where(c => c.Eligible).DistinctBy(c => c.Node).ToArray(); Require(constants.Length == 32, "Preparation census");
        var preparation = new List<object>(); var observations = new List<object>();
        using var journal = new StreamWriter(new FileStream(Path.Combine(args[5], "journal.jsonl"), FileMode.CreateNew));
        void Log(object row) { journal.WriteLine(JsonSerializer.Serialize(row)); journal.Flush(); }
        long frequency = Stopwatch.Frequency; int passes = benchmark ? 4 : 1;
        for (int pass = 0; pass < passes; pass++) foreach (var call in constants)
        {
            long start = benchmark ? Stopwatch.GetTimestamp() : 0;
            var graph = call.Create();
            long stop = benchmark ? Stopwatch.GetTimestamp() : 0;
            Require(!benchmark || stop > start, "Preparation clock");
            var map = (IDictionary)typeof(ComputationalGraph).GetField("PackedConvWeights", Members)!.GetValue(graph)!;
            Require(map.Count == 1 && graph.RetainedPackedWeightBytes == call.WeightBytes, "Actual graph preparation");
            var record = map.Values.Cast<object>().Single();
            var packed = (float[])record.GetType().GetProperty("Values", Members)!.GetValue(record)!;
            string hash = Hash(packed); Require(hash == call.PreparedHash, "Prepared graph weight changed");
            var row = new { kind = "preparation", role = args[1], pass, warmup = pass == 0, index = call.Index, node = call.Node,
                ticks = stop - start, frequency, bytes = call.WeightBytes, sha256 = hash };
            preparation.Add(row); Log(row);
        }
        for (int pass = 0; pass < passes; pass++) foreach (var call in calls)
        for (int iteration = 0; iteration < (benchmark ? call.Iterations : 1); iteration++)
        {
            long start = benchmark ? Stopwatch.GetTimestamp() : 0;
            // Both roles invoke the existing prepared ordinary graph caller.
            // There is no role-dependent unprepared-convolution control here.
            var output = call.Run(lanes);
            long stop = benchmark ? Stopwatch.GetTimestamp() : 0;
            Require(!benchmark || stop > start, "Call clock");
            string hash = Hash(output.Buffer.Span); bool exact = hash == hashes[(call.Name, call.Index)];
            var row = new { kind = "call", role = args[1], pass, warmup = pass == 0, name = call.Name, index = call.Index,
                form = call.Form, eligible = call.Eligible, iteration, iterations = benchmark ? call.Iterations : 1, work = call.Work,
                ticks = stop - start, frequency, values = output.Length, sha256 = hash, exact };
            observations.Add(row); Log(row); Require(exact, "Changed timed output");
        }
        fixtureType.GetMethod("Unchanged", Members)!.Invoke(fixtures, null);
        int dispatches = ((ICollection)graphType.GetField("Rows", Members)!.GetValue(null)!).Count;
        long retained = (long)graphType.GetProperty("RetainedBytes", Members)!.GetValue(null)!;
        Require(dispatches == observations.Count && retained == 3L * 21086208, "All graph calls and retained bytes");
        var report = new { passed = true, role = args[1], mode = args[0], cases = 108, calls = observations.Count,
            warmups = benchmark ? perPass : 108, measured = benchmark ? 3 * perPass : 0, protocol = "geometry-2pow31-v1",
            observations, preparation, read_only_operands = true, lanes, runtime = Environment.Version.ToString(), core = args[6], probe = args[7],
            executable = FileHash(typeof(Driver).Assembly.Location), pid = Environment.ProcessId, flags, prepared_bytes = 21086208,
            graph_retained_bytes = retained, graph_dispatches = dispatches, no_performance_measurement = !benchmark,
            setup = "Both roles use unchanged prepared graph callers. Fixture arrays/prepared clones load before timing. Graph creation/preparation is reported separately; warmup populates 108 graphs retaining 63,258,624 bytes. Complete call timers include graph caller assertions/recording, scans, rentals, conversions, kernels and graph epilogues; output hashing/journal IO are outside." };
        File.WriteAllText(Path.Combine(args[5], "result.json"), JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine(JsonSerializer.Serialize(new { report.passed, report.role, report.mode, report.calls })); return 0;
    }
}
