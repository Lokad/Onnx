using System.Collections;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

internal static class Extremes
{
    const BindingFlags Members = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static;
    static void Require(bool ok, string message) { if (!ok) throw new InvalidDataException(message); }
    static string FileHash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
    static string Hash(DenseTensor<float> tensor) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(tensor.Buffer.Span)));
    static string ArrayHash(float[] values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));

    static int Main(string[] args)
    {
        Require(args.Length == 6, "probe output core-hash probe-hash role width");
        Require((args[4] is "selected" or "candidate") && (args[5] is "256" or "512"), "Role/width");
        int lanes = int.Parse(args[5]) / 32;
        Require(Environment.ProcessorCount == 1 && Avx2.IsSupported && Fma.IsSupported && Avx512F.IsSupported == (lanes == 16), "Hardware/affinity");
        var flags = Environment.GetEnvironmentVariables().Keys.Cast<string>().Where(k =>
            k.StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || k.StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase)
            || k.StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase)).Order().ToArray();
        Require(flags.SequenceEqual(lanes == 8 ? new[] { "DOTNET_EnableAVX512" } : Array.Empty<string>())
            && (lanes == 16 || Environment.GetEnvironmentVariable("DOTNET_EnableAVX512") == "0"), "Runtime flags");
        Require(!File.Exists(args[1]) && FileHash(typeof(Tensor<float>).Assembly.Location) == args[2] && FileHash(args[0]) == args[3], "Pinned assemblies/output");
        var assembly = Assembly.LoadFrom(Path.GetFullPath(args[0]));
        var graph = assembly.GetType("GraphRaw", true)!;
        var check = graph.GetMethod("Check", Members)!;
        var create = graph.GetMethod("Create", Members)!;
        var run = graph.GetMethod("Run", Members)!;
        var observations = new List<object>();
        int index = 0;
        foreach (int c in new[] { 64, 80, 128, 256 })
        foreach (int m in new[] { 32, 48 })
        foreach (int width in new[] { 7, 13 })
        foreach (int stride in new[] { 1, 2 })
        for (int pattern = 0; pattern < 5; pattern++)
        {
            const int h = 3;
            int oh = (h + stride - 1) / stride, ow = (width + stride - 1) / stride;
            var x = new float[c * h * width]; var w = new float[m * c * 9];
            float[] extremes = [float.MaxValue, -float.MaxValue, float.Epsilon, -float.Epsilon, 1f, -1f, 0f, -0f];
            for (int i = 0; i < x.Length; i++) x[i] = pattern switch
            {
                0 => extremes[i % 8], 1 => (i % 2 == 0 ? 1f : -1f) * float.Epsilon,
                2 => i % 2 == 0 ? 0f : -0f, 3 => float.MaxValue,
                _ => i / (h * width) / 16 % 2 == 0 ? 128f : 1f / 128
            };
            for (int i = 0; i < w.Length; i++) w[i] = pattern switch
            {
                0 => extremes[(i * 3) % 8], 1 => i % 3 == 0 ? 2f : -1f,
                2 => i % 2 == 0 ? 1f : -1f, 3 => i / 9 % 2 == 0 ? 2f : -2f,
                _ => i / 9 % 2 == 0 ? 1f : -1f
            };
            var bias = Enumerable.Range(0, m).Select(i => pattern is 0 or 3 ? (i % 2 == 0 ? float.MaxValue : -float.MaxValue) : .25f).ToArray();
            var residual = Enumerable.Range(0, m * oh * ow).Select(i => i % 2 == 0 ? .5f : -.5f).ToArray();
            Require(x.All(float.IsFinite) && w.All(float.IsFinite) && bias.All(float.IsFinite) && residual.All(float.IsFinite), "Finite operands");
            for (int epilogue = 0; epilogue < 8; epilogue++)
            {
                var b = (epilogue & 1) != 0 ? bias : Array.Empty<float>();
                var r = (epilogue & 2) != 0 ? residual : Array.Empty<float>(); bool relu = (epilogue & 4) != 0;
                var inputs = new[] { x, w, b, r }; var inputHashes = inputs.Select(ArrayHash).ToArray();
                check.Invoke(null, new object[] { "finite-extremes", index, x, w, b, r, c, m, h, width, stride, lanes, relu });
                long budget = (long)((m + 2 * lanes - 1) / (2 * lanes) * 2 * lanes) * c * 9 * 4;
                var prepared = (ComputationalGraph)create.Invoke(null, new object[] { x, w, b, r, c, m, h, width, stride, relu, budget })!;
                var held = (DenseTensor<float>)run.Invoke(null, new object[] { prepared })!; string first = Hash(held);
                var second = (DenseTensor<float>)run.Invoke(null, new object[] { prepared })!;
                Require(!ReferenceEquals(held, second) && Hash(held) == first && Hash(second) == first, "Exact repeat/owned output");
                Require(held.Length == m * oh * ow, "Output shape");
                Require(inputs.Select(ArrayHash).SequenceEqual(inputHashes), "Readonly operands");
                observations.Add(new { index, c, m, h, w = width, stride, pattern, epilogue, values = held.Length, sha256 = first, exact_repeat = true, owned = true });
                index++;
            }
        }
        var graphRows = ((IEnumerable)graph.GetField("Rows", Members)!.GetValue(null)!).Cast<object>().ToArray();
        long differences = (long)graph.GetField("Differences", Members)!.GetValue(null)!;
        long payloads = (long)graph.GetField("NaNPayloadDifferences", Members)!.GetValue(null)!;
        int controls = (int)graph.GetField("ControlCalls", Members)!.GetValue(null)!;
        int candidates = (int)graph.GetField("CandidateCalls", Members)!.GetValue(null)!;
        Require(index == 1280 && graphRows.Length == index && controls == index && candidates == 2 * index && differences == 0, "Complete graph census/values");
        File.WriteAllText(args[1], JsonSerializer.Serialize(new { passed = true, core = args[2], probe = args[3], role = args[4],
            executable = FileHash(typeof(Extremes).Assembly.Location), pid = Environment.ProcessId, runtime = Environment.Version.ToString(),
            processor_count = Environment.ProcessorCount, lanes, avx512 = Avx512F.IsSupported, flags, cases = index,
            controls, candidates, additional_exact_calls = 2 * index, differences, nan_payload_differences = payloads,
            observations, graph_cases = graphRows, no_performance_measurement = true }, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine("Finite extreme coverage complete: 1280 cases, 6400 graph calls.");
        return 0;
    }
}
