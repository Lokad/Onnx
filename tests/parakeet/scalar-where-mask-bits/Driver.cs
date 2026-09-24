using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

static class Program
{
    static string Hash(byte[] b) => Convert.ToHexStringLower(SHA256.HashData(b));
    static int Main(string[] args)
    {
        if (args.Length != 5 || Environment.Version.ToString() != "10.0.8") throw new InvalidDataException("environment");
        var rows = new List<object>();
        using var cases = JsonDocument.Parse(File.ReadAllText(Path.Combine(args[0], "cases.json")));
        foreach (var entry in cases.RootElement.EnumerateArray())
        {
            var raw = entry.GetProperty("mask").EnumerateArray().Select(e => e.GetByte()).ToArray();
            var mask = MemoryMarshal.Cast<byte, bool>(raw).ToArray();
            var x = new float[] { -10000 }; var y = new float[] { 10, 20 };
            var c = new DenseTensor<bool>(mask.AsMemory(), new[] { 2 });
            var expected = raw.Select((b, i) => BitConverter.SingleToInt32Bits(b != 0 ? x[0] : y[i])).ToArray();
            var output = Tensor<float>.Where(c, new DenseTensor<float>(x.AsMemory(), Array.Empty<int>()), new DenseTensor<float>(y.AsMemory(), new[] { 2 }));
            var actual = output.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray();
            if (!raw.SequenceEqual(MemoryMarshal.AsBytes(mask.AsSpan()).ToArray()) || x[0] != -10000 || y[0] != 10 || y[1] != 20)
                throw new InvalidDataException("input mutation");
            rows.Add(new { name = entry.GetProperty("name").GetString(), mask = raw.Select(b => (int)b).ToArray(),
                expected, actual, matches_nonzero_selection = expected.SequenceEqual(actual), inputs_unchanged = true });
        }
        var flags = Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>()
            .Where(v => ((string)v.Key).StartsWith("DOTNET_", StringComparison.OrdinalIgnoreCase) || ((string)v.Key).StartsWith("COMPlus_", StringComparison.OrdinalIgnoreCase) || ((string)v.Key).StartsWith("LOKAD_", StringComparison.OrdinalIgnoreCase))
            .ToDictionary(v => (string)v.Key, v => (string)v.Value!);
        var value = new { completed = true, no_performance_measurement = true, pid = Environment.ProcessId,
            runtime = Environment.Version.ToString(), role = args[1], mode = args[2], width = int.Parse(args[3]), flags,
            assembly = Hash(File.ReadAllBytes(Assembly.GetExecutingAssembly().Location)),
            core_sha256 = Hash(File.ReadAllBytes(typeof(Tensor<float>).Assembly.Location)), rows };
        File.WriteAllText(args[4], JsonSerializer.Serialize(value, new JsonSerializerOptions { WriteIndented = true }));
        return 0;
    }
}
