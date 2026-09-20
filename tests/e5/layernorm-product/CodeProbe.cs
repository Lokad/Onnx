using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx;

internal static class CodeProbe
{
    static string Hash(float[] values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));
    static void Require(bool value, string why) { if (!value) throw new InvalidDataException(why); }

    internal static void Run(string output, string core, object hardware, Dictionary<string, string?> flags)
    {
        Require(OperatingSystem.IsLinux() && Environment.Version.ToString() == "10.0.8", "Declared AMD host");
        Require(flags["COMPlus_JitDisasm"] == "*LayerNormFloatInto*" &&
            flags["COMPlus_JitStdOutFile"] == Path.Combine(Path.GetDirectoryName(output)!, "jit.txt"), "Declared instrumentation");
        // 385 covers the wider transform and the scalar tail; both public
        // overloads reach the private product kernel being disassembled.
        const int width = 385, rows = 30;
        var data = Enumerable.Range(0, width * rows).Select(i => (i % 101 - 50) * .03125f).ToArray();
        var gamma = Enumerable.Range(0, width).Select(i => (i % 13 - 6) * .125f).ToArray();
        var beta = Enumerable.Range(0, width).Select(i => (i % 7 - 3) * .0625f).ToArray();
        var x = new DenseTensor<float>(data, new[] { rows, width });
        var scale = new DenseTensor<float>(gamma, new[] { width });
        var bias = new DenseTensor<float>(beta, new[] { width });
        var destination = new DenseTensor<float>(new[] { rows, width });
        string xHash = Hash(data), scaleHash = Hash(gamma), biasHash = Hash(beta);
        var held = Tensor<float>.LayerNormalization(x, scale, bias, -1, 1e-5f);
        string expected = Hash(held.ToArray());
        var watch = Stopwatch.StartNew(); long pairs = 0;
        do
        {
            Tensor<float>.LayerNormalization(x, scale, bias, destination, -1, 1e-5f);
            Tensor<float>.LayerNormalization(x, scale, null, destination, -1, 1e-5f);
            pairs++;
        } while (pairs < 128 || watch.Elapsed.TotalSeconds < 3);
        watch.Stop();
        Tensor<float>.LayerNormalization(x, scale, bias, destination, -1, 1e-5f);
        Require(Hash(destination.ToArray()) == expected && Hash(held.ToArray()) == expected, "Output changed");
        Require(Hash(data) == xHash && Hash(gamma) == scaleHash && Hash(beta) == biasHash, "Input changed");
        Require(!Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.ModuleName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase)), "Native ORT loaded");
        File.WriteAllText(Path.Combine(output, "result.json"), JsonSerializer.Serialize(new
        {
            passed = true, core_sha256 = core,
            probe_sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(Assembly.GetExecutingAssembly().Location))),
            runtime = Environment.Version.ToString(), hardware, flags, pairs, seconds = watch.Elapsed.TotalSeconds,
            inputs_unchanged = true, held_outputs_unchanged = true, input_sha256 = xHash,
            scale_sha256 = scaleHash, bias_sha256 = biasHash, output_sha256 = expected,
            scope = "Product code observation only; no latency comparison"
        }, new JsonSerializerOptions { WriteIndented = true }));
    }
}
