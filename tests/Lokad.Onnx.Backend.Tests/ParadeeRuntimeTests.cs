namespace Lokad.Onnx.Backend.Tests;

using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Lokad.Onnx.Tests.Support;
using Xunit.Abstractions;

// Private model lane: ordinary CI exercises the small operators without downloading
// weights. Explicitly supplying the fixture root makes missing/corrupt assets failures.
public class ParadeeRuntimeTests(ITestOutputHelper output)
{
    // Pitch errors accumulate as phase and are multiplied by 64 harmonics. Keep
    // checkpoints strict; qualify the final signal by peak error and residual energy.
    const double MaximumSampleError = 1e-3, MinimumSnrDb = 50, MaximumUpstreamError = 1e-4;
    [SkippableFact]
    public void Fp32ControlledNoiseMatchesOrt() => Qualify("fp32", true);
    [SkippableFact]
    public void Fp32OriginalExportRuns() => Qualify("fp32", false);
    [SkippableFact]
    public void Int8ControlledNoiseMatchesOrt() => Qualify("int8", true);
    [SkippableFact]
    public void Int8OriginalExportRuns() => Qualify("int8", false);

    static string FileHash(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }

    static string TensorHash<T>(Tensor<T> tensor) where T : unmanaged =>
        Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(tensor.ToArray().AsSpan())));

    static string InputHash(ITensor tensor) => tensor switch
    {
        Tensor<float> t => TensorHash(t),
        Tensor<long> t => TensorHash(t),
        _ => throw new InvalidDataException("Unexpected fixture input type."),
    };

    static int[] Shape(JsonElement value) => value.GetProperty("shape").EnumerateArray().Select(d => d.GetInt32()).ToArray();

    static string CheckedPath(string root, JsonElement descriptor, string pathProperty, string hashProperty)
    {
        string path = Path.Combine(root, descriptor.GetProperty(pathProperty).GetString()!);
        Assert.True(File.Exists(path), "Missing fixture: " + path);
        Assert.Equal(descriptor.GetProperty(hashProperty).GetString(), FileHash(path));
        return path;
    }

    static ITensor Read(string root, JsonElement descriptor)
    {
        var tensor = NpySupport.ReadTensor(CheckedPath(root, descriptor, "file", "sha256"));
        Assert.Equal(Shape(descriptor), tensor.Dims);
        return tensor;
    }

    static double MaximumScaledError(ITensor expected, ITensor actual)
    {
        Assert.Equal(expected.ElementType, actual.ElementType);
        Assert.Equal(expected.Dims, actual.Dims);
        static double[] Values(ITensor tensor) => tensor switch
        {
            Tensor<float> f => f.ToArray().Select(v => (double)v).ToArray(),
            Tensor<double> d => d.ToArray(),
            _ => throw new InvalidDataException("Expected floating checkpoint."),
        };
        var left = Values(expected);
        var right = Values(actual);
        double maximum = 0;
        for (int i = 0; i < left.Length; i++)
        {
            Assert.True(double.IsFinite(left[i]) && double.IsFinite(right[i]));
            maximum = Math.Max(maximum, Math.Abs(left[i] - right[i]) / Math.Max(1, Math.Abs(left[i])));
        }
        return maximum;
    }

    void Qualify(string variant, bool controlled)
    {
        string? root = Environment.GetEnvironmentVariable("LONNX_PARADEE_FIXTURES");
        Skip.If(string.IsNullOrWhiteSpace(root), "Set LONNX_PARADEE_FIXTURES to the frozen private fixture directory.");
        Assert.True(Directory.Exists(root), "Explicit Paradee fixture directory does not exist: " + root);
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root!, "manifest.json")));
        var manifest = document.RootElement;
        output.WriteLine("core_sha256={0}; runtime={1}", FileHash(typeof(ComputationalGraph).Assembly.Location), Environment.Version);
        Assert.Equal(2, manifest.GetProperty("schema").GetInt32());
        var contract = manifest.GetProperty("waveform_contract");
        Assert.Equal(MaximumSampleError, contract.GetProperty("maximum_absolute_error").GetDouble());
        Assert.Equal(MinimumSnrDb, contract.GetProperty("minimum_snr_db").GetDouble());
        Assert.Equal(MaximumUpstreamError, contract.GetProperty("upstream_maximum_scaled_error").GetDouble());
        Assert.Equal("ed2934d4df8a4fd891e29584c5e1104649409765", manifest.GetProperty("revision").GetString());
        Assert.Equal(manifest.GetProperty("input_fixture_sha256").GetString(), FileHash(Path.Combine(root!, "paradee-inputs.json")));
        using var inputsDocument = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "fixtures", "paradee-inputs.json")));
        var declared = inputsDocument.RootElement.GetProperty("cases").EnumerateArray().ToDictionary(c => c.GetProperty("key").GetString()!);
        Assert.Equal(8, declared.Count);
        var model = manifest.GetProperty("models").EnumerateArray().Single(m => m.GetProperty("variant").GetString() == variant);
        Assert.Equal(variant == "fp32" ? "8bac7b0bebef403cb22b54ead6bee423fa34a04fd91f73326b7f65cd9d0d25b9"
            : "a6989796692ac173597e15d405c971989202d5c73cfad65f22e90bd7b6c97da1", model.GetProperty("original_sha256").GetString());
        string role = controlled ? "controlled" : "original";
        string path = CheckedPath(root!, model, role, role + "_sha256");
        var graph = OnnxImport.Load(path);
        Assert.True(graph is not null, OnnxImport.LastErrorMessage);
        graph!.Prepare();
        var cases = model.GetProperty("cases").EnumerateArray().ToArray();
        Assert.Equal(declared.Keys.Order(), cases.Select(c => c.GetProperty("key").GetString()!).Order());
        var held = new List<(float[] Values, string Hash)>();
        var numericalFailures = new List<string>();
        foreach (var item in cases)
        {
            string key = item.GetProperty("key").GetString()!;
            var referenceInputs = declared[key];
            var tensors = item.GetProperty("tensors");
            var feed = new Dictionary<string, ITensor>();
            foreach (string name in new[] { "input_ids", "speed", "voice" }) feed[name] = Read(root!, tensors.GetProperty(name));
            Assert.Equal(referenceInputs.GetProperty("input_ids").EnumerateArray().Select(v => v.GetInt64()), ((Tensor<long>)feed["input_ids"]).ToArray());
            Assert.Equal((float)referenceInputs.GetProperty("speed").GetDouble(), ((Tensor<float>)feed["speed"]).GetValue(0));
            Assert.Equal(referenceInputs.GetProperty("voice").GetInt64(), ((Tensor<long>)feed["voice"]).GetValue(0));
            if (controlled) feed[model.GetProperty("noise_input").GetString()!] = Read(root!, tensors.GetProperty("paradee_noise"));
            var inputHashes = feed.ToDictionary(p => p.Key, p => InputHash(p.Value));
            var reference = (Tensor<float>)Read(root!, tensors.GetProperty("waveform"));
            float[]? first = null;
            double worst = 0;
            double peak = 0;
            double squaredError = 0, squaredReference = 0;
            for (int repeat = 0; repeat < 2; repeat++)
            {
                graph.Reset();
                Assert.True(graph.Execute(feed, false), variant + "/" + key + ": " + graph.LastErrorMessage);
                var waveform = Assert.IsAssignableFrom<Tensor<float>>(graph.Outputs["waveform"]);
                Assert.Equal(reference.Dimensions.ToArray(), waveform.Dimensions.ToArray());
                var actual = waveform.ToArray();
                Assert.All(actual, value => Assert.True(float.IsFinite(value), "Nonfinite waveform."));
                if (controlled)
                {
                    var expected = reference.ToArray();
                    for (int i = 0; i < actual.Length; i++)
                    {
                        double difference = Math.Abs((double)actual[i] - expected[i]);
                        peak = Math.Max(peak, difference);
                        worst = Math.Max(worst, difference / Math.Max(1, Math.Abs((double)expected[i])));
                        squaredError += ((double)actual[i] - expected[i]) * ((double)actual[i] - expected[i]);
                        squaredReference += (double)expected[i] * expected[i];
                    }
                    var durations = Assert.IsAssignableFrom<Tensor<float>>(graph.Outputs[model.GetProperty("duration_output").GetString()!]);
                    var expectedDurations = (Tensor<float>)Read(root!, tensors.GetProperty("durations"));
                    Assert.Equal(expectedDurations.Dimensions.ToArray(), durations.Dimensions.ToArray());
                    Assert.Equal(expectedDurations.ToArray(), durations.ToArray());
                    foreach (var checkpoint in model.GetProperty("upstream_outputs").EnumerateObject())
                    {
                        var expectedCheckpoint = Read(root!, tensors.GetProperty(checkpoint.Name));
                        double error = MaximumScaledError(expectedCheckpoint, graph.Outputs[checkpoint.Value.GetString()!]!);
                        Assert.True(error <= MaximumUpstreamError, key + "/" + checkpoint.Name + " scaled error " + error.ToString("R"));
                    }
                    if (first is not null) Assert.Equal(first, actual);
                }
                first ??= actual;
                foreach (var input in feed) Assert.Equal(inputHashes[input.Key], InputHash(input.Value));
                foreach (var previous in held)
                    Assert.Equal(previous.Hash, Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(previous.Values.AsSpan()))));
            }
            double relative = squaredReference == 0 ? (squaredError == 0 ? 0 : double.PositiveInfinity) : Math.Sqrt(squaredError / squaredReference);
            double snr = relative == 0 ? double.PositiveInfinity : -20 * Math.Log10(relative);
            if (controlled)
            {
                output.WriteLine("{0}/{1}/{2}: shape=[{3}], max_scaled_error={4:R}, peak_error={5:R}, relative_l2={6:R}, snr_db={7:R}",
                    variant, role, key, string.Join(",", reference.Dimensions.ToArray()), worst, peak, relative, snr);
                if (peak > MaximumSampleError || snr < MinimumSnrDb)
                    numericalFailures.Add(variant + "/" + key + " peak " + peak.ToString("R") + " SNR " + snr.ToString("R"));
            }
            else output.WriteLine("{0}/{1}/{2}: shape=[{3}], finite waveform; no cross-engine random-sample equality claim.",
                variant, role, key, string.Join(",", reference.Dimensions.ToArray()));
            held.Add((first!, Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(first!.AsSpan())))));
        }
        Assert.True(numericalFailures.Count == 0, string.Join(Environment.NewLine, numericalFailures));
        output.WriteLine("PASS {0}/{1}: all eight fixed inputs executed twice.", variant, role);
    }
}
