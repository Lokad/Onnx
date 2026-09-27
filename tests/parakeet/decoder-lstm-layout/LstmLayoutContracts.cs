using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Xunit;

namespace Lokad.Onnx.Backend.Tests;

public class LstmLayoutContracts
{
    const int H = 640, Outputs = 4 * H;
    static string Sha(string file) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(file)));
    static string Hash(ReadOnlySpan<float> values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values)));
    static void Equal(ReadOnlySpan<float> a, ReadOnlySpan<float> b) => Assert.True(MemoryMarshal.AsBytes(a).SequenceEqual(MemoryMarshal.AsBytes(b)), "Exact float bytes");
    static float[] Values(int count) => Enumerable.Range(0, count).Select(i => (i % 113 - 56) * .015625f).ToArray();
    static float[] Flat(ReadOnlySpan<float> weight, int n, int columns)
    {
        var result = new float[weight.Length];
        for (int k = 0; k < n; k++)
        for (int o = 0; o < columns; o++) result[k * columns + o] = weight[o * n + k];
        return result;
    }
    static float[] Pack(ReadOnlySpan<float> flat, int n, int columns)
    {
        var result = new float[flat.Length]; int block = PreparedLstmProjection.ColumnsPerBlock;
        for (int o = 0; o < columns; o++)
        for (int k = 0; k < n; k++)
        {
            int group = o / block * block, width = Math.Min(block, columns - group);
            result[group * n + k * width + o - group] = flat[k * columns + o];
        }
        return result;
    }
    static void Raw(float[] input, float[] flat, int columns)
    {
        var packed = Pack(flat, input.Length, columns);
        string inputHash = Hash(input), packedHash = Hash(packed);
        const float Guard = 192837f;
        var expected = Enumerable.Repeat(Guard, columns + 2).ToArray();
        var actual = (float[])expected.Clone();
        CPUExecutionProvider.LstmProjectOrdered(input, flat, expected.AsSpan(1, columns));
        PreparedLstmProjection.Multiply(input, packed, actual.AsSpan(1, columns));
        Equal(expected, actual);
        long start = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 8; i++) PreparedLstmProjection.Multiply(input, packed, actual.AsSpan(1, columns));
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - start);
        Equal(expected, actual); Assert.Equal(inputHash, Hash(input)); Assert.Equal(packedHash, Hash(packed));
    }

    [Fact]
    public void BoundariesRetainBitsGuardsAndZeroAllocation()
    {
        foreach (int n in new[] { 0, 1, 3, 17 })
        foreach (int columns in new[] { 0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 2560 })
            Raw(Values(n), Values(n * columns), columns);
        Raw(Values(H), Values(H * Outputs), Outputs);
    }

    [Fact]
    public void ExceptionalValuesRetainPayloadsAndOperandOrder()
    {
        float[] values = [float.NegativeInfinity, float.PositiveInfinity, float.MaxValue, -float.MaxValue,
            float.Epsilon, -float.Epsilon, -0f, 0f, BitConverter.Int32BitsToSingle(0x7fc12345),
            BitConverter.Int32BitsToSingle(unchecked((int)0xffc54321)), 1f, -1f];
        for (int shift = 0; shift < values.Length; shift++)
            Raw(Enumerable.Range(0, 3).Select(i => values[(i + shift) % values.Length]).ToArray(),
                Enumerable.Range(0, 3 * 71).Select(i => values[(i + shift + 4) % values.Length]).ToArray(), 71);
    }

    [Fact]
    public void EveryCapturedProjectionUsesTheActualPreparedLayout()
    {
        string root = Environment.GetEnvironmentVariable("LSTM_LAYOUT_BASE")!;
        using var capture = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, "fixtures/result.json")));
        var arrays = new Dictionary<string, float[]>();
        DenseTensor<float> Load(JsonElement descriptor)
        {
            string name = descriptor.GetProperty("file").GetString()!;
            Assert.Equal(Path.GetFileName(name), name);
            string path = Path.Combine(root, "fixtures", name);
            Assert.Equal(descriptor.GetProperty("sha256").GetString(), Sha(path));
            if (!arrays.TryGetValue(name, out var data)) arrays[name] = data = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(path)).ToArray();
            return new DenseTensor<float>(data, descriptor.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray());
        }
        var prepared = new Dictionary<int, (ComputationalGraph graph, float[][] packed, float[][] flat)>();
        var outputHashes = new List<string>(); int calls = 0;
        foreach (var call in capture.RootElement.GetProperty("calls").EnumerateArray())
        {
            int index = call.GetProperty("index").GetInt32();
            var inputs = call.GetProperty("inputs").EnumerateArray().ToArray();
            if (!prepared.TryGetValue(index, out var entry))
            {
                var graph = new ComputationalGraph(64L * 1024 * 1024);
                graph.Metadata["Name"] = "captured-lstm-layout"; graph.Opset[""] = 17;
                graph.Inputs["x"] = Load(inputs[0]); graph.Inputs["h"] = Load(inputs[5]); graph.Inputs["c"] = Load(inputs[6]);
                graph.Initializers["w"] = Load(inputs[1]); graph.Initializers["r"] = Load(inputs[2]); graph.Initializers["b"] = Load(inputs[3]);
                graph.Outputs["y"] = Tensor<float>.Zeros(1, 1, 1, H);
                graph.Outputs["yh"] = Tensor<float>.Zeros(1, 1, H); graph.Outputs["yc"] = Tensor<float>.Zeros(1, 1, H);
                graph.Nodes.Add(new Node { Name = "lstm", Op = OpType.LSTM, OpTypeName = "LSTM", Domain = "", OpsetVersion = 17,
                    Inputs = ["x", "w", "r", "b", "", "h", "c"], Outputs = ["y", "yh", "yc"], Attributes = new() { ["hidden_size"] = H } });
                graph.Prepare(); Assert.Equal(2, graph.PackedLstmWeights.Count);
                Assert.Equal(2L * H * Outputs * sizeof(float), graph.RetainedPackedWeightBytes);
                var weights = new[] { (Tensor<float>)graph.Initializers["w"], (Tensor<float>)graph.Initializers["r"] };
                var packed = weights.Select(w => GraphLstmPacking.Resolve(graph.PackedLstmWeights, w)!).ToArray();
                var flat = weights.Select(w => Flat(w.ToArray(), H, Outputs)).ToArray();
                for (int i = 0; i < 2; i++) Equal(Pack(flat[i], H, Outputs), packed[i]);
                prepared[index] = entry = (graph, packed, flat);
            }
            for (int i = 0; i < 2; i++)
            {
                var input = Load(inputs[i == 0 ? 0 : 5]).ToArray();
                var expected = new float[Outputs]; var actual = new float[Outputs];
                CPUExecutionProvider.LstmProjectOrdered(input, entry.flat[i], expected);
                PreparedLstmProjection.Multiply(input, entry.packed[i], actual);
                Equal(expected, actual); outputHashes.Add(Hash(actual));
            }
            calls++;
        }
        Assert.Equal(380, calls); Assert.Equal(2, prepared.Count); Assert.Equal(760, outputHashes.Count);
        foreach (var entry in prepared.Values)
            for (int i = 0; i < 2; i++) Equal(Pack(entry.flat[i], H, Outputs), entry.packed[i]);
        using var stream = new FileStream(Environment.GetEnvironmentVariable("LSTM_LAYOUT_PROJECTIONS")!, FileMode.CreateNew);
        JsonSerializer.Serialize(stream, new { passed = true, calls, projections = outputHashes.Count, values = outputHashes.Count * Outputs, hashes = outputHashes });
    }

    [Fact]
    public void LoadedCandidateAndModeMatchTheFrozenProducts()
    {
        if (!OperatingSystem.IsLinux()) throw new PlatformNotSupportedException();
        string root = Environment.GetEnvironmentVariable("LSTM_LAYOUT_BASE")!;
        string mode = Environment.GetEnvironmentVariable("LSTM_LAYOUT_MODE")!;
        using var built = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, "built.json")));
        string core = typeof(ComputationalGraph).Assembly.Location;
        Assert.Equal(built.RootElement.GetProperty("products").GetProperty("candidate").GetProperty("sha256").GetString(), Sha(core));
        Assert.Equal(mode == "normal", Avx512F.IsSupported);
        Assert.Equal(mode != "scalar", Vector.IsHardwareAccelerated);
        Assert.Equal(mode != "scalar", Avx2.IsSupported && Fma.IsSupported);
        Assert.Equal("10.0.8", Environment.Version.ToString()); Assert.Equal(1, Environment.ProcessorCount);
        using var process = Process.GetCurrentProcess();
        Assert.Equal(4L, process.ProcessorAffinity.ToInt64());
        Assert.DoesNotContain(process.Modules.Cast<ProcessModule>(), m => m.FileName.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase));
        using var stream = new FileStream(Environment.GetEnvironmentVariable("LSTM_LAYOUT_IDENTITY")!, FileMode.CreateNew);
        JsonSerializer.Serialize(stream, new { passed = true, mode, pid = process.Id, core_sha256 = Sha(core),
            consumer_sha256 = Sha(Assembly.GetExecutingAssembly().Location), vector_count = Vector<float>.Count,
            block = PreparedLstmProjection.ColumnsPerBlock, avx512 = Avx512F.IsSupported, hardware = Vector.IsHardwareAccelerated,
            runtime = Environment.Version.ToString(), affinity = process.ProcessorAffinity.ToInt64() });
    }
}
