using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;

namespace Lokad.Onnx.Backend.Tests;

public class SignalReferenceTests
{
    static JsonDocument Fixture() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "signal-reference.json")));

    public static IEnumerable<object[]> Cases()
    {
        using var document = Fixture();
        foreach (var value in document.RootElement.GetProperty("cases").EnumerateArray())
            for (int mode = 0; mode < 3; mode++)
                yield return new object[] { value.GetProperty("name").GetString() ?? throw new InvalidDataException(), mode };
    }

    static Tensor<T> Input<T>(JsonElement value, int mode) where T : unmanaged
    {
        int[] shape = value.GetProperty("shape").EnumerateArray().Select(d => d.GetInt32()).ToArray();
        T[] data = value.GetProperty("values").Deserialize<T[]>() ?? throw new InvalidDataException();
        var backing = new T[data.Length + 4];
        var tensor = new DenseTensor<T>(backing.AsMemory(2, data.Length), shape, mode == 2);
        for (int i = 0; i < data.Length; i++)
        {
            int flat = i;
            var indices = new int[shape.Length];
            for (int d = shape.Length - 1; d >= 0; d--) { indices[d] = flat % shape[d]; flat /= shape[d]; }
            tensor[indices] = data[i];
        }
        return tensor;
    }

    static ITensor Input(JsonElement value, int mode) => value.GetProperty("dtype").GetString() switch
    {
        "float32" => Input<float>(value, mode), "float64" => Input<double>(value, mode),
        "int32" => Input<int>(value, mode), "int64" => Input<long>(value, mode),
        _ => throw new InvalidDataException()
    };

    static string Bits(ITensor tensor) => tensor switch
    {
        Tensor<float> t => Convert.ToHexString(MemoryMarshal.AsBytes(t.ToArray().AsSpan())),
        Tensor<double> t => Convert.ToHexString(MemoryMarshal.AsBytes(t.ToArray().AsSpan())),
        Tensor<int> t => Convert.ToHexString(MemoryMarshal.AsBytes(t.ToArray().AsSpan())),
        Tensor<long> t => Convert.ToHexString(MemoryMarshal.AsBytes(t.ToArray().AsSpan())),
        _ => throw new InvalidDataException()
    };

    static void AssertMatches(JsonElement expected, ITensor actual)
    {
        var reference = Input(expected, 0);
        Assert.Equal(reference.ElementType, actual.ElementType);
        Assert.Equal(reference.Dims, actual.Dims);
        if (actual is Tensor<int> || actual is Tensor<long>) { Assert.Equal(Bits(reference), Bits(actual)); return; }
        double[] values = actual is Tensor<float> single ? single.ToArray().Select(v => (double)v).ToArray() : ((Tensor<double>)actual).ToArray();
        double[] wanted = reference is Tensor<float> floats ? floats.ToArray().Select(v => (double)v).ToArray() : ((Tensor<double>)reference).ToArray();
        double tolerance = actual is Tensor<float> ? 1e-4 : 1e-10;
        for (int i = 0; i < values.Length; i++)
            Assert.True(double.IsFinite(values[i]) && Math.Abs(values[i] - wanted[i]) <= tolerance * Math.Max(1, Math.Abs(wanted[i])),
                $"[{i}]: expected {wanted[i]:R}, got {values[i]:R}");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void ImportedGraphMatchesCompleteReferencesAndOwnsResults(string name, int mode)
    {
        using var document = Fixture();
        var item = document.RootElement.GetProperty("cases").EnumerateArray().Single(c => c.GetProperty("name").GetString() == name);
        byte[] model = Convert.FromBase64String(item.GetProperty("model").GetString() ?? throw new InvalidDataException());
        Assert.Equal(item.GetProperty("model_sha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(model)));
        var graph = OnnxImport.Load(model) ?? throw new InvalidDataException("Model import failed");
        if (mode == 2) graph = graph.CreateExecution(null);
        var inputs = item.GetProperty("inputs").EnumerateArray().Select((v, i) => (v, i))
            .Where(p => p.v.ValueKind != JsonValueKind.Null).ToDictionary(p => "x" + p.i, p => Input(p.v, mode));
        var before = inputs.ToDictionary(p => p.Key, p => Bits(p.Value));
        var held = new List<(ITensor Tensor, string Bits)>();
        var options = mode == 0 ? ExecutionOptions.Scalar : ExecutionOptions.Simd;
        for (int call = 0; call < 2; call++)
        {
            graph.Reset();
            Assert.True(graph.Execute(inputs, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
            var output = graph.Outputs["y"];
            AssertMatches(item.GetProperty("expected"), output);
            if (item.GetProperty("op").GetString() == "ReduceSum") Assert.Equal(Bits(Input(item.GetProperty("expected"), 0)), Bits(output));
            held.Add((output, Bits(output)));
            foreach (var pair in inputs) Assert.Equal(before[pair.Key], Bits(pair.Value));
            graph.Reset();
            Assert.False(graph.Execute(new Dictionary<string, ITensor>(), false, ExecutionProvider.CPU, options));
            foreach (var value in held) Assert.Equal(value.Bits, Bits(value.Tensor));
        }
    }
}
