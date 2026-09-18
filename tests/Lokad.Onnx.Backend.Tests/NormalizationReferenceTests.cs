using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;

namespace Lokad.Onnx.Backend.Tests;

public class NormalizationReferenceTests
{
    static JsonDocument Fixture() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "normalization-ort.json")));

    public static IEnumerable<object[]> Cases()
    {
        using var document = Fixture();
        foreach (var value in document.RootElement.GetProperty("cases").EnumerateArray())
            for (int mode = 0; mode < 3; mode++) yield return new object[] { value.GetProperty("name").GetString()!, mode };
    }

    static Tensor<T> Input<T>(JsonElement value, int mode) where T : unmanaged
    {
        int[] shape = value.GetProperty("shape").EnumerateArray().Select(d => d.GetInt32()).ToArray();
        T[] data = MemoryMarshal.Cast<byte, T>(Convert.FromHexString(value.GetProperty("bytes").GetString()!)).ToArray();
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

    static ITensor Input(string dtype, JsonElement value, int mode) => dtype switch
    {
        "float32" => Input<float>(value, mode), "float64" => Input<double>(value, mode),
        _ => throw new InvalidOperationException(dtype)
    };

    static string Bits(ITensor tensor) => tensor switch
    {
        Tensor<float> t => Convert.ToHexStringLower(MemoryMarshal.AsBytes(t.ToArray().AsSpan())),
        Tensor<double> t => Convert.ToHexStringLower(MemoryMarshal.AsBytes(t.ToArray().AsSpan())),
        _ => throw new InvalidOperationException(tensor.ElementType.ToString())
    };

    static void AssertMatches(JsonElement expected, ITensor actual)
    {
        var bytes = Convert.FromHexString(expected.GetProperty("bytes").GetString()!);
        double[] reference, values;
        double absolute, relative;
        if (actual is Tensor<float> single)
        {
            reference = MemoryMarshal.Cast<byte, float>(bytes).ToArray().Select(v => (double)v).ToArray();
            values = single.ToArray().Select(v => (double)v).ToArray();
            absolute = 1e-5; relative = 3e-6;
        }
        else
        {
            reference = MemoryMarshal.Cast<byte, double>(bytes).ToArray();
            values = ((Tensor<double>)actual).ToArray();
            absolute = 1e-12; relative = 1e-12;
        }
        Assert.Equal(reference.Length, values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            double e = reference[i], a = values[i];
            if (double.IsNaN(e)) Assert.True(double.IsNaN(a), $"[{i}]: expected NaN, got {a:R}");
            else if (double.IsInfinity(e)) Assert.Equal(e, a);
            else Assert.True(double.IsFinite(a) && Math.Abs(e - a) <= absolute + relative * Math.Abs(e),
                $"[{i}]: expected {e:R}, got {a:R}");
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void ImportedGraphMatchesNativeAcrossVersionsTypesAndLayouts(string name, int mode)
    {
        Skip.If(mode == 2 && !Fma.IsSupported, "Explicit intrinsic options require FMA");
        using var document = Fixture();
        var item = document.RootElement.GetProperty("cases").EnumerateArray().Single(c => c.GetProperty("name").GetString() == name);
        byte[] model = Convert.FromBase64String(item.GetProperty("model").GetString()!);
        Assert.Equal(item.GetProperty("model_sha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(model)));
        var plan = OnnxImport.Load(model)!;
        var graph = mode == 2 ? plan.CreateExecution(null) : plan;
        string dtype = item.GetProperty("dtype").GetString()!;
        var inputs = item.GetProperty("inputs").EnumerateObject().ToDictionary(p => p.Name, p => Input(dtype, p.Value, mode));
        var before = inputs.ToDictionary(p => p.Key, p => Bits(p.Value));
        var expected = item.GetProperty("output");
        var options = mode == 0 ? ExecutionOptions.Scalar : mode == 1 ? ExecutionOptions.Simd : ExecutionOptions.Intrinsics;
        var held = new List<(ITensor Tensor, string Bytes)>();
        for (int call = 0; call < 2; call++)
        {
            graph.Reset();
            Assert.True(graph.Execute(inputs, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
            var output = graph.Outputs["y"];
            Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(d => d.GetInt32()), output.Dims);
            AssertMatches(expected, output);
            held.Add((output, Bits(output)));
            foreach (var pair in inputs) Assert.Equal(before[pair.Key], Bits(pair.Value));
            graph.Reset();
            Assert.False(graph.Execute(new Dictionary<string, ITensor>(), false, ExecutionProvider.CPU, options));
            foreach (var value in held) Assert.Equal(value.Bytes, Bits(value.Tensor));
        }
    }
}
