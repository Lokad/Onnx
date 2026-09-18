using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;

namespace Lokad.Onnx.Backend.Tests;

public class ClipReferenceTests
{
    static JsonDocument Fixture() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "clip-ort.json")));

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
        "int32" => Input<int>(value, mode), "int64" => Input<long>(value, mode),
        _ => throw new InvalidOperationException(dtype)
    };

    static string Bits(ITensor tensor) => tensor switch
    {
        Tensor<float> t => Convert.ToHexStringLower(MemoryMarshal.AsBytes(t.ToArray().AsSpan())),
        Tensor<double> t => Convert.ToHexStringLower(MemoryMarshal.AsBytes(t.ToArray().AsSpan())),
        Tensor<int> t => Convert.ToHexStringLower(MemoryMarshal.AsBytes(t.ToArray().AsSpan())),
        Tensor<long> t => Convert.ToHexStringLower(MemoryMarshal.AsBytes(t.ToArray().AsSpan())),
        _ => throw new InvalidOperationException(tensor.ElementType.ToString())
    };

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void ImportedGraphMatchesNativeBytesAcrossVersionsTypesAndLayouts(string name, int mode)
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
        var held = new List<ITensor>();
        for (int call = 0; call < 2; call++)
        {
            graph.Reset();
            Assert.True(graph.Execute(inputs, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
            var output = graph.Outputs["y"];
            Assert.Equal(expected.GetProperty("shape").EnumerateArray().Select(d => d.GetInt32()), output.Dims);
            Assert.Equal(expected.GetProperty("bytes").GetString(), Bits(output));
            held.Add(output);
            foreach (var pair in inputs) Assert.Equal(before[pair.Key], Bits(pair.Value));
            graph.Reset();
            Assert.False(graph.Execute(new Dictionary<string, ITensor>(), false, ExecutionProvider.CPU, options));
            foreach (var value in held) Assert.Equal(expected.GetProperty("bytes").GetString(), Bits(value));
        }
    }
}
