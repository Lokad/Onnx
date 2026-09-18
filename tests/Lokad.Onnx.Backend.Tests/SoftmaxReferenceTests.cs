using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;

namespace Lokad.Onnx.Backend.Tests;

public class SoftmaxReferenceTests
{
    public static IEnumerable<object[]> Cases()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "softmax-ort.json")));
        foreach (var item in document.RootElement.GetProperty("cases").EnumerateArray())
            for (int mode = 0; mode < 3; mode++) yield return new object[] { item.GetProperty("name").GetString()!, mode, item.GetRawText() };
    }

    static float[] Values(JsonElement tensor) => MemoryMarshal.Cast<byte, float>(Convert.FromHexString(tensor.GetProperty("bytes").GetString()!)).ToArray();
    static Tensor<float> Input(JsonElement value, int mode)
    {
        int[] shape = value.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var data = Values(value);
        var storage = Enumerable.Repeat(1234567f, data.Length + 8).ToArray();
        var input = new DenseTensor<float>(storage.AsMemory(4, data.Length), shape, mode == 2);
        for (int i = 0; i < data.Length; i++)
        {
            int flat = i; var indices = new int[shape.Length];
            for (int d = shape.Length - 1; d >= 0; d--) { indices[d] = flat % shape[d]; flat /= shape[d]; }
            input[indices] = data[i];
        }
        return input;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void NativeGraphAgreementAcrossLayoutsAndModes(string name, int mode, string json)
    {
        using var document = JsonDocument.Parse(json); var fixture = document.RootElement;
        var raw = Convert.FromBase64String(fixture.GetProperty("model").GetString()!);
        Assert.Equal(fixture.GetProperty("sha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(raw)));
        var plan = OnnxImport.Load(raw); Assert.NotNull(plan);
        var graph = mode == 2 ? plan.CreateExecution(null) : plan;
        var input = fixture.GetProperty("inputs").EnumerateObject().ToDictionary(p => p.Name, p => (ITensor)Input(p.Value, mode));
        var before = input.ToDictionary(p => p.Key, p => ((Tensor<float>)p.Value).ToArray().Select(BitConverter.SingleToInt32Bits).ToArray());
        var options = mode == 0 ? ExecutionOptions.Scalar : mode == 1 ? ExecutionOptions.Simd : ExecutionOptions.Default;
        var expected = Values(fixture.GetProperty("output"));
        var held = new List<(Tensor<float> Tensor, int[] Bits)>();
        for (int call = 0; call < 2; call++)
        {
            Assert.True(graph.Execute(input, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
            var tensor = (Tensor<float>)graph.Outputs["y"]!; var actual = tensor.ToArray();
            Assert.Equal(fixture.GetProperty("output").GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()), tensor.Dimensions.ToArray());
            for (int i = 0; i < actual.Length; i++)
            {
                if (float.IsNaN(expected[i])) Assert.True(float.IsNaN(actual[i]), name + " expected NaN");
                else Assert.True(float.IsFinite(actual[i]) && Math.Abs((double)actual[i] - expected[i]) <= 1e-6,
                    $"{name}[{i}]: expected {expected[i]:R}, actual {actual[i]:R}");
            }
            held.Add((tensor, actual.Select(BitConverter.SingleToInt32Bits).ToArray()));
            foreach (var p in input) Assert.Equal(before[p.Key], ((Tensor<float>)p.Value).ToArray().Select(BitConverter.SingleToInt32Bits));
            graph.Reset();
            Assert.False(graph.Execute(new Dictionary<string, ITensor>(), true, ExecutionProvider.CPU, options));
            foreach (var saved in held) Assert.Equal(saved.Bits, saved.Tensor.ToArray().Select(BitConverter.SingleToInt32Bits));
        }
    }
}
