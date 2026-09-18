using System.Security.Cryptography;
using System.Text.Json;

namespace Lokad.Onnx.Backend.Tests;

public class IfOrtFixtureTests
{
    public static IEnumerable<object[]> Cases()
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "if-ort.json")));
        foreach (var item in doc.RootElement.GetProperty("cases").EnumerateArray())
            foreach (var mode in new[] { "scalar", "simd", "auto" })
                yield return new object[] { item.GetProperty("name").GetString()!, mode, item.GetRawText() };
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void ImportedGraphMatchesNativeOrt(string name, string mode, string record)
    {
        using var document = JsonDocument.Parse(record);
        var fixture = document.RootElement;
        var bytes = Convert.FromBase64String(fixture.GetProperty("model").GetString()!);
        Assert.Equal(fixture.GetProperty("sha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(bytes)));
        var graph = OnnxImport.Load(bytes);
        Assert.NotNull(graph);
        var inputs = new Dictionary<string, ITensor>();
        foreach (var item in fixture.GetProperty("inputs").EnumerateArray())
        {
            var dims = item.GetProperty("dims").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            inputs[item.GetProperty("name").GetString()!] = item.GetProperty("type").GetString() == "bool"
                ? new DenseTensor<bool>(item.GetProperty("data").EnumerateArray().Select(x => x.GetBoolean()).ToArray(), dims)
                : new DenseTensor<float>(item.GetProperty("data").EnumerateArray().Select(x => x.GetSingle()).ToArray(), dims);
        }
        var options = mode == "scalar" ? ExecutionOptions.Scalar : mode == "simd" ? ExecutionOptions.Simd : ExecutionOptions.Default;
        Assert.True(graph.Execute(inputs, true, ExecutionProvider.CPU, options), name + ": " + graph.LastErrorMessage);
        foreach (var output in fixture.GetProperty("outputs").EnumerateArray())
        {
            var tensor = (Tensor<float>)graph.Outputs[output.GetProperty("name").GetString()!]!;
            Assert.Equal(output.GetProperty("dims").EnumerateArray().Select(x => x.GetInt32()), ((ITensor)tensor).Dims);
            Assert.Equal(output.GetProperty("data").EnumerateArray().Select(x => BitConverter.SingleToInt32Bits(x.GetSingle())),
                tensor.ToArray().Select(BitConverter.SingleToInt32Bits));
        }
    }
}
