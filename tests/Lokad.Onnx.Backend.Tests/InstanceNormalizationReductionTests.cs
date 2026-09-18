using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;

namespace Lokad.Onnx.Backend.Tests;

public class InstanceNormalizationReductionTests
{
    static JsonDocument Fixture() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "instance-normalization-long.json")));
    static float Value(string bits) => BitConverter.Int32BitsToSingle(unchecked((int)uint.Parse(bits, NumberStyles.HexNumber, CultureInfo.InvariantCulture)));
    static string Hash(float[] values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));

    public static IEnumerable<object[]> Cases()
    {
        using var fixture = Fixture();
        foreach (var item in fixture.RootElement.GetProperty("cases").EnumerateArray())
            for (int mode = 0; mode < 3; mode++) yield return new object[] { item.GetProperty("name").GetString()!, mode };
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void LongAndConstantRowsAgreeWithNativeAcrossWindowsAndLayouts(string name, int mode)
    {
        using var fixture = Fixture();
        var item = fixture.RootElement.GetProperty("cases").EnumerateArray().Single(v => v.GetProperty("name").GetString() == name);
        int width = item.GetProperty("width").GetInt32();
        var pattern = item.GetProperty("pattern").EnumerateArray().Select(v => Value(v.GetString()!)).ToArray();
        var storage = Enumerable.Repeat(1234567f, 6 * width + 10).ToArray();
        var x = new DenseTensor<float>(storage.AsMemory(5, 6 * width), new[] { 2, 3, width }, mode == 2);
        var expected = new float[6 * width];
        for (int row = 0; row < 6; row++)
        for (int i = 0; i < width; i++)
        {
            float value = pattern[(i + row * 5) % pattern.Length];
            x[row / 3, row % 3, i] = value;
            string key = unchecked((uint)BitConverter.SingleToInt32Bits(value)).ToString("x8", CultureInfo.InvariantCulture);
            expected[row * width + i] = Value(item.GetProperty("expected")[row].GetProperty(key).GetString()!);
        }
        Assert.Equal(item.GetProperty("input_sha256").GetString(), Hash(x.ToArray()));
        Assert.Equal(item.GetProperty("output_sha256").GetString(), Hash(expected));
        var before = storage.Select(BitConverter.SingleToInt32Bits).ToArray();
        var scale = DenseTensor<float>.OfValues(fixture.RootElement.GetProperty("scale").EnumerateArray().Select(v => Value(v.GetString()!)).ToArray());
        var bias = DenseTensor<float>.OfValues(fixture.RootElement.GetProperty("bias").EnumerateArray().Select(v => Value(v.GetString()!)).ToArray());
        var options = mode == 0 ? ExecutionOptions.Scalar : mode == 1 ? ExecutionOptions.Simd : ExecutionOptions.Default;
        var result = CPUExecutionProvider.InstanceNorm(x, scale, bias, 1e-5f, options);
        Assert.Equal(OpStatus.Success, result.Status);
        var output = (Tensor<float>)Assert.Single(result.Outputs);
        Assert.Equal(new[] { 2, 3, width }, output.Dimensions.ToArray());
        Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), output.ToArray().Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(before, storage.Select(BitConverter.SingleToInt32Bits));
        // Results must own storage even for degenerate, constant and offset rows.
        x.SetValue(0, 999f);
        Assert.Equal(item.GetProperty("output_sha256").GetString(), Hash(output.ToArray()));
    }
}
