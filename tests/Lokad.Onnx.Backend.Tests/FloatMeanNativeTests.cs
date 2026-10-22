using System.Text.Json;

namespace Lokad.Onnx.Backend.Tests;

public class FloatMeanNativeTests
{
    public static IEnumerable<object[]> Cases()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "reducemean-ort.json")));
        foreach (var item in document.RootElement.GetProperty("cases").EnumerateArray())
            yield return new object[]
            {
                item.GetProperty("name").GetString()!,
                item.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
                item.GetProperty("axes").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
                item.GetProperty("pattern").GetString()!,
                item.GetProperty("expectedBits").EnumerateArray().Select(x => x.GetUInt32()).ToArray()
            };
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchesNativeAcrossAxesLengthsAndLayouts(string name, int[] shape, int[] axes, string pattern, uint[] expectedBits)
    {
        float[] cycle = pattern switch
        {
            "cancellation" => [100000000, 1, -100000000, 1],
            "extreme" => [float.MaxValue, float.MaxValue, -float.MaxValue, -float.MaxValue],
            "special" => new uint[] { 0x7fc12345, 0x7f800000, 0xff800000, 0x80000000, 0, 0x3f800000 }
                .Select(BitConverter.UInt32BitsToSingle).ToArray(),
            _ => throw new ArgumentOutOfRangeException(nameof(pattern))
        };
        int length = shape.Aggregate(1, (a, b) => a * b);
        float[] values = Enumerable.Range(0, length).Select(i => cycle[i % cycle.Length]).ToArray();
        foreach (bool reversed in new[] { false, true })
        {
            var input = new DenseTensor<float>(new float[length], shape, reversed);
            int flat = 0;
            foreach (var index in input.GetDimensionsIterator()) input[index] = values[flat++];
            foreach (var options in new[] { TensorExecutionOptions.Scalar, TensorExecutionOptions.Simd, TensorExecutionOptions.Auto })
            foreach (bool keep in new[] { false, true })
            {
                var result = Tensor<float>.ReduceMean(input, DenseTensor<int>.OfValues(axes), keep, false, options);
                float[] actual = result.ToArray();
                Assert.Equal(expectedBits.Length, actual.Length);
                for (int i = 0; i < actual.Length; i++)
                {
                    float expected = BitConverter.UInt32BitsToSingle(expectedBits[i]);
                    bool good = float.IsNaN(expected) ? float.IsNaN(actual[i])
                        : float.IsInfinity(expected) ? expected == actual[i]
                        : float.IsFinite(actual[i]) && Math.Abs((double)actual[i] - expected) / Math.Max(1d, Math.Abs(expected)) <= 1e-4;
                    Assert.True(good, $"{name} at {i}: expected {expected}, actual {actual[i]}, reversed={reversed}, keep={keep}");
                }
                int[] expectedShape = shape.Where((_, i) => !axes.Contains(i)).ToArray();
                if (keep) expectedShape = shape.Select((d, i) => axes.Contains(i) ? 1 : d).ToArray();
                Assert.Equal(expectedShape, result.Dimensions.ToArray());
                Assert.False(TensorAlias.SharesBackingMemory(input, result));
            }
            Assert.Equal(values.Select(BitConverter.SingleToUInt32Bits), input.ToArray().Select(BitConverter.SingleToUInt32Bits));
        }
    }
}
