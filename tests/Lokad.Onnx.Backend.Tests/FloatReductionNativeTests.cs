using System.Text.Json;

namespace Lokad.Onnx.Backend.Tests;

public class FloatReductionNativeTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (string operation in new[] { "ReduceMean", "ReduceSum" })
        {
            using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", operation.ToLowerInvariant() + "-ort.json")));
            foreach (var item in document.RootElement.GetProperty("cases").EnumerateArray())
                yield return new object[]
                {
                    operation, item.GetProperty("name").GetString()!,
                    item.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
                    item.GetProperty("axes").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
                    item.GetProperty("pattern").GetString()!,
                    item.GetProperty("expectedBits").EnumerateArray().Select(x => x.GetUInt32()).ToArray()
                };
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchesNativeAcrossAxesLengthsAndLayouts(string operation, string name, int[] shape, int[] axes, string pattern, uint[] expectedBits)
    {
        float[] values = InputValues(shape, pattern);
        int length = values.Length;
        foreach (bool reversed in new[] { false, true })
        {
            var input = new DenseTensor<float>(new float[length], shape, reversed);
            int flat = 0;
            foreach (var index in input.GetDimensionsIterator()) input[index] = values[flat++];
            foreach (var options in operation == "ReduceSum" ? new[] { TensorExecutionOptions.Auto }
                : new[] { TensorExecutionOptions.Scalar, TensorExecutionOptions.Simd, TensorExecutionOptions.Auto })
            foreach (bool keep in new[] { false, true })
            {
                var result = operation == "ReduceSum"
                    ? Tensor<float>.ReduceSum(input, DenseTensor<int>.OfValues(axes), keep, false)
                    : Tensor<float>.ReduceMean(input, DenseTensor<int>.OfValues(axes), keep, false, options);
                AssertNative(expectedBits, result.ToArray(), name);
                int[] expectedShape = shape.Where((_, i) => !axes.Contains(i)).ToArray();
                if (keep) expectedShape = shape.Select((d, i) => axes.Contains(i) ? 1 : d).ToArray();
                Assert.Equal(expectedShape, result.Dimensions.ToArray());
                Assert.False(TensorAlias.SharesBackingMemory(input, result));
            }
            Assert.Equal(values.Select(BitConverter.SingleToUInt32Bits), input.ToArray().Select(BitConverter.SingleToUInt32Bits));
        }
    }

    public static IEnumerable<object[]> GraphCases() => Cases().Where(c => (string)c[0] == "ReduceSum"
        && (string)c[1] is "first-4-cancellation" or "partial-2-cancellation" or "first-two-4-extreme" or "middle-4-extreme");

    [Theory]
    [MemberData(nameof(GraphCases))]
    public void ImportedSumsMatchNativeAcrossEquivalentAxes(string operation, string name, int[] shape, int[] axes, string pattern, uint[] expectedBits)
    {
        var input = new DenseTensor<float>(InputValues(shape, pattern), shape);
        foreach (var requested in new[] { axes, axes.Reverse().ToArray(), axes.Select(a => a - shape.Length).Concat(axes).ToArray() })
        foreach (bool keep in new[] { false, true })
        {
            var model = new OnnxModel { Name = name };
            model.Opset[""] = 18;
            model.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = shape });
            model.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float,
                Dims = keep ? shape.Select((d, i) => axes.Contains(i) ? 1 : d).ToArray() : shape.Where((_, i) => !axes.Contains(i)).ToArray() });
            model.Initializers.Add(new OnnxTensor { Name = "a", ElementType = TensorElementType.Int64,
                Dims = [requested.Length], Data = requested.Select(a => (long)a).ToArray() });
            model.Nodes.Add(new OnnxNode { Name = "sum", OpType = operation, Inputs = ["x", "a"], Outputs = ["y"], Attributes = new() { ["keepdims"] = keep ? 1 : 0 } });
            var graph = Model.Load(model)!;
            foreach (var options in new[] { ExecutionOptions.Default, ExecutionOptions.Scalar, ExecutionOptions.Simd })
            {
                Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
                AssertNative(expectedBits, ((Tensor<float>)graph.Outputs["y"]!).ToArray(), name);
            }
        }
    }

    static float[] InputValues(int[] shape, string pattern)
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
        return Enumerable.Range(0, length).Select(i => cycle[i % cycle.Length]).ToArray();
    }

    static void AssertNative(uint[] expectedBits, float[] actual, string name)
    {
        Assert.Equal(expectedBits.Length, actual.Length);
        for (int i = 0; i < actual.Length; i++)
        {
            float expected = BitConverter.UInt32BitsToSingle(expectedBits[i]);
            bool good = float.IsNaN(expected) ? float.IsNaN(actual[i]) : float.IsInfinity(expected) ? expected == actual[i]
                : float.IsFinite(actual[i]) && Math.Abs((double)actual[i] - expected) / Math.Max(1d, Math.Abs(expected)) <= 1e-4;
            Assert.True(good, $"{name} at {i}: expected {expected}, actual {actual[i]}");
        }
    }
}
