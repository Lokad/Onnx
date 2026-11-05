namespace Lokad.Onnx.Backend.Tests;

using System.Numerics;
using System.Runtime.InteropServices;

public class NormalizationValidationTests
{
    static DenseTensor<float> Data() => new(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 1, 2, 3 });
    static DenseTensor<float> Scale() => DenseTensor<float>.OfValues(new[] { 1f, 2f });

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    public void InstanceNormalizationRejectsInvalidInputsWithoutWrites(int invalid)
    {
        ITensor? input = Data(), scale = Scale(), bias = Scale();
        switch (invalid)
        {
            case 0: input = null; break;
            case 1: scale = null; break;
            case 2: bias = null; break;
            case 3: input = new DenseTensor<double>(new[] { 1, 2, 3 }); break;
            case 4: scale = DenseTensor<double>.OfValues(new[] { 1.0, 2.0 }); break;
            case 5: bias = DenseTensor<double>.OfValues(new[] { 1.0, 2.0 }); break;
            case 6: input = new DenseTensor<float>(new[] { 2, 3 }); break;
            case 7: scale = new DenseTensor<float>(new[] { 1, 2 }); break;
            case 8: bias = new DenseTensor<float>(new[] { 1 }); break;
            case 9: input = DenseTensor<float>.Scalar(1f); break;
        }
        float[]? before = (input as Tensor<float>)?.ToArray();
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.InstanceNorm(input, scale, bias, null, null).Status);
        if (input is Tensor<float> tensor) Assert.Equal(before, tensor.ToArray());
    }

    [Theory]
    [InlineData(-4)]
    [InlineData(3)]
    [InlineData(int.MinValue)]
    [InlineData(int.MaxValue)]
    public void LogSoftmaxRejectsInvalidAxis(int axis)
    {
        var input = Data();
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.LogSoftmax(input, axis, null, null, 17).Status);
        Assert.Equal(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, input.ToArray());
    }

    [Fact]
    public void UnaryOperationsRejectMissingAndUnsupportedInput()
    {
        foreach (ITensor? input in new ITensor?[] { null, DenseTensor<bool>.OfValues(new[] { true }), DenseTensor<int>.OfValues(new[] { 1 }) })
        {
            Assert.Equal(OpStatus.Failure, CPUExecutionProvider.LeakyRelu(input, null, null).Status);
            Assert.Equal(OpStatus.Failure, CPUExecutionProvider.LogSoftmax(input, 0, null, null, 17).Status);
        }
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.LogSoftmax(DenseTensor<float>.Scalar(1f), -1, null, null, 17).Status);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(11)]
    [InlineData(13)]
    [InlineData(17)]
    public void LogSoftmaxNodeUsesResolvedVersionForOmittedAxis(int version)
    {
        var graph = new ComputationalGraph { Opset = new Dictionary<string, int> { [""] = 13 } };
        var data = new DenseTensor<float>(new float[12], new[] { 2, 2, 3 });
        graph.Inputs["x"] = data;
        var node = new Node { Name = "log", Op = OpType.LogSoftmax, OpTypeName = "LogSoftmax", OpsetVersion = version,
            Inputs = new[] { "x" }, Outputs = new[] { "y" } };
        var result = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, result.Status);
        float expected = -MathF.Log(version == 11 ? 6 : 3);
        Assert.All(((Tensor<float>)result.Outputs[0]).ToArray(), actual => Assert.Equal(expected, actual));
    }

    [Fact]
    public void LeakyReluPreservesSignedZeroAndUsesFloatAttributeForDouble()
    {
        var input = DenseTensor<double>.OfValues(new[] { -0.0, 0.0, -1.0, 2.0 });
        var result = CPUExecutionProvider.LeakyRelu(input, null, null);
        var output = ((Tensor<double>)result.Outputs[0]).ToArray();
        Assert.Equal(long.MinValue, BitConverter.DoubleToInt64Bits(output[0]));
        Assert.Equal(0L, BitConverter.DoubleToInt64Bits(output[1]));
        Assert.Equal(-(double).01f, output[2]);
        Assert.Equal(2.0, output[3]);
    }

    public static IEnumerable<object?[]> LeakyAlphaValues()
    {
        foreach (float? alpha in new float?[] { null, .1f, .2f, 0f, -0f, -.5f,
            float.PositiveInfinity, float.NegativeInfinity, float.NaN })
            yield return new object?[] { alpha };
    }

    [Theory]
    [MemberData(nameof(LeakyAlphaValues))]
    public void LeakyReluFloatPreservesPayloadBitsAcrossModesAndLayouts(float? alpha)
    {
        var values = Enumerable.Range(0, 513).Select(i =>
            BitConverter.Int32BitsToSingle(unchecked((int)(0x9e3779b9u * (uint)i)))).ToArray();
        int[] exceptional = { 0, int.MinValue, 0x7f800000, unchecked((int)0xff800000),
            0x7fc01234, unchecked((int)0xffc02345), 0x7f801234, unchecked((int)0xff802345) };
        for (int i = 0; i < exceptional.Length; i++)
            values[i] = values[values.Length - 1 - i] = BitConverter.Int32BitsToSingle(exceptional[i]);
        AssertLeakyBits(values, alpha ?? .01f, alpha);
    }

    [Theory]
    [MemberData(nameof(LeakyAlphaValues))]
    public void LeakyReluDoublePreservesPayloadBitsAcrossModesAndLayouts(float? alpha)
    {
        var values = Enumerable.Range(0, 513).Select(i =>
            BitConverter.Int64BitsToDouble(unchecked((long)(0x9e3779b97f4a7c15ul * (ulong)i)))).ToArray();
        long[] exceptional = { 0, long.MinValue, 0x7ff0000000000000, unchecked((long)0xfff0000000000000),
            0x7ff8000000001234, unchecked((long)0xfff8000000002345),
            0x7ff0000000001234, unchecked((long)0xfff0000000002345) };
        for (int i = 0; i < exceptional.Length; i++)
            values[i] = values[values.Length - 1 - i] = BitConverter.Int64BitsToDouble(exceptional[i]);
        AssertLeakyBits(values, (double)(alpha ?? .01f), alpha);
    }

    static void AssertLeakyBits<T>(T[] values, T alpha, float? attribute)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var expected = values.Select(value => value < T.Zero ? alpha * value : value).ToArray();
        var storage = new T[values.Length + 6];
        values.CopyTo(storage, 3);
        var dense = new DenseTensor<T>(storage.AsMemory(3, values.Length), new[] { values.Length });
        var backing = new DenseTensor<T>(new[] { values.Length, 2 });
        for (int i = 0; i < values.Length; i++) backing[i, 0] = values[i];
        var view = backing.Slice(new SliceIndex(0, values.Length), SliceIndex.Index(0));
        var before = MemoryMarshal.AsBytes(storage.AsSpan()).ToArray();
        foreach (var input in new Tensor<T>[] { dense, view })
        foreach (var mode in new[] { ExecutionOptions.Scalar, ExecutionOptions.Simd, ExecutionOptions.Default })
        {
            var result = CPUExecutionProvider.LeakyRelu(input, attribute, mode);
            Assert.Equal(OpStatus.Success, result.Status);
            var output = Assert.IsType<DenseTensor<T>>(result.Outputs[0]);
            Assert.Equal(input.Dimensions.ToArray(), output.Dimensions.ToArray());
            Assert.True(MemoryMarshal.AsBytes(expected.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(output.Buffer.Span)));
            output.SetValue(0, T.One);
            Assert.True(MemoryMarshal.AsBytes(values.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(input.ToArray().AsSpan())));
        }
        Assert.Equal(before, MemoryMarshal.AsBytes(storage.AsSpan()).ToArray());
    }

    [Fact]
    public void LeakyReluScalarAndEmptyInputsRetainShape()
    {
        foreach (var input in new[] { DenseTensor<float>.Scalar(-2f), new DenseTensor<float>(new[] { 2, 0, 3 }) })
        {
            var result = CPUExecutionProvider.LeakyRelu(input, .25f, ExecutionOptions.Default);
            Assert.Equal(OpStatus.Success, result.Status);
            var output = (Tensor<float>)result.Outputs[0];
            Assert.Equal(input.Dimensions.ToArray(), output.Dimensions.ToArray());
            Assert.Equal(input.Length, output.Length);
            if (input.Length != 0) Assert.Equal(-.5f, output.GetValue(0));
        }
    }

    [Theory]
    [InlineData(OpType.InstanceNormalization, 5, 3, false)]
    [InlineData(OpType.InstanceNormalization, 6, 3, true)]
    [InlineData(OpType.InstanceNormalization, 17, 2, false)]
    [InlineData(OpType.InstanceNormalization, 17, 4, false)]
    [InlineData(OpType.LeakyRelu, 5, 1, false)]
    [InlineData(OpType.LeakyRelu, 6, 1, true)]
    [InlineData(OpType.LeakyRelu, 17, 2, false)]
    [InlineData(OpType.LogSoftmax, 11, 1, true)]
    [InlineData(OpType.LogSoftmax, 17, 0, false)]
    [InlineData(OpType.LogSoftmax, 17, 2, false)]
    public void RegistryEnforcesVersionAndArity(OpType op, int version, int inputs, bool supported)
    {
        var node = new Node { Name = "test", Op = op, OpTypeName = op.ToString(), OpsetVersion = version,
            Inputs = Enumerable.Range(0, inputs).Select(i => "x" + i).ToArray(), Outputs = new[] { "y" } };
        Assert.Equal(supported, CPUExecutionProvider.SupportsNode(node));
    }
}
