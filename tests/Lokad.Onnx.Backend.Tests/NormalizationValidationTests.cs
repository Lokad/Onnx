namespace Lokad.Onnx.Backend.Tests;

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
