namespace Lokad.Onnx.Backend.Tests;

public class ClipValidationTests
{
    static DenseTensor<float> Data() => DenseTensor<float>.OfValues(new[] { -4f, 0f, 1f, 4f });

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void InvalidBoundsFailWithoutChangingInput(int invalid)
    {
        var data = Data();
        ITensor bound = invalid switch
        {
            0 => DenseTensor<double>.OfValues(new[] { 1.0 }),
            1 => DenseTensor<float>.OfValues(Array.Empty<float>()),
            2 => DenseTensor<float>.OfValues(new[] { 1f, 2f }),
            3 => new DenseTensor<float>(new[] { 1f }, new[] { 1, 1 }),
            _ => new DenseTensor<float>(new[] { 1f }, new[] { 1, 1, 1 })
        };
        foreach (bool lower in new[] { true, false })
        {
            var result = CPUExecutionProvider.Clip(data, lower ? bound : null, lower ? null : bound, null, null, null, 17);
            Assert.Equal(OpStatus.Failure, result.Status);
            Assert.Equal(new[] { -4f, 0f, 1f, 4f }, data.ToArray());
        }
    }

    [Fact]
    public void SingleValueBoundReadsItsBackingOffsetAndIsNotMutated()
    {
        var backing = new[] { 99L, long.MaxValue - 2, 88L };
        var min = new DenseTensor<long>(backing.AsMemory(1, 1), Array.Empty<int>());
        var input = DenseTensor<long>.OfValues(new[] { long.MinValue, long.MaxValue - 1, long.MaxValue });
        var result = CPUExecutionProvider.Clip(input, min, null, null, null, null, 17);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new[] { long.MaxValue - 2, long.MaxValue - 1, long.MaxValue }, ((Tensor<long>)result.Outputs[0]).ToArray());
        Assert.Equal(new[] { 99L, long.MaxValue - 2, 88L }, backing);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(6)]
    [InlineData(10)]
    public void LegacyAttributesRequireOrderedBoundsAndNoTensorBounds(int version)
    {
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Clip(Data(), null, null, 3f, 1f, null, version).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Clip(Data(), null, null, float.NaN, null, null, version).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Clip(Data(), Data(), null, null, null, null, version).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Clip(DenseTensor<int>.OfValues(new[] { 1 }), null, null, null, null, null, version).Status);
    }

    [Theory]
    [InlineData(11)]
    [InlineData(12)]
    [InlineData(17)]
    public void ModernClipRefusesAttributeBoundsRatherThanIgnoringThem(int version)
    {
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Clip(Data(), null, null, -2f, null, null, version).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Clip(Data(), null, null, null, 2f, null, version).Status);
    }

    [Theory]
    [InlineData(11, false)]
    [InlineData(12, true)]
    public void IntegerInputSupportStartsAtVersionTwelve(int version, bool supported)
    {
        var result = CPUExecutionProvider.Clip(DenseTensor<long>.OfValues(new[] { long.MinValue, long.MaxValue }), null, null, null, null, null, version);
        Assert.Equal(supported ? OpStatus.Success : OpStatus.Failure, result.Status);
    }

    [Fact]
    public void LegacyDoubleDefaultsComeFromFloatAttributes()
    {
        var input = DenseTensor<double>.OfValues(new[] { double.NegativeInfinity, 0.0, double.PositiveInfinity });
        var result = CPUExecutionProvider.Clip(input, null, null, null, null, null, 10);
        Assert.Equal(new[] { (double)float.MinValue, 0.0, (double)float.MaxValue }, ((Tensor<double>)result.Outputs[0]).ToArray());
    }

    [Fact]
    public void MissingAndUnsupportedInputsFail()
    {
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Clip(null, null, null, null, null, null, 17).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Clip(DenseTensor<bool>.OfValues(new[] { true }), null, null, null, null, null, 17).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Clip(Data(), null, null, null, null, null, 1).Status);
    }

    [Theory]
    [InlineData(5, 1, false)]
    [InlineData(6, 1, true)]
    [InlineData(10, 2, false)]
    [InlineData(11, 3, true)]
    [InlineData(17, 4, false)]
    public void RegistryEnforcesVersionedInputArity(int version, int inputs, bool supported)
    {
        var node = new Node { Name = "clip", Op = OpType.Clip, OpTypeName = "Clip", OpsetVersion = version,
            Inputs = Enumerable.Range(0, inputs).Select(i => "i" + i).ToArray(), Outputs = new[] { "y" } };
        Assert.Equal(supported, CPUExecutionProvider.SupportsNode(node));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(10)]
    [InlineData(17)]
    public void NodeVersionOverridesGraphAndUnknownVersionUsesGraph(int nodeVersion)
    {
        var graph = new ComputationalGraph { Opset = new Dictionary<string, int> { [""] = 17 } };
        graph.Inputs["x"] = Data();
        var node = new Node { Name = "clip", Op = OpType.Clip, OpTypeName = "Clip", OpsetVersion = nodeVersion,
            Inputs = new[] { "x" }, Outputs = new[] { "y" }, Attributes = new Dictionary<string, object> { ["min"] = -1f, ["max"] = 1f } };
        var result = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(nodeVersion == 10 ? OpStatus.Success : OpStatus.Failure, result.Status);
        if (nodeVersion == 10) Assert.Equal(new[] { -1f, 0f, 1f, 1f }, ((Tensor<float>)result.Outputs[0]).ToArray());
    }
}
