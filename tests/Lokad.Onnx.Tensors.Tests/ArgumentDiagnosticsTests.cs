namespace Lokad.Onnx.Tensors.Tests;

public class ArgumentDiagnosticsTests
{
    [Fact]
    public void InvalidBroadcast_NamesTargetShapeAndExplainsMismatch()
    {
        var input = DenseTensor<float>.OfShape(2, 3);
        var error = Assert.Throws<ArgumentException>(() => Tensor<float>.BroadcastTo(input, new[] { 2, 4 }));
        Assert.Equal("targetShape", error.ParamName);
        Assert.Contains("Cannot broadcast dimension 3 to 4.", error.Message);
    }

    [Fact]
    public void InvalidUnsqueeze_NamesAxesThroughBothEntryPoints()
    {
        var input = DenseTensor<float>.OfShape(2, 3);
        var direct = Assert.Throws<ArgumentException>(() => Tensor<float>.Unsqueeze(input, new[] { 2, -2 }));
        var numeric = Assert.Throws<ArgumentException>(() => ((INumericTensor)input).Unsqueeze(new[] { 2, -2 }));
        foreach (var error in new[] { direct, numeric })
        {
            Assert.Equal("axes", error.ParamName);
            Assert.Contains("repeated dimension", error.Message);
        }
    }

    [Fact]
    public void InvalidMatMulDestination_NamesDestinationWithoutWriting()
    {
        var input = DenseTensor<float>.OfShape(2, 2);
        var destination = DenseTensor<float>.OfShape(2, 3);
        destination.Fill(7);
        var error = Assert.Throws<ArgumentException>(() => Tensor<float>.MatMul2D(input, input, destination, TensorExecutionOptions.Auto));
        Assert.Equal("destination", error.ParamName);
        Assert.Contains("Destination shape must match", error.Message);
        Assert.All(destination.ToArray(), value => Assert.Equal(7f, value));
    }

    [Fact]
    public void AliasedMatMulDestination_NamesDestinationWithoutWriting()
    {
        var input = new DenseTensor<float>(new float[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var error = Assert.Throws<ArgumentException>(() => Tensor<float>.MatMul2D(input, input, input, TensorExecutionOptions.Auto));
        Assert.Equal("destination", error.ParamName);
        Assert.Contains("Destination must not alias", error.Message);
        Assert.Equal(new float[] { 1, 2, 3, 4 }, input.ToArray());
    }

    [Fact]
    public void InputShapeException_PreservesContextAndNamesInput()
    {
        var input = DenseTensor<float>.OfShape(2, 3);
        var shape = new[] { 3, 2 };
        var error = new TensorInputShapeException("data", shape, input);
        Assert.Equal("data", error.ParamName);
        Assert.Contains("has shape", error.Message);
        Assert.Contains("required to be [3, 2].", error.Message);
        Assert.Equal("data", error.Name);
        Assert.Same(shape, error.Shape);
        Assert.Same(input, error.Input);
    }

    [Fact]
    public void InvalidGatherAxis_NamesThePublicParameter()
    {
        var error = Assert.Throws<ArgumentException>(() => Tensor<float>.Gather(
            DenseTensor<float>.OfShape(2, 3), DenseTensor<int>.OfValues(new[] { 0 }), 2));
        Assert.Equal("_axis", error.ParamName);
        Assert.Contains("exceeds the number of dimensions", error.Message);
    }

    [Fact]
    public void InvalidSliceEnds_NamesEndsInsteadOfStarts()
    {
        var error = Assert.Throws<ArgumentException>(() => Tensor<float>.Slice(
            DenseTensor<float>.OfShape(2, 3), DenseTensor<int>.OfValues(new[] { 0 }),
            DenseTensor<int>.OfValues(new int[,] { { 1 } }), null, null));
        Assert.Equal("ends", error.ParamName);
        Assert.Contains("rank of the end tensor", error.Message);
    }

    [Fact]
    public void InvalidReductionAxis_NamesAxes()
    {
        var error = Assert.Throws<ArgumentException>(() => Tensor<float>.ReduceSum(
            DenseTensor<float>.OfShape(2, 3), DenseTensor<int>.OfValues(new[] { 2 })));
        Assert.Equal("axes", error.ParamName);
        Assert.Contains("out of range for tensor rank 2", error.Message);
    }
}
