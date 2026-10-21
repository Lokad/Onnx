namespace Lokad.Onnx.Tensors.Tests;

public class ElementwiseLayoutTests
{
    static Tensor<U> Layout<U>(U[] values, int layout) where U : unmanaged
    {
        var dense = new DenseTensor<U>(new[] { 2, layout >= 2 ? 3 : 2 }, layout == 1 || layout == 3);
        Tensor<U> tensor = layout >= 2 ? dense.Slice(SliceIndex.All, new SliceIndex(1, 3)) : dense;
        int position = 0;
        foreach (var coordinate in tensor.GetDimensionsIterator()) tensor[coordinate] = values[position++];
        Assert.Equal(values, tensor.ToArray());
        return tensor;
    }

    static void Values(float[] expected, OpResult result)
    {
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(expected, ((Tensor<float>)result.Outputs[0]).ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void ComparisonsAndSelection_UseLogicalPositions(int layout)
    {
        var input = Layout(new float[] { 1, 2, 3, 4 }, layout);
        var canonical = Layout(new float[] { 1, 2, 3, 4 }, 0);
        Assert.Equal(new[] { true, true, true, true }, Tensor<float>.Equal(input, canonical).ToArray());
        Assert.Equal(new[] { true, true, true, true }, Tensor<float>.Less(input, Layout(new float[] { 2, 3, 4, 5 }, 0)).ToArray());
        foreach (var options in new[] { ExecutionOptions.Default, ExecutionOptions.Memory, ExecutionOptions.Scalar })
        {
            var result = CPUExecutionProvider.LessOrEqual(input, canonical, options);
            Assert.Equal(OpStatus.Success, result.Status);
            Assert.Equal(new[] { true, true, true, true }, ((Tensor<bool>)result.Outputs[0]).ToArray());
        }
        var condition = Layout(new[] { true, false, true, false }, layout);
        Assert.Equal(new float[] { 1, 0, 3, 0 }, Tensor<float>.Where(condition, input, DenseTensor<float>.OfShape(2, 2)).ToArray());
        Assert.Equal(new float[] { 1, 2, 3, 4 }, input.ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void ScalarArithmeticAndPow_KeepLogicalOrderOnEitherSide(int layout)
    {
        var input = Layout(new float[] { 1, 2, 3, 4 }, layout);
        foreach (var options in new[] { ExecutionOptions.Default, ExecutionOptions.Memory, ExecutionOptions.Scalar })
        foreach (var scalar in new Tensor<float>[] { DenseTensor<float>.Scalar(2), DenseTensor<float>.OfValues(new float[] { 2 }) })
        {
            Values(new float[] { 3, 4, 5, 6 }, CPUExecutionProvider.Add(input, scalar, options, null));
            Values(new float[] { 3, 4, 5, 6 }, CPUExecutionProvider.Add(scalar, input, options, null));
            Values(new float[] { 1, 0, -1, -2 }, CPUExecutionProvider.Sub(scalar, input, options));
            Values(new float[] { 2, 4, 6, 8 }, CPUExecutionProvider.Mul(input, scalar, options, null));
            Values(new float[] { 0.5f, 1, 1.5f, 2 }, CPUExecutionProvider.Div(input, scalar, options, null));
            Values(new float[] { 1, 4, 9, 16 }, CPUExecutionProvider.Pow(input, scalar, options));
            Values(new float[] { 2, 4, 8, 16 }, CPUExecutionProvider.Pow(scalar, input, options));
        }
    }

    [Fact]
    public void BroadcastViewsAndEmptyInputs_RetainTheirLogicalValues()
    {
        var input = DenseTensor<float>.OfValues(new float[] { 1, 2 }).Reshape(1, 2).BroadcastDim(0, 2);
        Values(new float[] { 3, 4, 3, 4 }, CPUExecutionProvider.Add(input, DenseTensor<float>.Scalar(2), null, null));
        var empty = new DenseTensor<float>(new[] { 0, 2 }, true);
        Values(Array.Empty<float>(), CPUExecutionProvider.Pow(empty, DenseTensor<float>.Scalar(2), null));
        Assert.Empty(Tensor<float>.Equal(empty, empty).ToArray());
    }
}
