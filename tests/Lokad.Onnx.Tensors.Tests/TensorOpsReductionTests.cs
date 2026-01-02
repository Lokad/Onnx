using System;
using System.Linq;

namespace Lokad.Onnx.Tensors.Tests;

public class TensorOpsReductionTests
{
    [Fact]
    public void ReduceSum_Mean_Max_Int()
    {
        var data = DenseTensor<int>.OfValues(new int[2, 2] { { 1, 2 }, { 3, 4 } });
        var axes = new int[] { 0 }.ToTensor<int>();

        var sum = Tensor<int>.ReduceSum(data, axes);
        var mean = Tensor<int>.ReduceMean(data, axes);

        Assert.Equal(new[] { 2 }, sum.Dimensions.ToArray());
        Assert.Equal(4, sum[0]);
        Assert.Equal(6, sum[1]);
        Assert.Equal(1, mean[0]);
        Assert.Equal(3, mean[1]);
    }

    [Fact]
    public void ReduceSum_Mean_Max_Float()
    {
        var data = DenseTensor<float>.OfValues(new float[2, 3] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        var axes = new int[] { 1 }.ToTensor<int>();

        var sum = Tensor<float>.ReduceSum(data, axes);
        var mean = Tensor<float>.ReduceMean(data, axes);
        var max = Tensor<float>.ReduceMax(data, axes);

        Assert.Equal(new[] { 2 }, sum.Dimensions.ToArray());
        Assert.Equal(6f, sum[0], 5);
        Assert.Equal(15f, sum[1], 5);
        Assert.Equal(2f, mean[0], 5);
        Assert.Equal(5f, mean[1], 5);
        Assert.Equal(3f, max[0], 5);
        Assert.Equal(6f, max[1], 5);
    }

    [Fact]
    public void ReduceSum_Mean_Max_Double()
    {
        var data = DenseTensor<double>.OfValues(new double[2, 2] { { 1d, 2d }, { 3d, 4d } });
        var axes = new int[] { 1 }.ToTensor<int>();

        var sum = Tensor<double>.ReduceSum(data, axes);
        var mean = Tensor<double>.ReduceMean(data, axes);
        var max = Tensor<double>.ReduceMax(data, axes);

        Assert.Equal(new[] { 2 }, sum.Dimensions.ToArray());
        Assert.Equal(3d, sum[0], 10);
        Assert.Equal(7d, sum[1], 10);
        Assert.Equal(1.5d, mean[0], 10);
        Assert.Equal(3.5d, mean[1], 10);
        Assert.Equal(2d, max[0], 10);
        Assert.Equal(4d, max[1], 10);
    }

    [Fact]
    public void Softmax_NormalizesAlongAxis()
    {
        var data = DenseTensor<float>.OfValues(new float[2, 2] { { 0f, 1f }, { -1f, 1f } });
        var output = Tensor<float>.Softmax(data, axis: 1);

        var expected0 = MathF.Exp(0f) / (MathF.Exp(0f) + MathF.Exp(1f));
        var expected1 = MathF.Exp(1f) / (MathF.Exp(0f) + MathF.Exp(1f));
        var expected2 = MathF.Exp(-1f) / (MathF.Exp(-1f) + MathF.Exp(1f));
        var expected3 = MathF.Exp(1f) / (MathF.Exp(-1f) + MathF.Exp(1f));

        Assert.Equal(expected0, output[0, 0], 5);
        Assert.Equal(expected1, output[0, 1], 5);
        Assert.Equal(expected2, output[1, 0], 5);
        Assert.Equal(expected3, output[1, 1], 5);
        Assert.Equal(1f, output[0, 0] + output[0, 1], 5);
        Assert.Equal(1f, output[1, 0] + output[1, 1], 5);
    }
}
