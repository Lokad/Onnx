namespace Lokad.Onnx.Tensors.Tests;

public class TensorReshapeTransposeTests
{
    [Fact]
    public void CanReshape()
    {
        var X = DenseTensor<int>.Ones(2, 3, 4);
        
        var s = DenseTensor<long>.OfValues(new long[] {4,2,3});
        var Y = Tensor<int>.Reshape(X, s);
        Assert.Equal(new int[] {4,2,3}, Y.Dimensions.ToArray());

        s = DenseTensor<long>.OfValues(new long[] { 2, 4, 3 });
        Y = Tensor<int>.Reshape(X, s);
        Assert.Equal(new int[] { 2, 4, 3 }, Y.Dimensions.ToArray());

        s = DenseTensor<long>.OfValues(new long[] { 2, 12 });
        Y = Tensor<int>.Reshape(X, s);
        Assert.Equal(new int[] { 2, 12 }, Y.Dimensions.ToArray());

        s = DenseTensor<long>.OfValues(new long[] { 2,  -1, 2});
        Y = Tensor<int>.Reshape(X, s);
        Assert.Equal(new int[] { 2, 6, 2 }, Y.Dimensions.ToArray());

        s = DenseTensor<long>.OfValues(new long[] { -1, 2, 3, 4 });
        Y = Tensor<int>.Reshape(X, s);
        Assert.Equal(new int[] { 1, 2, 3, 4 }, Y.Dimensions.ToArray());

        s = DenseTensor<long>.OfValues(new long[] { 2, 0, 1, -1 });
        Y = Tensor<int>.Reshape(X, s);
        Assert.Equal(new int[] { 2, 3, 1, 4 }, Y.Dimensions.ToArray());
    }

    [Fact]
    public void CanTranspose()
    {
        var X = DenseTensor<int>.Ones(2, 3, 4);
        var tX = Tensor<int>.Transpose(X);
        Assert.Equal(2, tX.Dimensions[2]);
        tX = Tensor<int>.Transpose(X, new int[] { 2, 0, 1 });
        Assert.Equal(4, tX.Dimensions[0]);
        Assert.Equal(3, tX.Dimensions[2]);
        Assert.Throws<ArgumentException>(() => Tensor<int>.Transpose(X, new int[] { 0, 0, 2 }));
        Assert.Throws<ArgumentException>(() => Tensor<int>.Transpose(X, new int[] { 0 }));
        Assert.Throws<ArgumentException>(() => Tensor<int>.Transpose(X, new int[] { 2, 6 }));
    }
}

