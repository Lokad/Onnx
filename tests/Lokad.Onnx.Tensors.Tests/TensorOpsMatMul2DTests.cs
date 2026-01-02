using System;
using System.Linq;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Tensors.Tests;

public class TensorOpsMatMul2DTests
{
    [Fact]
    public void MatMul2D_Int_ComputesExpected()
    {
        var a = DenseTensor<int>.OfValues(new int[,] { { 1, 2 }, { 3, 4 } });
        var b = DenseTensor<int>.OfValues(new int[,] { { 5, 6 }, { 7, 8 } });

        var c = Tensor<int>.MatMul2D(a, b);

        Assert.Equal(new[] { 2, 2 }, c.Dimensions.ToArray());
        Assert.Equal(19, c[0, 0]);
        Assert.Equal(22, c[0, 1]);
        Assert.Equal(43, c[1, 0]);
        Assert.Equal(50, c[1, 1]);
    }

    [Fact]
    public void MatMul2D_Float_ComputesExpected()
    {
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var b = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 7f, 8f } });

        var c = Tensor<float>.MatMul2D(a, b);

        Assert.Equal(new[] { 2, 2 }, c.Dimensions.ToArray());
        Assert.Equal(19f, c[0, 0], 5);
        Assert.Equal(22f, c[0, 1], 5);
        Assert.Equal(43f, c[1, 0], 5);
        Assert.Equal(50f, c[1, 1], 5);
    }

    [Fact]
    public void MatMul2D_Double_ComputesExpected()
    {
        var a = DenseTensor<double>.OfValues(new double[,] { { 1d, 2d }, { 3d, 4d } });
        var b = DenseTensor<double>.OfValues(new double[,] { { 5d, 6d }, { 7d, 8d } });

        var c = Tensor<double>.MatMul2D(a, b);

        Assert.Equal(new[] { 2, 2 }, c.Dimensions.ToArray());
        Assert.Equal(19d, c[0, 0], 10);
        Assert.Equal(22d, c[0, 1], 10);
        Assert.Equal(43d, c[1, 0], 10);
        Assert.Equal(50d, c[1, 1], 10);
    }

    [Fact]
    public void MatMul2D_Int_MatchesManagedOnSlices()
    {
        var a = Tensor<int>.Arange(0, 2 * 2 * 3).Reshape(2, 2, 3);
        var b = Tensor<int>.Arange(0, 2 * 3 * 2).Reshape(2, 3, 2);

        var sliceA = a[1, ..];
        var sliceB = b[1, ..];

        var expected = Tensor<int>.MatMul2D_managed(sliceA, sliceB);
        var actual = Tensor<int>.MatMul2D(sliceA, sliceB);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void MatMul2D_Float_MatchesManaged()
    {
        var a = DenseTensor<float>.OfValues(new float[,] { { 0.5f, 1.5f, -2f }, { 3f, 0f, 1f } });
        var b = DenseTensor<float>.OfValues(new float[,] { { 2f, 1f }, { -1f, 0.5f }, { 4f, -3f } });

        var expected = Tensor<float>.MatMul2D_managed(a, b);
        var actual = Tensor<float>.MatMul2D(a, b);

        Assert.Equal(expected[0, 0], actual[0, 0], 5);
        Assert.Equal(expected[0, 1], actual[0, 1], 5);
        Assert.Equal(expected[1, 0], actual[1, 0], 5);
        Assert.Equal(expected[1, 1], actual[1, 1], 5);
    }

    [Fact]
    public void MatMul2D_Float_RespectsHardwareFlags()
    {
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var b = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 7f, 8f } });

        using var baseline = new HardwareConfigScope(useSimd: false, useIntrinsics: false);
        var scalar = Tensor<float>.MatMul2D(a, b);

        using var simdOnly = new HardwareConfigScope(useSimd: true, useIntrinsics: false);
        var simd = Tensor<float>.MatMul2D(a, b);
        Assert.Equal(scalar, simd);

        if (Fma.IsSupported)
        {
            using var intrinsics = new HardwareConfigScope(useSimd: true, useIntrinsics: true);
            var intrinsicsResult = Tensor<float>.MatMul2D(a, b);
            Assert.Equal(scalar, intrinsicsResult);
        }
    }

    [Fact]
    public void MatMul2D_RejectsRankMismatch()
    {
        Assert.Throws<ArgumentException>(() => Tensor<int>.MatMul2D(Tensor<int>.Ones(2, 2, 2), Tensor<int>.Ones(2, 2)));
        Assert.Throws<ArgumentException>(() => Tensor<float>.MatMul2D(Tensor<float>.Ones(2, 2), Tensor<float>.Ones(2)));
        Assert.Throws<ArgumentException>(() => Tensor<double>.MatMul2D(Tensor<double>.Ones(2, 2, 2), Tensor<double>.Ones(2, 2)));
    }

    [Fact]
    public void MatMul2D_RejectsDimensionMismatch()
    {
        Assert.Throws<ArgumentException>(() => Tensor<int>.MatMul2D(Tensor<int>.Ones(2, 3), Tensor<int>.Ones(2, 2)));
        Assert.Throws<ArgumentException>(() => Tensor<float>.MatMul2D(Tensor<float>.Ones(3, 2), Tensor<float>.Ones(3, 1)));
        Assert.Throws<ArgumentException>(() => Tensor<double>.MatMul2D(Tensor<double>.Ones(1, 4), Tensor<double>.Ones(3, 2)));
    }
}
