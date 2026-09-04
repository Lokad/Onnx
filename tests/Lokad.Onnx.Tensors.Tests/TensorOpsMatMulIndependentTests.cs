using System;
using System.Linq;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Tensors.Tests;

public class TensorOpsMatMulIndependentTests
{
    static float IndependentCell(Tensor<float> a, Tensor<float> b, int i, int j)
    {
        float s = 0f;
        for (int p = 0; p < a.Dimensions[1]; p++)
        {
            s += a[i, p] * b[p, j];
        }
        return s;
    }

    static void AssertMatMulMatchesIndependent(Tensor<float> a, Tensor<float> b, Tensor<float> actual, float eps = 1e-4f)
    {
        Assert.Equal(new[] { a.Dimensions[0], b.Dimensions[1] }, actual.Dimensions.ToArray());
        for (int i = 0; i < actual.Dimensions[0]; i++)
        {
            for (int j = 0; j < actual.Dimensions[1]; j++)
            {
                Assert.Equal(IndependentCell(a, b, i, j), actual[i, j], 4);
            }
        }
    }

    static DenseTensor<float> SeqFloat(int rows, int cols, int start = 1)
    {
        var t = new DenseTensor<float>(new[] { rows, cols });
        int v = start;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                t[i, j] = v++;
            }
        }
        return t;
    }

    static void RunAllModes(Tensor<float> a, Tensor<float> b, Action<Tensor<float>> assert)
    {
        Tensor<float> scalar;
        using (new HardwareConfigScope(useSimd: false, useIntrinsics: false))
        {
            scalar = Tensor<float>.MatMul2D(a, b);
        }
        assert(scalar);

        using (new HardwareConfigScope(useSimd: true, useIntrinsics: false))
        {
            assert(Tensor<float>.MatMul2D(a, b));
        }

        if (Fma.IsSupported)
        {
            using (new HardwareConfigScope(useSimd: true, useIntrinsics: true))
            {
                assert(Tensor<float>.MatMul2D(a, b));
            }
        }
    }

    [Fact]
    public void MatMul2D_IntrinsicsThirdRow_UsesCorrectPointer()
    {
        var a = SeqFloat(4, 3);
        var b = SeqFloat(3, 32);
        RunAllModes(a, b, actual => AssertMatMulMatchesIndependent(a, b, actual));
    }

    [Fact]
    public void MatMul2D_RowCounts_AllModes()
    {
        foreach (int m in new[] { 1, 2, 3, 4, 5 })
        {
            var a = SeqFloat(m, 3);
            var b = SeqFloat(3, 4);
            RunAllModes(a, b, actual => AssertMatMulMatchesIndependent(a, b, actual));
        }
    }

    [Fact]
    public void MatMul2D_OutputBoundaries_AllModes()
    {
        foreach (int k in new[] { 31, 32, 33 })
        {
            var a = SeqFloat(4, 3);
            var b = SeqFloat(3, k);
            RunAllModes(a, b, actual => AssertMatMulMatchesIndependent(a, b, actual));
        }
    }

    [Fact]
    public void MatMul2D_ReverseStride_MatchesIndependent()
    {
        var ax = new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } };
        var by = new float[,] { { 7f, 8f }, { 9f, 10f }, { 11f, 12f } };
        var revX = ax.ToTensor<float>(reverseStride: true);
        var revY = by.ToTensor<float>(reverseStride: true);
        RunAllModes(revX, revY, actual => AssertMatMulMatchesIndependent(revX, revY, actual));
    }

    [Fact]
    public void MatMul2D_NonContiguousSlice_MatchesIndependent()
    {
        var big = SeqFloat(4, 6);
        var a = big[0..2, 0..3];
        var b = SeqFloat(3, 2);
        var aa = (Tensor<float>)a;
        RunAllModes(aa, b, actual => AssertMatMulMatchesIndependent(aa, b, actual));
    }

    [Fact]
    public void ToDenseTensor_ReversedStride_ReturnsRowMajorCopy()
    {
        var src = new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } };
        var rev = src.ToTensor<float>(reverseStride: true);
        Assert.True(rev.IsReversedStride);
        var dense = rev.ToDenseTensor();
        Assert.False(dense.IsReversedStride);
        Assert.Equal(new[] { 2, 3 }, dense.Dimensions.ToArray());
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.Equal(rev[i, j], dense[i, j], 6);
            }
        }
        Assert.Equal(1f, dense[0, 0], 6);
        Assert.Equal(6f, dense[1, 2], 6);
    }

    [Fact]
    public void ToDenseTensor_RowMajor_ReturnsSame()
    {
        var dense = SeqFloat(2, 3);
        Assert.Same(dense, dense.ToDenseTensor());
    }

    [Fact]
    public void ToDenseTensor_IntAndDouble_ReversedStride_ReturnsRowMajor()
    {
        var ai = new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } };
        var revi = ai.ToTensor<int>(reverseStride: true);
        var densei = revi.ToDenseTensor();
        Assert.False(densei.IsReversedStride);
        Assert.Equal(6, densei[2, 1]);

        var ad = new double[,] { { 1d, 2d }, { 3d, 4d } };
        var revd = ad.ToTensor<double>(reverseStride: true);
        var densed = revd.ToDenseTensor();
        Assert.False(densed.IsReversedStride);
        Assert.Equal(4d, densed[1, 1], 10);
    }

    [Fact]
    public void MatMul2D_Int_MatchesIndependent()
    {
        var a = new int[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 10, 11, 12 } };
        var b = new int[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } };
        var ta = a.ToTensor<int>();
        var tb = b.ToTensor<int>();
        Tensor<int> actual;
        using (new HardwareConfigScope(false, false))
        {
            actual = Tensor<int>.MatMul2D(ta, tb);
        }
        Assert.Equal(22, actual[0, 0]);
        Assert.Equal(28, actual[0, 1]);
        Assert.Equal(49, actual[1, 0]);
        Assert.Equal(64, actual[1, 1]);
        Assert.Equal(76, actual[2, 0]);
        Assert.Equal(100, actual[2, 1]);
        Assert.Equal(103, actual[3, 0]);
        Assert.Equal(136, actual[3, 1]);
    }

    [Fact]
    public void MatMul2D_Double_MatchesIndependent()
    {
        var a = new double[,] { { 1d, 2d }, { 3d, 4d } };
        var b = new double[,] { { 5d, 6d }, { 7d, 8d } };
        var ta = a.ToTensor<double>();
        var tb = b.ToTensor<double>();
        Tensor<double> actual;
        using (new HardwareConfigScope(false, false))
        {
            actual = Tensor<double>.MatMul2D(ta, tb);
        }
        Assert.Equal(19d, actual[0, 0], 10);
        Assert.Equal(22d, actual[0, 1], 10);
        Assert.Equal(43d, actual[1, 0], 10);
        Assert.Equal(50d, actual[1, 1], 10);
    }

    [Fact]
    public void MatMul2D_OddRowsWideOutput_AllModes()
    {
        foreach (int rows in new[] { 3, 5 })
        {
            var left = SeqFloat(rows, 5);
            var right = SeqFloat(5, 64);
            RunAllModes(left, right, actual => AssertMatMulMatchesIndependent(left, right, actual));
        }
    }
}
