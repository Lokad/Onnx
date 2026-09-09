using System;
using System.Linq;
using System.Runtime.Intrinsics;
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

    static void AssertMatMulMatchesIndependent(Tensor<float> a, Tensor<float> b, Tensor<float> actual, float eps)
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

    static DenseTensor<float> SeqFloat(int rows, int cols, int start)
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
        assert(Tensor<float>.MatMul2D(a, b, TensorExecutionOptions.Scalar));
        assert(Tensor<float>.MatMul2D(a, b, TensorExecutionOptions.Simd));
        if (Fma.IsSupported)
        {
            assert(Tensor<float>.MatMul2D(a, b, TensorExecutionOptions.Intrinsics));
        }
    }

    [Fact]
    public void MatMul2D_IntrinsicsThirdRow_UsesCorrectPointer()
    {
        var a = SeqFloat(4, 3, 1);
        var b = SeqFloat(3, 32, 1);
        RunAllModes(a, b, actual => AssertMatMulMatchesIndependent(a, b, actual, 1e-4f));
    }

    [Fact]
    public void MatMul2D_RowCounts_AllModes()
    {
        foreach (int m in new[] { 1, 2, 3, 4, 5 })
        {
            var a = SeqFloat(m, 3, 1);
            var b = SeqFloat(3, 4, 1);
            RunAllModes(a, b, actual => AssertMatMulMatchesIndependent(a, b, actual, 1e-4f));
        }
    }

    [Fact]
    public void MatMul2D_OutputBoundaries_AllModes()
    {
        foreach (int k in new[] { 31, 32, 33 })
        {
            var a = SeqFloat(4, 3, 1);
            var b = SeqFloat(3, k, 1);
            RunAllModes(a, b, actual => AssertMatMulMatchesIndependent(a, b, actual, 1e-4f));
        }
    }

    static DenseTensor<float> RandDense(int rows, int cols, Random rnd)
    {
        var t = new DenseTensor<float>(new[] { rows, cols });
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t[i, j] = (float)rnd.NextDouble() - 0.5f;
        return t;
    }

    static float FmaScalar(float x, float y, float z) =>
        Fma.MultiplyAddScalar(Vector128.Create(x), Vector128.Create(y), Vector128.Create(z)).GetElement(0);

    static DenseTensor<float> FmaReference(Tensor<float> a, Tensor<float> b)
    {
        // Mirrors the kernel split exactly: FMA axpy below the 8-wide ceiling,
        // plain multiply-add at and above it, all ascending, so agreement is bitwise.
        int m = a.Dimensions[0], n = a.Dimensions[1], k = b.Dimensions[1];
        int ceiling = (k / Vector256<float>.Count) * Vector256<float>.Count;
        var c = new DenseTensor<float>(new[] { m, k });
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float aij = a[i, j];
                for (int p = 0; p < ceiling; p++)
                    c[i, p] = FmaScalar(aij, b[j, p], c[i, p]);
                for (int p = ceiling; p < k; p++)
                    c[i, p] += aij * b[j, p];
            }
        return c;
    }

    static void AssertBitwiseEqual(DenseTensor<float> expected, Tensor<float> actual)
    {
        var ea = expected.ToArray();
        var aa = actual.ToArray();
        Assert.Equal(ea.Length, aa.Length);
        for (int i = 0; i < ea.Length; i++)
            Assert.True(BitConverter.SingleToUInt32Bits(ea[i]) == BitConverter.SingleToUInt32Bits(aa[i]), "bit mismatch at " + i);
    }

    [Fact]
    public void MatMul2DKRemainder_MatchesFmaOrderBitwise()
    {
        // k%32 != 0 exercises the 2x4 remainder tail (plus the odd-row tail for
        // odd m). The per-element operation order matches the non-unrolled kernel
        // exactly (FMA below the 8-wide ceiling, scalar above), so intrinsics
        // results agree with the split reference bitwise.
        var rnd = new Random(20260909);
        var shapes = new[] { (m: 4, n: 17, k: 49), (m: 5, n: 65, k: 100), (m: 2, n: 9, k: 8) };
        foreach (var (m, n, k) in shapes)
        {
            var a = RandDense(m, n, rnd);
            var b = RandDense(n, k, rnd);
            if (Fma.IsSupported)
            {
                var expected = FmaReference(a, b);
                AssertBitwiseEqual(expected, Tensor<float>.MatMul2D(a, b, TensorExecutionOptions.Intrinsics));
                AssertBitwiseEqual(expected, Tensor<float>.MatMul2D(a, b, TensorExecutionOptions.Auto));
            }
            RunAllModes(a, b, actual => AssertMatMulMatchesIndependent(a, b, actual, 1e-4f));
        }
    }

    [Fact]
    public void MatMul2D_ReverseStride_MatchesIndependent()
    {
        var ax = new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } };
        var by = new float[,] { { 7f, 8f }, { 9f, 10f }, { 11f, 12f } };
        var revX = ax.ToTensor<float>(reverseStride: true);
        var revY = by.ToTensor<float>(reverseStride: true);
        RunAllModes(revX, revY, actual => AssertMatMulMatchesIndependent(revX, revY, actual, 1e-4f));
    }

    [Fact]
    public void MatMul2D_NonContiguousSlice_MatchesIndependent()
    {
        var big = SeqFloat(4, 6, 1);
        var a = big[0..2, 0..3];
        var b = SeqFloat(3, 2, 1);
        var aa = (Tensor<float>)a;
        RunAllModes(aa, b, actual => AssertMatMulMatchesIndependent(aa, b, actual, 1e-4f));
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
        var dense = SeqFloat(2, 3, 1);
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
        Tensor<int> actual = Tensor<int>.MatMul2D(ta, tb, TensorExecutionOptions.Scalar);
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
        Tensor<double> actual = Tensor<double>.MatMul2D(ta, tb, TensorExecutionOptions.Scalar);
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
            var left = SeqFloat(rows, 5, 1);
            var right = SeqFloat(5, 64, 1);
            RunAllModes(left, right, actual => AssertMatMulMatchesIndependent(left, right, actual, 1e-4f));
        }
    }

    static int[] SweepI(int n)
    {
        var a = new int[n];
        for (int i = 0; i < n; i++) a[i] = (i * 37 + 11) % 17 - 8;
        return a;
    }

    static float[] SweepF(int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = ((i * 37 + 11) % 17 - 8) * 0.5f;
        return a;
    }

    static double[] SweepD(int n)
    {
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = ((i * 37 + 11) % 17 - 8) * 0.5;
        return a;
    }

    static int[] NaiveMmAccumulate(int[] a, int[] b, int[] c0, int m, int n, int k)
    {
        var c = (int[])c0.Clone();
        for (int i = 0; i < m; i++)
            for (int kk = 0; kk < k; kk++)
                for (int j = 0; j < n; j++)
                    c[i * k + kk] += a[i * n + j] * b[j * k + kk];
        return c;
    }

    static float[] NaiveMmAccumulate(float[] a, float[] b, float[] c0, int m, int n, int k)
    {
        var c = (float[])c0.Clone();
        for (int i = 0; i < m; i++)
            for (int kk = 0; kk < k; kk++)
            {
                float total = c[i * k + kk];
                for (int j = 0; j < n; j++)
                    total += a[i * n + j] * b[j * k + kk];
                c[i * k + kk] = total;
            }
        return c;
    }

    static double[] NaiveMmAccumulate(double[] a, double[] b, double[] c0, int m, int n, int k)
    {
        var c = (double[])c0.Clone();
        for (int i = 0; i < m; i++)
            for (int kk = 0; kk < k; kk++)
            {
                double total = c[i * k + kk];
                for (int j = 0; j < n; j++)
                    total += a[i * n + j] * b[j * k + kk];
                c[i * k + kk] = total;
            }
        return c;
    }

    static int[] RunMm(int[] a, int[] b, int[] c, int m, int n, int k)
    {
        unsafe
        {
            fixed (int* pa = a)
            fixed (int* pb = b)
            fixed (int* pc = c)
            {
                MathOps.mm(m, n, k, pa, pb, pc);
            }
        }
        return c;
    }

    static float[] RunMm(float[] a, float[] b, float[] c, int m, int n, int k)
    {
        unsafe
        {
            fixed (float* pa = a)
            fixed (float* pb = b)
            fixed (float* pc = c)
            {
                MathOps.mm(m, n, k, pa, pb, pc);
            }
        }
        return c;
    }

    static double[] RunMm(double[] a, double[] b, double[] c, int m, int n, int k)
    {
        unsafe
        {
            fixed (double* pa = a)
            fixed (double* pb = b)
            fixed (double* pc = c)
            {
                MathOps.mm(m, n, k, pa, pb, pc);
            }
        }
        return c;
    }

    [Fact]
    public void ScalarMm_MatchesNaiveAccumulate()
    {
        var shapes = new[] { (m: 1, n: 1, k: 1), (m: 2, n: 3, k: 4), (m: 4, n: 4, k: 4), (m: 3, n: 7, k: 2), (m: 5, n: 1, k: 6) };
        foreach (var (m, n, k) in shapes)
        {
            foreach (bool preload in new[] { false, true })
            {
                var ia = SweepI(m * n);
                var ib = SweepI(n * k);
                var ic = preload ? SweepI(m * k) : new int[m * k];
                Assert.Equal(NaiveMmAccumulate(ia, ib, ic, m, n, k), RunMm(ia, ib, (int[])ic.Clone(), m, n, k));

                var fa = SweepF(m * n);
                var fb = SweepF(n * k);
                var fc = preload ? SweepF(m * k) : new float[m * k];
                Assert.Equal(NaiveMmAccumulate(fa, fb, fc, m, n, k), RunMm(fa, fb, (float[])fc.Clone(), m, n, k));

                var da = SweepD(m * n);
                var db = SweepD(n * k);
                var dc = preload ? SweepD(m * k) : new double[m * k];
                Assert.Equal(NaiveMmAccumulate(da, db, dc, m, n, k), RunMm(da, db, (double[])dc.Clone(), m, n, k));
            }
        }
    }
}
