namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Covers shared-right batch folding in the float matrix dispatcher: batches
/// over one right matrix collapse to a single product, while genuinely
/// batched, broadcast, or strided operands keep the per-batch fallback.
/// An independent double-accumulated oracle judges every case; agreement is
/// within float rounding, not bit identity, since the shared dispatcher may
/// select different vectorized kernels per width.
/// </summary>
public class MatMulBatchFoldingTests
{
    static float Pattern(int i) => ((i * 37 + 11) % 97 - 48) * 0.02f;

    static DenseTensor<float> FilledTensor(int[] dims)
    {
        var t = DenseTensor<float>.OfShape(dims);
        var span = t.Buffer.Span;
        for (int i = 0; i < span.Length; i++) span[i] = Pattern(i);
        return t;
    }

    static double[] NaiveBatched(float[] x, int[] xd, float[] y, int[] yd)
    {
        int rx = xd.Length, ry = yd.Length;
        int m = xd[rx - 2], n = xd[rx - 1], k = yd[ry - 1];
        int rb = System.Math.Max(rx, ry) - 2;
        int[] bd = new int[rb];
        for (int d = 0; d < rb; d++)
        {
            int dxd = d < rb - (rx - 2) ? 1 : xd[d - (rb - (rx - 2))];
            int dyd = d < rb - (ry - 2) ? 1 : yd[d - (rb - (ry - 2))];
            bd[d] = System.Math.Max(dxd, dyd);
        }
        int count = 1;
        foreach (var d in bd) count *= d;
        var z = new double[count * m * k];
        for (int b = 0; b < count; b++)
        {
            int rem = b;
            int xo = 0, yo = 0;
            for (int d = rb - 1; d >= 0; d--)
            {
                int c = rem % bd[d];
                rem /= bd[d];
                int xi = d - (rb - (rx - 2));
                int yi = d - (rb - (ry - 2));
                int xcs = (xi < 0 || xd[xi] == 1) ? 0 : c;
                int ycs = (yi < 0 || yd[yi] == 1) ? 0 : c;
                int xss = m * n, yss = n * k;
                for (int j = d + 1; j < rb; j++)
                {
                    int xj = j - (rb - (rx - 2));
                    int yj = j - (rb - (ry - 2));
                    xss *= xj < 0 ? 1 : xd[xj];
                    yss *= yj < 0 ? 1 : yd[yj];
                }
                xo += xcs * xss;
                yo += ycs * yss;
            }
            for (int i = 0; i < m; i++)
                for (int j = 0; j < k; j++)
                {
                    double acc = 0.0;
                    for (int l = 0; l < n; l++)
                        acc += (double)x[xo + i * n + l] * y[yo + l * k + j];
                    z[(b * m + i) * k + j] = acc;
                }
        }
        return z;
    }

    static void AssertNear(double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double tol = 2e-4 * (1.0 + System.Math.Abs(expected[i]));
            Assert.True(System.Math.Abs(actual[i] - expected[i]) <= tol, "index " + i);
        }
    }

    static float[] RunAllocating(Tensor<float> x, Tensor<float> y, TensorExecutionOptions opts)
    {
        return ((Tensor<float>)Tensor<float>.MatMul(x, y, opts)).ToArray();
    }

    [Fact]
    public void Fold_JointLikeBatch_MatchesNaive()
    {
        var x = FilledTensor(new[] { 1, 8, 5, 16 });
        var y = FilledTensor(new[] { 16, 24 });
        var e = NaiveBatched(x.Buffer.ToArray(), new[] { 1, 8, 5, 16 }, y.Buffer.ToArray(), new[] { 16, 24 });
        AssertNear(e, RunAllocating(x, y, TensorExecutionOptions.Auto));
        AssertNear(e, RunAllocating(x, y, TensorExecutionOptions.Scalar));
    }

    [Fact]
    public void Fold_SingleRowBatches_MatchesNaive()
    {
        var x = FilledTensor(new[] { 4, 1, 7 });
        var y = FilledTensor(new[] { 7, 9 });
        var e = NaiveBatched(x.Buffer.ToArray(), new[] { 4, 1, 7 }, y.Buffer.ToArray(), new[] { 7, 9 });
        AssertNear(e, RunAllocating(x, y, TensorExecutionOptions.Auto));
    }

    [Fact]
    public void Fold_MultiDimBatch_MatchesNaive()
    {
        var x = FilledTensor(new[] { 2, 3, 4, 6 });
        var y = FilledTensor(new[] { 6, 8 });
        var e = NaiveBatched(x.Buffer.ToArray(), new[] { 2, 3, 4, 6 }, y.Buffer.ToArray(), new[] { 6, 8 });
        AssertNear(e, RunAllocating(x, y, TensorExecutionOptions.Auto));
    }

    [Fact]
    public void NoFold_TrueBatchedRight_MatchesNaive()
    {
        var x = FilledTensor(new[] { 2, 3, 4, 6 });
        var y = FilledTensor(new[] { 2, 3, 6, 8 });
        var e = NaiveBatched(x.Buffer.ToArray(), new[] { 2, 3, 4, 6 }, y.Buffer.ToArray(), new[] { 2, 3, 6, 8 });
        AssertNear(e, RunAllocating(x, y, TensorExecutionOptions.Auto));
    }

    [Fact]
    public void NoFold_BroadcastLeft_MatchesNaive()
    {
        var x = FilledTensor(new[] { 4, 6 });
        var y = FilledTensor(new[] { 2, 3, 6, 8 });
        var e = NaiveBatched(x.Buffer.ToArray(), new[] { 4, 6 }, y.Buffer.ToArray(), new[] { 2, 3, 6, 8 });
        AssertNear(e, RunAllocating(x, y, TensorExecutionOptions.Auto));
    }

    [Fact]
    public void NoFold_SingleBatch_MatchesNaive()
    {
        // batchCount 1 (the single-step regime) keeps the original product.
        var x = FilledTensor(new[] { 1, 1, 5, 16 });
        var y = FilledTensor(new[] { 16, 24 });
        var e = NaiveBatched(x.Buffer.ToArray(), new[] { 1, 1, 5, 16 }, y.Buffer.ToArray(), new[] { 16, 24 });
        AssertNear(e, RunAllocating(x, y, TensorExecutionOptions.Auto));
    }

    [Fact]
    public void Fold_IntoDestinationApi_MatchesNaive()
    {
        var x = FilledTensor(new[] { 1, 8, 5, 16 });
        var y = FilledTensor(new[] { 16, 24 });
        var dest = DenseTensor<float>.OfShape(new[] { 1, 8, 5, 24 });
        Tensor<float>.MatMul(x, y, dest, TensorExecutionOptions.Auto);
        var e = NaiveBatched(x.Buffer.ToArray(), new[] { 1, 8, 5, 16 }, y.Buffer.ToArray(), new[] { 16, 24 });
        AssertNear(e, dest.ToArray());
    }

    [Fact]
    public void Fold_OffsetRightView_MatchesNaive()
    {
        var big = FilledTensor(new[] { 32, 8 });
        var w = big.Slice(new SliceIndex[] { new SliceIndex(8, 24), SliceIndex.All });
        var x = FilledTensor(new[] { 2, 4, 16 });
        var e = NaiveBatched(x.Buffer.ToArray(), new[] { 2, 4, 16 }, w.ToArray(), new[] { 16, 8 });
        AssertNear(e, RunAllocating(x, w, TensorExecutionOptions.Auto));
    }
}
