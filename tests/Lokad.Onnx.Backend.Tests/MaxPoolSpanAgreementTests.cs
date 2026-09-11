namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// The span MaxPool loop agrees with an independent scalar reference on
/// windows, strides, dilations, pads and batches, and non-dense views keep
/// flowing through the generic fallback with identical values.
/// </summary>
public class MaxPoolSpanAgreementTests
{
    static float[] Naive(float[] x, int n, int c, int h, int w, int kH, int kW, int sH, int sW, int dH, int dW, int pt, int pl, int pb, int pr, out int[] dims)
    {
        int effKH = (kH - 1) * dH + 1;
        int effKW = (kW - 1) * dW + 1;
        int oh = (h + pt + pb - effKH) / sH + 1;
        int ow = (w + pl + pr - effKW) / sW + 1;
        dims = new int[] { n, c, oh, ow };
        var y = new float[n * c * oh * ow];
        for (int nn = 0; nn < n; nn++)
            for (int dd = 0; dd < c; dd++)
                for (int oy = 0; oy < oh; oy++)
                    for (int ox = 0; ox < ow; ox++)
                    {
                        float best = -float.MaxValue;
                        for (int ky = 0; ky < kH; ky++)
                        {
                            int xr = oy * sH - pt + ky * dH;
                            if (xr < 0 || xr >= h) continue;
                            for (int kx = 0; kx < kW; kx++)
                            {
                                int xc = ox * sW - pl + kx * dW;
                                if (xc < 0 || xc >= w) continue;
                                float v = x[((nn * c + dd) * h + xr) * w + xc];
                                if (v > best) best = v;
                            }
                        }
                        y[((nn * c + dd) * oh + oy) * ow + ox] = best;
                    }
        return y;
    }

    static DenseTensor<float> Fill(int[] dims, int seed)
    {
        var rnd = new Random(seed);
        var t = Tensor<float>.Zeros(dims).ToDenseTensor();
        for (int i = 0; i < t.Length; i++)
        {
            float v = (float)(rnd.NextDouble() * 4 - 2);
            if (i % 31 == 0) v = float.NaN;
            if (i % 37 == 0) v = float.PositiveInfinity;
            t.SetValue(i, v);
        }
        return t;
    }

    public static System.Collections.Generic.IEnumerable<object[]> Cases()
    {
        yield return new object[] { new int[] { 1, 64, 112, 112 }, 3, 3, 2, 2, 1, 1, new int[] { 1, 1, 1, 1 } };
        yield return new object[] { new int[] { 1, 4, 7, 7 }, 3, 3, 2, 2, 1, 1, new int[] { 1, 1, 1, 1 } };
        yield return new object[] { new int[] { 2, 3, 6, 6 }, 2, 2, 2, 2, 1, 1, new int[] { 0, 0, 0, 0 } };
        yield return new object[] { new int[] { 1, 2, 5, 5 }, 3, 3, 1, 1, 2, 2, new int[] { 0, 0, 0, 0 } };
        yield return new object[] { new int[] { 1, 1, 4, 6 }, 2, 3, 1, 2, 1, 1, new int[] { 0, 0, 0, 0 } };
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void SpanLoop_MatchesNaive(int[] dims, int kH, int kW, int sH, int sW, int dH, int dW, int[] pads)
    {
        var x = Fill(dims, 77);
        var got = Tensor<float>.MaxPool2D(x, new int[] { kH, kW }, pads, new int[] { sH, sW }, new int[] { dH, dW }, false);
        var want = Naive(x.ToArray(), dims[0], dims[1], dims[2], dims[3], kH, kW, sH, sW, dH, dW, pads[0], pads[1], pads[2], pads[3], out var wantDims);
        Assert.Equal(wantDims, got.Dimensions.ToArray());
        Assert.Equal(want, got.ToArray());
    }

    [Fact]
    public void ViewInput_FallsBackWithIdenticalValues()
    {
        var x = Fill(new int[] { 1, 4, 6, 6 }, 5);
        var view = x.Slice(new SliceIndex(0, 1), new SliceIndex(1, 3), new SliceIndex(1, 5), new SliceIndex(1, 5));
        var got = Tensor<float>.MaxPool2D(view, new int[] { 3, 3 }, new int[] { 1, 1, 1, 1 }, new int[] { 2, 2 }, null, false);
        var sub = new float[1 * 2 * 4 * 4];
        for (int d = 0; d < 2; d++)
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    sub[(d * 4 + r) * 4 + c] = x[0, 1 + d, 1 + r, 1 + c];
        var want = Naive(sub, 1, 2, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, out var wantDims);
        Assert.Equal(wantDims, got.Dimensions.ToArray());
        Assert.Equal(want, got.ToArray());
    }
}
