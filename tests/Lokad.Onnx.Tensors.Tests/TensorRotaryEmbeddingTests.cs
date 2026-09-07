using System;
using System.Linq;

namespace Lokad.Onnx.Tensors.Tests;

public class TensorRotaryEmbeddingTests
{
    static int[] Strides(int[] dims)
    {
        var s = new int[dims.Length];
        int acc = 1;
        for (int d = dims.Length - 1; d >= 0; d--) { s[d] = acc; acc *= dims[d]; }
        return s;
    }

    static float[] Oracle(float[] xd, int[] dims, float[] cosd, int[] cosDims, float[] sind, int half, int axis)
    {
        int rank = dims.Length;
        int a = axis < 0 ? axis + rank : axis;
        int inner = dims[a];
        int span = inner - half;
        var xs = Strides(dims);
        var cs = Strides(cosDims);
        int off = rank - cosDims.Length;
        var idx = new int[rank];
        var os = Strides(dims);
        var output = new float[xd.Length];
        int cx = 0;
        for (int n = 0; n < xd.Length; n++)
        {
            int cc = 0;
            for (int d = 0; d < rank; d++)
            {
                int cd = d - off;
                int ci = cd < 0 ? 0 : idx[d] % cosDims[cd];
                if (cd >= 0) cc += ci * cs[cd];
            }
            int pos = idx[a];
            int rflat = pos < span ? cx + half * xs[a] : cx - span * xs[a];
            float rotated = pos < span ? -xd[rflat] : xd[rflat];
            output[cx] = xd[cx] * cosd[cc] + rotated * sind[cc];
            for (int d = rank - 1; d >= 0; d--)
            {
                idx[d]++;
                cx += os[d];
                if (idx[d] < dims[d]) break;
                idx[d] = 0;
                cx -= os[d] * dims[d];
            }
        }
        return output;
    }

    static void AssertParity(int[] dims, int[] cosDims, int half, int axis, int seed)
    {
        var rnd = new Random(seed);
        int n = dims.Aggregate(1, (x, y) => x * y);
        int m = cosDims.Aggregate(1, (x, y) => x * y);
        var xd = new float[n];
        var cd = new float[m];
        var sd = new float[m];
        for (int i = 0; i < n; i++) xd[i] = (float)(rnd.NextDouble() * 4.0 - 2.0);
        for (int i = 0; i < m; i++) { cd[i] = (float)(rnd.NextDouble() * 2.0 - 1.0); sd[i] = (float)(rnd.NextDouble() * 2.0 - 1.0); }
        var x = new DenseTensor<float>((float[])xd.Clone(), (int[])dims.Clone());
        var cos = new DenseTensor<float>((float[])cd.Clone(), (int[])cosDims.Clone());
        var sin = new DenseTensor<float>((float[])sd.Clone(), (int[])cosDims.Clone());
        var expected = Oracle(xd, dims, cd, cosDims, sd, half, axis);
        var actual = Tensor<float>.RotaryEmbedding(x, cos, sin, half, axis).ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 5);
    }

    [Fact]
    public void MatchesOracle_LastAxis_EvenDim()
    {
        AssertParity(new[] { 2, 3, 8 }, new[] { 3, 8 }, 4, -1, 11);
    }

    [Fact]
    public void MatchesOracle_BroadcastCos_FirstAxis()
    {
        AssertParity(new[] { 2, 6 }, new[] { 6 }, 3, 1, 12);
    }

    [Fact]
    public void MatchesOracle_OddDim()
    {
        AssertParity(new[] { 2, 5 }, new[] { 5 }, 2, -1, 13);
    }

    [Fact]
    public void KnownRotation_Values()
    {
        var x = new DenseTensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        var ones = new DenseTensor<float>(new[] { 1f, 1f, 1f, 1f }, new[] { 4 });
        var zeros = new DenseTensor<float>(new[] { 0f, 0f, 0f, 0f }, new[] { 4 });
        var through = Tensor<float>.RotaryEmbedding(x, ones, zeros, 2, 0).ToArray();
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, through);
        var rotated = Tensor<float>.RotaryEmbedding(x, zeros, ones, 2, 0).ToArray();
        Assert.Equal(new[] { -3f, -4f, 1f, 2f }, rotated);
    }

    [Fact]
    public void Rejects_BadBroadcast_Half_Axis()
    {
        var x = new DenseTensor<float>(new float[2 * 3 * 8], new[] { 2, 3, 8 });
        var cos = new DenseTensor<float>(new float[3 * 8], new[] { 3, 8 });
        var sin = new DenseTensor<float>(new float[3 * 8], new[] { 3, 8 });
        Assert.Throws<ArgumentException>(() => Tensor<float>.RotaryEmbedding(x, new DenseTensor<float>(new float[3 * 7], new[] { 3, 7 }), sin, 4));
        Assert.Throws<ArgumentException>(() => Tensor<float>.RotaryEmbedding(x, cos, sin, 9));
        Assert.Throws<ArgumentException>(() => Tensor<float>.RotaryEmbedding(x, cos, sin, 4, 3));
    }
}
