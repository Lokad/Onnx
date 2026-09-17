namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// E62: the vectorized softmax maximum reproduces the scalar break-on-first-NaN
/// loop exactly (comparison uses ==, which treats -0 and +0 as equal; downstream
/// exp absorbs the sign bit-identically). The scalar path of the same helper is
/// the reference, so these tests pin routing equivalence, not a second formula.
/// </summary>
public class SoftmaxMaxTests
{
    static void MaxEqual(float[] x)
    {
        float vec = Tensor<float>.SoftmaxContiguousMax(x, 0, x.Length, true);
        float sc = Tensor<float>.SoftmaxContiguousMax(x, 0, x.Length, false);
        bool same = vec == sc || (float.IsNaN(vec) && float.IsNaN(sc));
        Assert.True(same, $"max differs: vec={vec:R} scalar={sc:R} on len {x.Length}.");
    }

    static float[] RandomRow(int block, int seed)
    {
        var rnd = new Random(seed);
        var x = new float[block];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rnd.NextDouble() * 20 - 10);
        return x;
    }

    [Fact]
    public void MaxMatchesScalarOnWidthEdges()
    {
        foreach (int block in new[] { 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 384 })
            MaxEqual(RandomRow(block, 1000 + block));
    }

    [Fact]
    public void MaxMatchesScalarOnExceptionalRows()
    {
        foreach (int block in new[] { 8, 30, 128 })
        {
            var x = RandomRow(block, 2000 + block);
            x[0] = float.NaN; MaxEqual(x);
            x = RandomRow(block, 2001 + block);
            x[block / 2] = float.NaN; MaxEqual(x);
            x = RandomRow(block, 2002 + block);
            x[block - 1] = float.NaN; MaxEqual(x);
            var all = new float[block];
            for (int i = 0; i < all.Length; i++) all[i] = float.NaN;
            MaxEqual(all);
            x = RandomRow(block, 2003 + block);
            x[3 % block] = float.PositiveInfinity; MaxEqual(x);
            x = RandomRow(block, 2004 + block);
            for (int i = 0; i < x.Length; i++) x[i] = float.NegativeInfinity;
            MaxEqual(x);
        }
    }

    [Fact]
    public void MaxMatchesScalarOnZeroMixAndTies()
    {
        var z = new float[64];
        for (int i = 0; i < z.Length; i++) z[i] = (i % 2 == 0) ? 0f : float.NegativeZero;
        MaxEqual(z);
        var t = new float[128];
        for (int i = 0; i < t.Length; i++) t[i] = 3.25f;
        MaxEqual(t);
    }
}

