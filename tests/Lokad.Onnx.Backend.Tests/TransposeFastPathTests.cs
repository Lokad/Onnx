namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// The line-copy transpose path agrees bit-wise with an independent naive
/// reference on head-split attention permutes, random permutes, tails and edge
/// shapes, proving the division-free odometer visits the same blocks.
/// </summary>
public class TransposeFastPathTests
{
    static float[] Naive(float[] x, int[] dims, int[] perm, out int[] outDims)
    {
        int rank = dims.Length;
        outDims = new int[rank];
        for (int d = 0; d < rank; d++) outDims[d] = dims[perm[d]];
        var inStrides = new int[rank];
        int s = 1;
        for (int d = rank - 1; d >= 0; d--) { inStrides[d] = s; s *= dims[d]; }
        var y = new float[x.Length];
        var coords = new int[rank];
        for (int i = 0; i < y.Length; i++)
        {
            int src = 0;
            for (int d = 0; d < rank; d++) src += coords[d] * inStrides[perm[d]];
            y[i] = x[src];
            for (int d = rank - 1; d >= 0; d--)
            {
                coords[d]++;
                if (coords[d] < outDims[d]) break;
                coords[d] = 0;
            }
        }
        return y;
    }

    static void Agree(int[] dims, int[] perm, int seed)
    {
        var rnd = new Random(seed);
        var x = new float[dims.Aggregate(1, (a, q) => a * q)];
        for (int i = 0; i < x.Length; i++) x[i] = (float)rnd.NextDouble() * 2f - 1f;
        var t = new DenseTensor<float>((float[])x.Clone(), (int[])dims.Clone());
        var y = Tensor<float>.Transpose(t, (int[])perm.Clone()).ToDenseTensor();
        var expect = Naive(x, dims, perm, out var outDims);
        Assert.True(y.Dimensions.ToArray().SequenceEqual(outDims), "dims moved.");
        Assert.True(y.Buffer.Span.SequenceEqual(expect), $"fast path diverges on [{string.Join(",", dims)}] perm [{string.Join(",", perm)}].");
    }

    [Fact]
    public void FastPathMatchesNaiveOnAttentionPermutes()
    {
        Agree(new[] { 1, 30, 12, 32 }, new[] { 0, 2, 1, 3 }, 71);
        Agree(new[] { 1, 30, 12, 32 }, new[] { 0, 2, 3, 1 }, 72);
        Agree(new[] { 1, 201, 6, 64 }, new[] { 0, 2, 1, 3 }, 73);
        Agree(new[] { 1, 12, 30, 32 }, new[] { 0, 2, 1, 3 }, 74);
    }

    [Fact]
    public void FastPathMatchesNaiveOnRandomPermutes()
    {
        var rnd = new Random(75);
        for (int trial = 0; trial < 12; trial++)
        {
            int rank = 2 + rnd.Next(4);
            var dims = new int[rank];
            for (int d = 0; d < rank; d++) dims[d] = 1 + rnd.Next(7);
            var perm = Enumerable.Range(0, rank).OrderBy(_ => rnd.Next()).ToArray();
            Agree(dims, perm, 1000 + trial);
        }
    }

    [Fact]
    public void FastPathMatchesNaiveOnEdges()
    {
        Agree(new[] { 1, 1, 12, 32 }, new[] { 0, 2, 1, 3 }, 81);
        Agree(new[] { 2, 1, 5 }, new[] { 2, 1, 0 }, 82);
        Agree(new[] { 7 }, new[] { 0 }, 83);
        Agree(new[] { 1, 8, 1536 }, new[] { 0, 2, 1 }, 84);
    }
}
