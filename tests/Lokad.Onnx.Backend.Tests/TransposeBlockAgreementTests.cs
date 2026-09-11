namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// The transpose fast paths (contiguous-run line copies, 4D head-merge face
/// tiles) agree with the generic odometer on attention shapes, other ranks,
/// other perms and other dtypes; hand values pin the element mapping.
/// </summary>
public class TransposeBlockAgreementTests
{
    static T[] Oracle<T>(T[] src, int[] dims, int[] perm) where T : unmanaged
    {
        int rank = dims.Length;
        var destDims = new int[rank];
        for (int i = 0; i < rank; i++) destDims[i] = dims[perm[i]];
        var srcStrides = new int[rank];
        int acc = 1;
        for (int d = rank - 1; d >= 0; d--) { srcStrides[d] = acc; acc *= dims[d]; }
        var dst = new T[src.Length];
        var coords = new int[rank];
        for (int i = 0; i < dst.Length; i++)
        {
            int off = 0;
            for (int d = 0; d < rank; d++) off += coords[d] * srcStrides[perm[d]];
            dst[i] = src[off];
            for (int d = rank - 1; d >= 0; d--)
            {
                coords[d]++;
                if (coords[d] < destDims[d]) break;
                coords[d] = 0;
            }
        }
        return dst;
    }

    static DenseTensor<float> FillFloat(int[] dims, int seed)
    {
        var rnd = new Random(seed);
        var t = Tensor<float>.Zeros(dims).ToDenseTensor();
        for (int i = 0; i < t.Length; i++) t.SetValue(i, rnd.NextSingle());
        return t;
    }

    [Theory]
    [InlineData(new int[] { 1, 30, 12, 32 }, new int[] { 0, 2, 1, 3 })]
    [InlineData(new int[] { 1, 12, 30, 32 }, new int[] { 0, 2, 3, 1 })]
    [InlineData(new int[] { 2, 8, 4, 16 }, new int[] { 0, 2, 1, 3 })]
    [InlineData(new int[] { 2, 4, 8, 16 }, new int[] { 0, 2, 3, 1 })]
    [InlineData(new int[] { 6, 8 }, new int[] { 1, 0 })]
    [InlineData(new int[] { 2, 3, 4 }, new int[] { 2, 1, 0 })]
    [InlineData(new int[] { 2, 3, 4 }, new int[] { 0, 2, 1 })]
    [InlineData(new int[] { 2, 3, 4 }, new int[] { 1, 0, 2 })]
    [InlineData(new int[] { 2, 3, 4 }, new int[] { 0, 1, 2 })]
    [InlineData(new int[] { 5 }, new int[] { 0 })]
    public void FloatTranspose_MatchesOracle(int[] dims, int[] perm)
    {
        var x = FillFloat(dims, 7);
        var got = Tensor<float>.Transpose(x, perm);
        var want = Oracle(x.ToArray(), dims, perm);
        Assert.Equal(want, got.ToArray());
        var dest = DenseTensor<float>.OfShape(got.Dimensions.ToArray());
        Tensor<float>.Transpose(x, dest, perm);
        Assert.Equal(want, dest.ToArray());
    }

    [Theory]
    [InlineData(new int[] { 1, 30, 12, 32 }, new int[] { 0, 2, 1, 3 })]
    [InlineData(new int[] { 1, 12, 30, 32 }, new int[] { 0, 2, 3, 1 })]
    [InlineData(new int[] { 6, 8 }, new int[] { 1, 0 })]
    public void DoubleTranspose_MatchesOracle(int[] dims, int[] perm)
    {
        var rnd = new Random(11);
        var x = Tensor<double>.Zeros(dims).ToDenseTensor();
        for (int i = 0; i < x.Length; i++) x.SetValue(i, rnd.NextDouble());
        var got = Tensor<double>.Transpose(x, perm);
        Assert.Equal(Oracle(x.ToArray(), dims, perm), got.ToArray());
    }

    [Theory]
    [InlineData(new int[] { 1, 30, 12, 32 }, new int[] { 0, 2, 1, 3 })]
    [InlineData(new int[] { 1, 12, 30, 32 }, new int[] { 0, 2, 3, 1 })]
    [InlineData(new int[] { 6, 8 }, new int[] { 1, 0 })]
    public void IntTranspose_MatchesOracle(int[] dims, int[] perm)
    {
        var rnd = new Random(13);
        var x = Tensor<int>.Zeros(dims).ToDenseTensor();
        for (int i = 0; i < x.Length; i++) x.SetValue(i, rnd.Next());
        var got = Tensor<int>.Transpose(x, perm);
        Assert.Equal(Oracle(x.ToArray(), dims, perm), got.ToArray());
    }

    [Fact]
    public void HeadSplit_MatchesHandValues()
    {
        // [1,2,2,2] with values 0..7, perm (0,2,1,3): depth lines of 2 move whole.
        var x = DenseTensor<float>.OfValues(new float[] { 0, 1, 2, 3, 4, 5, 6, 7 }, new int[] { 1, 2, 2, 2 });
        var got = Tensor<float>.Transpose(x, new int[] { 0, 2, 1, 3 });
        Assert.Equal(new int[] { 1, 2, 2, 2 }, got.Dimensions.ToArray());
        Assert.Equal(new float[] { 0, 1, 4, 5, 2, 3, 6, 7 }, got.ToArray());
    }

    [Fact]
    public void HeadMerge_MatchesHandValues()
    {
        // [1,2,2,2] with values 0..7, perm (0,2,3,1): each (i,j) face rotates.
        var x = DenseTensor<float>.OfValues(new float[] { 0, 1, 2, 3, 4, 5, 6, 7 }, new int[] { 1, 2, 2, 2 });
        var got = Tensor<float>.Transpose(x, new int[] { 0, 2, 3, 1 });
        Assert.Equal(new int[] { 1, 2, 2, 2 }, got.Dimensions.ToArray());
        Assert.Equal(new float[] { 0, 4, 1, 5, 2, 6, 3, 7 }, got.ToArray());
    }
}