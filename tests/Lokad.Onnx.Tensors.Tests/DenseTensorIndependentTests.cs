namespace Lokad.Onnx.Tensors.Tests;

/// <summary>
/// Independent oracle for DenseTensor storage against the row-major contract.
/// Strides, flat indices, and expected sequences are recomputed here from the
/// layout definition, not by calling into the tensor implementation, so this
/// passes on the pre-rewrite and rewritten bodies and proves spec parity.
/// </summary>
public class DenseTensorIndependentTests
{
    static int[] NaiveStrides(int[] dims)
{
        var strides = new int[dims.Length];
        int stride = 1;
        for (int i = dims.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= dims[i];
        }
        return strides;
}

    static int NaiveLength(int[] dims)
    {
        int length = 1;
        foreach (int d in dims) length *= d;
        return length;
    }

    static int NaiveFlat(int[] indices, int[] strides)
    {
        int flat = 0;
        for (int i = 0; i < indices.Length; i++) flat += indices[i] * strides[i];
        return flat;
    }

    static int[] NaiveCoords(int flat, int[] dims)
    {
        var indices = new int[dims.Length];
        for (int i = dims.Length - 1; i >= 0; i--)
        {
            indices[i] = flat % dims[i];
            flat /= dims[i];
        }
        return indices;
    }

    static readonly int[][] Shapes = new int[][]
    {
        new int[] { 1 },
        new int[] { 5 },
        new int[] { 2, 3 },
        new int[] { 3, 1 },
        new int[] { 1, 4 },
        new int[] { 2, 3, 4 },
        new int[] { 4, 4 },
        new int[0],
    };

    [Fact]
    public void Storage_MatchesNaiveLayout()
    {
        foreach (int[] dims in Shapes)
        {
            int length = NaiveLength(dims);
            int[] strides = NaiveStrides(dims);
            var t = new DenseTensor<int>(dims);
            Assert.Equal(dims.Length, t.Rank);
            Assert.Equal(length, (int)t.Length);
            Assert.Equal(dims, t.Dimensions.ToArray());
            Assert.Equal(strides, t.Strides.ToArray());
            for (int i = 0; i < length; i++) t.SetValue(i, i * 7 + 3);
            for (int i = 0; i < length; i++) Assert.Equal(i * 7 + 3, t.GetValue(i));
            Assert.Equal(IntoList(length), t.Buffer.ToArray());
            for (int i = 0; i < length; i++)
            {
                int[] coords = NaiveCoords(i, dims);
                Assert.Equal(i * 7 + 3, t[coords]);
                Assert.Equal(i, NaiveFlat(coords, strides));
            }
        }
    }

    static int[] IntoList(int length)
    {
        var expected = new int[length];
        for (int i = 0; i < length; i++) expected[i] = i * 7 + 3;
        return expected;
    }
    [Fact]
    public void Ctors_MatchNaiveContents()
    {
        var values = new int[] { 10, 20, 30, 40, 50, 60 };
        var rank1 = DenseTensor<int>.OfValues(values);
        values[0] = -1;
        Assert.Equal(new int[] { 10, 20, 30, 40, 50, 60 }, rank1.ToArray());
        Assert.Equal(new int[] { 6 }, rank1.Dimensions.ToArray());

        var span = new int[] { 1, 2, 3, 4 }.AsSpan();
        var shaped = DenseTensor<int>.OfValues(span, new int[] { 2, 2 });
        Assert.Equal(new int[] { 1, 2, 3, 4 }, shaped.ToArray());
        Assert.Equal(3, shaped[1, 0]);

        var grid = new int[,] { { 1, 2, 3 }, { 4, 5, 6 } };
        var fromArray = DenseTensor<int>.OfValues(grid);
        Assert.Equal(new int[] { 2, 3 }, fromArray.Dimensions.ToArray());
        Assert.Equal(new int[] { 1, 2, 3, 4, 5, 6 }, fromArray.ToArray());
        Assert.Equal(5, fromArray[1, 1]);

        var backing = new int[] { 7, 8, 9, 10 };
        var wrapped = new DenseTensor<int>(backing.AsMemory(), new int[] { 2, 2 });
        backing[0] = 70;
        Assert.Equal(70, wrapped.GetValue(0));
        wrapped.SetValue(1, 80);
        Assert.Equal(80, backing[1]);

        var ex = Assert.Throws<ArgumentException>(() => new DenseTensor<int>(backing.AsMemory(), new int[] { 3 }));
        Assert.Contains("must match product", ex.Message);
    }

    [Fact]
    public void CloneReshape_MatchNaiveSemantics()
    {
        var t = DenseTensor<int>.OfValues(new int[] { 0, 1, 2, 3, 4, 5 });
        var reshaped = t.Reshape(new int[] { 2, 3 });
        Assert.Equal(new int[] { 0, 1, 2, 3, 4, 5 }, reshaped.ToArray());
        Assert.Equal(4, reshaped[1, 1]);
        Assert.Throws<ArgumentException>(() => t.Reshape(new int[] { 4 }));

        var clone = t.Clone();
        clone.SetValue(0, -5);
        Assert.Equal(0, t.GetValue(0));
        Assert.Equal(-5, clone.GetValue(0));
    }

    [Fact]
    public void ReversedTensor_ToDenseTensor_MatchesNaiveOrder()
    {
        int[] dims = new int[] { 2, 3 };
        int[] rowStrides = NaiveStrides(dims);
        var rev = new DenseTensor<int>(dims, reverseStride: true);
        for (int i = 0; i < 6; i++)
        {
            int[] coords = NaiveCoords(i, dims);
            rev[coords] = i;
        }
        var dense = rev.ToDenseTensor();
        Assert.Equal(new int[] { 0, 1, 2, 3, 4, 5 }, dense.ToArray());
        Assert.Equal(rowStrides, dense.Strides.ToArray());
    }
}
