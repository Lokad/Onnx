namespace Lokad.Onnx.Tensors.Tests;

public class TensorLayoutTests
{
    [Fact]
    public void Dims_ReturnsCopy()
    {
        ITensor t = DenseTensor<int>.OfValues(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var dims = t.Dims;
        dims[0] = 10;
        Assert.Equal(new int[] { 2, 3 }, t.Dims);
        Assert.Equal(6, t.Length);
        Assert.Equal(1, ((Tensor<int>)t).GetValue(0));
    }

    [Fact]
    public void BroadcastClone_Isolated()
    {
        Tensor<int> t = DenseTensor<int>.OfValues(new int[,] { { 1, 2, 3 } });
        var b = t.BroadcastDim(0, 2);
        var c = b.Clone();
        c.SetValue(0, 99);
        Assert.Equal(99, c.GetValue(0));
        Assert.Equal(1, t.GetValue(0));
        Assert.Equal(new int[] { 1, 3 }, t.Dimensions.ToArray());
    }

    [Fact]
    public void BroadcastDim_PreservesSource()
    {
        Tensor<int> t = DenseTensor<int>.OfValues(new int[,] { { 7 } });
        var b1 = t.BroadcastDim(0, 2);
        Assert.Equal(new int[] { 2, 1 }, b1.Dimensions.ToArray());
        var b2 = b1.BroadcastDim(1, 4);
        Assert.Equal(new int[] { 2, 4 }, b2.Dimensions.ToArray());
        Assert.Equal(new int[] { 2, 1 }, b1.Dimensions.ToArray());
        Assert.Equal(2, b1.Length);
        Assert.Equal(new int[] { 7, 7 }, b1.ToArray());
        Assert.Equal(new int[] { 1, 1 }, t.Dimensions.ToArray());
        Assert.Equal(1, t.Length);
    }

    [Fact]
    public void NestedSliceBroadcast_ViewWritesThrough()
    {
        Tensor<int> t = DenseTensor<int>.OfValues(new int[,] { { 1, 2, 3 } });
        var b = t.BroadcastDim(0, 2);
        var s = b[0..2, 1];
        Assert.Equal(new int[] { 2 }, s.Dimensions.ToArray());
        Assert.Equal(2, s.GetValue(0));
        Assert.Equal(2, s.GetValue(1));
        s.SetValue(0, 9);
        Assert.Equal(9, t.GetValue(1));
        s.SetValue(0, 2);
        Assert.Equal(2, t.GetValue(1));
    }

    [Fact]
    public void ReversedStrides_Roundtrip()
    {
        var r = new int[2, 3].ToTensor<int>(reverseStride: true);
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) r[i, j] = i * 3 + j;
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) Assert.Equal(i * 3 + j, r[i, j]);
        var d = r.ToDenseTensor();
        Assert.Equal(new int[] { 2, 3 }, d.Dimensions.ToArray());
        Assert.Equal(new int[] { 0, 1, 2, 3, 4, 5 }, d.ToArray());
    }

    [Fact]
    public void InsertDim_IsView_CloneIsCopy()
    {
        Tensor<int> t = DenseTensor<int>.OfValues(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var v = t.InsertDim(0);
        Assert.Equal(new int[] { 1, 2, 3 }, v.Dimensions.ToArray());
        v.SetValue(0, 99);
        Assert.Equal(99, t.GetValue(0));
        v.SetValue(0, 1);
        var c = t.Clone();
        c.SetValue(0, 7);
        Assert.Equal(7, c.GetValue(0));
        Assert.Equal(1, t.GetValue(0));
    }

    [Fact]
    public void DenseStorage_AgreesWithIndexing()
    {
        var t = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        Tensor<float> tt = t;
        var span = tt.Storage.Span;
        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                int offset = tt.GetStorageIndex(new int[] { r, c });
                Assert.Equal(r * 3 + c, offset);
                Assert.Equal(tt.GetValue(r * 3 + c), span[offset], 5);
            }
        }
    }

    [Fact]
    public void OverlappingCopy_ReversedIntoSelf()
    {
        Tensor<int> x = Tensor<int>.Arange(0, 12).Reshape(3, 4);
        var rev = new DenseTensor<int>(((DenseTensor<int>)x).Buffer, new int[] { 3, 4 }, true);
        x[0..3, 0..4] = rev;
        Assert.Equal(new int[] { 0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11 }, x.ToArray());
    }
}
