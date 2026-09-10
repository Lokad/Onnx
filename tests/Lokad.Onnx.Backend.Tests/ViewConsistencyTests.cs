namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins that sliced and broadcast views, which share storage with exotic
/// strides, agree with dense computation: every kernel densifies defensively
/// today, and these tests catch any future fast path that misreads a view.
/// Expectations are hand-computed; the differential corpus cannot express
/// view inputs by its dense-only format.
/// </summary>
public class ViewConsistencyTests
{
    static DenseTensor<float> Base() =>
        DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });

    static Tensor<float> MiddleCols() =>
        Base().Slice(new SliceIndex(0, 2), new SliceIndex(1, 3));

    static Tensor<float> BroadcastRow() =>
        DenseTensor<float>.OfValues(new float[] { 5f, 6f }).PadLeft().BroadcastDim(0, 2);

    [Fact]
    public void Slice_ReduceMean_MatchesDense()
    {
        var axes = new int[] { 1 }.ToTensor<int>();
        var got = Tensor<float>.ReduceMean(MiddleCols(), axes, false, false);
        Assert.Equal(new float[] { 2.5f, 5.5f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 2f, 3f }, { 5f, 6f } });
        Assert.Equal(Tensor<float>.ReduceMean(dense, axes, false, false).ToArray(), got.ToArray());
    }

    [Fact]
    public void Slice_Softmax_MatchesDense()
    {
        var got = Tensor<float>.Softmax(MiddleCols(), -1, null, 13);
        var values = got.ToArray();
        Assert.Equal(0.26894f, values[0], 5);
        Assert.Equal(0.73106f, values[1], 5);
        Assert.Equal(0.26894f, values[2], 5);
        Assert.Equal(0.73106f, values[3], 5);
        var dense = DenseTensor<float>.OfValues(new float[,] { { 2f, 3f }, { 5f, 6f } });
        Assert.Equal(Tensor<float>.Softmax(dense, -1, null, 13).ToArray(), values);
    }

    [Fact]
    public void Broadcast_MatMul_MatchesDense()
    {
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var expected = new float[] { 15f, 18f, 35f, 42f };
        var got = Tensor<float>.MatMul(a, BroadcastRow(), TensorExecutionOptions.Scalar);
        Assert.Equal(new int[] { 2, 2 }, got.Dimensions.ToArray());
        Assert.Equal(expected, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 5f, 6f } });
        Assert.Equal(Tensor<float>.MatMul(a, dense, TensorExecutionOptions.Scalar).ToArray(), got.ToArray());
    }

    [Fact]
    public void Broadcast_Add_MatchesDense()
    {
        var a = DenseTensor<float>.OfValues(new float[,] { { 10f, 20f }, { 30f, 40f } });
        var expected = new float[] { 15f, 26f, 35f, 46f };
        var got = Tensor<float>.Add(BroadcastRow(), a);
        Assert.Equal(expected, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 5f, 6f } });
        Assert.Equal(Tensor<float>.Add(dense, a).ToArray(), got.ToArray());
    }

    [Fact]
    public void Slice_Conv2D_MatchesDense()
    {
        // ORT 1.29: [18, 22, 34, 38].
        var parent = DenseTensor<float>.OfValues(new float[1, 1, 3, 4] { { { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f }, { 9f, 10f, 11f, 12f } } } });
        var view = parent.Slice(new SliceIndex(0, 1), new SliceIndex(0, 1), new SliceIndex(0, 3), new SliceIndex(1, 4));
        Assert.IsType<TensorSlice<float>>(view);
        var w = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 1f }, { 1f, 1f } } } });
        var got = Tensor<float>.Conv2D(view, w, 1, new int[] { 0, 0, 0, 0 }, null, null, new int[] { 1, 1 }, null);
        Assert.Equal(new float[] { 18f, 22f, 34f, 38f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[1, 1, 3, 3] { { { { 2f, 3f, 4f }, { 6f, 7f, 8f }, { 10f, 11f, 12f } } } });
        Assert.Equal(Tensor<float>.Conv2D(dense, w, 1, new int[] { 0, 0, 0, 0 }, null, null, new int[] { 1, 1 }, null).ToArray(), got.ToArray());
    }

    [Fact]
    public void Slice_LayerNorm_MatchesDense()
    {
        // ORT 1.29: rows of [-1.2247356, 0, 1.2247356].
        var parent = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } });
        var view = parent.Slice(new SliceIndex(0, 2), new SliceIndex(0, 3));
        Assert.IsType<TensorSlice<float>>(view);
        var scale = DenseTensor<float>.OfValues(new float[] { 1f, 1f, 1f });
        var bias = DenseTensor<float>.OfValues(new float[] { 0f, 0f, 0f });
        var got = Tensor<float>.LayerNormalization(view, scale, bias, -1, 1e-5f);
        var expected = new float[] { -1.2247357f, 0f, 1.2247357f, -1.2247357f, 0f, 1.2247357f };
        Assert.Equal(expected.Length, got.ToArray().Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], got.ToArray()[i], 6);
        var dense = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 5f, 6f, 7f } });
        Assert.Equal(Tensor<float>.LayerNormalization(dense, scale, bias, -1, 1e-5f).ToArray(), got.ToArray());
    }

    [Fact]
    public void Slice_Gather_MatchesDense()
    {
        // Gathering rows out of a strided view exercises the generic
        // index path (the span fast path requires standard strides).
        var idx = DenseTensor<int>.OfValues(new int[] { 1, 0 });
        var got = Tensor<float>.Gather(MiddleCols(), idx, 0);
        Assert.Equal(new int[] { 2, 2 }, got.Dimensions.ToArray());
        Assert.Equal(new float[] { 5f, 6f, 2f, 3f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 2f, 3f }, { 5f, 6f } });
        Assert.Equal(Tensor<float>.Gather(dense, idx, 0).ToArray(), got.ToArray());
    }

    [Fact]
    public void Slice_Concat_MatchesDense()
    {
        // Concatenating two views stacked on the same storage walks the
        // generic iterator (the chunk copier needs dense inputs).
        var view = MiddleCols();
        var got = Tensor<float>.Concat(view, view, 0);
        Assert.Equal(new int[] { 4, 2 }, got.Dimensions.ToArray());
        Assert.Equal(new float[] { 2f, 3f, 5f, 6f, 2f, 3f, 5f, 6f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 2f, 3f }, { 5f, 6f } });
        Assert.Equal(Tensor<float>.Concat(dense, dense, 0).ToArray(), got.ToArray());
    }

    [Fact]
    public void Slice_Transpose_MatchesDense()
    {
        // Permuting a strided view must read through the view strides,
        // not the parent storage order.
        var got = Tensor<float>.Transpose(MiddleCols(), new int[] { 1, 0 });
        Assert.Equal(new int[] { 2, 2 }, got.Dimensions.ToArray());
        Assert.Equal(new float[] { 2f, 5f, 3f, 6f }, got.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 2f, 3f }, { 5f, 6f } });
        Assert.Equal(Tensor<float>.Transpose(dense, new int[] { 1, 0 }).ToArray(), got.ToArray());
    }

    [Fact]
    public void NestedSlice_MatchesDense()
    {
        // A slice of a slice chains two indirections; values must match
        // the equivalent dense window.
        var nested = MiddleCols().Slice(new SliceIndex(1, 2), new SliceIndex(0, 2));
        Assert.Equal(new int[] { 1, 2 }, nested.Dimensions.ToArray());
        Assert.Equal(new float[] { 5f, 6f }, nested.ToArray());
        var dense = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f } });
        Assert.Equal(dense.ToArray(), nested.ToArray());
    }
}
