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
}
