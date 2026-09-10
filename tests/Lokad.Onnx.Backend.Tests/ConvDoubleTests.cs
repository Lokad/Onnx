namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins the double Conv path with hand-computed values: ORT CPU has no
/// double Conv kernel (NOT_IMPLEMENTED), so no differential reference
/// exists; exact small-integer arithmetic needs none.
/// </summary>
public class ConvDoubleTests
{
    [Fact]
    public void GemmDiagonal_SumsExactly()
    {
        // 1*1 + 2*0 + 3*0 + 4*1 = 5.
        var x = DenseTensor<double>.OfValues(new double[1, 1, 2, 2] { { { { 1.0, 2.0 }, { 3.0, 4.0 } } } });
        var w = DenseTensor<double>.OfValues(new double[1, 1, 2, 2] { { { { 1.0, 0.0 }, { 0.0, 1.0 } } } });
        var y = Tensor<double>.Conv2D(x, w, 1, new int[] { 0, 0, 0, 0 }, null, null, new int[] { 1, 1 }, null);
        Assert.Equal(new int[] { 1, 1, 1, 1 }, y.Dimensions.ToArray());
        Assert.Equal(new double[] { 5.0 }, y.ToArray());
    }

    [Fact]
    public void ChannelPicker_MapsChannels()
    {
        var x = DenseTensor<double>.OfValues(new double[1, 2, 2, 2]
        {
            { { { 1.0, 2.0 }, { 3.0, 4.0 } }, { { 5.0, 6.0 }, { 7.0, 8.0 } } },
        });
        var w = DenseTensor<double>.OfValues(new double[2, 2, 1, 1]
        {
            { { { 1.0 } }, { { 0.0 } } },
            { { { 0.0 } }, { { 1.0 } } },
        });
        var y = Tensor<double>.Conv2D(x, w, 1, new int[] { 0, 0, 0, 0 }, null, null, new int[] { 1, 1 }, null);
        Assert.Equal(new int[] { 1, 2, 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 }, y.ToArray());
    }
}