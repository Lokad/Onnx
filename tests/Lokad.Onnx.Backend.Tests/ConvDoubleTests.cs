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
    public void NaNInput_YieldsNaN()
    {
        // No ORT CPU double Conv kernel exists (NOT_IMPLEMENTED), so
        // this is hand-exact like its neighbors: NaN poisons the
        // multiply-accumulate.
        var x = DenseTensor<double>.OfValues(new double[1, 1, 1, 1] { { { { double.NaN } } } });
        var w = DenseTensor<double>.OfValues(new double[1, 1, 1, 1] { { { { 1.0 } } } });
        var y = Tensor<double>.Conv2D(x, w, 1, new int[] { 0, 0, 0, 0 }, null, null, new int[] { 1, 1 }, null);
        Assert.True(double.IsNaN(y.ToArray()[0]));
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

    [Fact]
    public void InfInput_Propagates()
    {
        // No ORT CPU double Conv kernel exists (NOT_IMPLEMENTED); exact
        // IEEE like the float twin: inf*0 is NaN on the direct and padded
        // paths, opposing infinities cancel across the accumulation, and
        // an infinite bias propagates through the epilogue.
        static double[] RunConv(double[,,,] xv, double[,,,] wv, double[]? bv, int[] pads)
        {
            var y = Tensor<double>.Conv2D(
                DenseTensor<double>.OfValues(xv), DenseTensor<double>.OfValues(wv),
                1, pads, bv is null ? null : DenseTensor<double>.OfValues(bv),
                null, new int[] { 1, 1 }, null);
            return y.ToArray();
        }
        var nopad = new int[] { 0, 0, 0, 0 };
        var y = RunConv(
            new double[1, 1, 1, 1] { { { { double.PositiveInfinity } } } },
            new double[1, 1, 1, 1] { { { { 0.0 } } } }, null, nopad);
        Assert.True(double.IsNaN(y[0]));
        y = RunConv(
            new double[1, 1, 1, 1] { { { { double.PositiveInfinity } } } },
            new double[1, 1, 1, 1] { { { { 2.0 } } } }, null, nopad);
        Assert.Equal(double.PositiveInfinity, y[0]);
        y = RunConv(
            new double[1, 1, 1, 2] { { { { double.PositiveInfinity, 5.0 } } } },
            new double[1, 1, 1, 2] { { { { 1.0, double.NegativeInfinity } } } }, null, nopad);
        Assert.True(double.IsNaN(y[0]));
        y = RunConv(
            new double[1, 1, 1, 1] { { { { 1.0 } } } },
            new double[1, 1, 1, 1] { { { { double.PositiveInfinity } } } }, null,
            new int[] { 0, 0, 1, 1 });
        Assert.Equal(double.PositiveInfinity, y[0]);
        Assert.True(double.IsNaN(y[1]));
        Assert.True(double.IsNaN(y[2]));
        Assert.True(double.IsNaN(y[3]));
        y = RunConv(
            new double[1, 1, 1, 1] { { { { double.PositiveInfinity } } } },
            new double[1, 1, 1, 1] { { { { 1.0 } } } },
            new double[] { double.NegativeInfinity }, nopad);
        Assert.True(double.IsNaN(y[0]));
        y = RunConv(
            new double[1, 1, 1, 1] { { { { 1.0 } } } },
            new double[1, 1, 1, 1] { { { { 1.0 } } } },
            new double[] { double.PositiveInfinity }, nopad);
        Assert.Equal(double.PositiveInfinity, y[0]);
    }

}
