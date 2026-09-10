using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins the double pooling paths with zero prior coverage. ORT CPU
/// implements double MaxPool (differential pin) but not double
/// GlobalAveragePool (NOT_IMPLEMENTED, like double Conv), so the GAP pin
/// is hand-exact. All values are exactly representable.
/// </summary>
public class PoolDoubleTests
{
    [Fact]
    public void MaxPoolDouble_MatchesOrt()
    {
        // ORT 1.29: 2x2 kernel, stride 1 on [[1..9]] -> [[5, 6], [8, 9]].
        var x = DenseTensor<double>.OfValues(new double[1, 1, 3, 3] { { { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } } } });
        var r = CPU.MaxPool(x, null, null, null, new int[] { 2, 2 }, null, null, new int[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<double>)r.Outputs[0];
        Assert.Equal(new int[] { 1, 1, 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new double[] { 5.0, 6.0, 8.0, 9.0 }, y.ToArray());
    }

    [Fact]
    public void GlobalAveragePoolDoubleNaN_YieldsNaN()
    {
        // No ORT reference exists (NOT_IMPLEMENTED); hand-exact like its
        // float twin: NaN poisons the spatial mean.
        var x = DenseTensor<double>.OfValues(new double[1, 1, 2, 2] { { { { 1.0, double.NaN }, { 3.0, 4.0 } } } });
        var r = CPU.GlobalAveragePool(x, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.True(double.IsNaN(((Tensor<double>)r.Outputs[0])[0, 0, 0, 0]));
    }

    [Fact]
    public void MaxPoolDoubleNaN_Propagates()
    {
        // ORT 1.29 double: NaN wins even in mixed windows ([NaN,2,3,4]
        // and all-NaN both yield NaN), unlike the float core.
        var mixed = DenseTensor<double>.OfValues(new double[1, 1, 2, 2] { { { { double.NaN, 2.0 }, { 3.0, 4.0 } } } });
        var rm = CPU.MaxPool(mixed, null, null, null, new int[] { 2, 2 }, null, null, new int[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, rm.Status);
        Assert.True(double.IsNaN(((Tensor<double>)rm.Outputs[0])[0, 0, 0, 0]));
        var all = DenseTensor<double>.OfValues(new double[1, 1, 2, 2] { { { { double.NaN, double.NaN }, { double.NaN, double.NaN } } } });
        var ra = CPU.MaxPool(all, null, null, null, new int[] { 2, 2 }, null, null, new int[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, ra.Status);
        Assert.True(double.IsNaN(((Tensor<double>)ra.Outputs[0])[0, 0, 0, 0]));
    }

    [Fact]
    public void GlobalAveragePoolDouble_IsExactMean()
    {
        // No ORT reference exists (NOT_IMPLEMENTED); (1+2+3+4)/4 = 2.5.
        var x = DenseTensor<double>.OfValues(new double[1, 1, 2, 2] { { { { 1.0, 2.0 }, { 3.0, 4.0 } } } });
        var r = CPU.GlobalAveragePool(x, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<double>)r.Outputs[0];
        Assert.Equal(new int[] { 1, 1, 1, 1 }, y.Dimensions.ToArray());
        Assert.Equal(new double[] { 2.5 }, y.ToArray());
    }

    [Fact]
    public void MaxPoolDouble_CeilDilationPadded()
    {
        // ORT 1.29: ceil + dilation 2 + pads 1 over arange(49), mirroring
        // the float combined-geometry pin on the separate double kernel.
        var x = DenseTensor<double>.OfValues(new double[1, 1, 7, 7]);
        for (int i = 0; i < 49; i++) x.Buffer.Span[i] = i;
        var r = CPU.MaxPool(x, null, 1, new int[] { 2, 2 }, new int[] { 3, 3 }, new int[] { 1, 1, 1, 1 }, null, new int[] { 2, 2 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<double>)r.Outputs[0];
        Assert.Equal(new int[] { 1, 1, 3, 3 }, y.Dimensions.ToArray());
        Assert.Equal(new double[] { 24.0, 26.0, 26.0, 38.0, 40.0, 40.0, 38.0, 40.0, 40.0 }, y.ToArray());
    }

    [Fact]
    public void MaxPoolDoubleInf_MatchesOrt()
    {
        // ORT 1.29 double: unlike the float -FLT_MAX seed, the double core
        // seeds -inf, so an all -inf window (padded or not) stays -inf.
        var k = new[] { 2, 2 };
        var s = new[] { 1, 1 };
        var plain = new (double[] v, double expected)[]
        {
            (new double[] { double.PositiveInfinity, 1.0, 2.0, 3.0 }, double.PositiveInfinity),
            (new double[] { double.NegativeInfinity, double.NegativeInfinity, double.NegativeInfinity, double.NegativeInfinity }, double.NegativeInfinity),
            (new double[] { double.NegativeInfinity, 5.0, 6.0, 7.0 }, 7.0),
        };
        foreach (var (v, expected) in plain)
        {
            var x = DenseTensor<double>.OfValues(new double[1, 1, 2, 2] { { { { v[0], v[1] }, { v[2], v[3] } } } });
            var r = CPU.MaxPool(x, null, null, null, k, null, null, s, null);
            Assert.Equal(OpStatus.Success, r.Status);
            Assert.Equal(expected, ((Tensor<double>)r.Outputs[0]).ToArray()[0]);
        }
        var padded = new (double v, double expected)[]
        {
            (double.NegativeInfinity, double.NegativeInfinity),
            (double.PositiveInfinity, double.PositiveInfinity),
            (5.0, 5.0),
        };
        foreach (var (v, expected) in padded)
        {
            var x = DenseTensor<double>.OfValues(new double[1, 1, 1, 1] { { { { v } } } });
            var r = CPU.MaxPool(x, null, null, null, k, new[] { 0, 0, 1, 1 }, null, s, null);
            Assert.Equal(OpStatus.Success, r.Status);
            Assert.Equal(expected, ((Tensor<double>)r.Outputs[0]).ToArray()[0]);
        }
    }

    [Fact]
    public void GlobalAveragePoolDoubleInf_IsExact()
    {
        // No ORT reference exists (NOT_IMPLEMENTED); hand-exact like the
        // float twin: inf/4 is inf, while (inf - inf + 1 + 2)/4 is NaN.
        var x = DenseTensor<double>.OfValues(new double[1, 1, 2, 2] { { { { double.PositiveInfinity, 1.0 }, { 2.0, 3.0 } } } });
        var r = CPU.GlobalAveragePool(x, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(double.PositiveInfinity, ((Tensor<double>)r.Outputs[0])[0, 0, 0, 0]);
        var xc = DenseTensor<double>.OfValues(new double[1, 1, 2, 2] { { { { double.PositiveInfinity, double.NegativeInfinity }, { 1.0, 2.0 } } } });
        var rc = CPU.GlobalAveragePool(xc, null);
        Assert.Equal(OpStatus.Success, rc.Status);
        Assert.True(double.IsNaN(((Tensor<double>)rc.Outputs[0])[0, 0, 0, 0]));
    }

}
