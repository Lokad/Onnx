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
}
