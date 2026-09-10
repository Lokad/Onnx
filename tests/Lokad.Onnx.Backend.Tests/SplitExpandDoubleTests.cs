using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins double basics for Split and Expand, whose typed kernels had zero
/// double coverage (neighboring Split/Expand pins are float-only), against
/// ORT 1.29 probe values. All values are exact small integers.
/// </summary>
public class SplitExpandDoubleTests
{
    [Fact]
    public void SplitDouble_MatchesOrt()
    {
        // ORT 1.29: axis 1, sizes [2, 2] -> [1, 2] + [3, 4].
        var x = DenseTensor<double>.OfValues(new double[,] { { 1.0, 2.0, 3.0, 4.0 } });
        var r = CPU.Split(x, DenseTensor<long>.OfValues(new long[] { 2L, 2L }), 1, null, null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new double[] { 1.0, 2.0 }, ((Tensor<double>)r.Outputs[0]).ToArray());
        Assert.Equal(new double[] { 3.0, 4.0 }, ((Tensor<double>)r.Outputs[1]).ToArray());
    }

    [Fact]
    public void ExpandDouble_MatchesOrt()
    {
        // ORT 1.29: [[1, 2]] to [2, 2] -> [[1, 2], [1, 2]].
        var x = DenseTensor<double>.OfValues(new double[,] { { 1.0, 2.0 } });
        var r = CPU.Expand(x, DenseTensor<long>.OfValues(new long[] { 2L, 2L }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<double>)r.Outputs[0];
        Assert.Equal(new int[] { 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new double[] { 1.0, 2.0, 1.0, 2.0 }, y.ToArray());
    }
}
