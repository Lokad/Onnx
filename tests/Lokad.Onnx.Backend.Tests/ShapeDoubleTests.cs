using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins double basics for the shape ops that route doubles through typed
/// kernels with zero prior double coverage (Gather, Slice, Transpose,
/// Concat, Tile), against ORT 1.29 probe values. All values are exact
/// small integers, so assertions are exact.
/// </summary>
public class ShapeDoubleTests
{
    static DenseTensor<double> D(double[,] v) => DenseTensor<double>.OfValues(v);

    [Fact]
    public void GatherDouble_MatchesOrt()
    {
        // ORT 1.29: indices [2, 0] on axis 1 -> [[3, 1], [6, 4]].
        var r = CPU.Gather(D(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } }), DenseTensor<int>.OfValues(new int[] { 2, 0 }), 1, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<double>)r.Outputs[0];
        Assert.Equal(new int[] { 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new double[] { 3.0, 1.0, 6.0, 4.0 }, y.ToArray());
    }

    [Fact]
    public void SliceDouble_MatchesOrt()
    {
        // ORT 1.29: rows [0, 2), cols [1, 3) -> [[2, 3], [5, 6]].
        var r = CPU.Slice(D(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } }), DenseTensor<long>.OfValues(new long[] { 0L, 1L }), DenseTensor<long>.OfValues(new long[] { 2L, 3L }), null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new double[] { 2.0, 3.0, 5.0, 6.0 }, ((Tensor<double>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void TransposeDouble_MatchesOrt()
    {
        // ORT 1.29: perm [1, 0] -> [[1, 4], [2, 5], [3, 6]].
        var r = CPU.Transpose(D(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } }), new int[] { 1, 0 }, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<double>)r.Outputs[0];
        Assert.Equal(new int[] { 3, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new double[] { 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 }, y.ToArray());
    }

    [Fact]
    public void ConcatDouble_MatchesOrt()
    {
        // ORT 1.29: axis 0 -> [[1, 2], [3, 4]].
        var r = CPU.Concat(new ITensor[] { D(new double[,] { { 1.0, 2.0 } }), D(new double[,] { { 3.0, 4.0 } }) }, 0, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new double[] { 1.0, 2.0, 3.0, 4.0 }, ((Tensor<double>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void TileDouble_MatchesOrt()
    {
        // ORT 1.29: repeats [2, 1] -> [[1, 2], [3, 4], [1, 2], [3, 4]].
        var r = CPU.Tile(D(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }), DenseTensor<long>.OfValues(new long[] { 2L, 1L }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<double>)r.Outputs[0];
        Assert.Equal(new int[] { 4, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new double[] { 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0 }, y.ToArray());
    }
}
