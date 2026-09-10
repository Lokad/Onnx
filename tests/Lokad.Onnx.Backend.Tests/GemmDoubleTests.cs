namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins the double Gemm path with ORT 1.29-probed values: unlike double
/// Conv (NOT_IMPLEMENTED in ORT CPU), double Gemm has a real differential
/// reference, and every GemmDouble branch (plain, transA, transB with
/// alpha/beta/vector bias, scalar bias, matrix bias) previously had zero
/// coverage. All values are exact small integers, so assertions are exact.
/// </summary>
public class GemmDoubleTests
{
    static DenseTensor<double> D(double[,] v) => DenseTensor<double>.OfValues(v);

    [Fact]
    public void Basic_MatchesOrt()
    {
        // ORT 1.29: [[19, 22], [43, 50]].
        var r = CPUExecutionProvider.Gemm(D(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }), D(new double[,] { { 5.0, 6.0 }, { 7.0, 8.0 } }), null, 1f, 1f, null, 0, 0);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<double>)r.Outputs[0];
        Assert.Equal(new int[] { 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new double[] { 19.0, 22.0, 43.0, 50.0 }, y.ToArray());
    }

    [Fact]
    public void TransA_MatchesOrt()
    {
        // ORT 1.29: [[58, 64], [139, 154]].
        var r = CPUExecutionProvider.Gemm(D(new double[,] { { 1.0, 4.0 }, { 2.0, 5.0 }, { 3.0, 6.0 } }), D(new double[,] { { 7.0, 8.0 }, { 9.0, 10.0 }, { 11.0, 12.0 } }), null, 1f, 1f, null, 1, 0);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new double[] { 58.0, 64.0, 139.0, 154.0 }, ((Tensor<double>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void TransB_AlphaBetaVectorC_MatchesOrt()
    {
        // ORT 1.29: 2 * (A @ B^T) + 0.5 * [100, 200] = [[150, 236], [294, 434]].
        var r = CPUExecutionProvider.Gemm(D(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } }), D(new double[,] { { 7.0, 8.0, 9.0 }, { 10.0, 11.0, 12.0 } }), DenseTensor<double>.OfValues(new double[] { 100.0, 200.0 }), 2f, 0.5f, null, 0, 1);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new double[] { 150.0, 236.0, 294.0, 434.0 }, ((Tensor<double>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void ScalarC_MatchesOrt()
    {
        // ORT 1.29: (A @ B) + 2 * 10 = [[39, 42], [63, 70]].
        var c = new DenseTensor<double>(new double[] { 10.0 }, Array.Empty<int>());
        var r = CPUExecutionProvider.Gemm(D(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }), D(new double[,] { { 5.0, 6.0 }, { 7.0, 8.0 } }), c, 1f, 2f, null, 0, 0);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new double[] { 39.0, 42.0, 63.0, 70.0 }, ((Tensor<double>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void MatrixC_MatchesOrt()
    {
        // ORT 1.29: (A @ B) + [[1, 2], [3, 4]] = [[20, 24], [46, 54]].
        var r = CPUExecutionProvider.Gemm(D(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }), D(new double[,] { { 5.0, 6.0 }, { 7.0, 8.0 } }), D(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }), 1f, 1f, null, 0, 0);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new double[] { 20.0, 24.0, 46.0, 54.0 }, ((Tensor<double>)r.Outputs[0]).ToArray());
    }
}
