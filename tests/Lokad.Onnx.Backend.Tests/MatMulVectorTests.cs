using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins rank-1 MatMul promotion against ORT 1.29 probe values: vector-vector
/// contracts to a scalar, matrix-vector and vector-matrix squeeze the
/// promoted axis back. No vector coverage existed anywhere.
/// </summary>
public class MatMulVectorTests
{
    [Fact]
    public void VectorDot_ContractsToScalar()
    {
        // ORT 1.29: scalar 32.
        var result = CPU.MatMul(
            DenseTensor<float>.OfValues(new float[] { 1f, 2f, 3f }),
            DenseTensor<float>.OfValues(new float[] { 4f, 5f, 6f }), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<float>)result.Outputs![0];
        Assert.Equal(new int[0], y.Dimensions.ToArray());
        Assert.Equal(new float[] { 32f }, y.ToArray());
    }

    [Fact]
    public void MatrixVector_SqueezesToVector()
    {
        // ORT 1.29: [6, 15].
        var result = CPU.MatMul(
            DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } }),
            DenseTensor<float>.OfValues(new float[] { 1f, 1f, 1f }), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<float>)result.Outputs![0];
        Assert.Equal(new int[] { 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 6f, 15f }, y.ToArray());
    }

    [Fact]
    public void VectorMatrix_SqueezesToVector()
    {
        // ORT 1.29: [13, 16].
        var result = CPU.MatMul(
            DenseTensor<float>.OfValues(new float[] { 1f, 2f }),
            DenseTensor<float>.OfValues(new float[,] { { 3f, 4f }, { 5f, 6f } }), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<float>)result.Outputs![0];
        Assert.Equal(new int[] { 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 13f, 16f }, y.ToArray());
    }
}