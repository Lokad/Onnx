using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins elementwise empty broadcast under the relaxed BroadcastTo guard:
/// dense [2] with empty [0,2] flows to empty [0,2] per NumPy/ONNX rules on
/// both sides (ORT 1.29 probe shapes), in both operand orders.
/// </summary>
public class BroadcastEmptyTests
{
    [Fact]
    public void AddDenseWithEmpty_ReturnsEmpty()
    {
        // ORT 1.29: shape (0, 2), no elements.
        var result = CPU.Add(
            DenseTensor<float>.OfValues(new float[] { 1f, 2f }),
            DenseTensor<float>.OfShape(0, 2), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<float>)result.Outputs![0];
        Assert.Equal(new int[] { 0, 2 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
    }

    [Fact]
    public void AddEmptyWithDense_ReturnsEmpty()
    {
        // ORT 1.29: shape (0, 2), no elements.
        var result = CPU.Add(
            DenseTensor<float>.OfShape(0, 2),
            DenseTensor<float>.OfValues(new float[] { 1f, 2f }), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<float>)result.Outputs![0];
        Assert.Equal(new int[] { 0, 2 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
    }

    [Fact]
    public void MulDenseWithEmpty_ReturnsEmpty()
    {
        // ORT 1.29: shape (0, 2), no elements.
        var result = CPU.Mul(
            DenseTensor<float>.OfValues(new float[] { 1f, 2f }),
            DenseTensor<float>.OfShape(0, 2), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<float>)result.Outputs![0];
        Assert.Equal(new int[] { 0, 2 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
    }
}