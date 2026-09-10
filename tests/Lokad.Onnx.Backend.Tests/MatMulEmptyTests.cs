using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Completes the empty-MatMul matrix against ORT 1.29 probe values: zero-K
/// is frozen differentially, zero-M and zero-batch rows flow to empty outputs
/// here, and the zero-K empty reduction sums to zero like the Conv
/// empty-channel precedent.
/// </summary>
public class MatMulEmptyTests
{
    static Tensor<float> Run(float[,] a, float[,] b)
    {
        var result = CPU.MatMul(DenseTensor<float>.OfValues(a), DenseTensor<float>.OfValues(b), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        return (Tensor<float>)result.Outputs![0];
    }

    [Fact]
    public void BoolInputs_RejectedCleanly()
    {
        // ORT 1.29 refuses bool MatMul inputs at load; the provider must
        // fail descriptively (verified differentially via OpDump).
        var a = DenseTensor<bool>.OfValues(new bool[,] { { true, false }, { false, true } });
        var b = DenseTensor<bool>.OfValues(new bool[,] { { true, true }, { false, false } });
        var r = CPU.MatMul(a, b, null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void ZeroM_ReturnsEmpty()
    {
        // ORT 1.29: [0,3] x [3,2] -> [0,2], no elements.
        var y = Run(new float[0, 3], new float[,] { { 1f, 2f }, { 3f, 4f }, { 5f, 6f } });
        Assert.Equal(new int[] { 0, 2 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
    }

    [Fact]
    public void ZeroBatch_ReturnsEmpty()
    {
        // ORT 1.29: [0,2,3] x [3,4] -> [0,2,4], no elements.
        var a = DenseTensor<float>.OfShape(0, 2, 3);
        var b = DenseTensor<float>.OfValues(new float[,] { { 1f, 1f, 1f, 1f }, { 1f, 1f, 1f, 1f }, { 1f, 1f, 1f, 1f } });
        var result = CPU.MatMul(a, b, null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<float>)result.Outputs![0];
        Assert.Equal(new int[] { 0, 2, 4 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
    }

    [Fact]
    public void ZeroK_SumsToZero()
    {
        // ORT 1.29: [2,0] x [0,3] -> [2,3] zeros (also frozen differentially).
        var y = Run(new float[2, 0], new float[0, 3]);
        Assert.Equal(new int[] { 2, 3 }, y.Dimensions.ToArray());
        Assert.Equal(new float[6], y.ToArray());
    }
}