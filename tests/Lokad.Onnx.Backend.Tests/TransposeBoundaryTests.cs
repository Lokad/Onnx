namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins Transpose permutation validation against ORT 1.29 probe values:
/// scalars pass through, duplicate and post-normalization-duplicate perms fail.
/// </summary>
public class TransposeBoundaryTests
{
    [Fact]
    public void ScalarNoPerm_PassesThrough()
    {
        // ORT 1.29: scalar transpose is scalar 7.
        var s = DenseTensor<float>.OfShape();
        s.SetValue(0, 7f);
        var y = Tensor<float>.Transpose(s, null);
        Assert.Equal(new int[0], y.Dimensions.ToArray());
        Assert.Equal(new float[] { 7f }, y.ToArray());
    }

    [Fact]
    public void DuplicatePerm_Throws()
    {
        // ORT 1.29 refuses perm=[0,0] at load.
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Transpose(x, new int[] { 0, 0 }));
    }

    [Fact]
    public void NegativeDuplicatePerm_Throws()
    {
        // ORT 1.29 refuses perm=[0,-2] at load (duplicate after normalization).
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Transpose(x, new int[] { 0, -2 }));
    }

    [Fact]
    public void EmptyInput_PermutesShape()
    {
        // ORT 1.29: transposing [0,3] with perm=[1,0] yields [3,0], empty.
        var y = Tensor<float>.Transpose(DenseTensor<float>.OfShape(0, 3), new int[] { 1, 0 });
        Assert.Equal(new int[] { 3, 0 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
    }
}