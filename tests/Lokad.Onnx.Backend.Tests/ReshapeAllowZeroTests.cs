namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins allowzero zero-size inference against ORT 1.29 probe values: literal
/// zeros count toward the volume (empty [2,0] reshapes to [2,0]), and -1
/// beside a zero known dim infers from nonzero extents on both sides for
/// empty inputs ([2,0,3]@[0,-1,0] -> [0,6,0]) while nonempty mismatches fail.
/// </summary>
public class ReshapeAllowZeroTests
{
    [Fact]
    public void EmptyLiteralZero_Succeeds()
    {
        // ORT 1.29: [2,0] allowzero=1 [2,0] -> [2,0]. Previously threw: the
        // literal zero never entered the volume product.
        var y = Tensor<float>.Reshape(DenseTensor<float>.OfShape(2, 0), DenseTensor<long>.OfValues(new long[] { 2L, 0L }), true);
        Assert.Equal(new int[] { 2, 0 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
    }

    [Fact]
    public void ZeroKnownInfersNonzeroExtents()
    {
        // ORT 1.29: [2,0,3]@[0,-1,0] -> [0,6,0]; [0,4]@[0,-1] -> [0,4].
        var a = Tensor<float>.Reshape(DenseTensor<float>.OfShape(2, 0, 3), DenseTensor<long>.OfValues(new long[] { 0L, -1L, 0L }), true);
        Assert.Equal(new int[] { 0, 6, 0 }, a.Dimensions.ToArray());
        var b = Tensor<float>.Reshape(DenseTensor<float>.OfShape(0, 4), DenseTensor<long>.OfValues(new long[] { 0L, -1L }), true);
        Assert.Equal(new int[] { 0, 4 }, b.Dimensions.ToArray());
    }

    [Fact]
    public void AllZeroInfersOne()
    {
        // ORT 1.29: [0,0]@[0,-1] -> [0,1] (empty nonzero products are 1).
        var y = Tensor<float>.Reshape(DenseTensor<float>.OfShape(0, 0), DenseTensor<long>.OfValues(new long[] { 0L, -1L }), true);
        Assert.Equal(new int[] { 0, 1 }, y.Dimensions.ToArray());
    }

    [Fact]
    public void NonemptyMismatch_Fails()
    {
        // ORT 1.29 fails the run; Lokad throws descriptively (never a
        // DivideByZero or a silent [0,6] accept).
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Reshape(
            DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } }),
            DenseTensor<long>.OfValues(new long[] { 0L, -1L }), true));
    }

    [Fact]
    public void CopyZeroNonempty_InfersLikeOrt()
    {
        // ORT 1.29 with allowzero=0: a copied zero on a nonempty input
        // behaves like its extent, so -1 infers against it ([2,6]@[-1,0],
        // [0,-1] and [2,0] all yield [2,6]; verified differentially).
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f, 5f, 6f }, { 7f, 8f, 9f, 10f, 11f, 12f } });
        foreach (var shape in new long[][] { new long[] { -1L, 0L }, new long[] { 0L, -1L }, new long[] { 2L, 0L } })
        {
            var y = Tensor<float>.Reshape(x, DenseTensor<long>.OfValues(shape), false);
            Assert.Equal(new int[] { 2, 6 }, y.Dimensions.ToArray());
            Assert.Equal(Enumerable.Range(1, 12).Select(v => (float)v).ToArray(), y.ToArray());
        }
    }

    [Fact]
    public void CopyZeroInfersSameRule()
    {
        // allowzero=0 copies the zero, then the same nonzero inference applies.
        var y = Tensor<float>.Reshape(DenseTensor<float>.OfShape(0, 4), DenseTensor<long>.OfValues(new long[] { 0L, -1L }), false);
        Assert.Equal(new int[] { 0, 4 }, y.Dimensions.ToArray());
    }
}