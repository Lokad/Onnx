using System.Numerics;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Complex128 data-movement arms (Reshape, Transpose, Concat, Gather, Slice)
/// succeed through the public API even though import refuses complex tensors,
/// so their values are pinned here hand-computed; arithmetic and Cast refuse
/// cleanly. The differential corpus has no complex dtype support.
/// </summary>
public class ComplexDataMovementTests
{
    static DenseTensor<Complex> Base() =>
        DenseTensor<Complex>.OfValues(new Complex[,] { { new Complex(1, 1), new Complex(2, 2) }, { new Complex(3, 3), new Complex(4, 4) } });

    [Fact]
    public void ComplexConcat_MatchesHandValues()
    {
        var a = DenseTensor<Complex>.OfValues(new Complex[] { new Complex(1, 1), new Complex(2, 2) });
        var b = DenseTensor<Complex>.OfValues(new Complex[] { new Complex(3, 3) });
        var got = Tensor<Complex>.Concat(a, b, 0);
        Assert.Equal(new int[] { 3 }, got.Dimensions.ToArray());
        Assert.Equal(new Complex[] { new Complex(1, 1), new Complex(2, 2), new Complex(3, 3) }, got.ToArray());
    }

    [Fact]
    public void ComplexSlice_MatchesHandValues()
    {
        var got = Base().Slice(new SliceIndex(0, 2), new SliceIndex(1, 3));
        Assert.Equal(new int[] { 2, 1 }, got.Dimensions.ToArray());
        Assert.Equal(new Complex[] { new Complex(2, 2), new Complex(4, 4) }, got.ToArray());
    }

    [Fact]
    public void ComplexTranspose_MatchesHandValues()
    {
        var got = Tensor<Complex>.Transpose(Base(), new int[] { 1, 0 });
        Assert.Equal(new int[] { 2, 2 }, got.Dimensions.ToArray());
        Assert.Equal(new Complex[] { new Complex(1, 1), new Complex(3, 3), new Complex(2, 2), new Complex(4, 4) }, got.ToArray());
    }

    [Fact]
    public void ComplexReshape_MatchesHandValues()
    {
        var got = Tensor<Complex>.Reshape(Base(), new long[] { 4 }.ToTensor<long>(), false);
        Assert.Equal(new int[] { 4 }, got.Dimensions.ToArray());
        Assert.Equal(new Complex[] { new Complex(1, 1), new Complex(2, 2), new Complex(3, 3), new Complex(4, 4) }, got.ToArray());
    }

    [Fact]
    public void ComplexGather_MatchesHandValues()
    {
        var idx = new int[] { 1, 0 }.ToTensor<int>();
        var got = Tensor<Complex>.Gather(Base(), idx, 0);
        Assert.Equal(new int[] { 2, 2 }, got.Dimensions.ToArray());
        Assert.Equal(new Complex[] { new Complex(3, 3), new Complex(4, 4), new Complex(1, 1), new Complex(2, 2) }, got.ToArray());
    }

    [Fact]
    public void ComplexCast_RefusedBothDirections()
    {
        // Complex casts remain unsupported in either direction.
        var c = DenseTensor<Complex>.OfValues(new Complex[] { new Complex(1, 1) });
        var f = DenseTensor<float>.OfValues(new float[] { 1f });
        Assert.Equal(OpStatus.Failure, CPU.Cast(c, TensorElementType.Float, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Cast(f, TensorElementType.Complex64, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Cast(f, TensorElementType.Complex128, null).Status);
    }

    [Fact]
    public void ComplexArithmetic_RefusedCleanly()
    {
        // No arithmetic arm exists for Complex128 (movement ops only), so every
        // form fails descriptively like the half-width boundary.
        var a = DenseTensor<Complex>.OfValues(new Complex[] { new Complex(1, 1), new Complex(2, 2) });
        var b = DenseTensor<Complex>.OfValues(new Complex[] { new Complex(3, 3), new Complex(4, 4) });
        Assert.Equal(OpStatus.Failure, CPU.Add(a, b, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Sub(a, b, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Mul(a, b, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Div(a, b, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Neg(a, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Abs(a, null).Status);
    }

    [Fact]
    public void ComplexSplit_RefusedCleanly()
    {
        // Split remains unsupported for complex inputs.
        var x = DenseTensor<System.Numerics.Complex>.OfValues(new System.Numerics.Complex[] { new System.Numerics.Complex(1, 0), new System.Numerics.Complex(2, 0), new System.Numerics.Complex(3, 0), new System.Numerics.Complex(4, 0) }, new int[] { 1, 4 });
        var r = CPU.Split(x, DenseTensor<long>.OfValues(new long[] { 2L, 2L }), 1, null, null, null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }
}
