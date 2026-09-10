using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// First Where coverage anywhere: basic selection, multidirectional broadcast
/// with a higher-rank condition, scalar condition/branches, and clean rejection
/// of incompatible shapes, all against ORT 1.29 probe values.
/// </summary>
public class WhereBoundaryTests
{
    [Fact]
    public void DoubleBasic_MatchesOrt()
    {
        // ORT 1.29 double: [1.5, 20.5].
        var result = CPU.Where(
            DenseTensor<bool>.OfValues(new bool[] { true, false }),
            DenseTensor<double>.OfValues(new double[] { 1.5, 2.5 }),
            DenseTensor<double>.OfValues(new double[] { 10.5, 20.5 }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new double[] { 1.5, 20.5 }, ((Tensor<double>)result.Outputs![0]).ToArray());
    }

    [Fact]
    public void BasicSelect_MatchesOrt()
    {
        // ORT 1.29: [1, 20].
        var result = CPU.Where(
            DenseTensor<bool>.OfValues(new bool[] { true, false }),
            DenseTensor<float>.OfValues(new float[] { 1f, 2f }),
            DenseTensor<float>.OfValues(new float[] { 10f, 20f }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new float[] { 1f, 20f }, ((Tensor<float>)result.Outputs![0]).ToArray());
    }

    [Fact]
    public void HigherRankCondition_BroadcastsMultidirectionally()
    {
        // ORT 1.29: [[1, 20], [10, 2]]. Previously threw: the two-way x/y
        // broadcast blinded the kernel to a condition outranking both.
        var c = DenseTensor<bool>.OfShape(2, 2);
        new bool[] { true, false, false, true }.CopyTo(c.Buffer.Span);
        var result = CPU.Where(c,
            DenseTensor<float>.OfValues(new float[] { 1f, 2f }),
            DenseTensor<float>.OfValues(new float[] { 10f, 20f }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var output = (Tensor<float>)result.Outputs![0];
        Assert.Equal(new int[] { 2, 2 }, output.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 20f, 10f, 2f }, output.ToArray());
    }

    [Fact]
    public void BoolSelection_MatchesHandComputed()
    {
        // Deliberate ORT-superset: ORT 1.29 has no bool-selection Where
        // kernel (NOT_IMPLEMENTED at 9 and 16); selection itself is exact.
        var c = DenseTensor<bool>.OfValues(new bool[] { true, false, true });
        var x = DenseTensor<bool>.OfValues(new bool[] { true, true, false });
        var y = DenseTensor<bool>.OfValues(new bool[] { false, false, true });
        var result = CPU.Where(c, x, y, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new bool[] { true, false, false }, ((Tensor<bool>)result.Outputs![0]).ToArray());
    }

    [Fact]
    public void ScalarCondition_Broadcasts()
    {
        var c = DenseTensor<bool>.OfShape();
        c.SetValue(0, true);
        var result = CPU.Where(c,
            DenseTensor<float>.OfValues(new float[] { 1f, 2f }),
            DenseTensor<float>.OfValues(new float[] { 10f, 20f }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new float[] { 1f, 2f }, ((Tensor<float>)result.Outputs![0]).ToArray());
    }

    [Fact]
    public void AllScalar_SelectsBranch()
    {
        // ORT 1.29: scalar cond/branches yield the selected scalar.
        foreach (var (cond, expected) in new[] { (true, 5f), (false, 7f) })
        {
            var c = DenseTensor<bool>.OfShape();
            c.SetValue(0, cond);
            var x = DenseTensor<float>.OfShape();
            x.SetValue(0, 5f);
            var y = DenseTensor<float>.OfShape();
            y.SetValue(0, 7f);
            var result = CPU.Where(c, x, y, null);
            Assert.Equal(OpStatus.Success, result.Status);
            Assert.Equal(new float[] { expected }, ((Tensor<float>)result.Outputs![0]).ToArray());
        }
    }

    [Fact]
    public void ScalarBranches_Broadcast()
    {
        var x = DenseTensor<float>.OfShape();
        x.SetValue(0, 5f);
        var y = DenseTensor<float>.OfShape();
        y.SetValue(0, 7f);
        var result = CPU.Where(
            DenseTensor<bool>.OfValues(new bool[] { true, false }),
            x, y, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new float[] { 5f, 7f }, ((Tensor<float>)result.Outputs![0]).ToArray());
    }

    [Fact]
    public void IncompatibleCondition_ThrowsDescriptively()
    {
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Where(
            DenseTensor<bool>.OfValues(new bool[] { true, false, true }),
            DenseTensor<float>.OfValues(new float[] { 1f, 2f }),
            DenseTensor<float>.OfValues(new float[] { 10f, 20f })));
    }

    [Fact]
    public void Unsigned_SelectsVerbatim()
    {
        // No ORT reference exists (uint32 Where is NOT_IMPLEMENTED in ORT
        // CPU); selection copies inputs verbatim, so hand-exact values need
        // no reference. uint32/uint64 are schema-valid Where types.
        var cond = DenseTensor<bool>.OfValues(new bool[] { true, false, true });
        var x32 = DenseTensor<uint>.OfValues(new uint[] { 1u, 2u, 4294967295u });
        var y32 = DenseTensor<uint>.OfValues(new uint[] { 10u, 20u, 30u });
        var r32 = CPU.Where(cond, x32, y32, null);
        Assert.Equal(OpStatus.Success, r32.Status);
        Assert.Equal(new uint[] { 1u, 20u, 4294967295u }, ((Tensor<uint>)r32.Outputs[0]).ToArray());
        var x64 = DenseTensor<ulong>.OfValues(new ulong[] { 1ul, 18446744073709551615ul });
        var y64 = DenseTensor<ulong>.OfValues(new ulong[] { 10ul, 20ul });
        var r64 = CPU.Where(DenseTensor<bool>.OfValues(new bool[] { false, true }), x64, y64, null);
        Assert.Equal(OpStatus.Success, r64.Status);
        Assert.Equal(new ulong[] { 10ul, 18446744073709551615ul }, ((Tensor<ulong>)r64.Outputs[0]).ToArray());
    }

    [Fact]
    public void MismatchedInputs_RejectedCleanly()
    {
        // ORT refuses both at load; the provider fails descriptively instead.
        var cond = DenseTensor<bool>.OfValues(new bool[] { true });
        var xf = DenseTensor<float>.OfValues(new float[] { 1f });
        var xi = DenseTensor<int>.OfValues(new int[] { 1 });
        Assert.Equal(OpStatus.Failure, CPU.Where(cond, xf, xi, null).Status);
        var badCond = DenseTensor<int>.OfValues(new int[] { 1 });
        Assert.Equal(OpStatus.Failure, CPU.Where(badCond, xf, xf, null).Status);
    }

    [Fact]
    public void EmptyInputs_YieldEmpty()
    {
        // ORT 1.29: all-empty inputs select to an empty output, not a failure.
        var cond = DenseTensor<bool>.OfShape(0);
        var xf = DenseTensor<float>.OfShape(0);
        var yf = DenseTensor<float>.OfShape(0);
        var r = CPU.Where(cond, xf, yf, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var z = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new int[] { 0 }, z.Dimensions.ToArray());
        Assert.Empty(z.ToArray());
    }

    [Fact]
    public void ExceptionalPayload_MatchesOrt()
    {
        // ORT 1.29: selection is pure - a NaN or infinity in the unselected
        // lane never leaks, and a selected exceptional value propagates
        // verbatim, including under a broadcasting condition.
        static float[] RunWhere(bool[] c, float[] x, float[] y)
        {
            var r = CPU.Where(DenseTensor<bool>.OfValues(c), DenseTensor<float>.OfValues(x), DenseTensor<float>.OfValues(y), null);
            Assert.Equal(OpStatus.Success, r.Status);
            return ((Tensor<float>)r.Outputs![0]).ToArray();
        }
        var y = RunWhere(new bool[] { true, false }, new float[] { float.NaN, 1f }, new float[] { 2f, float.NaN });
        Assert.True(float.IsNaN(y[0]));
        Assert.True(float.IsNaN(y[1]));
        y = RunWhere(new bool[] { true, false }, new float[] { float.PositiveInfinity, 1f }, new float[] { 2f, 3f });
        Assert.Equal(float.PositiveInfinity, y[0]);
        Assert.Equal(3f, y[1]);
        y = RunWhere(new bool[] { true, false }, new float[] { 1f, 2f }, new float[] { float.PositiveInfinity, float.NaN });
        Assert.Equal(1f, y[0]);
        Assert.True(float.IsNaN(y[1]));
        y = RunWhere(new bool[] { false, true }, new float[] { 1f, 2f }, new float[] { float.NegativeInfinity, float.NegativeInfinity });
        Assert.Equal(float.NegativeInfinity, y[0]);
        Assert.Equal(2f, y[1]);
        var rb = CPU.Where(
            DenseTensor<bool>.OfValues(new bool[2, 1] { { true }, { false } }),
            DenseTensor<float>.OfValues(new float[2, 2] { { 1f, 2f }, { 3f, 4f } }),
            DenseTensor<float>.OfValues(new float[2, 2] { { float.NaN, float.NaN }, { float.NaN, float.NaN } }), null);
        Assert.Equal(OpStatus.Success, rb.Status);
        var yb = ((Tensor<float>)rb.Outputs![0]).ToArray();
        Assert.Equal(1f, yb[0]);
        Assert.Equal(2f, yb[1]);
        Assert.True(float.IsNaN(yb[2]));
        Assert.True(float.IsNaN(yb[3]));
    }

    [Fact]
    public void ExceptionalPayloadDouble_MatchesOrt()
    {
        // ORT 1.29 double: same pure-selection table as the float guard.
        static double[] RunWhereDouble(bool[] c, double[] x, double[] y)
        {
            var r = CPU.Where(DenseTensor<bool>.OfValues(c), DenseTensor<double>.OfValues(x), DenseTensor<double>.OfValues(y), null);
            Assert.Equal(OpStatus.Success, r.Status);
            return ((Tensor<double>)r.Outputs![0]).ToArray();
        }
        var y = RunWhereDouble(new bool[] { true, false }, new double[] { double.NaN, 1.0 }, new double[] { 2.0, double.NaN });
        Assert.True(double.IsNaN(y[0]));
        Assert.True(double.IsNaN(y[1]));
        y = RunWhereDouble(new bool[] { true, false }, new double[] { double.PositiveInfinity, 1.0 }, new double[] { 2.0, 3.0 });
        Assert.Equal(double.PositiveInfinity, y[0]);
        Assert.Equal(3.0, y[1]);
        y = RunWhereDouble(new bool[] { true, false }, new double[] { 1.0, 2.0 }, new double[] { double.PositiveInfinity, double.NaN });
        Assert.Equal(1.0, y[0]);
        Assert.True(double.IsNaN(y[1]));
        y = RunWhereDouble(new bool[] { false, true }, new double[] { 1.0, 2.0 }, new double[] { double.NegativeInfinity, double.NegativeInfinity });
        Assert.Equal(double.NegativeInfinity, y[0]);
        Assert.Equal(2.0, y[1]);
        var rb = CPU.Where(
            DenseTensor<bool>.OfValues(new bool[2, 1] { { true }, { false } }),
            DenseTensor<double>.OfValues(new double[2, 2] { { 1.0, 2.0 }, { 3.0, 4.0 } }),
            DenseTensor<double>.OfValues(new double[2, 2] { { double.NaN, double.NaN }, { double.NaN, double.NaN } }), null);
        Assert.Equal(OpStatus.Success, rb.Status);
        var yb = ((Tensor<double>)rb.Outputs![0]).ToArray();
        Assert.Equal(1.0, yb[0]);
        Assert.Equal(2.0, yb[1]);
        Assert.True(double.IsNaN(yb[2]));
        Assert.True(double.IsNaN(yb[3]));
    }


    [Fact]
    public void UInt8Select_MatchesOrt()
    {
        // ORT 1.29 runs uint8 Where at opsets 13 and 14 (probed [1,20,3],
        // no version gate); int8/int16/uint16 stay NOT_IMPLEMENTED there,
        // so they keep failing descriptively here.
        var result = CPU.Where(
            DenseTensor<bool>.OfValues(new bool[] { true, false, true }),
            DenseTensor<byte>.OfValues(new byte[] { 1, 2, 3 }),
            DenseTensor<byte>.OfValues(new byte[] { 10, 20, 30 }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new byte[] { 1, 20, 3 }, ((Tensor<byte>)result.Outputs![0]).ToArray());
        var c = DenseTensor<bool>.OfValues(new bool[] { true });
        Assert.Equal(OpStatus.Failure, CPU.Where(c,
            DenseTensor<sbyte>.OfValues(new sbyte[] { 1 }),
            DenseTensor<sbyte>.OfValues(new sbyte[] { 2 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Where(c,
            DenseTensor<short>.OfValues(new short[] { 1 }),
            DenseTensor<short>.OfValues(new short[] { 2 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Where(c,
            DenseTensor<ushort>.OfValues(new ushort[] { 1 }),
            DenseTensor<ushort>.OfValues(new ushort[] { 2 }), null).Status);
    }
}
