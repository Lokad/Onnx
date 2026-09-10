using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

public class CastNumericTests
{
    [Fact]
    public void FloatFractions_TruncateTowardZero()
    {
        var x = DenseTensor<float>.OfValues(new float[] { 1.9f, -1.9f, 2.5f, -2.5f, 0.5f });
        var r = CPUExecutionProvider.Cast(x, TensorElementType.Int32, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new int[] { 1, -1, 2, -2, 0 }, ((Tensor<int>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void FloatOverflowAndNan_SaturateToMin()
    {
        var x = DenseTensor<float>.OfValues(new float[] { 1e20f, -1e20f, float.NaN, float.PositiveInfinity });
        var r = CPUExecutionProvider.Cast(x, TensorElementType.Int32, null);
        Assert.Equal(OpStatus.Success, r.Status);
        foreach (var v in ((Tensor<int>)r.Outputs[0]).ToArray()) Assert.Equal(int.MinValue, v);
        var d = DenseTensor<double>.OfValues(new double[] { 1e300, double.NaN });
        var rd = CPUExecutionProvider.Cast(d, TensorElementType.Int64, null);
        Assert.Equal(OpStatus.Success, rd.Status);
        foreach (var v in ((Tensor<long>)rd.Outputs[0]).ToArray()) Assert.Equal(long.MinValue, v);
    }

    [Fact]
    public void IntegerNarrowing_Wraps()
    {
        var x = DenseTensor<int>.OfValues(new int[] { 300, -1, 128, 257 });
        var r = CPUExecutionProvider.Cast(x, TensorElementType.Int8, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new sbyte[] { 44, -1, -128, 1 }, ((Tensor<sbyte>)r.Outputs[0]).ToArray());
        var u = DenseTensor<int>.OfValues(new int[] { 300, -1 });
        var ru = CPUExecutionProvider.Cast(u, TensorElementType.UInt8, null);
        Assert.Equal(OpStatus.Success, ru.Status);
        Assert.Equal(new byte[] { 44, 255 }, ((Tensor<byte>)ru.Outputs[0]).ToArray());
        var l = DenseTensor<long>.OfValues(new long[] { 4294967297L, -1L });
        var rl = CPUExecutionProvider.Cast(l, TensorElementType.Int32, null);
        Assert.Equal(OpStatus.Success, rl.Status);
        Assert.Equal(new int[] { 1, -1 }, ((Tensor<int>)rl.Outputs[0]).ToArray());
    }

    [Fact]
    public void NegativeIntToUint_Wraps()
    {
        // ORT 1.29: int-to-uint casts reinterpret bits ([-1, min] ->
        // [max, 2^31]); narrowing wrap is pinned above, this is the
        // same-width signed-to-unsigned twin.
        var r = CPUExecutionProvider.Cast(DenseTensor<int>.OfValues(new int[] { -1, -2147483648 }), TensorElementType.UInt32, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new uint[] { 4294967295u, 2147483648u }, ((Tensor<uint>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void LongToFloat_RoundsAtPrecisionBoundary()
    {
        // ORT 1.29: longs beyond 2^24 round like ints ([2^24+1,
        // -(2^24+1), 100] -> [2^24, -2^24, 100]).
        var r = CPUExecutionProvider.Cast(DenseTensor<long>.OfValues(new long[] { 16777217L, -16777217L, 100L }), TensorElementType.Float, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 16777216f, -16777216f, 100f }, ((Tensor<float>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void NegativeIntToUlong_Wraps()
    {
        // ORT 1.29: [-1, min] int32 casts to [2^64-1, 2^64-2^31],
        // exercising the explicit 64-bit truncate-and-wrap helper.
        var r = CPUExecutionProvider.Cast(DenseTensor<int>.OfValues(new int[] { -1, -2147483648 }), TensorElementType.UInt64, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new ulong[] { 18446744073709551615ul, 18446744071562067968ul }, ((Tensor<ulong>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void IntToFloat_RoundsAtPrecisionBoundary()
    {
        // ORT 1.29: ints beyond 2^24 round to nearest float ([2^24,
        // 2^24+1, -(2^24+1), 100] -> [2^24, 2^24, -2^24, 100]).
        var r = CPUExecutionProvider.Cast(DenseTensor<int>.OfValues(new int[] { 16777216, 16777217, -16777217, 100 }), TensorElementType.Float, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 16777216f, 16777216f, -16777216f, 100f }, ((Tensor<float>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void DoubleToFloat_OverflowsToInfinity()
    {
        // ORT 1.29: out-of-float-range doubles become signed infinities
        // ([1e300, -1e300, 1.5] -> [inf, -inf, 1.5]).
        var r = CPUExecutionProvider.Cast(DenseTensor<double>.OfValues(new double[] { 1e300, -1e300, 1.5 }), TensorElementType.Float, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = ((Tensor<float>)r.Outputs[0]).ToArray();
        Assert.Equal(float.PositiveInfinity, y[0]);
        Assert.Equal(float.NegativeInfinity, y[1]);
        Assert.Equal(1.5f, y[2]);
    }

    [Fact]
    public void DoubleOverflowAndNaN_SaturateSignedMin32()
    {
        // ORT 1.29: like the int64 twin, out-of-range doubles and NaN
        // saturate int32 to the minimum ([1e300, -1e300, nan, 1.9] ->
        // [min, min, min, 1]).
        var r = CPUExecutionProvider.Cast(DenseTensor<double>.OfValues(new double[] { 1e300, -1e300, double.NaN, 1.9 }), TensorElementType.Int32, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new int[] { -2147483648, -2147483648, -2147483648, 1 }, ((Tensor<int>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void FloatOverflowAndNaN_SaturateSignedMin64()
    {
        // ORT 1.29: like the double twin, out-of-range floats and NaN
        // saturate int64 to the minimum ([1e30, -1e30, nan, 1.9] ->
        // [min, min, min, 1]).
        var r = CPUExecutionProvider.Cast(DenseTensor<float>.OfValues(new float[] { 1e30f, -1e30f, float.NaN, 1.9f }), TensorElementType.Int64, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new long[] { -9223372036854775808L, -9223372036854775808L, -9223372036854775808L, 1L }, ((Tensor<long>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void DoubleOverflowAndNaN_SaturateSignedMin()
    {
        // ORT 1.29: out-of-range doubles and NaN saturate int64 to the
        // minimum ([1e300, -1e300, nan, 1.9] -> [min, min, min, 1]),
        // mirroring the float saturation pin.
        var r = CPUExecutionProvider.Cast(DenseTensor<double>.OfValues(new double[] { 1e300, -1e300, double.NaN, 1.9 }), TensorElementType.Int64, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new long[] { -9223372036854775808L, -9223372036854775808L, -9223372036854775808L, 1L }, ((Tensor<long>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void UlongToDouble_RoundsAtPrecisionBoundary()
    {
        // ORT 1.29: ulongs beyond 2^53 round ([max, 2^53+1, 100] ->
        // [2^64, 2^53, 100]), the unsigned twin of the long boundary.
        var r = CPUExecutionProvider.Cast(DenseTensor<ulong>.OfValues(new ulong[] { 18446744073709551615ul, 9007199254740993ul, 100ul }), TensorElementType.Double, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new double[] { 18446744073709551616.0, 9007199254740992.0, 100.0 }, ((Tensor<double>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void LongToDouble_RoundsAtPrecisionBoundary()
    {
        // ORT 1.29: longs beyond 2^53 round to nearest double ([max,
        // 2^53+1, 100] -> [2^63, 2^53, 100]), mirroring the int/float
        // boundary pinned above.
        var r = CPUExecutionProvider.Cast(DenseTensor<long>.OfValues(new long[] { 9223372036854775807L, 9007199254740993L, 100L }), TensorElementType.Double, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new double[] { 9223372036854775808.0, 9007199254740992.0, 100.0 }, ((Tensor<double>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void UnsignedExtremes_MatchOracle()
    {
        var x = DenseTensor<float>.OfValues(new float[] { 5e9f, 3e9f });
        var r = CPUExecutionProvider.Cast(x, TensorElementType.UInt32, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new uint[] { 705032704u, 3000000000u }, ((Tensor<uint>)r.Outputs[0]).ToArray());
        var nan = DenseTensor<float>.OfValues(new float[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity });
        var rn = CPUExecutionProvider.Cast(nan, TensorElementType.UInt32, null);
        Assert.Equal(OpStatus.Success, rn.Status);
        Assert.Equal(new uint[] { 0u, 0u, 0u }, ((Tensor<uint>)rn.Outputs[0]).ToArray());
        var big = DenseTensor<float>.OfValues(new float[] { 1e19f });
        var rb = CPUExecutionProvider.Cast(big, TensorElementType.UInt64, null);
        Assert.Equal(OpStatus.Success, rb.Status);
        Assert.Equal(new ulong[] { 9999999980506447872ul }, ((Tensor<ulong>)rb.Outputs[0]).ToArray());
    }

    [Fact]
    public void Bools_ConvertBothWays()
    {
        var b = DenseTensor<bool>.OfValues(new bool[] { true, false });
        var r = CPUExecutionProvider.Cast(b, TensorElementType.Int32, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new int[] { 1, 0 }, ((Tensor<int>)r.Outputs[0]).ToArray());
        var z = DenseTensor<int>.OfValues(new int[] { 0, 1, -3 });
        var rz = CPUExecutionProvider.Cast(z, TensorElementType.Bool, null);
        Assert.Equal(OpStatus.Success, rz.Status);
        Assert.Equal(new bool[] { false, true, true }, ((Tensor<bool>)rz.Outputs[0]).ToArray());
        var f = DenseTensor<float>.OfValues(new float[] { 0f, 0.5f, float.NaN });
        var rf = CPUExecutionProvider.Cast(f, TensorElementType.Bool, null);
        Assert.Equal(OpStatus.Success, rf.Status);
        Assert.Equal(new bool[] { false, true, true }, ((Tensor<bool>)rf.Outputs[0]).ToArray());
    }

    [Fact]
    public void SameType_ReturnsCopy()
    {
        var x = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        var r = CPUExecutionProvider.Cast(x, TensorElementType.Float, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 1f, 2f }, ((Tensor<float>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void UnsupportedPairs_FailExplicitly()
    {
        var x = DenseTensor<float>.OfValues(new float[] { 1f });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Cast(x, TensorElementType.Float16, null).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Cast(x, TensorElementType.String, null).Status);
    }

    [Fact]
    public void DoubleFractions_TruncateTowardZero()
    {
        var x = DenseTensor<double>.OfValues(new double[] { 300.7, -300.7 });
        var r = CPUExecutionProvider.Cast(x, TensorElementType.UInt16, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new ushort[] { 300, 65236 }, ((Tensor<ushort>)r.Outputs[0]).ToArray());
    }

    static ComputationalGraph SaturateGraph(long? saturate)
    {
        // C08: opset 24 Cast threads saturate; only the default is supported.
        var mp = new OnnxModel { Name = "cast-saturate" };
        mp.Opset[""] = 24;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new int[] { 2 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Int32, Dims = new int[] { 2 } });
        var attrs = new Dictionary<string, object> { { "to", (int)TensorElementType.Int32 } };
        if (saturate.HasValue) attrs["saturate"] = saturate.Value;
        mp.Nodes.Add(new OnnxNode { Name = "c", OpType = "Cast", Inputs = new string[] { "x" }, Outputs = new string[] { "y" }, Attributes = attrs });
        return Model.Load(mp)!;
    }

    static Dictionary<string, ITensor> SaturateFeed()
    {
        return new Dictionary<string, ITensor>
        {
            ["x"] = DenseTensor<float>.OfValues(new float[] { 1.9f, -1.9f }),
        };
    }

    [Fact]
    public void SaturateAbsent_CastsNormally()
    {
        var graph = SaturateGraph(null);
        Assert.True(graph.Execute(SaturateFeed(), true), graph.LastErrorMessage);
        Assert.Equal(new int[] { 1, -1 }, ((Tensor<int>)graph.Outputs["y"]).ToArray());
    }

    [Fact]
    public void SaturateOne_CastsNormally()
    {
        var graph = SaturateGraph(1L);
        Assert.True(graph.Execute(SaturateFeed(), true), graph.LastErrorMessage);
        Assert.Equal(new int[] { 1, -1 }, ((Tensor<int>)graph.Outputs["y"]).ToArray());
    }

    [Fact]
    public void SaturateZero_FailsNamingAttribute()
    {
        // ORT rejects saturate outside float 8 casts at session build.
        var graph = SaturateGraph(0L);
        Assert.False(graph.Execute(SaturateFeed(), true));
        Assert.Contains("saturate", graph.LastErrorMessage ?? "");
    }
}
