using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// First Range coverage anywhere: basic, empty, negative-delta and float
/// ranges against ORT 1.29 probe values, plus the zero-delta clean failure
/// both sides enforce (ORT InvalidArgument, Lokad descriptive Failure).
/// </summary>
public class RangeBoundaryTests
{
    static DenseTensor<long> Scalar64(long value)
    {
        var s = DenseTensor<long>.OfShape();
        s.SetValue(0, value);
        return s;
    }

    static long[] RunLong(long start, long limit, long delta)
    {
        var result = CPU.Range(Scalar64(start), Scalar64(limit), Scalar64(delta), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var output = (Tensor<long>)result.Outputs![0];
        return output.ToArray();
    }

    [Fact]
    public void IntBasic_MatchesOrt()
    {
        // ORT 1.29: [0, 2, 4].
        Assert.Equal(new long[] { 0L, 2L, 4L }, RunLong(0L, 5L, 2L));
    }

    [Fact]
    public void EmptyRanges_ReturnEmpty()
    {
        // ORT 1.29: start == limit and wrong-direction ranges are empty.
        Assert.Empty(RunLong(3L, 3L, 1L));
        Assert.Empty(RunLong(5L, 0L, 1L));
    }

    [Fact]
    public void NegativeDelta_CountsDown()
    {
        // ORT 1.29: [5, 4, 3, 2, 1].
        Assert.Equal(new long[] { 5L, 4L, 3L, 2L, 1L }, RunLong(5L, 0L, -1L));
    }

    [Fact]
    public void FloatRange_MatchesOrt()
    {
        // ORT 1.29: [0.5, 1.5], and wrong-direction float ranges are empty.
        var fwd = CPU.Range(ScalarF(0.5f), ScalarF(2.5f), ScalarF(1f), null);
        Assert.Equal(OpStatus.Success, fwd.Status);
        Assert.Equal(new float[] { 0.5f, 1.5f }, ((Tensor<float>)fwd.Outputs![0]).ToArray());
        var back = CPU.Range(ScalarF(2.5f), ScalarF(0.5f), ScalarF(1f), null);
        Assert.Equal(OpStatus.Success, back.Status);
        Assert.Empty(((Tensor<float>)back.Outputs![0]).ToArray());
    }

    [Fact]
    public void DoubleRange_MatchesOrt()
    {
        // ORT 1.29: [0.5, 1.0, 1.5].
        var r = CPU.Range(ScalarD(0.5), ScalarD(2.0), ScalarD(0.5), null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new double[] { 0.5, 1.0, 1.5 }, ((Tensor<double>)r.Outputs![0]).ToArray());
    }

    static DenseTensor<double> ScalarD(double value)
    {
        var s = DenseTensor<double>.OfShape();
        s.SetValue(0, value);
        return s;
    }

    static DenseTensor<float> ScalarF(float value)
    {
        var s = DenseTensor<float>.OfShape();
        s.SetValue(0, value);
        return s;
    }

    [Fact]
    public void BoolInputs_RejectedCleanly()
    {
        // ORT 1.29 refuses bool Range at load (not in Range types);
        // the provider fails descriptively instead.
        var t = DenseTensor<bool>.OfValues(new bool[] { true });
        Assert.Equal(OpStatus.Failure, CPU.Range(t, t, t, null).Status);
    }

    [Fact]
    public void UintInputs_RejectedCleanly()
    {
        // ORT 1.29 refuses uint Range at load (not in Range types);
        // the provider fails descriptively instead.
        var t = DenseTensor<uint>.OfValues(new uint[] { 1u });
        Assert.Equal(OpStatus.Failure, CPU.Range(t, t, t, null).Status);
    }

    [Fact]
    public void MixedDtypes_RejectedCleanly()
    {
        // ORT 1.29 refuses mixed Range dtypes at load (single type T);
        // the provider must fail descriptively instead of converting
        // (float limits would truncate silently toward int).
        Assert.Equal(OpStatus.Failure, CPU.Range(Scalar64(0L), ScalarF(3f), ScalarF(1f), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Range(ScalarF(0f), Scalar64(3L), Scalar64(1L), null).Status);
    }

    [Fact]
    public void ZeroDelta_FailsCleanly()
    {
        Assert.Throws<System.ArgumentException>(() => Tensor<long>.Range(0L, 5L, 0L));
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 13 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = Scalar64(0L);
        graph.Inputs["y"] = Scalar64(5L);
        graph.Inputs["d"] = Scalar64(0L);
        var node = new Node
        {
            Name = "n", Op = OpType.Range, OpTypeName = OpType.Range.ToString(), Domain = "",
            OpsetVersion = 13, IsFused = false,
            Inputs = new[] { "x", "y", "d" }, Outputs = new[] { "z" },
            Attributes = new Dictionary<string, object>(),
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("delta", r.Message ?? "");
    }

    [Fact]
    public void Int16Range_MatchesOrt()
    {
        // ORT 1.29 int16: (0,5,2) -> [0,2,4]; (5,0,-2) -> [5,3,1].
        var up = CPU.Range(Scalar16(0), Scalar16(5), Scalar16(2), null);
        Assert.Equal(OpStatus.Success, up.Status);
        Assert.Equal(new short[] { 0, 2, 4 }, ((Tensor<short>)up.Outputs[0]).ToArray());
        var down = CPU.Range(Scalar16(5), Scalar16(0), Scalar16(-2), null);
        Assert.Equal(OpStatus.Success, down.Status);
        Assert.Equal(new short[] { 5, 3, 1 }, ((Tensor<short>)down.Outputs[0]).ToArray());
    }

    static DenseTensor<short> Scalar16(short value)
    {
        var s = DenseTensor<short>.OfShape();
        s.SetValue(0, value);
        return s;
    }

    [Fact]
    public void FloatFractionalCountAndValues_MatchOrt()
    {
        // ORT 1.29: the count walks start + i*delta against the limit in
        // double precision, so Range(0, 0.3, 0.1f) keeps its 4th element
        // (a float ceil-quotient miscounts it as 3); emitted values iterate
        // v += delta, which drifts from start + i*delta by ulps on long
        // spans (Range(0, 10, 0.1f) ends at 9.90000152, not the
        // multiplied 9.90000057...).
        var r = CPU.Range(ScalarF(0f), ScalarF(0.3f), ScalarF(0.1f), null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 0f, 0.1f, 0.2f, 0.3f }, ((Tensor<float>)r.Outputs![0]).ToArray());
        var rlong = CPU.Range(ScalarF(0f), ScalarF(10f), ScalarF(0.1f), null);
        Assert.Equal(OpStatus.Success, rlong.Status);
        var ylong = ((Tensor<float>)rlong.Outputs![0]).ToArray();
        Assert.Equal(100, ylong.Length);
        Assert.Equal(9.900001525878906f, ylong[99]);
        var r3 = CPU.Range(ScalarF(0f), ScalarF(1f), ScalarF(1f / 3f), null);
        Assert.Equal(OpStatus.Success, r3.Status);
        Assert.Equal(3, ((Tensor<float>)r3.Outputs![0]).ToArray().Length);
    }

    [Fact]
    public void DoubleFractionalValues_MatchOrt()
    {
        // ORT 1.29 double: the count stays a ceil-quotient, but emitted
        // values iterate v += delta like float (Range(0, 10, 0.1) ends at
        // 9.89999999999998, not the multiplied 9.9000000000000004).
        var r = CPU.Range(ScalarD(0.0), ScalarD(10.0), ScalarD(0.1), null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = ((Tensor<double>)r.Outputs![0]).ToArray();
        Assert.Equal(100, y.Length);
        Assert.Equal(9.89999999999998, y[99]);
        var r3 = CPU.Range(ScalarD(0.0), ScalarD(1.0), ScalarD(1.0 / 3.0), null);
        Assert.Equal(OpStatus.Success, r3.Status);
        Assert.Equal(3, ((Tensor<double>)r3.Outputs![0]).ToArray().Length);
    }

    [Fact]
    public void NonFiniteRange_BehavesLikeOrt()
    {
        // ORT 1.29 fails ranges with no finite count (NaN anywhere, an
        // infinite start/limit); an infinite delta yields an empty range.
        // The provider surfaces the failures as descriptive throws, like
        // zero delta at tensor level and graph Failure through dispatch.
        var empty = CPU.Range(ScalarF(0f), ScalarF(1f), ScalarF(float.PositiveInfinity), null);
        Assert.Equal(OpStatus.Success, empty.Status);
        Assert.Empty(((Tensor<float>)empty.Outputs![0]).ToArray());
        Assert.Throws<System.ArgumentException>(() => CPU.Range(ScalarF(0f), ScalarF(1f), ScalarF(float.NaN), null));
        Assert.Throws<System.ArgumentException>(() => CPU.Range(ScalarF(0f), ScalarF(float.PositiveInfinity), ScalarF(1f), null));
        Assert.Throws<System.ArgumentException>(() => CPU.Range(ScalarD(0.0), ScalarD(1.0), ScalarD(double.NaN), null));
    }



    [Fact]
    public void HugeCount_FailsCleanly()
    {
        // ORT 1.29 fails huge-count ranges at allocation (8TB refused);
        // counts beyond int.MaxValue are unrepresentable in int32 dims, so
        // they must throw instead of collapsing to empty (the float kernel
        // already guards this way).
        Assert.Throws<System.ArgumentException>(() => Tensor<double>.Range(0.0, 1099511627776.0, 1.0));
        Assert.Throws<System.ArgumentException>(() => Tensor<long>.Range(0L, 1099511627776L, 1L));
        Assert.Throws<System.ArgumentException>(() => Tensor<int>.Range(-2147483648, 2147483647, 1));
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 13 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = ScalarD(0.0);
        graph.Inputs["y"] = ScalarD(1099511627776.0);
        graph.Inputs["d"] = ScalarD(1.0);
        var node = new Node
        {
            Name = "n", Op = OpType.Range, OpTypeName = OpType.Range.ToString(), Domain = "",
            OpsetVersion = 13, IsFused = false,
            Inputs = new[] { "x", "y", "d" }, Outputs = new[] { "z" },
            Attributes = new Dictionary<string, object>(),
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }
}
