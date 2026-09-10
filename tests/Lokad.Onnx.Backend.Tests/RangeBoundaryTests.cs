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
}
