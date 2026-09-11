using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Int64 ReduceMax values with ORT 1.29 semantics: plain maximum, empty
/// extents yield the dtype minimum. Expectations are ORT-probed.
/// </summary>
public class ReduceMaxInt64Tests
{
    static long[] Run(long[] values, int[] dims, int[] axes, int keepDims)
    {
        var x = DenseTensor<long>.OfValues(values, dims);
        var r = CPU.ReduceMax(x, DenseTensor<int>.OfValues(axes), keepDims, 0, null);
        Assert.Equal(OpStatus.Success, r.Status);
        return ((Tensor<long>)r.Outputs![0]).ToArray();
    }

    [Fact]
    public void Basics_MatchOrt()
    {
        Assert.Equal(new long[] { 5 }, Run(new long[] { 1L, 5L, 3L }, new int[] { 3 }, new int[] { 0 }, 0));
        Assert.Equal(new long[] { -1 }, Run(new long[] { -5L, -1L, -3L }, new int[] { 3 }, new int[] { 0 }, 0));
    }

    [Fact]
    public void EmptyExtent_YieldsMinValue()
    {
        Assert.Equal(new long[] { long.MinValue, long.MinValue }, Run(new long[0], new int[] { 2, 0 }, new int[] { 1 }, 0));
    }
}
