using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Int64 ReduceMean values with ORT 1.29 semantics: double accumulation,
// divided, truncated toward zero. Expectations are ORT-probed, including
/// the truncation discriminators (half-up and half-even both fail them).
/// </summary>
public class ReduceMeanInt64Tests
{
    static DenseTensor<long> Vec(params long[] values) => DenseTensor<long>.OfValues(values);

    static long[] Run(long[] values)
    {
        var axes = DenseTensor<int>.OfValues(new int[] { 0 });
        var r = CPU.ReduceMean(Vec(values), axes, 1, 0, null);
        Assert.Equal(OpStatus.Success, r.Status);
        return ((Tensor<long>)r.Outputs![0]).ToArray();
    }

    [Fact]
    public void Basics_MatchOrt()
    {
        Assert.Equal(new long[] { 2 }, Run(new long[] { 1L, 2L, 3L, 4L }));
        Assert.Equal(new long[] { 42 }, Run(new long[] { 42L }));
        Assert.Equal(new long[] { 7 }, Run(new long[] { 7L, 7L, 7L, 8L }));
    }

    [Fact]
    public void Truncation_MatchesOrt()
    {
        // 1.5 truncates to 1 (half-up and half-even would give 2);
        // -1.5 truncates to -1 (floor would give -2).
        Assert.Equal(new long[] { 1 }, Run(new long[] { 1L, 2L }));
        Assert.Equal(new long[] { -1 }, Run(new long[] { -1L, -2L }));
        Assert.Equal(new long[] { 5 }, Run(new long[] { 5L, 6L }));
        Assert.Equal(new long[] { -5 }, Run(new long[] { -5L, -6L }));
        // Precision: double accumulation rounds 2^53+1 before dividing.
        Assert.Equal(new long[] { 4503599627370496L }, Run(new long[] { 9007199254740993L, 1L }));
    }

    [Fact]
    public void SaturationAndEmpty_MatchOrt()
    {
        Assert.Equal(new long[] { long.MaxValue }, Run(new long[] { long.MaxValue, long.MaxValue }));
        var axes = DenseTensor<int>.OfValues(new int[] { 1 });
        var r = CPU.ReduceMean(DenseTensor<long>.OfValues(new long[0], new int[] { 2, 0 }), axes, 0, 0, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new long[] { 0L, 0L }, ((Tensor<long>)r.Outputs![0]).ToArray());
    }
}
