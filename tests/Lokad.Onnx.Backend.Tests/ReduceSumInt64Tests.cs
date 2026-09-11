using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Int64 ReduceSum values with ORT 1.29 semantics: double accumulation with
/// saturation. Expectations are ORT-probed, including the vectors that
/// separate double accumulation from running saturation and wide-then-clamp.
/// </summary>
public class ReduceSumInt64Tests
{
    static DenseTensor<long> Vec(params long[] values) => DenseTensor<long>.OfValues(values);

    static long[] Run(long[] values)
    {
        var axes = DenseTensor<int>.OfValues(new int[] { 0 });
        var r = CPU.ReduceSum(Vec(values), axes, 1, 0, null);
        Assert.Equal(OpStatus.Success, r.Status);
        return ((Tensor<long>)r.Outputs![0]).ToArray();
    }

    [Fact]
    public void Basics_MatchOrt()
    {
        Assert.Equal(new long[] { 6 }, Run(new long[] { 1L, 2L, 3L }));
        Assert.Equal(new long[] { 42 }, Run(new long[] { 42L }));
        Assert.Equal(new long[] { 2 }, Run(new long[] { -5L, 10L, -3L }));
    }

    [Fact]
    public void Saturation_MatchesOrt()
    {
        // ORT 1.29: [Max, 1] clamps to Max, [Min, -1] to Min.
        Assert.Equal(new long[] { long.MaxValue }, Run(new long[] { long.MaxValue, 1L }));
        Assert.Equal(new long[] { long.MinValue }, Run(new long[] { long.MinValue, -1L }));
        // Discriminators: [Max, Max, Min, Min] is 0 (wide-then-clamp would
        // give -2, running saturation Min); [Max, Max, Max, Min] is Max.
        Assert.Equal(new long[] { 0 }, Run(new long[] { long.MaxValue, long.MaxValue, long.MinValue, long.MinValue }));
        Assert.Equal(new long[] { long.MaxValue }, Run(new long[] { long.MaxValue, long.MaxValue, long.MaxValue, long.MinValue }));
        // Precision: 2^53+1 rounds in double accumulation (running exact
        // arithmetic would give 2^53+2).
        Assert.Equal(new long[] { 9007199254740992L }, Run(new long[] { 9007199254740993L, 1L }));
    }
}
