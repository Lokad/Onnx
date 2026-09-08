using System;
using System.Linq;
using Xunit;

namespace Lokad.Onnx.Tensors.Tests;

sealed class FloatArrayComparer : System.Collections.Generic.IEqualityComparer<float[]>
{
    public bool Equals(float[]? x, float[]? y) => x is not null && y is not null && x.SequenceEqual(y);
    public int GetHashCode(float[] o) => o.Length;
}

public class ReductionEngineTests
{
    [Fact]
    public void IntMean_SumsBeforeDividing_OracleVerified()
    {
        // Verified against ORT 1.29 out of process: column sums [4,6] average to [2,3],
        // not the divide-first [1,3].
        var data = DenseTensor<int>.OfValues(new int[2, 2] { { 1, 2 }, { 3, 4 } });
        var mean0 = Tensor<int>.ReduceMean(data, new int[] { 0 }.ToTensor<int>());
        Assert.Equal(new[] { 2 }, mean0.Dimensions.ToArray());
        Assert.Equal(2, mean0[0]);
        Assert.Equal(3, mean0[1]);
        var mean1 = Tensor<int>.ReduceMean(data, new int[] { 1 }.ToTensor<int>());
        Assert.Equal(1, mean1[0]);
        Assert.Equal(3, mean1[1]);
    }

    [Fact]
    public void FloatMean_ScalesWithoutSecondTensor()
    {
        var data = new float[100000];
        for (int i = 0; i < data.Length; i++) data[i] = i * 0.5f + 1f;
        var x = new DenseTensor<float>(data, new[] { 100000 });
        var axes = new DenseTensor<int>(new int[] { 0 }, new[] { 1 });
        for (int i = 0; i < 3; i++) Tensor<float>.ReduceMean(x, axes);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10; i++) Tensor<float>.ReduceMean(x, axes);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        var m = Tensor<float>.ReduceMean(x, axes);
        Assert.True(System.Math.Abs(25000.75f - m.GetValue(0)) < 1f);
        Assert.True(allocated < 6000000L, $"float mean allocated {allocated} bytes for 10x400KB reductions");
    }

    [Fact]
    public void EmptyReductions_KeepOracleEdges()
    {
        var empty = new DenseTensor<float>(new float[0], new[] { 0, 3 });
        var sum = Tensor<float>.ReduceSum(empty, new int[] { 0 }.ToTensor<int>());
        Assert.Equal(new[] { 3 }, sum.Dimensions.ToArray());
        Assert.Equal(new float[] { 0f, 0f, 0f }, sum.ToArray(), new FloatArrayComparer());
        var max = Tensor<float>.ReduceMax(empty, new int[] { 0 }.ToTensor<int>());
        Assert.Equal(new float[] { float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity }, max.ToArray(), new FloatArrayComparer());
        var mean = Tensor<float>.ReduceMean(empty, new int[] { 0 }.ToTensor<int>());
        foreach (var v in mean.ToArray()) Assert.True(float.IsNaN(v));
    }

    [Fact]
    public void ArbitraryAxes_Values_WithAndWithoutKeepDims()
    {
        var data = new DenseTensor<float>(Enumerable.Range(0, 24).Select(i => (float)i).ToArray(), new[] { 2, 3, 4 });
        var axes = new int[] { 0, 2 }.ToTensor<int>();
        double expected(int j)
        {
            double s = 0;
            for (int i = 0; i < 2; i++)
                for (int k = 0; k < 4; k++)
                    s += i * 12 + j * 4 + k;
            return s;
        }
        var sum = Tensor<float>.ReduceSum(data, axes);
        Assert.Equal(new[] { 3 }, sum.Dimensions.ToArray());
        for (int j = 0; j < 3; j++) Assert.True(System.Math.Abs((float)expected(j) - sum[j]) < 0.001f);
        var kept = Tensor<float>.ReduceSum(data, axes, true, null);
        Assert.Equal(new[] { 1, 3, 1 }, kept.Dimensions.ToArray());
        var mean = Tensor<float>.ReduceMean(data, axes);
        for (int j = 0; j < 3; j++) Assert.True(System.Math.Abs((float)(expected(j) / 8.0) - mean[j]) < 0.001f);
    }

    [Fact]
    public void FloatMean_PrecisionMatchesDoubleComputation()
    {
        var data = DenseTensor<float>.OfValues(new float[] { 1000000f, 1000001f, 1000002f, 1000003f });
        var mean = Tensor<float>.ReduceMean(data, new int[] { 0 }.ToTensor<int>());
        double expected = (1000000.0 + 1000001.0 + 1000002.0 + 1000003.0) / 4.0;
        Assert.True(System.Math.Abs((float)expected - mean.GetValue(0)) < 0.1f);
    }
}
