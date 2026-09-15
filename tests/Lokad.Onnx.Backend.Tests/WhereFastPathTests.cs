using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Covers the pooled span fast path of Tensor.Where: agreement with the legacy
/// view-based implementation across broadcast geometries and dtypes, pool
/// engagement and reuse, fallback for non-dense layouts, and preserved errors.
/// </summary>
public class WhereFastPathTests
{
    static DenseTensor<bool> BoolTensor(int[] dims, bool[] values) => new DenseTensor<bool>(values, dims);

    static float[] Range(float start, float step, int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = start + step * i;
        return v;
    }

    [Fact]
    public void IdenticalShape_BitMatchesLegacyAndRentsPool()
    {
        var c = BoolTensor(new[] { 2, 3 }, new[] { true, false, true, false, true, false });
        var x = new DenseTensor<float>(Range(1f, 1f, 6), new[] { 2, 3 });
        var y = new DenseTensor<float>(Range(-1f, -0.5f, 6), new[] { 2, 3 });
        var pool = new TensorBufferPool();
        var actual = Tensor<float>.Where(c, x, y, TensorExecutionOptions.Auto, pool);
        var expected = Tensor<float>.Where(c, x, y);
        Assert.Equal(expected.ToArray(), actual.ToArray());
        Assert.Equal(1, pool.AllocatedNew);
    }

    [Fact]
    public void AttentionBroadcast_BitMatchesLegacy()
    {
        var condValues = new bool[256];
        for (int i = 0; i < 256; i++) condValues[i] = (i % 5) != 0;
        var c = BoolTensor(new[] { 1, 1, 16, 16 }, condValues);
        var x = new DenseTensor<float>(new[] { -100f }, new[] { 1 });
        var y = new DenseTensor<float>(Range(-2f, 0.01f, 2048), new[] { 1, 8, 16, 16 });
        var pool = new TensorBufferPool();
        var actual = Tensor<float>.Where(c, x, y, TensorExecutionOptions.Auto, pool);
        var expected = Tensor<float>.Where(c, x, y);
        Assert.Equal(new[] { 1, 8, 16, 16 }, actual.Dimensions.ToArray());
        Assert.Equal(expected.ToArray(), actual.ToArray());
        Assert.Equal(1, pool.AllocatedNew);
    }

    [Fact]
    public void ScalarCondition_SelectsWholeLeg()
    {
        var c = BoolTensor(new[] { 1 }, new[] { false });
        var x = new DenseTensor<float>(Range(1f, 1f, 4), new[] { 2, 2 });
        var y = new DenseTensor<float>(Range(10f, 10f, 4), new[] { 2, 2 });
        var actual = Tensor<float>.Where(c, x, y, TensorExecutionOptions.Auto, null);
        Assert.Equal(y.ToArray(), actual.ToArray());
    }

    [Fact]
    public void RankZero_AllScalar()
    {
        var c = new DenseTensor<bool>(new[] { true }, new[] { 1 });
        var x = new DenseTensor<float>(new[] { 3f }, new int[0]);
        var y = new DenseTensor<float>(new[] { 4f }, new int[0]);
        var actual = Tensor<float>.Where(c, x, y, TensorExecutionOptions.Auto, null);
        Assert.Equal(new[] { 3f }, actual.ToArray());
    }

    [Fact]
    public void ReversedStride_FallsBackWithAgreement()
    {
        var c = new DenseTensor<bool>(new[] { 2, 2 }, true);
        bool[] cv = new[] { true, false, false, true };
        for (int i = 0; i < 4; i++) c.SetValue(i, cv[i]);
        var x = new DenseTensor<float>(Range(1f, 1f, 4), new[] { 2, 2 });
        var y = new DenseTensor<float>(Range(-1f, -1f, 4), new[] { 2, 2 });
        var pool = new TensorBufferPool();
        var actual = Tensor<float>.Where(c, x, y, TensorExecutionOptions.Auto, pool);
        Assert.Equal(Tensor<float>.Where(c, x, y).ToArray(), actual.ToArray());
        Assert.Equal(0, pool.AllocatedNew);
    }

    [Fact]
    public void OtherDtypes_AgreeWithLegacy()
    {
        var c = BoolTensor(new[] { 4 }, new[] { true, false, false, true });
        var xi = new DenseTensor<int>(new[] { 1, 2, 3, 4 }, new[] { 4 });
        var yi = new DenseTensor<int>(new[] { 5, 6, 7, 8 }, new[] { 4 });
        Assert.Equal(Tensor<int>.Where(c, xi, yi).ToArray(), Tensor<int>.Where(c, xi, yi, TensorExecutionOptions.Auto, null).ToArray());
        var xd = new DenseTensor<double>(new[] { 1.0, 2.0 }, new[] { 2 });
        var yd = new DenseTensor<double>(new[] { 3.0, 4.0 }, new[] { 2 });
        var cd = BoolTensor(new[] { 2 }, new[] { false, true });
        Assert.Equal(Tensor<double>.Where(cd, xd, yd).ToArray(), Tensor<double>.Where(cd, xd, yd, TensorExecutionOptions.Auto, null).ToArray());
    }

    [Fact]
    public void PooledOutput_RecyclesAfterReturn()
    {
        var c = BoolTensor(new[] { 4 }, new[] { true, false, true, false });
        var x = new DenseTensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        var y = new DenseTensor<float>(new[] { 5f, 6f, 7f, 8f }, new[] { 4 });
        var pool = new TensorBufferPool();
        var first = (DenseTensor<float>)Tensor<float>.Where(c, x, y, TensorExecutionOptions.Auto, pool);
        Assert.True(MemoryMarshal.TryGetArray<float>(first.Buffer, out var window) && window.Array is not null);
        pool.Return(window.Array);
        var second = Tensor<float>.Where(c, x, y, TensorExecutionOptions.Auto, pool);
        Assert.Equal(first.ToArray(), second.ToArray());
        Assert.Equal(1, pool.AllocatedNew);
        Assert.Equal(1, pool.Reused);
    }

    [Fact]
    public void IncompatibleShapes_ThrowOnBothPaths()
    {
        var c = BoolTensor(new[] { 3 }, new[] { true, false, true });
        var x = new DenseTensor<float>(new[] { 1f, 2f }, new[] { 2 });
        var y = new DenseTensor<float>(new[] { 3f, 4f }, new[] { 2 });
        Assert.Throws<ArgumentException>(() => Tensor<float>.Where(c, x, y));
        Assert.Throws<ArgumentException>(() => Tensor<float>.Where(c, x, y, TensorExecutionOptions.Auto, null));
    }
}

