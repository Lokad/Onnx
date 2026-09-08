using System;
using Xunit;

namespace Lokad.Onnx.Tensors.Tests;

public class TensorCreationTests
{
    [Fact]
    public void CopyFromExistingArray_Values_And_Allocation()
    {
        var data = new float[100000];
        for (int i = 0; i < data.Length; i++) data[i] = i * 0.5f;
        for (int i = 0; i < 5; i++) DenseTensor<float>.OfValues(data);
        long before = GC.GetAllocatedBytesForCurrentThread();
        DenseTensor<float> t = DenseTensor<float>.OfValues(data);
        for (int i = 0; i < 10; i++) t = DenseTensor<float>.OfValues(data);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(99999 * 0.5f, t.GetValue(99999));
        Assert.Equal(new int[] { 100000 }, t.Dimensions.ToArray());
        Assert.True(allocated < 5000000L, $"array copy allocated {allocated} bytes for 10x400KB payloads");
    }

    [Fact]
    public void Copy_IsIndependent_Wrap_SharesStorage()
    {
        var data = new float[] { 1f, 2f, 3f };
        var copy = DenseTensor<float>.OfValues(data);
        data[0] = 99f;
        Assert.Equal(1f, copy.GetValue(0));
        var shared = new float[] { 1f, 2f, 3f };
        var wrap = new DenseTensor<float>(new Memory<float>(shared), new[] { 3 });
        shared[0] = 99f;
        Assert.Equal(99f, wrap.GetValue(0));
    }

    [Fact]
    public void ReversedLayout_Values()
    {
        var data = new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } };
        var t = data.ToTensor<float>(reverseStride: true);
        Assert.Equal(new int[] { 2, 3 }, t.Dimensions.ToArray());
        Assert.Equal(1f, t[0, 0]);
        Assert.Equal(2f, t[0, 1]);
        Assert.Equal(3f, t[0, 2]);
        Assert.Equal(4f, t[1, 0]);
        Assert.Equal(5f, t[1, 1]);
        Assert.Equal(6f, t[1, 2]);
    }

    [Fact]
    public void Multidimensional_Values_WithoutBoxes()
    {
        var data = new float[100, 100];
        for (int i = 0; i < 100; i++)
            for (int j = 0; j < 100; j++)
                data[i, j] = i * 100 + j;
        for (int i = 0; i < 3; i++) DenseTensor<float>.OfValues(data);
        long before = GC.GetAllocatedBytesForCurrentThread();
        var t = DenseTensor<float>.OfValues(data);
        for (int i = 0; i < 5; i++) t = DenseTensor<float>.OfValues(data);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(new int[] { 100, 100 }, t.Dimensions.ToArray());
        Assert.Equal(1234f, t[12, 34]);
        Assert.True(allocated < 300000L, $"multidim copy allocated {allocated} bytes for 5x40KB payloads");
    }

    [Fact]
    public void SpanOverload_Values_And_Mismatch()
    {
        var data = new float[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var t = DenseTensor<float>.OfValues(data.AsSpan(), new int[] { 2, 3 });
        Assert.Equal(new int[] { 2, 3 }, t.Dimensions.ToArray());
        Assert.Equal(6f, t[1, 2]);
        data[0] = 99f;
        Assert.Equal(1f, t.GetValue(0));
        Assert.Throws<ArgumentException>(() => DenseTensor<float>.OfValues(data.AsSpan(), new int[] { 2, 2 }));
    }

    [Fact]
    public void NullInputs_Throw()
    {
        Assert.Throws<ArgumentNullException>(() => DenseTensor<float>.OfValues((float[])null!));
        Assert.Throws<ArgumentNullException>(() => DenseTensor<float>.OfValues((Array)null!));
    }
}
