using System;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using Xunit;

namespace Lokad.Onnx.Tensors.Tests;

public class DestinationContractTests
{
    static DenseTensor<float> Dirty2x2() =>
        DenseTensor<float>.OfValues(new float[,] { { 7.5f, 7.5f }, { 7.5f, 7.5f } }).ToDenseTensor();

    static void AssertMatMul2x2(Tensor<float> actual)
    {
        // [1,2;3,4] @ [5,6;7,8] = [19,22;43,50].
        Assert.Equal(new float[] { 19f, 22f, 43f, 50f }, actual.ToArray());
    }

    static (Tensor<float> x, Tensor<float> y) MatMulPair() =>
        (DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } }),
         DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 7f, 8f } }));

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    public void DirtyDestination_MatMul2D_Overwrites(bool simd, bool intrinsics)
    {
        if (intrinsics && !Fma.IsSupported) return;
        var (x, y) = MatMulPair();
        var options = new TensorExecutionOptions(simd, intrinsics, 1);
        var dest = Dirty2x2();
        Tensor<float>.MatMul2D(x, y, dest, options);
        AssertMatMul2x2(dest);
    }

    [SkippableFact]
    public void DirtyDestination_MatMul2D_Overwrites_Parallel()
    {
        Skip.If(!System.Runtime.Intrinsics.X86.Fma.IsSupported, "x86 FMA not available on this machine.");
        var x = DenseTensor<float>.Ones(64, 8);
        var y = DenseTensor<float>.Ones(8, 4);
        var dest = DenseTensor<float>.OfValues(Enumerable.Repeat(3.25f, 64 * 4).ToArray()).ToDenseTensor();
        dest = new DenseTensor<float>(dest.Buffer, new int[] { 64, 4 });
        Tensor<float>.MatMul2D(x, y, dest, TensorExecutionOptions.Parallel(4));
        Assert.Equal(Enumerable.Repeat(8f, 64 * 4).ToArray(), dest.ToArray());
    }

    [Fact]
    public void Destination_MatMul2D_ReuseAcrossProducts()
    {
        var (x, y) = MatMulPair();
        var dest = Dirty2x2();
        Tensor<float>.MatMul2D(x, y, dest, TensorExecutionOptions.Simd);
        AssertMatMul2x2(dest);
        var y2 = DenseTensor<float>.OfValues(new float[,] { { 1f, 0f }, { 0f, 1f } });
        Tensor<float>.MatMul2D(x, y2, dest, TensorExecutionOptions.Simd);
        Assert.Equal(new float[] { 1f, 2f, 3f, 4f }, dest.ToArray());
    }

    [Fact]
    public void AliasedDestination_MatMul2D_Throws()
    {
        var (x, y) = MatMulPair();
        var dx = x.ToDenseTensor();
        Assert.Throws<ArgumentException>(() => Tensor<float>.MatMul2D(x, y, dx, TensorExecutionOptions.Scalar));
    }

    [Fact]
    public void AliasedDestination_BatchedMatMul_Throws()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var y = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 7f, 8f } });
        Assert.Throws<ArgumentException>(() => Tensor<float>.MatMul(x, y, x, TensorExecutionOptions.Scalar));
    }

    [Fact]
    public void AliasedDestination_Transpose_Throws()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        Assert.Throws<ArgumentException>(() => Tensor<float>.Transpose(x, x, new int[] { 1, 0 }));
    }

    [Fact]
    public void AliasedBackingMemory_MatMul2D_ThrowsBeforeMutation()
    {
        // C06 reproducer: a distinct destination object over the input array.
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var b = DenseTensor<float>.OfValues(new float[,] { { 1f, 0f }, { 0f, 1f } });
        var dest = new DenseTensor<float>(a.Buffer, new int[] { 2, 2 });
        var ex = Assert.Throws<ArgumentException>(() => Tensor<float>.MatMul2D(a, b, dest, TensorExecutionOptions.Scalar));
        Assert.Contains("alias", ex.Message);
        Assert.Equal(new float[] { 1f, 2f, 3f, 4f }, a.ToArray());
    }

    [Fact]
    public void DisjointWindows_MatMul2D_Works()
    {
        var store = new float[] { 1f, 2f, 3f, 4f, 0f, 0f, 0f, 0f };
        var x = new DenseTensor<float>(store.AsMemory(0, 4), new int[] { 2, 2 });
        var dest = new DenseTensor<float>(store.AsMemory(4, 4), new int[] { 2, 2 });
        var id = DenseTensor<float>.OfValues(new float[,] { { 1f, 0f }, { 0f, 1f } });
        Tensor<float>.MatMul2D(x, id, dest, TensorExecutionOptions.Scalar);
        Assert.Equal(new float[] { 1f, 2f, 3f, 4f }, dest.ToArray());
        Assert.Equal(new float[] { 1f, 2f, 3f, 4f }, x.ToArray());
    }

    [Fact]
    public void OverlappingWindows_MatMul2D_Throws()
    {
        var store = new float[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var x = new DenseTensor<float>(store.AsMemory(0, 4), new int[] { 2, 2 });
        var dest = new DenseTensor<float>(store.AsMemory(2, 4), new int[] { 2, 2 });
        var id = DenseTensor<float>.OfValues(new float[,] { { 1f, 0f }, { 0f, 1f } });
        Assert.Throws<ArgumentException>(() => Tensor<float>.MatMul2D(x, id, dest, TensorExecutionOptions.Scalar));
    }

    [Fact]
    public void SliceView_OverlappingDestination_Throws()
    {
        var p = DenseTensor<float>.Ones(4, 4).ToDenseTensor();
        Tensor<float> rows = p[0..2, ..];
        var dest = new DenseTensor<float>(p.Buffer.Slice(4, 8), new int[] { 2, 4 });
        var id = DenseTensor<float>.OfValues(new float[4, 4]
        {
            { 1f, 0f, 0f, 0f }, { 0f, 1f, 0f, 0f }, { 0f, 0f, 1f, 0f }, { 0f, 0f, 0f, 1f },
        });
        Assert.Throws<ArgumentException>(() => Tensor<float>.MatMul2D(rows, id, dest, TensorExecutionOptions.Scalar));
    }

    [Fact]
    public void DisjointWindows_Transpose_Works()
    {
        var store = new float[] { 1f, 2f, 3f, 4f, 0f, 0f, 0f, 0f };
        var x = new DenseTensor<float>(store.AsMemory(0, 4), new int[] { 2, 2 });
        var dest = new DenseTensor<float>(store.AsMemory(4, 4), new int[] { 2, 2 });
        Tensor<float>.Transpose(x, dest, new int[] { 1, 0 });
        Assert.Equal(new float[] { 1f, 3f, 2f, 4f }, dest.ToArray());
    }

    [Fact]
    public void AliasedBackingMemory_Transpose_Throws()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var dest = new DenseTensor<float>(x.Buffer, new int[] { 2, 2 });
        var ex = Assert.Throws<ArgumentException>(() => Tensor<float>.Transpose(x, dest, new int[] { 1, 0 }));
        Assert.Contains("alias", ex.Message);
        Assert.Equal(new float[] { 1f, 2f, 3f, 4f }, x.ToArray());
    }

    [Fact]
    public void Apply_DestinationLengthMismatch_Throws()
    {
        var x = DenseTensor<float>.OfValues(new float[] { 1f, 2f, 3f, 4f, 5f, 6f });
        var short_ = DenseTensor<float>.Zeros(5);
        var long_ = DenseTensor<float>.Zeros(8);
        Assert.Throws<ArgumentException>(() => x.Apply(v => v, short_));
        Assert.Throws<ArgumentException>(() => x.Apply(v => v, long_));
    }

    [Fact]
    public void BinaryApply_OperandLengthMismatch_Throws()
    {
        var x = DenseTensor<float>.OfValues(new float[] { 1f, 2f, 3f, 4f, 5f, 6f });
        var y = DenseTensor<float>.OfValues(new float[] { 1f, 1f, 1f, 1f, 1f, 1f, 1f });
        var dest = DenseTensor<float>.Zeros(6);
        Assert.Throws<ArgumentException>(() => x.VectorizedApply((l, r) => l + r, (l, r) => l + r, y, dest, TensorExecutionOptions.Simd));
    }

    [Fact]
    public void ReversedStrideExactBuffer_MatchesScalarPath()
    {
        var xr = new DenseTensor<float>(new int[] { 4, 8 }, true);
        for (int i = 0; i < 32; i++) xr.SetValue(i, i + 1f);
        var y = new DenseTensor<float>(Enumerable.Repeat(10f, 32).ToArray(), new[] { 4, 8 });
        var viaSimd = DenseTensor<float>.Zeros(4, 8);
        var viaScalar = DenseTensor<float>.Zeros(4, 8);
        Tensor<float>.Add(xr, y, viaSimd, TensorExecutionOptions.Simd);
        Tensor<float>.Add(xr, y, viaScalar, TensorExecutionOptions.Scalar);
        Assert.Equal(viaScalar.ToArray(), viaSimd.ToArray());
    }

    [Fact]
    public void InPlaceAliasing_AllowedForElementwise()
    {
        var x = DenseTensor<float>.OfValues(new float[] { 1f, 2f, 3f });
        x.Apply(v => v * 2f, x);
        Assert.Equal(new float[] { 2f, 4f, 6f }, x.ToArray());
        var y = DenseTensor<float>.OfValues(new float[] { 10f, 10f, 10f });
        Tensor<float>.Add(x, y, x, TensorExecutionOptions.Simd);
        Assert.Equal(new float[] { 12f, 14f, 16f }, x.ToArray());
    }

    [Fact]
    public void SoftmaxDestinationAliasing_MatchesAllocating()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var expected = Tensor<float>.Softmax(x, -1, null, 13);
        var actual = x.ToDenseTensor();
        Tensor<float>.Softmax(x, actual, -1, null, 13);
        Assert.Equal(expected.ToArray(), actual.ToArray());
    }
}
