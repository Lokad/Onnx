namespace Lokad.Onnx.Backend.Tests;

using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

[CollectionDefinition("NonParallelAllocation", DisableParallelization = true)]
public class NonParallelAllocationCollection { }

/// <summary>
/// Guards the P02 rule: the N-D destination and pool overloads compute
/// directly into the destination with identical numbers to the allocating
/// path, across promotion, batching, broadcast, dirty buffers, zero dims,
/// and kernel modes, without a second product-sized payload on pooled calls.
/// </summary>
[Collection("NonParallelAllocation")]
public class MatMulDestinationTests
{
    static TensorExecutionOptions[] Modes()
    {
        var modes = new List<TensorExecutionOptions> { TensorExecutionOptions.Scalar, TensorExecutionOptions.Simd };
        if (Fma.IsSupported) modes.Add(TensorExecutionOptions.Intrinsics);
        return modes.ToArray();
    }

    static DenseTensor<float> Rand(int[] dims, Random rnd)
    {
        int n = 1;
        foreach (var d in dims) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rnd.NextDouble() * 2.0 - 1.0);
        return new DenseTensor<float>(data, dims);
    }

    static DenseTensor<float> Dirty(int[] dims)
    {
        int n = 1;
        foreach (var d in dims) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = 7.5f;
        return new DenseTensor<float>(data, dims);
    }

    static void AgreesElementwise(float[] actual, float[] expected, string variant)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float tol = 1e-4f * Math.Max(1f, Math.Abs(expected[i]));
            Assert.True(Math.Abs(actual[i] - expected[i]) <= tol, variant + " differs at " + i + ": " + actual[i] + " vs " + expected[i] + ".");
        }
    }

    public static IEnumerable<object[]> DestinationCases()
    {
        yield return new object[] { new int[] { 2, 3 }, new int[] { 3, 4 } };
        yield return new object[] { new int[] { 4 }, new int[] { 4, 3 } };
        yield return new object[] { new int[] { 2, 4 }, new int[] { 4 } };
        yield return new object[] { new int[] { 4 }, new int[] { 4 } };
        yield return new object[] { new int[] { 2, 3, 4 }, new int[] { 4, 5 } };
        yield return new object[] { new int[] { 2, 3, 4 }, new int[] { 2, 4, 5 } };
        yield return new object[] { new int[] { 1, 3, 4 }, new int[] { 2, 4, 5 } };
        yield return new object[] { new int[] { 2, 2, 3, 4 }, new int[] { 4, 5 } };
    }

    [Theory]
    [MemberData(nameof(DestinationCases))]
    public void DestinationMatchesAllocating(int[] xd, int[] yd)
    {
        foreach (var mode in Modes())
        {
            var rnd = new Random(1234);
            var x = Rand(xd, rnd);
            var y = Rand(yd, rnd);
            var expected = Tensor<float>.MatMul(x, y, mode);
            var dest = Dirty(expected.Dimensions.ToArray());
            var got = Tensor<float>.MatMul(x, y, dest, mode);
            Assert.Same(dest, got);
            AgreesElementwise(dest.ToArray(), ((Tensor<float>)expected).ToArray(),
                "mode dims=[" + string.Join(",", xd) + "]x[" + string.Join(",", yd) + "]");
        }
    }

    public static IEnumerable<object[]> BroadcastCases()
    {
        yield return new object[] { new int[] { 4, 8, 16 }, new int[] { 16, 32 } };
        yield return new object[] { new int[] { 2, 8, 16 }, new int[] { 1, 16, 32 } };
        yield return new object[] { new int[] { 3, 5, 7 }, new int[] { 3, 7, 2 } };
    }

    [Theory]
    [MemberData(nameof(BroadcastCases))]
    public void BroadcastSharedWeight_Agrees(int[] xd, int[] yd)
    {
        foreach (var mode in Modes())
        {
            var rnd = new Random(5150);
            var x = Rand(xd, rnd);
            var y = Rand(yd, rnd);
            var expected = Tensor<float>.MatMul(x, y, mode);
            var dest = Dirty(expected.Dimensions.ToArray());
            Tensor<float>.MatMul(x, y, dest, mode);
            AgreesElementwise(dest.ToArray(), ((Tensor<float>)expected).ToArray(), "broadcast");
        }
    }

    [Fact]
    public void ExplicitBroadcastView_Agrees()
    {
        var mode = TensorExecutionOptions.Simd;
        var rnd = new Random(5151);
        var x = Rand(new int[] { 4, 8, 16 }, rnd);
        var y = Rand(new int[] { 16, 32 }, rnd);
        Assert.True(Tensor<float>.Broadcast(y, new int[] { 4, 16, 32 }, out var yv));
        var expected = Tensor<float>.MatMul(x, y, mode);
        var got = Tensor<float>.MatMul(x, yv!, mode);
        AgreesElementwise(((Tensor<float>)got).ToArray(), ((Tensor<float>)expected).ToArray(), "explicit-view");
    }

    [Fact]
    public void BroadcastOverStridedSource_FallsBackCorrectly()
    {
        var rnd = new Random(5152);
        var src = new DenseTensor<float>(new int[] { 16, 32 }, true);
        for (int i = 0; i < src.Length; i++) src.SetValue(i, (float)rnd.NextDouble());
        Assert.True(Tensor<float>.Broadcast(src, new int[] { 4, 16, 32 }, out var yv));
        var x = Rand(new int[] { 4, 8, 16 }, rnd);
        var expected = Tensor<float>.MatMul(x, src, TensorExecutionOptions.Scalar);
        var fallback = Tensor<float>.MatMul(x, yv!, TensorExecutionOptions.Scalar);
        AgreesElementwise(((Tensor<float>)fallback).ToArray(), ((Tensor<float>)expected).ToArray(), "strided-fallback");
    }

    [Fact]
    public void PooledBroadcastWeight_SkipsBatchCopy()
    {
        var opts = TensorExecutionOptions.Simd;
        var pool = new TensorBufferPool();
        var rnd = new Random(5153);
        var x = Rand(new int[] { 8, 32, 64 }, rnd);
        var y = Rand(new int[] { 64, 128 }, rnd);
        long tile = 8L * 64 * 128 * 4;
        long best = long.MaxValue;
        Tensor<float>? last = null;
        for (int i = 0; i < 10; i++)
        {
            long before = GC.GetTotalAllocatedBytes(false);
            last = Tensor<float>.MatMul(x, y, opts, pool);
            best = Math.Min(best, GC.GetTotalAllocatedBytes(false) - before);
            ReturnToPool(pool, last);
        }
        Assert.Equal(new int[] { 8, 32, 128 }, last!.Dimensions.ToArray());
        Assert.True(best < tile, "pooled broadcast call allocated " + best + " bytes (>= one weight-tile payload).");
    }

    [Fact]
    public void ZeroInnerDims_Agree()
    {
        var mode = TensorExecutionOptions.Simd;
        var x = Rand(new int[] { 2, 0 }, new Random(42));
        var y = Rand(new int[] { 0, 3 }, new Random(43));
        var expected = Tensor<float>.MatMul(x, y, mode);
        Assert.Equal(new int[] { 2, 3 }, expected.Dimensions.ToArray());
        Assert.Equal(new float[6], expected.ToArray());
        var dest = Dirty(new int[] { 2, 3 });
        Tensor<float>.MatMul(x, y, dest, mode);
        Assert.Equal(new float[6], dest.ToArray());
        var xb = Rand(new int[] { 0, 2, 3 }, new Random(44));
        var yb = Rand(new int[] { 0, 3, 4 }, new Random(45));
        var eb = Tensor<float>.MatMul(xb, yb, mode);
        Assert.Equal(new int[] { 0, 2, 4 }, eb.Dimensions.ToArray());
        var db = Dirty(new int[] { 0, 2, 4 });
        Tensor<float>.MatMul(xb, yb, db, mode);
        Assert.Empty(db.ToArray());
    }

    [Fact]
    public void PooledCall_AllocatesNoSecondPayload()
    {
        var opts = TensorExecutionOptions.Simd;
        var pool = new TensorBufferPool();
        var rnd = new Random(99);
        var x = Rand(new int[] { 128, 256 }, rnd);
        var y = Rand(new int[] { 256, 128 }, rnd);
        long best = long.MaxValue;
        Tensor<float>? last = null;
        for (int i = 0; i < 10; i++)
        {
            long before = GC.GetTotalAllocatedBytes(false);
            last = Tensor<float>.MatMul(x, y, opts, pool);
            best = Math.Min(best, GC.GetTotalAllocatedBytes(false) - before);
            ReturnToPool(pool, last);
        }
        Assert.Equal(new int[] { 128, 128 }, last!.Dimensions.ToArray());
        Assert.True(best < 128L * 128 * 4, "pooled rank-2 call allocated " + best + " bytes (>= one product-sized payload).");
        // Batched broadcast-operand materialization is P05 scope: the shared
        // operand tile still allocates there. Batched routing itself (no fresh
        // output plus copy) is covered by the agreement cases above.
    }

    static void ReturnToPool(TensorBufferPool pool, Tensor<float> z)
    {
        if (MemoryMarshal.TryGetArray<float>(((DenseTensor<float>)z).Buffer, out var seg) && seg.Array is not null)
            pool.Return(seg.Array);
    }

}
