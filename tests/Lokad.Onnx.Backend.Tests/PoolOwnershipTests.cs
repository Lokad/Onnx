using System.Runtime.CompilerServices;

namespace Lokad.Onnx.Backend.Tests;

public class PoolOwnershipTests
{
    struct ManagedValue { public object? Item; }
    readonly record struct UnmanagedValue(int Count, double Weight);
    enum Tag : byte { First, Second }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void NonRentableReturnsRejectWithoutStateChangesOrRetention(int kind)
    {
        var pool = new TensorBufferPool();
        var rejected = RejectUnrentable(pool, kind);
        Assert.Equal(0, pool.Returned);
        Assert.Equal(0, pool.Dropped);
        Assert.Equal(0, pool.AllocatedNew);
        Assert.Equal(0, pool.Reused);
        Assert.Equal(0, pool.PeakOutstandingBytes);
        Collect();
        Assert.False(rejected.IsAlive);
        var valid = pool.Rent<float>(2);
        pool.Return(valid);
        Assert.Same(valid, pool.Rent<float>(2));
        GC.KeepAlive(pool);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    static WeakReference RejectUnrentable(TensorBufferPool pool, int kind)
    {
        Array value = kind switch
        {
            0 => new string[] { "a", "b" },
            1 => new object[] { new object() },
            2 => new float[][] { [1], [2] },
            3 => new ManagedValue[] { new() { Item = new object() } },
            _ => new int?[] { 1, null }
        };
        var failure = Assert.Throws<ArgumentException>(() => pool.Return(value));
        Assert.Equal("array", failure.ParamName);
        Assert.False(pool.IsOwned(value));
        return new WeakReference(value);
    }

    [Fact]
    public void ForeignUnmanagedValuesRemainAdoptableAndOwnedReturnsRemainReusable()
    {
        CheckAdoption(new UnmanagedValue[] { new(3, .5) });
        CheckAdoption(new[] { Tag.First, Tag.Second });
        CheckAdoption(new[] { DateTime.UnixEpoch });
        CheckAdoption(new System.Numerics.Complex[] { new(1, 2) });
    }

    static void CheckAdoption<T>(T[] array) where T : unmanaged
    {
        var pool = new TensorBufferPool();
        pool.Return(array);
        Assert.False(pool.IsOwned(array));
        Assert.Same(array, pool.Rent<T>(array.Length));
        Assert.True(pool.IsOwned(array));
        pool.Return(array);
        Assert.Same(array, pool.Rent<T>(array.Length));
        Assert.Equal(2, pool.Returned);
        Assert.Equal(2, pool.Reused);
        Assert.Equal(0, pool.AllocatedNew);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DiscardedRentsCanBeCollectedWhileThePoolLives(bool retainedCache)
    {
        var cache = retainedCache ? new ReleasedBufferCache(4096, 8) : null;
        var pool = cache is null ? new TensorBufferPool() : new TensorBufferPool(cache);
        var discarded = ReturnBeyondCap(pool);
        Collect();
        Assert.False(discarded.IsAlive);
        Assert.Equal(33, pool.AllocatedNew);
        Assert.Equal(32, pool.Returned);
        Assert.Equal(1, pool.Dropped);
        Assert.Equal(33 * 4096, pool.PeakOutstandingBytes);
        GC.KeepAlive(pool);
    }

    [Fact]
    public void StorageRejectedByTheRetainedCacheCanAlsoBeCollected()
    {
        var cache = new ReleasedBufferCache(0, 0);
        var pool = new TensorBufferPool(cache);
        var discarded = ReturnAndTransfer(pool);
        Collect();
        Assert.False(discarded.IsAlive);
        Assert.Equal(0, cache.Bytes);
        Assert.Equal(0, cache.Count);
        GC.KeepAlive(pool);
    }

    [Fact]
    public void StillHeldDiscardedArraysRemainOwnedAndReusable()
    {
        var pool = new TensorBufferPool();
        var arrays = Enumerable.Range(0, 33).Select(_ => pool.Rent<int>(4)).ToArray();
        foreach (var array in arrays) pool.Return(array);
        int[] held = arrays[^1];
        held[0] = 731;
        Collect();
        Assert.True(pool.IsOwned(held));
        Assert.Equal(731, held[0]);
        var reused = pool.Rent<int>(4);
        Assert.True(arrays.Take(32).Any(array => ReferenceEquals(array, reused)));
        pool.Return(held);
        Assert.Same(held, pool.Rent<int>(4));
        Assert.Equal(1, pool.Dropped);
        Assert.Equal(2, pool.Reused);
        Assert.Throws<ArgumentException>(() => pool.Return(arrays[0]));
    }

    [Fact]
    public void NonSzReturnIsRejectedWithoutPoisoningThePool()
    {
        var pool = new TensorBufferPool();
        var invalid = Array.CreateInstance(typeof(float), new[] { 3 }, new[] { 1 });
        Assert.False(invalid.GetType().IsSZArray);
        Assert.Throws<ArgumentException>(() => pool.Return(invalid));
        Assert.Equal(0, pool.Returned);
        Assert.Equal(0, pool.Dropped);
        var valid = pool.Rent<float>(3);
        Assert.Equal(new float[3], valid);
        Assert.Equal(1, pool.AllocatedNew);
    }

    [Fact]
    public void ForeignAdoptionBecomesOwnedOnlyAfterARent()
    {
        var cache = new ReleasedBufferCache(4096, 8);
        var pool = new TensorBufferPool(cache);
        var foreign = new bool[] { true, false, true };
        pool.Return(foreign);
        Assert.False(pool.IsOwned(foreign));
        Assert.Same(foreign, pool.Rent<bool>(3));
        Assert.True(pool.IsOwned(foreign));
        pool.Return(foreign);
        pool.RetainReleasedBuffers();
        Assert.Equal(3, cache.Bytes);
        Assert.Equal(1, cache.Count);
        Assert.Equal(3, pool.PeakOutstandingBytes);
        Assert.Same(foreign, new TensorBufferPool(cache).Rent<bool>(3));
    }

    static void Collect()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    static WeakReference ReturnBeyondCap(TensorBufferPool pool)
    {
        var arrays = Enumerable.Range(0, 33).Select(_ => pool.Rent<float>(1024)).ToArray();
        foreach (var array in arrays) pool.Return(array);
        return new WeakReference(arrays[^1]);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    static WeakReference ReturnAndTransfer(TensorBufferPool pool)
    {
        var array = pool.Rent<float>(1024);
        pool.Return(array);
        pool.RetainReleasedBuffers();
        return new WeakReference(array);
    }
}
