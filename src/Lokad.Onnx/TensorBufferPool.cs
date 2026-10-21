namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

/// <summary>Per-execution pool of exact-size backing arrays for dense tensor outputs.</summary>
/// <remarks>
/// Rent and return are single-threaded by construction: only graph dispatch rents and only the
/// owning execution releases at recorded last uses, while parallel kernel workers merely write disjoint
/// spans. Kernels stay destination-based and never see the pool. Misses allocate; returns beyond a small
/// per-shape cap are dropped. Never static: one pool per graph execution.
/// Ownership: every array handed out by Rent is recognized while it remains alive, without rooting
/// discarded arrays solely to remember their identity. Release logic
/// must only return arrays for which IsOwned reports true, so caller inputs, model constants,
/// attribute tensors and unrelated fresh buffers are never adopted. See C01.
/// Metrics: AllocatedNew/AllocatedNewBytes count fresh rents (GC allocation); Reused/ReusedBytes count
/// rents served from previously returned storage (pool-served bytes, not new GC bytes); PeakOutstandingBytes tracks the high-water mark of checked-out bytes (adoptions excluded, drops released); Returned counts
/// dead buffers adopted into the pool; Dropped counts returns discarded by the per-shape cap. None is a
/// substitute for the others: GC bytes come from the collector, live payload from tensor shapes, scratch
/// from temporary kernel buffers, and pool-served bytes from these counters.
/// Return contract: only single-dimensional (SZ) arrays are pooled. Null, multidimensional, and duplicate
/// (already buffered) arrays are rejected with an exception. Foreign arrays (never rented from this pool)
/// are adopted for reuse; release logic still only returns owned storage.
/// The bounded released-buffer cache supplies/retains only already-free arrays
/// between executions. Ownership, live/outstanding records and metrics remain
/// per execution; no still-checked-out array is exported when a run ends.
/// </remarks>
public sealed class TensorBufferPool
{
    internal const int MaxBufferedPerShape = 32;

    readonly ReleasedBufferCache? releasedCache;

    // Per-pool override lets tests compare both routes without global mutation.
    internal bool ReleaseFusedTemporaries { get; set; } = AblationSwitches.EnableFusedTempRelease;

    public TensorBufferPool() { }

    internal TensorBufferPool(ReleasedBufferCache releasedCache)
    {
        this.releasedCache = releasedCache;
    }

    readonly Dictionary<(Type, int), Stack<Array>> free = new();

    sealed class Ownership(long bytes)
    {
        internal readonly long Bytes = bytes;
    }

    readonly ConditionalWeakTable<Array, Ownership> owned = new();

    readonly HashSet<Array> buffered = new();

    public int AllocatedNew { get; private set; }

    public int Reused { get; private set; }

    public int Returned { get; private set; }

    public int Dropped { get; private set; }

    public long AllocatedNewBytes { get; private set; }

    public long ReusedBytes { get; private set; }

    /// <summary>High-water mark of pool bytes checked out during this execution.</summary>
    /// <remarks>Rents raise the current outstanding total and returns lower it;
    /// adopted foreign storage was never checked out, so it never moves the gauge.</remarks>
    public long PeakOutstandingBytes { get; private set; }

    readonly Dictionary<Array, long> outstanding = new();

    long outstandingBytes;

    /// <summary>Memoized run-static alias roots for release probes, built lazily on the first probe so executions without pool-owned releases never pay for the snapshot. Null with StaticRootsBuilt set selects the legacy per-release scan.</summary>
    internal HashSet<Array>? StaticRoots;

    internal bool StaticRootsBuilt;

    public T[] Rent<T>(int length) where T : unmanaged
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        var key = (typeof(T), length);
        long bytes = (long)length * Unsafe.SizeOf<T>();
        if (free.TryGetValue(key, out var stack) && stack.Count > 0)
        {
            Reused++;
            ReusedBytes += bytes;
            var reused = (T[])stack.Pop();
            buffered.Remove(reused);
            TrackRent(reused, bytes);
            return reused;
        }
        var retained = releasedCache?.Take<T>(length);
        if (retained is not null)
        {
            Reused++;
            ReusedBytes += bytes;
            TrackRent(retained, bytes);
            return retained;
        }
        AllocatedNew++;
        AllocatedNewBytes += bytes;
        var fresh = new T[length];
        TrackRent(fresh, bytes);
        return fresh;
    }

    /// <summary>Rents an array zeroed throughout, for kernels with accumulate semantics.</summary>
    /// <remarks>Fresh arrays are already zeroed; reused arrays are cleared here because float MatMul
    /// kernels accumulate into their destination and historically relied on zeroed fresh outputs.</remarks>
    public T[] RentCleared<T>(int length) where T : unmanaged
    {
        int reusedBefore = Reused;
        var rented = Rent<T>(length);
        if (Reused != reusedBefore && rented.Length > 0) Array.Clear(rented, 0, rented.Length);
        return rented;
    }

    /// <summary>Reports whether the array was rented from this pool instance.</summary>
    public bool IsOwned(Array array) => array is not null && owned.TryGetValue(array, out _);

    public void Return(Array array)
    {
        if (array is null) throw new ArgumentNullException(nameof(array));
        if (!array.GetType().IsSZArray) throw new ArgumentException("Only zero-based SZ arrays can be pooled.", nameof(array));
        var element = array.GetType().GetElementType();
        if (element is null) throw new ArgumentException("Only single-dimensional arrays can be pooled.", nameof(array));
        if (buffered.Contains(array)) throw new ArgumentException("Array was already returned to the pool.", nameof(array));
        if (IsOwned(array)) TrackReturn(array);
        var key = (element, array.Length);
        if (!free.TryGetValue(key, out var stack))
        {
            stack = new Stack<Array>();
            free[key] = stack;
        }
        if (stack.Count < MaxBufferedPerShape)
        {
            stack.Push(array);
            buffered.Add(array);
            Returned++;
        }
        else
        {
            Dropped++;
        }
    }

    void TrackRent(Array array, long bytes)
    {
        if (!owned.TryGetValue(array, out _)) owned.Add(array, new Ownership(bytes));
        outstanding[array] = bytes;
        outstandingBytes += bytes;
        if (outstandingBytes > PeakOutstandingBytes) PeakOutstandingBytes = outstandingBytes;
    }

    void TrackReturn(Array array)
    {
        if (outstanding.Remove(array, out var bytes)) outstandingBytes -= bytes;
    }

    /// <summary>End-of-execution transfer of already-free, rent-owned arrays only.</summary>
    internal void RetainReleasedBuffers()
    {
        if (releasedCache is null) return;
        foreach (var entry in free)
            foreach (var array in entry.Value)
                if (owned.TryGetValue(array, out var ownership)) releasedCache.Store(entry.Key, array, ownership.Bytes);
        free.Clear();
        buffered.Clear();
    }
}
