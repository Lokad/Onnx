namespace Lokad.Onnx;

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

/// <summary>Counter snapshot marking the start of one execution run on a shared pool.</summary>
internal readonly record struct PoolMark(int AllocatedNew, int Reused, int Returned, int Dropped, long AllocatedNewBytes, long ReusedBytes);

/// <summary>Per-run pool activity: counter deltas between run marks plus the run peak.</summary>
internal readonly record struct PoolRunDelta(int AllocatedNew, int Reused, int Returned, int Dropped, long AllocatedNewBytes, long ReusedBytes, long PeakOutstandingBytes);

/// <summary>Pool of exact-size backing arrays for dense tensor outputs, shared across executions of one plan.</summary>
/// <remarks>
/// Rent and return are serialized by a leaf lock: executions sharing a plan may run concurrently, while parallel kernel workers merely write disjoint
/// spans. Kernels stay destination-based and never see the pool. Misses allocate; returns beyond a small
/// per-shape cap are dropped. Never static: one pool per prepared plan, shared by reference with its contexts.
/// Ownership: every array ever handed out by Rent on this pool instance is remembered, except entries forgotten at run boundaries (see BeginRun).
/// Release logic must only return arrays for which IsOwned reports true, so caller inputs, model constants,
/// attribute tensors and unrelated fresh buffers are never adopted. See C01.
/// Metrics: AllocatedNew/AllocatedNewBytes count fresh rents (GC allocation); Reused/ReusedBytes count
/// rents served from previously returned storage (pool-served bytes, not new GC bytes); PeakOutstandingBytes tracks the running high-water mark of checked-out bytes (adoptions excluded, drops released); Returned counts
/// dead buffers adopted into the pool; Dropped counts returns discarded by the per-shape cap. None is a
/// substitute for the others: GC bytes come from the collector, live payload from tensor shapes, scratch
/// from temporary kernel buffers, and pool-served bytes from these counters. Per-run Last* reporting uses
/// mark deltas, which are exact for serial executions and approximate under concurrency.
/// Return contract: only single-dimensional (SZ) arrays are pooled. Null, multidimensional, and duplicate
/// (already buffered) arrays are rejected with an exception. Foreign arrays (never rented from this pool)
/// are adopted for reuse; release logic still only returns owned storage.
/// </remarks>
public sealed class TensorBufferPool
{
    const int MaxBufferedPerShape = 32;

    readonly object sync = new();

    readonly Dictionary<(Type, int), Stack<Array>> free = new();

    readonly HashSet<Array> owned = new();

    readonly HashSet<Array> buffered = new();

    public int AllocatedNew { get; private set; }

    public int Reused { get; private set; }

    public int Returned { get; private set; }

    public int Dropped { get; private set; }

    public long AllocatedNewBytes { get; private set; }

    public long ReusedBytes { get; private set; }

    /// <summary>Running high-water mark of checked-out pool bytes.</summary>
    /// <remarks>Rents raise the current outstanding total and returns lower it;
    /// adopted foreign storage was never checked out, so it never moves the gauge.</remarks>
    public long PeakOutstandingBytes { get; private set; }

    readonly Dictionary<Array, long> outstanding = new();

    long outstandingBytes;

    long peakWindow;

    /// <summary>Starts one execution run: forgets ownership and gauges for storage that outlived previous runs, keeping only free-stack contents.</summary>
    /// <remarks>Arrays checked out past a run boundary (for example caller-held outputs) can never re-enter circulation: without ownership their later release probes refuse them, so recycling them is impossible. Under concurrent executions this conservatively also forgets the other run live set, which only costs reuse, never correctness.</remarks>
    internal PoolMark BeginRun()
    {
        lock (sync)
        {
            owned.Clear();
            foreach (var kv in free)
                foreach (var arr in kv.Value)
                    owned.Add(arr);
            outstanding.Clear();
            outstandingBytes = 0;
            peakWindow = 0;
            return new PoolMark(AllocatedNew, Reused, Returned, Dropped, AllocatedNewBytes, ReusedBytes);
        }
    }

    /// <summary>Ends one execution run, reporting counter deltas against its mark plus the run peak.</summary>
    internal PoolRunDelta EndRun(PoolMark mark)
    {
        lock (sync)
        {
            return new PoolRunDelta(
                AllocatedNew - mark.AllocatedNew,
                Reused - mark.Reused,
                Returned - mark.Returned,
                Dropped - mark.Dropped,
                AllocatedNewBytes - mark.AllocatedNewBytes,
                ReusedBytes - mark.ReusedBytes,
                peakWindow);
        }
    }

    public T[] Rent<T>(int length) where T : unmanaged
    {
        lock (sync)
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
                owned.Add(reused);
                TrackRent(reused, bytes);
                return reused;
            }
            AllocatedNew++;
            AllocatedNewBytes += bytes;
            var fresh = new T[length];
            owned.Add(fresh);
            TrackRent(fresh, bytes);
            return fresh;
        }
    }

    /// <summary>Rents an array zeroed throughout, for kernels with accumulate semantics.</summary>
    /// <remarks>Fresh arrays are already zeroed; reused arrays are cleared here because float MatMul
    /// kernels accumulate into their destination and historically relied on zeroed fresh outputs.</remarks>
    public T[] RentCleared<T>(int length) where T : unmanaged
    {
        lock (sync)
        {
            int reusedBefore = Reused;
            var rented = Rent<T>(length);
            if (Reused != reusedBefore && rented.Length > 0) Array.Clear(rented, 0, rented.Length);
            return rented;
        }
    }

    /// <summary>Reports whether the array was rented from this pool instance.</summary>
    public bool IsOwned(Array array)
    {
        lock (sync)
        {
            return array is not null && owned.Contains(array);
        }
    }

    public void Return(Array array)
    {
        lock (sync)
        {
            if (array is null) throw new ArgumentNullException(nameof(array));
            if (array.Rank != 1) throw new ArgumentException("Only single-dimensional arrays can be pooled.", nameof(array));
            var element = array.GetType().GetElementType();
            if (element is null) throw new ArgumentException("Only single-dimensional arrays can be pooled.", nameof(array));
            if (buffered.Contains(array)) throw new ArgumentException("Array was already returned to the pool.", nameof(array));
            if (owned.Contains(array)) TrackReturn(array);
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
    }

    void TrackRent(Array array, long bytes)
    {
        outstanding[array] = bytes;
        outstandingBytes += bytes;
        if (outstandingBytes > PeakOutstandingBytes) PeakOutstandingBytes = outstandingBytes;
        if (outstandingBytes > peakWindow) peakWindow = outstandingBytes;
    }

    void TrackReturn(Array array)
    {
        if (outstanding.Remove(array, out var bytes)) outstandingBytes -= bytes;
    }
}

