namespace Lokad.Onnx;

using System;
using System.Collections.Generic;

/// <summary>Per-execution pool of exact-size backing arrays for dense tensor outputs.</summary>
/// <remarks>Rent and return are single-threaded by construction: only graph dispatch rents and only the
/// owning execution releases at recorded last uses, while parallel kernel workers merely write disjoint
/// spans. Kernels stay destination-based and never see the pool. Misses allocate; returns beyond a small
/// per-shape cap are dropped. Never static: one pool per graph execution.</remarks>
public sealed class TensorBufferPool
{
    const int MaxBufferedPerShape = 32;

    readonly Dictionary<(Type, int), Stack<Array>> free = new();

    public int AllocatedNew { get; private set; }

    public int Reused { get; private set; }

    public int Returned { get; private set; }

    public int Dropped { get; private set; }

    public long AllocatedNewBytes { get; private set; }

    public long ReusedBytes { get; private set; }

    public T[] Rent<T>(int length)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        var key = (typeof(T), length);
        long bytes = (long)length * System.Runtime.InteropServices.Marshal.SizeOf<T>();
        if (free.TryGetValue(key, out var stack) && stack.Count > 0)
        {
            Reused++;
            ReusedBytes += bytes;
            return (T[])stack.Pop();
        }
        AllocatedNew++;
        AllocatedNewBytes += bytes;
        return new T[length];
    }

    /// <summary>Rents an array zeroed throughout, for kernels with accumulate semantics.</summary>
    /// <remarks>Fresh arrays are already zeroed; reused arrays are cleared here because float MatMul
    /// kernels accumulate into their destination and historically relied on zeroed fresh outputs.</remarks>
    public T[] RentCleared<T>(int length)
    {
        int reusedBefore = Reused;
        var rented = Rent<T>(length);
        if (Reused != reusedBefore && rented.Length > 0) Array.Clear(rented, 0, rented.Length);
        return rented;
    }

    public void Return(Array array)
    {
        if (array is null) throw new ArgumentNullException(nameof(array));
        var element = array.GetType().GetElementType();
        if (element is null) throw new ArgumentException("Only single-dimensional arrays can be pooled.", nameof(array));
        var key = (element, array.Length);
        if (!free.TryGetValue(key, out var stack))
        {
            stack = new Stack<Array>();
            free[key] = stack;
        }
        if (stack.Count < MaxBufferedPerShape)
        {
            stack.Push(array);
            Returned++;
        }
        else
        {
            Dropped++;
        }
    }
}
