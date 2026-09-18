namespace Lokad.Onnx;

/// <summary>
/// Bounded storage for arrays already released by a completed execution.
/// Never scans bindings or adopts live values. Each serialized graph/context
/// owns its own cache; explicit contexts never share one with their parent.
/// </summary>
internal sealed class ReleasedBufferCache
{
    internal const long DefaultByteLimit = 128L * 1024 * 1024;
    internal const int DefaultCountLimit = 256;

    readonly long byteLimit;
    readonly int countLimit;
    readonly Dictionary<(Type, int), Stack<Array>> free = new();

    internal long Bytes { get; private set; }
    internal int Count { get; private set; }
    internal int Shapes => free.Count;

    internal ReleasedBufferCache(long byteLimit, int countLimit)
    {
        if (byteLimit < 0) throw new ArgumentOutOfRangeException(nameof(byteLimit));
        if (countLimit < 0) throw new ArgumentOutOfRangeException(nameof(countLimit));
        this.byteLimit = byteLimit;
        this.countLimit = countLimit;
    }

    internal T[]? Take<T>(int length) where T : unmanaged
    {
        var key = (typeof(T), length);
        if (!free.TryGetValue(key, out var stack)) return null;
        var result = (T[])stack.Pop();
        if (stack.Count == 0) free.Remove(key);
        Bytes -= (long)length * System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        Count--;
        return result;
    }

    // The pool supplies exact managed payload bytes recorded by Rent<T>.
    // Drop excess returns; no eviction or retention of oversized arrays.
    internal void Store((Type, int) key, Array array, long bytes)
    {
        if (bytes > byteLimit - Bytes || Count >= countLimit) return;
        if (!free.TryGetValue(key, out var stack))
        {
            stack = new Stack<Array>();
            free.Add(key, stack);
        }
        if (stack.Count >= TensorBufferPool.MaxBufferedPerShape) return;
        stack.Push(array);
        Bytes += bytes;
        Count++;
    }

    internal void Clear()
    {
        free.Clear();
        Bytes = 0;
        Count = 0;
    }
}
