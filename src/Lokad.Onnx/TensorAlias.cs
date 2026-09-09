namespace Lokad.Onnx;

using System;
using System.Runtime.InteropServices;

/// <summary>
/// Backing-memory overlap probes backing the destination alias rules.
/// Each tensor resolves to its root storage array plus its exact touched
/// range by sampling the mapping over the shape corners (offsets are linear
/// in coordinates, so extrema sit at corners). Ranges are conservative only
/// when storage is unresolvable or rank exceeds the corner budget, in which
/// case callers keep their previous behavior.
/// </summary>
internal static class TensorAlias
{
    const int MaxCornerRank = 8;

    public static bool SharesBackingMemory<T>(Tensor<T>? a, Tensor<T>? b) where T : unmanaged
    {
        if (a is null || b is null) return false;
        if (ReferenceEquals(a, b)) return true;
        if (!TryGetTouchedRange(a, out var arrA, out var startA, out var lenA)) return false;
        if (!TryGetTouchedRange(b, out var arrB, out var startB, out var lenB)) return false;
        if (!ReferenceEquals(arrA, arrB)) return false;
        if (lenA <= 0 || lenB <= 0) return false;
        return startA < startB + lenB && startB < startA + lenA;
    }

    static bool TryGetTouchedRange<T>(Tensor<T> t, out Array? array, out long start, out long length) where T : unmanaged
    {
        array = null;
        start = 0;
        length = 0;
        Memory<T> storage;
        try
        {
            storage = t.Storage;
        }
        catch (Exception)
        {
            return false;
        }
        if (!MemoryMarshal.TryGetArray((ReadOnlyMemory<T>)storage, out var seg) || seg.Array is null) return false;
        if (t.Length == 0)
        {
            array = seg.Array;
            start = seg.Offset;
            length = 0;
            return true;
        }
        int rank = t.Rank;
        var dims = t.Dimensions;
        if (rank > MaxCornerRank || dims.Length != rank)
        {
            array = seg.Array;
            start = seg.Offset;
            length = seg.Count;
            return true;
        }
        var coords = new int[rank];
        long lo = long.MaxValue;
        long hi = long.MinValue;
        int corners = 1 << rank;
        for (int m = 0; m < corners; m++)
        {
            for (int d = 0; d < rank; d++) coords[d] = ((m >> d) & 1) == 0 ? 0 : dims[d] - 1;
            long off = (long)seg.Offset + t.GetStorageIndex(coords);
            if (off < lo) lo = off;
            if (off > hi) hi = off;
        }
        array = seg.Array;
        start = lo;
        length = hi - lo + 1;
        return true;
    }
}
