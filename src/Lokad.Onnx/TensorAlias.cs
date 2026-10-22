namespace Lokad.Onnx;

using System;
using System.Runtime.InteropServices;

/// <summary>
/// Backing-memory overlap probes backing the destination alias rules.
/// Distinct root storage arrays never overlap. Shared arrays need their exact
/// touched ranges, sampled at shape corners after composing every parent mapping.
/// Unresolvable nonempty storage conservatively reports possible overlap;
/// high-rank layouts use their complete backing-storage range.
/// </summary>
internal static class TensorAlias
{
    const int MaxCornerRank = 8;

    public static bool SharesBackingMemory<T>(Tensor<T>? a, Tensor<T>? b) where T : unmanaged
    {
        if (a is null || b is null) return false;
        if (a.Length == 0 || b.Length == 0) return false;
        if (ReferenceEquals(a, b)) return true;
        if (!TryGetStorage(a, out var arrA, out var startA, out var lenA, out bool packedA)) return true;
        if (!TryGetStorage(b, out var arrB, out var startB, out var lenB, out bool packedB)) return true;
        if (!ReferenceEquals(arrA, arrB)) return false;
        GetTouchedRange(a, packedA, ref startA, ref lenA);
        GetTouchedRange(b, packedB, ref startB, ref lenB);
        if (lenA <= 0 || lenB <= 0) return false;
        return startA < startB + lenB && startB < startA + lenA;
    }

    static bool TryGetStorage<T>(Tensor<T> t, out Array? array, out long start, out long length, out bool packed) where T : unmanaged
    {
        array = null;
        start = 0;
        length = 0;
        packed = false;
        if (t is Tensor<float> typed && OwnedPackedTensor.FindStorage(typed) is { } owner)
        {
            array = owner.PackedArray;
            length = owner.PackedArray.Length;
            packed = true;
            return true;
        }
        Memory<T> storage;
        try
        {
            storage = t.Storage;
        }
        catch (Exception ex) when (!Runtime.IsFatal(ex))
        {
            return false;
        }
        if (!MemoryMarshal.TryGetArray((ReadOnlyMemory<T>)storage, out var seg) || seg.Array is null) return false;
        array = seg.Array;
        start = seg.Offset;
        length = seg.Count;
        return true;
    }

    static void GetTouchedRange<T>(Tensor<T> t, bool packed, ref long start, ref long length) where T : unmanaged
    {
        if (packed) return;
        int rank = t.Rank;
        var dims = t.Dimensions;
        if (rank > MaxCornerRank || dims.Length != rank)
            return;
        var coords = new int[rank];
        long lo = long.MaxValue;
        long hi = long.MinValue;
        int corners = 1 << rank;
        for (int m = 0; m < corners; m++)
        {
            for (int d = 0; d < rank; d++) coords[d] = ((m >> d) & 1) == 0 ? 0 : dims[d] - 1;
            long off = start + t.GetStorageIndex(coords);
            if (off < lo) lo = off;
            if (off > hi) hi = off;
        }
        start = lo;
        length = hi - lo + 1;
    }
}
