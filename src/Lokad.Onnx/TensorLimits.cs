namespace Lokad.Onnx;

/// <summary>Bound geometry at entry points; numeric tensor payloads keep their dtype domain.</summary>
internal static class TensorLimits
{
    internal const long MaximumGeometryMagnitude = 1L << 48;

    internal static void ValidateGeometry(long value, string name)
    {
        if (value < -MaximumGeometryMagnitude || value > MaximumGeometryMagnitude)
            throw new ArgumentException("Tensor geometry must be within [-2^48, 2^48] (inclusive).", name);
    }

    internal static void ValidateGeometry(double value, string name)
    {
        if (!double.IsFinite(value) || value < -MaximumGeometryMagnitude || value > MaximumGeometryMagnitude)
            throw new ArgumentException("Tensor geometry must be finite and within [-2^48, 2^48] (inclusive).", name);
    }

    internal static long MultiplyDimensions(long product, long dimension, string name)
    {
        if (dimension < 0) throw new ArgumentException("Dimensions must be non-negative.", name);
        if (dimension > MaximumGeometryMagnitude || (dimension != 0 && product > MaximumGeometryMagnitude / dimension))
            throw new ArgumentException("Tensor shape product exceeds the 2^48 geometry limit.", name);
        return product * dimension;
    }

    internal static int IntegerRangeCount(long start, long limit, long delta)
    {
        ValidateGeometry(start, nameof(start));
        ValidateGeometry(limit, nameof(limit));
        ValidateGeometry(delta, nameof(delta));
        if (delta == 0) throw new ArgumentException("Range step cannot be zero.", nameof(delta));
        // Bounded inputs make distance and negation safe in long arithmetic.
        long distance = delta > 0 ? limit - start : start - limit;
        if (distance <= 0) return 0;
        long step = delta > 0 ? delta : -delta;
        long count = (distance - 1) / step + 1;
        if (count > int.MaxValue) throw new ArgumentException("Range count exceeds the maximum tensor length.", nameof(limit));
        return (int)count;
    }

    internal static void ValidateDescriptor(OnnxValueInfo description)
    {
        // Unknown/symbolic extents do not define an element count yet.
        if (description.Dims.Any(d => d < 0) || description.DimParams?.Any(p => p is not null) == true) return;
        ArrayUtilities.ComputeOffsetForReductionLong(description.Dims, 0);
    }
}
