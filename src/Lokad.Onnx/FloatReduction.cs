namespace Lokad.Onnx;

using System;

/// <summary>Fixed four-lane float reduction, independent of managed buffer address.</summary>
internal static class FloatReduction
{
    internal static float Sum(ReadOnlySpan<float> row, int alignedStart, float mean, bool squared)
    {
        if (row.Length == 0) return 0;
        int alignedEnd = alignedStart + (row.Length - alignedStart) / 4 * 4;
        if (alignedEnd == alignedStart)
        {
            float scalar = Value(row[0], mean, squared);
            for (int i = 1; i < row.Length; i++) scalar += Value(row[i], mean, squared);
            return scalar;
        }
        int index = alignedStart;
        float a = Value(row[index], mean, squared);
        float b = Value(row[index + 1], mean, squared);
        float c = Value(row[index + 2], mean, squared);
        float d = Value(row[index + 3], mean, squared);
        index += 4;
        if (index < alignedEnd)
        {
            float e = Value(row[index], mean, squared);
            float f = Value(row[index + 1], mean, squared);
            float g = Value(row[index + 2], mean, squared);
            float h = Value(row[index + 3], mean, squared);
            index += 4;
            for (; index + 8 <= alignedEnd; index += 8)
            {
                a += Value(row[index], mean, squared);
                b += Value(row[index + 1], mean, squared);
                c += Value(row[index + 2], mean, squared);
                d += Value(row[index + 3], mean, squared);
                e += Value(row[index + 4], mean, squared);
                f += Value(row[index + 5], mean, squared);
                g += Value(row[index + 6], mean, squared);
                h += Value(row[index + 7], mean, squared);
            }
            a += e; b += f; c += g; d += h;
            if (index < alignedEnd)
            {
                a += Value(row[index], mean, squared);
                b += Value(row[index + 1], mean, squared);
                c += Value(row[index + 2], mean, squared);
                d += Value(row[index + 3], mean, squared);
            }
        }
        float sum = (a + c) + (b + d);
        for (int i = 0; i < alignedStart; i++) sum += Value(row[i], mean, squared);
        for (int i = alignedEnd; i < row.Length; i++) sum += Value(row[i], mean, squared);
        return sum;
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    static float Value(float value, float mean, bool squared)
    {
        if (!squared) return value;
        float delta = value - mean;
        return delta * delta;
    }

}
