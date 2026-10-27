namespace Lokad.Onnx;

/// <summary>
/// ORT's serial Int64 reduction policy: compensate double accumulation, then
/// truncate/saturate the result. Input conversion can still round above 2^53.
/// </summary>
internal struct Int64ReductionAccumulator
{
    double total;
    double correction;

    public void Add(long value) => AddConverted(value);

    public void AddSquare(long value)
    {
        double converted = value;
        AddConverted(converted * converted);
    }

    void AddConverted(double value)
    {
        double adjusted = value - correction;
        double next = total + adjusted;
        correction = (next - total) - adjusted;
        total = next;
    }

    public readonly long Sum => Saturate(total);

    public readonly long Mean(int count) => count == 0 ? 0 : Saturate(total / count);

    static long Saturate(double value) => value >= 9223372036854775808.0 ? long.MaxValue
        : value <= -9223372036854775808.0 ? long.MinValue : (long)value;
}
