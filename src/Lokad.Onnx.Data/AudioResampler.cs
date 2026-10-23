namespace Lokad.Onnx;

using System;

/// <summary>Managed, centered low-pass conversion of mono float PCM sample rates.</summary>
public static class AudioResampler
{
    /// <summary>Returns owned PCM at the destination rate, without modifying the input.</summary>
    /// <remarks>Rates must be 8000..192000 Hz. The symmetric Kaiser-windowed sinc filter
    /// uses beta 8.6, half length 32 times max(up,down) and cutoff 0.94/max(up,down),
    /// with reduced integer up/down factors. The output has ceil(length*up/down)
    /// samples, starts at the input's time origin, and assumes zero outside the clip.
    /// Short requests compute used coefficients on demand when a full table
    /// would be disproportionate; normalization still scans the complete filter.
    /// Equal rates copy exactly. No clipping or volume normalization is applied.</remarks>
    public static float[] Resample(ReadOnlySpan<float> samples, int sourceRate, int destinationRate)
    {
        if (sourceRate < 8000 || sourceRate > 192000) throw new ArgumentOutOfRangeException(nameof(sourceRate));
        if (destinationRate < 8000 || destinationRate > 192000) throw new ArgumentOutOfRangeException(nameof(destinationRate));
        foreach (float sample in samples)
            if (!float.IsFinite(sample)) throw new ArgumentException("Audio samples must be finite.", nameof(samples));
        if (sourceRate == destinationRate || samples.IsEmpty) return samples.ToArray();
        int gcd = Gcd(sourceRate, destinationRate), up = destinationRate / gcd, down = sourceRate / gcd;
        long length = ((long)samples.Length * up + down - 1) / down;
        if (length > int.MaxValue) throw new ArgumentException("Resampled audio is too long.", nameof(samples));
        var output = new float[(int)length];
        int half = 32 * Math.Max(up, down);
        double cutoff = 0.94 / Math.Max(up, down);
        long tapWork = (long)output.Length * Math.Min(samples.Length, 2 * half / up + 1);
        bool materialize = tapWork >= (2 * half + 1) / 4;
        var coefficients = Filter(half, cutoff, up, materialize, out double gain, out double denominator);
        for (int i = 0; i < output.Length; i++)
        {
            long center = (long)i * down;
            long first = Math.Max(0, (center - half + up - 1) / up);
            long last = Math.Min(samples.Length - 1, (center + half) / up);
            double sum = 0;
            if (coefficients is null)
            {
                for (long input = first; input <= last; input++)
                {
                    double coefficient = Coefficient((int)(center - input * up), half, cutoff, denominator) * gain;
                    sum += samples[(int)input] * coefficient;
                }
            }
            else
            {
                for (long input = first; input <= last; input++)
                    sum += samples[(int)input] * coefficients[(int)(half + center - input * up)];
            }
            float sample = (float)sum;
            if (!float.IsFinite(sample)) throw new ArgumentException("Resampled audio exceeds the float32 range.", nameof(samples));
            output[i] = sample;
        }
        return output;
    }

    static int Gcd(int a, int b)
    {
        while (b != 0) { int remainder = a % b; a = b; b = remainder; }
        return a;
    }

    static double[]? Filter(int half, double cutoff, int up, bool materialize, out double gain, out double denominator)
    {
        var values = materialize ? new double[2 * half + 1] : null;
        denominator = BesselZero(8.6);
        double sum = 0;
        for (int i = -half; i <= half; i++)
        {
            double value = Coefficient(i, half, cutoff, denominator);
            if (values is not null) values[i + half] = value;
            sum += value;
        }
        gain = up / sum;
        if (values is not null)
            for (int i = 0; i < values.Length; i++) values[i] *= gain;
        return values;
    }

    static double Coefficient(int offset, int half, double cutoff, double denominator)
    {
        double x = offset / (double)half;
        double window = BesselZero(8.6 * Math.Sqrt(Math.Max(0, 1 - x * x))) / denominator;
        double sinc = offset == 0 ? cutoff : Math.Sin(Math.PI * cutoff * offset) / (Math.PI * offset);
        return window * sinc;
    }

    static double BesselZero(double x)
    {
        double sum = 1, term = 1, square = x * x / 4;
        for (int k = 1; k < 100; k++)
        {
            term *= square / (k * k);
            sum += term;
            if (term <= sum * 1e-17) break;
        }
        return sum;
    }
}
