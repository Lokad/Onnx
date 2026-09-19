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
        var coefficients = Filter(half, 0.94 / Math.Max(up, down), up);
        for (int i = 0; i < output.Length; i++)
        {
            long center = (long)i * down;
            long first = Math.Max(0, (center - half + up - 1) / up);
            long last = Math.Min(samples.Length - 1, (center + half) / up);
            double sum = 0;
            for (long input = first; input <= last; input++)
                sum += samples[(int)input] * coefficients[(int)(half + center - input * up)];
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

    static double[] Filter(int half, double cutoff, int up)
    {
        const double beta = 8.6;
        var values = new double[2 * half + 1];
        double denominator = BesselZero(beta), sum = 0;
        for (int i = -half; i <= half; i++)
        {
            double x = i / (double)half;
            double window = BesselZero(beta * Math.Sqrt(Math.Max(0, 1 - x * x))) / denominator;
            double sinc = i == 0 ? cutoff : Math.Sin(Math.PI * cutoff * i) / (Math.PI * i);
            values[i + half] = window * sinc;
            sum += values[i + half];
        }
        for (int i = 0; i < values.Length; i++) values[i] *= up / sum;
        return values;
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
