namespace Lokad.Onnx.Backend.Tests;

using System;
using System.Numerics;
using System.Threading;

/// <summary>Managed 80-bin filterbanks for the pyannote Community-1 WeSpeaker embedding model.</summary>
internal static class DenseWeSpeakerReference
{
    public const int SampleRate = 16000;
    public const int MinimumSamples = 400;
    public const int MaximumSamples = 480000;
    public const int MelBins = 80;
    const int WindowSize = 400, HopSize = 160, FourierSize = 512;
    const float Epsilon = 1.1920928955078125e-7f;
    static readonly float[] Window = CreateWindow();
    static readonly float[] MelWeights = CreateMelWeights();
    static readonly Complex[] Roots = CreateRoots();

    /// <summary>Converts 400 through 480000 finite mono 16 kHz PCM samples in [-1,1]
    /// to owned float32 [1,1+(samples-400)/160,80] features.</summary>
    /// <remarks>Uses complete 25 ms frames at 10 ms intervals, frame DC removal,
    /// .97 preemphasis, a nonperiodic Hamming window, a 512-point transform,
    /// Kaldi mel bands, natural log and global feature centering. No padding,
    /// resampling, channel mixing, silence removal or truncation is performed.
    /// Inputs are unchanged and concurrent calls own separate scratch.</remarks>
    public static DenseTensor<float> LogMelFilterbank(ReadOnlySpan<float> samples, int sampleRate, CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();
        if (sampleRate != SampleRate) throw new ArgumentOutOfRangeException(nameof(sampleRate), "WeSpeaker requires mono 16000 Hz PCM.");
        if (samples.Length < MinimumSamples || samples.Length > MaximumSamples)
            throw new ArgumentOutOfRangeException(nameof(samples), "WeSpeaker accepts 400 samples through 30 seconds per request.");
        foreach (float sample in samples)
            if (!float.IsFinite(sample) || sample < -1 || sample > 1)
                throw new ArgumentException("Audio must contain finite normalized samples in [-1,1].", nameof(samples));
        int frames = 1 + (samples.Length - WindowSize) / HopSize;
        var result = new DenseTensor<float>(new[] { 1, frames, MelBins });
        var output = result.Buffer.Span;
        // Preserve frame precision until the FFT; rounding this preprocessing
        // to float can dominate the error of low-energy spectral components.
        var centered = new double[WindowSize];
        var spectrum = new Complex[FourierSize];
        var powers = new float[FourierSize / 2 + 1];
        for (int frame = 0; frame < frames; frame++)
        {
            cancellation.ThrowIfCancellationRequested();
            int start = frame * HopSize;
            double total = 0;
            for (int j = 0; j < WindowSize; j++) { centered[j] = samples[start + j] * 32768f; total += centered[j]; }
            double mean = total / WindowSize;
            for (int j = 0; j < WindowSize; j++) centered[j] -= mean;
            Array.Clear(spectrum, 0, spectrum.Length);
            for (int j = 0; j < WindowSize; j++)
            {
                double previous = .97f * centered[j == 0 ? 0 : j - 1];
                double value = (centered[j] - previous) * Window[j];
                spectrum[j] = new Complex(value, 0);
            }
            Fourier(spectrum);
            for (int k = 0; k < powers.Length; k++)
            {
                float real = (float)spectrum[k].Real, imaginary = (float)spectrum[k].Imaginary;
                float magnitude = (float)Math.Sqrt((double)real * real + (double)imaginary * imaginary);
                powers[k] = magnitude * magnitude;
            }
            for (int mel = 0; mel < MelBins; mel++)
            {
                double energy = 0;
                for (int k = 0; k < FourierSize / 2; k++) energy += (double)powers[k] * MelWeights[mel * (FourierSize / 2) + k];
                output[frame * MelBins + mel] = MathF.Log(Math.Max(Epsilon, (float)energy));
            }
        }
        for (int mel = 0; mel < MelBins; mel++)
        {
            cancellation.ThrowIfCancellationRequested();
            double total = 0;
            for (int frame = 0; frame < frames; frame++) total += output[frame * MelBins + mel];
            float mean = (float)(total / frames);
            for (int frame = 0; frame < frames; frame++) output[frame * MelBins + mel] -= mean;
        }
        return result;
    }

    static void Fourier(Span<Complex> data)
    {
        for (int i = 1, j = 0; i < FourierSize; i++)
        {
            int bit = FourierSize >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) { var value = data[i]; data[i] = data[j]; data[j] = value; }
        }
        for (int size = 2; size <= FourierSize; size <<= 1)
        {
            int half = size / 2, step = FourierSize / size;
            for (int start = 0; start < FourierSize; start += size)
                for (int j = 0; j < half; j++)
                {
                    Complex even = data[start + j], odd = data[start + j + half] * Roots[j * step];
                    data[start + j] = even + odd; data[start + j + half] = even - odd;
                }
        }
    }

    static float[] CreateWindow()
    {
        var result = new float[WindowSize]; float step = (float)(2 * Math.PI / (WindowSize - 1));
        for (int i = 0; i < result.Length; i++) result[i] = MathF.Cos(i * step) * -.46f + .54f;
        return result;
    }

    static float[] CreateMelWeights()
    {
        double low = 1127 * Math.Log(1 + 20.0 / 700), high = 1127 * Math.Log(1 + 8000.0 / 700);
        float delta = (float)((high - low) / (MelBins + 1));
        var result = new float[MelBins * (FourierSize / 2)];
        for (int band = 0; band < MelBins; band++)
        {
            float left = band * delta + (float)low, middle = (band + 1) * delta + (float)low, right = (band + 2) * delta + (float)low;
            for (int k = 0; k < FourierSize / 2; k++)
            {
                float mel = 1127f * MathF.Log(1 + (SampleRate / (float)FourierSize * k) / 700f);
                result[band * (FourierSize / 2) + k] = Math.Max(0, Math.Min((mel - left) / (middle - left), (right - mel) / (right - middle)));
            }
        }
        return result;
    }

    static Complex[] CreateRoots()
    {
        var roots = new Complex[FourierSize / 2];
        for (int i = 0; i < roots.Length; i++) { double angle = -2 * Math.PI * i / FourierSize; roots[i] = new Complex(Math.Cos(angle), Math.Sin(angle)); }
        return roots;
    }
}
