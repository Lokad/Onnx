namespace Lokad.Onnx;

using System;
using System.Numerics;

/// <summary>Audio features for the 128-bin, 30-second Whisper Large V3/Turbo encoder.</summary>
public static class WhisperAudio
{
    public const int SampleRate = 16000;
    public const int SampleCount = 480000;
    public const int MelBins = 128;
    public const int Frames = 3000;
    const int WindowSize = 400;
    const int HopSize = 160;

    static readonly double[] Window = CreateWindow();
    static readonly Complex[] Roots = CreateRoots();
    static readonly MelBand[] Bands = CreateBands();

    /// <summary>
    /// Converts mono PCM samples to float32 [1,128,3000] log-mel features.
    /// Samples normally lie in [-1,1]. All supplied samples must be finite.
    /// Short clips are right-padded with silence; longer clips are truncated to
    /// 30 seconds. This method does not resample, mix channels or normalize volume.
    /// </summary>
    /// <remarks>
    /// Uses a periodic Hann window, centered reflection padding, an unnormalized
    /// 400-point Fourier transform, and Slaney mel scale/area normalization.
    /// Log10 powers are floored at 1e-10, clipped to eight below the clip maximum,
    /// then transformed by (value + 4) / 4. Empty input represents silence.
    /// Scratch and returned storage belong to this call; input storage is untouched.
    /// </remarks>
    public static DenseTensor<float> LogMelSpectrogram(ReadOnlySpan<float> samples, int sampleRate)
    {
        if (sampleRate != SampleRate)
            throw new ArgumentOutOfRangeException(nameof(sampleRate), "Whisper features require mono 16000 Hz audio.");
        for (int i = 0; i < samples.Length; i++)
            if (!float.IsFinite(samples[i]))
                throw new ArgumentException("Audio samples must be finite.", nameof(samples));

        int used = Math.Min(samples.Length, SampleCount);
        var windowed = new double[WindowSize];
        var spectrum = new Complex[WindowSize];
        var powers = new double[WindowSize / 2 + 1];
        var result = new DenseTensor<float>(new[] { 1, MelBins, Frames });
        var output = result.Buffer.Span;
        float maximum = float.NegativeInfinity;
        for (int frame = 0; frame < Frames; frame++)
        {
            bool nonzero = false;
            for (int j = 0; j < WindowSize; j++)
            {
                int index = frame * HopSize + j - WindowSize / 2;
                // Reflection is relative to the entire padded/truncated clip,
                // not the original short input. Neither endpoint is repeated.
                if (index < 0) index = -index;
                if (index >= SampleCount) index = 2 * SampleCount - 2 - index;
                double value = index < used ? samples[index] * Window[j] : 0;
                windowed[j] = value;
                nonzero |= value != 0;
            }
            if (nonzero)
            {
                Fourier(windowed, 0, 1, spectrum);
                for (int k = 0; k < powers.Length; k++)
                    powers[k] = spectrum[k].Real * spectrum[k].Real + spectrum[k].Imaginary * spectrum[k].Imaginary;
            }
            for (int mel = 0; mel < MelBins; mel++)
            {
                double power = 0;
                if (nonzero)
                {
                    MelBand band = Bands[mel];
                    for (int k = 0; k < band.Weights.Length; k++)
                        power += powers[band.Start + k] * band.Weights[k];
                }
                float value = (float)Math.Log10(Math.Max(1e-10, power));
                output[mel * Frames + frame] = value;
                maximum = Math.Max(maximum, value);
            }
        }
        float floor = maximum - 8;
        for (int i = 0; i < output.Length; i++) output[i] = (Math.Max(output[i], floor) + 4) / 4;
        return result;
    }

    // A 400-point transform factors entirely into radices two and five.
    // Each recursive child writes a disjoint contiguous segment. The small
    // temporary below preserves child values during the in-place combination.
    // No frame/recursion allocates heap storage, and FFT length stays exactly 400.
    static void Fourier(ReadOnlySpan<double> input, int offset, int stride, Span<Complex> output)
    {
        int length = output.Length;
        if (length == 1) { output[0] = new Complex(input[offset], 0); return; }
        int radix = length % 2 == 0 ? 2 : 5;
        int childLength = length / radix;
        for (int j = 0; j < radix; j++)
            Fourier(input, offset + j * stride, stride * radix, output.Slice(j * childLength, childLength));
        Span<Complex> values = stackalloc Complex[5];
        int rootStep = WindowSize / length;
        for (int k = 0; k < childLength; k++)
        {
            for (int j = 0; j < radix; j++) values[j] = output[j * childLength + k];
            for (int q = 0; q < radix; q++)
            {
                int frequency = k + q * childLength;
                Complex sum = values[0];
                for (int j = 1; j < radix; j++)
                    sum += values[j] * Roots[j * frequency * rootStep % WindowSize];
                output[frequency] = sum;
            }
        }
    }

    static double[] CreateWindow()
    {
        var result = new double[WindowSize];
        for (int i = 0; i < result.Length; i++) result[i] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * i / WindowSize);
        return result;
    }

    static Complex[] CreateRoots()
    {
        var result = new Complex[WindowSize];
        for (int i = 0; i < result.Length; i++)
        {
            double angle = -2 * Math.PI * i / WindowSize;
            result[i] = new Complex(Math.Cos(angle), Math.Sin(angle));
        }
        return result;
    }

    sealed class MelBand
    {
        public readonly int Start;
        public readonly double[] Weights;
        public MelBand(int start, double[] weights) { Start = start; Weights = weights; }
    }

    static MelBand[] CreateBands()
    {
        // Slaney's linear scale below 1 kHz, logarithmic scale above it.
        double logStep = Math.Log(6.4) / 27;
        double maximumMel = 15 + Math.Log((SampleRate / 2.0) / 1000) / logStep;
        var edges = new double[MelBins + 2];
        for (int i = 0; i < edges.Length; i++)
        {
            double mel = i * maximumMel / (MelBins + 1);
            edges[i] = mel < 15 ? mel * 200 / 3 : 1000 * Math.Exp((mel - 15) * logStep);
        }
        var result = new MelBand[MelBins];
        for (int mel = 0; mel < result.Length; mel++)
        {
            int first = Math.Max(0, (int)Math.Ceiling(edges[mel] * WindowSize / SampleRate));
            int last = Math.Min(WindowSize / 2, (int)Math.Floor(edges[mel + 2] * WindowSize / SampleRate));
            var weights = new double[Math.Max(0, last - first + 1)];
            for (int k = 0; k < weights.Length; k++)
            {
                double hz = (first + k) * (double)SampleRate / WindowSize;
                double lower = (hz - edges[mel]) / (edges[mel + 1] - edges[mel]);
                double upper = (edges[mel + 2] - hz) / (edges[mel + 2] - edges[mel + 1]);
                weights[k] = Math.Max(0, Math.Min(lower, upper)) * 2 / (edges[mel + 2] - edges[mel]);
            }
            result[mel] = new MelBand(first, weights);
        }
        return result;
    }
}
