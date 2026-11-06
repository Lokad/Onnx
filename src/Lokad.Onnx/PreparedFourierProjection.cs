namespace Lokad.Onnx;

using System;
using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

/// <summary>
/// Private windowed-Fourier recipe. Every live convolution weight bit must
/// still match; preparation never makes an initializer immutable.
/// </summary>
internal sealed class PreparedFourierProjection
{
    readonly float[] expected;
    readonly double[] window;
    readonly FrameFourierTransform transform;
    internal int Length { get; }
    internal long RetainedBytes => expected.LongLength * sizeof(float)
        + window.LongLength * sizeof(double) + transform.RetainedBytes;

    PreparedFourierProjection(int length, float[] expected, double[] window)
    {
        Length = length;
        this.expected = expected;
        this.window = window;
        transform = new FrameFourierTransform(length);
    }

    internal static bool SupportedLength(int length)
    {
        if (length < 16 || length > 1024) return false;
        foreach (int factor in new[] { 2, 3, 5 }) while (length % factor == 0) length /= factor;
        return length == 1;
    }

    internal static long RequiredBytes(int length) => checked(
        2L * (length / 2 + 1) * length * sizeof(float) + (long)length * sizeof(double)
        + 2L * length * Vector<float>.Count * sizeof(float));

    internal static PreparedFourierProjection Hann(int length)
    {
        if (!SupportedLength(length)) throw new ArgumentOutOfRangeException(nameof(length));
        int bins = length / 2 + 1;
        var window = new double[length];
        var expected = new float[2 * bins * length];
        for (int j = 0; j < length; j++) window[j] = .5 - .5 * Math.Cos(2 * Math.PI * j / length);
        for (int k = 0; k < bins; k++)
            for (int j = 0; j < length; j++)
            {
                double phase = 2 * Math.PI * ((k * j) % length) / length;
                expected[k * length + j] = (float)(Math.Cos(phase) * window[j]);
                expected[(bins + k) * length + j] = (float)(-Math.Sin(phase) * window[j]);
            }
        return new(length, expected, window);
    }

    internal static PreparedFourierProjection? FloatWindow(int length, ReadOnlySpan<float> sourceWindow)
    {
        if (!SupportedLength(length) || sourceWindow.Length != length) return null;
        foreach (float value in sourceWindow) if (!float.IsFinite(value) || MathF.Abs(value) > 1) return null;
        int bins = length / 2 + 1;
        var window = new double[length];
        for (int j = 0; j < length; j++) window[j] = sourceWindow[j];
        float coefficient = (float)(2 * Math.PI / length);
        var phases = new float[bins * length];
        for (int k = 0; k < bins; k++)
            for (int j = 0; j < length; j++) phases[k * length + j] = ((k * j) % length) * coefficient;
        var data = new DenseTensor<float>(phases, new[] { bins, length });
        var cosine = CPUExecutionProvider.Cos(data, null);
        var sine = CPUExecutionProvider.Sin(data, null);
        if (cosine.Status != OpStatus.Success || sine.Status != OpStatus.Success) return null;
        var c = ((Tensor<float>)cosine.Outputs[0]).ToArray();
        var s = ((Tensor<float>)sine.Outputs[0]).ToArray();
        var expected = new float[2 * bins * length];
        for (int k = 0; k < bins; k++)
            for (int j = 0; j < length; j++)
            {
                expected[k * length + j] = c[k * length + j] * sourceWindow[j];
                expected[(bins + k) * length + j] = -(s[k * length + j] * sourceWindow[j]);
            }
        return new(length, expected, window);
    }

    internal bool Matches(ReadOnlySpan<float> weights) =>
        MemoryMarshal.AsBytes(weights).SequenceEqual(MemoryMarshal.AsBytes(expected.AsSpan()));

    internal bool SameWeights(PreparedFourierProjection other) => Matches(other.expected);

    internal bool TryWrite(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, int step,
        Span<float> output, IScratchAccountant? scratch)
    {
        if (step < 1 || input.Length < Length || !Matches(weights)
            || input.Overlaps(output) || weights.Overlaps(output)) return false;
        int frames = (input.Length - Length) / step + 1, bins = Length / 2 + 1;
        if ((long)output.Length != 2L * bins * frames) return false;
        foreach (float value in input)
        {
            float magnitude = MathF.Abs(value);
            if (!float.IsFinite(value) || magnitude > 1e20f || magnitude != 0 && magnitude < 1e-30f) return false;
        }
        if (Vector512.IsHardwareAccelerated)
        {
            WriteWide(input, step, frames, output, scratch);
            return true;
        }
        int width = Vector<float>.Count;
        Vector<float>[]? dataR = null, dataI = null, spectrumR = null, spectrumI = null;
        try
        {
            dataR = ArrayPool<Vector<float>>.Shared.Rent(Length);
            dataI = ArrayPool<Vector<float>>.Shared.Rent(Length);
            spectrumR = ArrayPool<Vector<float>>.Shared.Rent(Length);
            spectrumI = ArrayPool<Vector<float>>.Shared.Rent(Length);
            scratch?.AddScratchBytes(4L * Length * width * sizeof(float));
            Span<float> values = stackalloc float[width];
            for (int frame = 0; frame < frames; frame += width)
            {
                int count = Math.Min(width, frames - frame);
                for (int j = 0; j < Length; j++)
                {
                    for (int lane = 0; lane < count; lane++) values[lane] = input[(frame + lane) * step + j];
                    values[count..].Clear();
                    dataR[j] = new Vector<float>(values) * new Vector<float>((float)window[j]);
                    dataI[j] = Vector<float>.Zero;
                }
                transform.Forward(dataR, dataI, spectrumR, spectrumI);
                for (int k = 0; k < bins; k++)
                {
                    int real = k * frames + frame, imaginary = (bins + k) * frames + frame;
                    if (count == width)
                    {
                        spectrumR[k].CopyTo(output.Slice(real, width));
                        spectrumI[k].CopyTo(output.Slice(imaginary, width));
                    }
                    else for (int lane = 0; lane < count; lane++)
                    {
                        output[real + lane] = spectrumR[k][lane];
                        output[imaginary + lane] = spectrumI[k][lane];
                    }
                }
            }
            return true;
        }
        finally
        {
            if (dataR is not null) ArrayPool<Vector<float>>.Shared.Return(dataR);
            if (dataI is not null) ArrayPool<Vector<float>>.Shared.Return(dataI);
            if (spectrumR is not null) ArrayPool<Vector<float>>.Shared.Return(spectrumR);
            if (spectrumI is not null) ArrayPool<Vector<float>>.Shared.Return(spectrumI);
        }
    }
    void WriteWide(ReadOnlySpan<float> input, int step, int frames, Span<float> output, IScratchAccountant? scratch)
    {
        int bins = Length / 2 + 1;
        int width = Vector512<float>.Count;
        Vector512<float>[]? dataR = null, dataI = null, spectrumR = null, spectrumI = null;
        try
        {
            dataR = ArrayPool<Vector512<float>>.Shared.Rent(Length);
            dataI = ArrayPool<Vector512<float>>.Shared.Rent(Length);
            spectrumR = ArrayPool<Vector512<float>>.Shared.Rent(Length);
            spectrumI = ArrayPool<Vector512<float>>.Shared.Rent(Length);
            scratch?.AddScratchBytes(4L * Length * width * sizeof(float));
            Span<float> values = stackalloc float[width];
            for (int frame = 0; frame < frames; frame += width)
            {
                int count = Math.Min(width, frames - frame);
                for (int j = 0; j < Length; j++)
                {
                    for (int lane = 0; lane < count; lane++) values[lane] = input[(frame + lane) * step + j];
                    values[count..].Clear();
                    dataR[j] = Vector512.Create((ReadOnlySpan<float>)values) * Vector512.Create((float)window[j]);
                    dataI[j] = Vector512<float>.Zero;
                }
                transform.Forward(dataR, dataI, spectrumR, spectrumI);
                for (int k = 0; k < bins; k++)
                {
                    int real = k * frames + frame, imaginary = (bins + k) * frames + frame;
                    if (count == width)
                    {
                        spectrumR[k].CopyTo(output.Slice(real, width));
                        spectrumI[k].CopyTo(output.Slice(imaginary, width));
                    }
                    else for (int lane = 0; lane < count; lane++)
                    {
                        output[real + lane] = spectrumR[k][lane];
                        output[imaginary + lane] = spectrumI[k][lane];
                    }
                }
            }
            return;
        }
        finally
        {
            if (dataR is not null) ArrayPool<Vector512<float>>.Shared.Return(dataR);
            if (dataI is not null) ArrayPool<Vector512<float>>.Shared.Return(dataI);
            if (spectrumR is not null) ArrayPool<Vector512<float>>.Shared.Return(spectrumR);
            if (spectrumI is not null) ArrayPool<Vector512<float>>.Shared.Return(spectrumI);
        }
    }
}
