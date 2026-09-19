namespace Lokad.Onnx;

using System;
using System.Numerics;
using static OpResult;

public partial class CPUExecutionProvider
{
    /// <summary>Natural logarithm with IEEE floating-point domain behavior.</summary>
    public static OpResult Log(ITensor? data, ExecutionOptions? options)
    {
        var op = OpType.Log;
        if (data is null) return MissingInput(op, nameof(data));
        (options ?? ExecutionOptions.Default).Validated();
        Profiler.StartOpStage(OpStage.Math);
        return data.ElementType switch
        {
            TensorElementType.Float => Success(op, LogCore((Tensor<float>)data)),
            TensorElementType.Double => Success(op, LogCore((Tensor<double>)data)),
            _ => InputTypeNotSupported(op, nameof(data), data)
        };
    }

    static DenseTensor<T> LogCore<T>(Tensor<T> data) where T : unmanaged, IFloatingPointIeee754<T>
    {
        var values = data.ToArray();
        for (int i = 0; i < values.Length; i++) values[i] = T.Log(values[i]);
        return new DenseTensor<T>(values, data.Dimensions.ToArray());
    }

    /// <summary>Unnormalized short-time Fourier transform, with owned real/imaginary output.</summary>
    public static OpResult STFT(ITensor? signal, ITensor? frameStep, ITensor? window, ITensor? frameLength, int? onesided, ExecutionOptions? options)
    {
        var op = OpType.STFT;
        if (signal is null) return MissingInput(op, nameof(signal));
        if (frameStep is null) return MissingInput(op, nameof(frameStep));
        (options ?? ExecutionOptions.Default).Validated();
        if (signal.ElementType is not (TensorElementType.Float or TensorElementType.Double))
            return InputTypeNotSupported(op, nameof(signal), signal);
        if (signal.Rank is not (2 or 3) || (signal.Rank == 3 && signal.Dims[2] is not (1 or 2)))
            return WrongInputShape(op, nameof(signal), signal, "STFT requires [batch,samples], [batch,samples,1] or [batch,samples,2].");
        int components = signal.Rank == 2 ? 1 : signal.Dims[2];
        int one = onesided ?? 1;
        if (one is not (0 or 1) || (one == 1 && components == 2))
            return AttributeNotSupported(op, nameof(onesided), one.ToString(), "onesided must be 0 or 1 and must be 0 for complex input.");
        if (frameStep.ElementType is not (TensorElementType.Int32 or TensorElementType.Int64))
            return WrongInputType(op, nameof(frameStep), "Frame step must be int32 or int64.", frameStep);
        if (frameStep.Length != 1) return WrongInputShape(op, nameof(frameStep), frameStep, "Frame step must contain one value.");
        long step = frameStep is Tensor<long> longStep ? longStep.GetValue(0) : ((Tensor<int>)frameStep).GetValue(0);
        if (step <= 0) return WrongInputShape(op, nameof(frameStep), frameStep, "Frame step must be positive.");
        if (window is not null)
        {
            if (window.ElementType != signal.ElementType) return WrongInputType(op, nameof(window), signal.ElementType, window);
            if (window.Rank != 1) return WrongInputShape(op, nameof(window), window, "Window must be rank one.");
        }
        long length = window?.Length ?? signal.Dims[1];
        if (frameLength is not null)
        {
            if (frameLength.ElementType is not (TensorElementType.Int32 or TensorElementType.Int64))
                return WrongInputType(op, nameof(frameLength), "Frame length must be int32 or int64.", frameLength);
            if (frameLength.Length != 1) return WrongInputShape(op, nameof(frameLength), frameLength, "Frame length must contain one value.");
            length = frameLength is Tensor<long> longLength ? longLength.GetValue(0) : ((Tensor<int>)frameLength).GetValue(0);
            if (window is not null && window.Length != length)
                return WrongInputShape(op, nameof(window), window, "Window and frame length must agree.");
        }
        if (length <= 0 || length > signal.Dims[1])
            return WrongInputShape(op, nameof(signal), signal, "Frame length must be positive and no greater than the signal length.");
        long frames = (signal.Dims[1] - length) / step + 1;
        long bins = one == 1 ? length / 2 + 1 : length;
        // Check division first: batch*frames*bins*2 may otherwise overflow int64.
        long maximum = int.MaxValue / 2 / bins;
        if (signal.Dims[0] != 0 && frames > maximum / signal.Dims[0])
            return WrongInputShape(op, nameof(signal), signal, "STFT output length must fit int32.");
        int[] shape = [signal.Dims[0], (int)frames, (int)bins, 2];
        Profiler.StartOpStage(OpStage.Math);
        return signal.ElementType == TensorElementType.Float
            ? Success(op, StftCore((Tensor<float>)signal, (Tensor<float>?)window, components, (int)length, step, shape))
            : Success(op, StftCore((Tensor<double>)signal, (Tensor<double>?)window, components, (int)length, step, shape));
    }

    static DenseTensor<T> StftCore<T>(Tensor<T> signal, Tensor<T>? window, int components, int length, long step, int[] shape)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        var output = DenseTensor<T>.OfShape(shape);
        if (output.Length == 0) return output;
        var samples = signal.ToArray();
        var weights = window?.ToArray();
        var spectrum = new Complex[length];
        var scratch = new Complex[length];
        var roots = new Complex[length];
        for (int i = 0; i < length; i++)
        {
            double angle = -2 * Math.PI * i / length;
            roots[i] = new Complex(Math.Cos(angle), Math.Sin(angle));
        }
        var destination = output.Buffer.Span;
        int position = 0;
        for (int batch = 0; batch < shape[0]; batch++)
            for (int frame = 0; frame < shape[1]; frame++)
            {
                long origin = (long)batch * signal.Dimensions[1] + frame * step;
                for (int i = 0; i < length; i++)
                {
                    int input = checked((int)((origin + i) * components));
                    double weight = weights is null ? 1 : double.CreateChecked(weights[i]);
                    double real = double.CreateChecked(samples[input]) * weight;
                    double imaginary = components == 2 ? double.CreateChecked(samples[input + 1]) * weight : 0;
                    spectrum[i] = new Complex(real, imaginary);
                }
                ForwardSpectrum(spectrum, scratch, roots);
                for (int bin = 0; bin < shape[2]; bin++)
                {
                    destination[position++] = T.CreateTruncating(spectrum[bin].Real);
                    destination[position++] = T.CreateTruncating(spectrum[bin].Imaginary);
                }
            }
        return output;
    }

    static void ForwardSpectrum(Complex[] values, Complex[] scratch, Complex[] roots)
    {
        int n = values.Length;
        if ((n & (n - 1)) != 0)
        {
            // Direct DFT keeps arbitrary lengths correct. The audio export uses
            // the radix-two path below; neither path allocates per frame.
            for (int k = 0; k < n; k++)
            {
                Complex sum = Complex.Zero;
                for (int j = 0; j < n; j++) sum += values[j] * roots[(int)((long)j * k % n)];
                scratch[k] = sum;
            }
            scratch.CopyTo(values, 0);
            return;
        }
        for (int i = 1, reversed = 0; i < n; i++)
        {
            int bit = n >> 1;
            while ((reversed & bit) != 0) { reversed ^= bit; bit >>= 1; }
            reversed ^= bit;
            if (i < reversed) (values[i], values[reversed]) = (values[reversed], values[i]);
        }
        for (int width = 2; width <= n;)
        {
            int half = width / 2, stride = n / width;
            for (int start = 0; start < n; start += width)
                for (int j = 0; j < half; j++)
                {
                    Complex even = values[start + j], odd = values[start + j + half] * roots[j * stride];
                    values[start + j] = even + odd;
                    values[start + j + half] = even - odd;
                }
            if (width == n) break;
            width *= 2;
        }
    }
}
