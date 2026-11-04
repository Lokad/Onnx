namespace Lokad.Onnx;

using System;
using System.Numerics;
using static OpResult;

public partial class CPUExecutionProvider
{
    /// <summary>One-dimensional transposed convolution with grouped, overlapping kernels.</summary>
    public static OpResult ConvTranspose(ITensor? input, ITensor? weights, ITensor? bias, string? autoPad,
        int[]? dilations, int? group, int[]? kernelShape, int[]? pads, int[]? strides,
        int[]? outputPadding, int[]? outputShape, ExecutionOptions? options)
    {
        const OpType op = OpType.ConvTranspose;
        if (input is null) return MissingInput(op, nameof(input));
        if (weights is null) return MissingInput(op, nameof(weights));
        var opts = (options ?? ExecutionOptions.Default).Validated();
        if (input.Rank != 3) return WrongInputShape(op, nameof(input), 3, input);
        if (weights.Rank != 3) return WrongInputShape(op, nameof(weights), 3, weights);
        if (weights.ElementType != input.ElementType) return WrongInputType(op, nameof(weights), input.ElementType, weights);
        if (bias is not null && bias.ElementType != input.ElementType) return WrongInputType(op, nameof(bias), input.ElementType, bias);
        if (autoPad is not null and not "NOTSET" and not "VALID")
            return AttributeNotSupported(op, "auto_pad", autoPad, "Only NOTSET and VALID are supported for 1D transposed convolution.");
        if (outputShape is not null)
            return AttributeNotSupported(op, "output_shape", string.Join(",", outputShape), "Explicit output_shape is not supported.");
        if (dilations is not null && dilations.Length != 1 || strides is not null && strides.Length != 1
            || outputPadding is not null && outputPadding.Length != 1 || pads is not null && pads.Length != 2
            || kernelShape is not null && kernelShape.Length != 1)
            return Failure(op, "One-dimensional convolution attributes have inconsistent lengths.");
        int dilation = dilations?[0] ?? 1, stride = strides?[0] ?? 1, groups = group ?? 1;
        int adjustment = outputPadding?[0] ?? 0, left = pads?[0] ?? 0, right = pads?[1] ?? 0;
        int channels = input.Dims[1], length = input.Dims[2], kernel = weights.Dims[2], outputsPerGroup = weights.Dims[1];
        if (groups <= 0 || channels <= 0 || channels % groups != 0 || weights.Dims[0] != channels || outputsPerGroup <= 0)
            return Failure(op, "Input/weight channels and group are inconsistent.");
        if (length <= 0 || kernel <= 0 || stride <= 0 || dilation <= 0 || left < 0 || right < 0
            || adjustment < 0 || adjustment >= Math.Max(stride, dilation))
            return Failure(op, "Lengths, stride and dilation must be positive; padding must be nonnegative and output_padding smaller than stride or dilation.");
        if (kernelShape is not null && kernelShape[0] != kernel)
            return Failure(op, "kernel_shape must match the weight tensor.");
        if (autoPad == "VALID" && (left != 0 || right != 0))
            return Failure(op, "VALID padding cannot be combined with nonzero explicit pads.");
        long outputLength = (long)(length - 1) * stride - left - right + (long)dilation * (kernel - 1) + adjustment + 1;
        long outputChannels = (long)groups * outputsPerGroup;
        TensorLimits.ValidateGeometry(outputLength, "output length");
        TensorLimits.ValidateGeometry(outputChannels, "output channels");
        if (outputLength <= 0 || outputLength > int.MaxValue || outputChannels > int.MaxValue)
            return Failure(op, "Output dimensions exceed supported allocation limits or are not positive.");
        TensorLimits.MultiplyDimensions(TensorLimits.MultiplyDimensions(input.Dims[0], outputChannels, "output"), outputLength, "output");
        if (bias is not null && (bias.Rank != 1 || bias.Dims[0] != outputChannels))
            return WrongInputShape(op, nameof(bias), bias, "Bias must have one element per output channel.");
        return input.ElementType switch
        {
            TensorElementType.Float => Success(op, groups == 1 && channels >= 16 && kernel >= 16 && opts.Tensor.UseSimd && Vector.IsHardwareAccelerated
                && outputChannels * kernel <= (16L * 1024 * 1024) / length
                ? ConvTransposeMatrix((Tensor<float>)input, (Tensor<float>)weights, (Tensor<float>?)bias,
                    stride, dilation, left, (int)outputChannels, (int)outputLength, opts.Tensor)
                : ConvTransposeValues((Tensor<float>)input, (Tensor<float>)weights, (Tensor<float>?)bias,
                    groups, stride, dilation, left, (int)outputChannels, (int)outputLength)),
            TensorElementType.Double => Success(op, ConvTransposeValues((Tensor<double>)input, (Tensor<double>)weights, (Tensor<double>?)bias,
                groups, stride, dilation, left, (int)outputChannels, (int)outputLength)),
            _ => InputTypeNotSupported(op, nameof(input), input),
        };
    }

    static DenseTensor<float> ConvTransposeMatrix(Tensor<float> input, Tensor<float> weights, Tensor<float>? bias,
        int stride, int dilation, int left, int outputChannels, int outputLength, TensorExecutionOptions options)
    {
        // Each matrix row contains one frame's reconstructed channels/kernels.
        // Sum channels through the existing GEMM dispatcher, then overlap-add frames.
        var x = input.ToRowMajor();
        var w = weights.ToRowMajor();
        int channels = input.Dimensions[1], length = input.Dimensions[2], kernel = weights.Dimensions[2];
        var matrix = w.Reshape(channels, outputChannels * kernel);
        var result = new DenseTensor<float>(new[] { input.Dimensions[0], outputChannels, outputLength });
        var y = result.Buffer.Span;
        for (int batch = 0; batch < input.Dimensions[0]; batch++)
        {
            var frame = new DenseTensor<float>(x.Buffer.Slice(batch * channels * length, channels * length), new[] { channels, length });
            var transposed = Tensor<float>.Transpose(frame, new[] { 1, 0 });
            var expanded = Tensor<float>.MatMul2D(transposed, matrix, options).ToRowMajor();
            options.ScratchReporter?.AddScratchBytes(expanded.Length * sizeof(float));
            var values = expanded.Buffer.Span;
            for (int sample = 0; sample < length; sample++)
                for (int channel = 0; channel < outputChannels; channel++)
                {
                    int source = (sample * outputChannels + channel) * kernel;
                    int destination = (batch * outputChannels + channel) * outputLength;
                    long start = (long)sample * stride - left;
                    for (int tap = 0; tap < kernel; tap++)
                    {
                        long position = start + (long)tap * dilation;
                        if ((ulong)position < (ulong)outputLength) y[destination + (int)position] += values[source + tap];
                    }
                }
        }
        if (bias is not null)
            for (int batch = 0; batch < input.Dimensions[0]; batch++)
                for (int channel = 0; channel < outputChannels; channel++)
                {
                    float value = bias.GetLogicalValue(channel);
                    var row = y.Slice((batch * outputChannels + channel) * outputLength, outputLength);
                    for (int i = 0; i < row.Length; i++) row[i] += value;
                }
        return result;
    }

    static DenseTensor<T> ConvTransposeValues<T>(Tensor<T> input, Tensor<T> weights, Tensor<T>? bias,
        int groups, int stride, int dilation, int left, int outputChannels, int outputLength) where T : unmanaged, INumber<T>
    {
        var x = input.ToArray();
        var w = weights.ToArray();
        var b = bias?.ToArray();
        var result = new DenseTensor<T>(new[] { input.Dimensions[0], outputChannels, outputLength });
        var y = result.Buffer.Span;
        int channels = input.Dimensions[1], length = input.Dimensions[2], kernel = weights.Dimensions[2];
        int inputsPerGroup = channels / groups, outputsPerGroup = outputChannels / groups;
        for (int batch = 0; batch < input.Dimensions[0]; batch++)
        {
            for (int channel = 0; channel < channels; channel++)
                for (int sample = 0; sample < length; sample++)
                {
                    T value = x[(batch * channels + channel) * length + sample];
                    long start = (long)sample * stride - left;
                    for (int output = 0; output < outputsPerGroup; output++)
                    {
                        int destination = (batch * outputChannels + channel / inputsPerGroup * outputsPerGroup + output) * outputLength;
                        int source = (channel * outputsPerGroup + output) * kernel;
                        for (int tap = 0; tap < kernel; tap++)
                        {
                            long position = start + (long)tap * dilation;
                            if ((ulong)position < (ulong)outputLength)
                                y[destination + (int)position] += value * w[source + tap];
                        }
                    }
                }
            if (b is not null)
                for (int channel = 0; channel < outputChannels; channel++)
                {
                    var row = y.Slice((batch * outputChannels + channel) * outputLength, outputLength);
                    for (int sample = 0; sample < row.Length; sample++) row[sample] += b[channel];
                }
        }
        return result;
    }
}
