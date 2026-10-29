namespace Lokad.Onnx;

using System.Buffers;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using static OpResult;

public partial class CPUExecutionProvider
{
    // The optimizer supplies the original live inputs, including pads and fill.
    // Unprepared or
    // unsupported calls retain the original operator sequence below.
    internal static OpResult ConformerConv(ITensor? input, ITensor? mask, ITensor? pointwise1,
        ITensor? depthwise, ITensor? bias, ITensor? pointwise2, ITensor? pads, ITensor? fill, ExecutionOptions? options)
    {
        const int channels = 1024;
        var op = OpType.FusedConformerConv;
        if (input is null) return MissingInput(op, nameof(input));
        if (mask is null) return MissingInput(op, nameof(mask));
        if (pointwise1 is null) return MissingInput(op, nameof(pointwise1));
        if (depthwise is null) return MissingInput(op, nameof(depthwise));
        if (bias is null) return MissingInput(op, nameof(bias));
        if (pointwise2 is null) return MissingInput(op, nameof(pointwise2));
        if (pads is null) return MissingInput(op, nameof(pads));
        if (fill is null) return MissingInput(op, nameof(fill));
        var opts = (options ?? ExecutionOptions.Default).Validated();
        if (opts.Tensor.UseSimd && opts.Tensor.UseIntrinsics && opts.Tensor.MaxDegreeOfParallelism == 1
            && Avx512F.IsSupported && Fma.IsSupported
            && pads is Tensor<long> padding && padding.Rank == 1 && padding.Length == 6
            && padding.GetValue(0) == 0 && padding.GetValue(1) == 0 && padding.GetValue(2) == 4
            && padding.GetValue(3) == 0 && padding.GetValue(4) == 0 && padding.GetValue(5) == 4
            && fill is Tensor<float> zero && zero.Length == 1 && zero.Rank <= 3
            && BitConverter.SingleToInt32Bits(zero.GetValue(0)) == 0
            && input is DenseTensor<float> x && x.Rank == 3 && x.Dimensions[0] == 1 && x.Dimensions[2] == channels
            && x.Dimensions[1] >= 48 && x.Dimensions[1] <= 1024 && GraphConvPacking.Standard(x)
            && mask is Tensor<bool> condition && condition.Rank == 3
            && condition.Dimensions[0] == 1 && condition.Dimensions[1] == 1 && condition.Dimensions[2] == x.Dimensions[1]
            && pointwise1 is OwnedPackedTensor { IsTransposedLogical: true } p1
            && PointwiseShape(p1, channels, 2 * channels)
            && pointwise2 is OwnedPackedTensor { IsTransposedLogical: true } p2 && PointwiseShape(p2, channels, channels)
            && depthwise is DenseTensor<float> dw && dw.Rank == 3 && dw.Dimensions[0] == channels
            && dw.Dimensions[1] == 1 && dw.Dimensions[2] == 9 && GraphConvPacking.Standard(dw)
            && bias is DenseTensor<float> b && b.Rank == 1 && b.Length == channels && GraphConvPacking.Standard(b))
            return Success(op, RunConformerConv(x, condition, p1, dw, b, p2, opts.Tensor));
        return ConformerConvFallback(input, mask, pointwise1, depthwise, bias, pointwise2, pads, fill, opts);
    }

    static bool PointwiseShape(OwnedPackedTensor value, int channels, int outputs) =>
        value.Rank == 3 && value.Dimensions[0] == outputs && value.Dimensions[1] == channels && value.Dimensions[2] == 1
        && value.Reduction == channels && value.Columns == outputs;

    static Tensor<float> RunConformerConv(DenseTensor<float> input, Tensor<bool> mask, OwnedPackedTensor p1,
        DenseTensor<float> depthwise, DenseTensor<float> bias, OwnedPackedTensor p2, TensorExecutionOptions options)
    {
        const int channels = 1024;
        int time = input.Dimensions[1], count = time * channels;
        float[]? upBuffer = null, gateBuffer = null, activationBuffer = null, kernelBuffer = null;
        try
        {
            upBuffer = RentConformerScratch(2 * count, options);
            gateBuffer = RentConformerScratch(count, options);
            activationBuffer = RentConformerScratch(count, options);
            kernelBuffer = RentConformerScratch(9 * channels, options);
            var x = new DenseTensor<float>(input.Buffer, new[] { time, channels });
            var up = new DenseTensor<float>(upBuffer.AsMemory(0, 2 * count), new[] { time, 2 * channels });
            var gate = new DenseTensor<float>(gateBuffer.AsMemory(0, count), new[] { time, channels });
            var activation = new DenseTensor<float>(activationBuffer.AsMemory(0, count), new[] { time, channels });
            Tensor<float>.MatMul2D(x, p1.PhysicalMatrix, up, options);
            ConformerNarrowTimeRows(x.Buffer.Span, p1.PackedArray, up.Buffer.Span, time, 2 * channels);
            for (int t = 0; t < time; t++)
            {
                var destination = gate.Buffer.Span.Slice(t * channels, channels);
                if (mask.GetValue(t)) { destination.Clear(); continue; }
                SigmoidRationalVector(up.Buffer.Span.Slice(t * 2 * channels + channels, channels), destination, true);
                var first = up.Buffer.Span.Slice(t * 2 * channels, channels);
                for (int c = 0; c < channels; c++) destination[c] = first[c] * destination[c];
            }
            for (int c = 0; c < channels; c++)
                for (int tap = 0; tap < 9; tap++) kernelBuffer[tap * channels + c] = depthwise.Buffer.Span[c * 9 + tap];
            ConformerDepthwise(gate.Buffer.Span, kernelBuffer, bias.Buffer.Span, activation.Buffer.Span, time);
            SigmoidRationalVector(activation.Buffer.Span, activation.Buffer.Span, true, true);
            var output = DenseTensor<float>.OfShape(time, channels);
            Tensor<float>.MatMul2D(activation, p2.PhysicalMatrix, output, options);
            ConformerNarrowTimeRows(activation.Buffer.Span, p2.PackedArray, output.Buffer.Span, time, channels);
            return output.Reshape(new[] { 1, time, channels });
        }
        finally
        {
            if (kernelBuffer is not null) ArrayPool<float>.Shared.Return(kernelBuffer);
            if (activationBuffer is not null) ArrayPool<float>.Shared.Return(activationBuffer);
            if (gateBuffer is not null) ArrayPool<float>.Shared.Return(gateBuffer);
            if (upBuffer is not null) ArrayPool<float>.Shared.Return(upBuffer);
        }
    }

    static float[] RentConformerScratch(int elements, TensorExecutionOptions options)
    {
        var buffer = ArrayPool<float>.Shared.Rent(elements);
        try { options.ScratchReporter?.AddScratchBytes((long)buffer.Length * sizeof(float)); }
        catch { ArrayPool<float>.Shared.Return(buffer); throw; }
        return buffer;
    }

    static unsafe void ConformerDepthwise(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> bias, Span<float> output, int time)
    {
        const int channels = 1024;
        fixed (float* x = input, w = weights, b = bias, y = output)
            for (int t = 0; t < time; t++)
                for (int c = 0; c < channels; c += 16)
                {
                    var sum = Vector512<float>.Zero;
                    for (int tap = 0; tap < 9; tap++)
                    {
                        int source = t + tap - 4;
                        var value = (uint)source < (uint)time ? Vector512.Load(x + source * channels + c) : Vector512<float>.Zero;
                        var weight = Vector512.Load(w + tap * channels + c);
                        sum = t < time / 8 * 8 ? Avx512F.FusedMultiplyAdd(value, weight, sum) : sum + value * weight;
                    }
                    (sum + Vector512.Load(b + c)).Store(y + t * channels + c);
                }
    }

    static unsafe void ConformerNarrowTimeRows(ReadOnlySpan<float> input,float[] packed,Span<float> output,int time,int columns)
    {
        const int Channels=1024;
        int remaining=time%8;if(remaining==0)return;int first=time-remaining;
        fixed(float* x=input,p=packed,y=output)
        for(int column=0;column<columns;column+=32)
        {
            var s00=Vector512<float>.Zero;var s01=Vector512<float>.Zero;
            var s10=Vector512<float>.Zero;var s11=Vector512<float>.Zero;
            var s20=Vector512<float>.Zero;var s21=Vector512<float>.Zero;
            var s30=Vector512<float>.Zero;var s31=Vector512<float>.Zero;
            var s40=Vector512<float>.Zero;var s41=Vector512<float>.Zero;
            var s50=Vector512<float>.Zero;var s51=Vector512<float>.Zero;
            var s60=Vector512<float>.Zero;var s61=Vector512<float>.Zero;
            for(int j=0;j<Channels;j++)
            {
                float* weights=p+column*Channels+j*32;
                var b0=Vector512.Load(weights);var b1=Vector512.Load(weights+16);
                if(remaining>0){var a=Vector512.Create(x[(first+0)*Channels+j]);s00=s00+a*b0;s01=s01+a*b1;}
                if(remaining>1){var a=Vector512.Create(x[(first+1)*Channels+j]);s10=s10+a*b0;s11=s11+a*b1;}
                if(remaining>2){var a=Vector512.Create(x[(first+2)*Channels+j]);s20=s20+a*b0;s21=s21+a*b1;}
                if(remaining>3){var a=Vector512.Create(x[(first+3)*Channels+j]);s30=s30+a*b0;s31=s31+a*b1;}
                if(remaining>4){var a=Vector512.Create(x[(first+4)*Channels+j]);s40=s40+a*b0;s41=s41+a*b1;}
                if(remaining>5){var a=Vector512.Create(x[(first+5)*Channels+j]);s50=s50+a*b0;s51=s51+a*b1;}
                if(remaining>6){var a=Vector512.Create(x[(first+6)*Channels+j]);s60=s60+a*b0;s61=s61+a*b1;}
            }
            if(remaining>0){s00.Store(y+(first+0)*columns+column);s01.Store(y+(first+0)*columns+column+16);}
            if(remaining>1){s10.Store(y+(first+1)*columns+column);s11.Store(y+(first+1)*columns+column+16);}
            if(remaining>2){s20.Store(y+(first+2)*columns+column);s21.Store(y+(first+2)*columns+column+16);}
            if(remaining>3){s30.Store(y+(first+3)*columns+column);s31.Store(y+(first+3)*columns+column+16);}
            if(remaining>4){s40.Store(y+(first+4)*columns+column);s41.Store(y+(first+4)*columns+column+16);}
            if(remaining>5){s50.Store(y+(first+5)*columns+column);s51.Store(y+(first+5)*columns+column+16);}
            if(remaining>6){s60.Store(y+(first+6)*columns+column);s61.Store(y+(first+6)*columns+column+16);}
        }
    }

    static OpResult ConformerConvFallback(ITensor input, ITensor mask, ITensor p1, ITensor dw, ITensor bias, ITensor p2, ITensor pads, ITensor fill, ExecutionOptions options)
    {
        var transpose = Transpose(input, new[] { 0, 2, 1 }, options, null);
        if (transpose.Status != OpStatus.Success) return transpose;
        var up = Conv(transpose.Outputs[0], p1, null, "NOTSET", null, 1, new[] { 1 }, new[] { 0, 0 }, new[] { 1 }, options);
        if (up.Status != OpStatus.Success) return up;
        var split = Split(up.Outputs[0], null, 1, null, null, options, 2);
        if (split.Status != OpStatus.Success) return split;
        var sigmoid = Sigmoid(split.Outputs[1], options);
        if (sigmoid.Status != OpStatus.Success) return sigmoid;
        var gate = Mul(split.Outputs[0], sigmoid.Outputs[0], options, null);
        if (gate.Status != OpStatus.Success) return gate;
        var masked = Where(mask, fill, gate.Outputs[0], options);
        if (masked.Status != OpStatus.Success) return masked;
        var padded = Pad(masked.Outputs[0], pads, null, "constant", null, null, options);
        if (padded.Status != OpStatus.Success) return padded;
        var filtered = Conv(padded.Outputs[0], dw, bias, "NOTSET", null, 1024, new[] { 9 }, new[] { 0, 0 }, new[] { 1 }, options);
        if (filtered.Status != OpStatus.Success) return filtered;
        var activated = SigmoidMul(filtered.Outputs[0], options);
        if (activated.Status != OpStatus.Success) return activated;
        var down = Conv(activated.Outputs[0], p2, null, "NOTSET", null, 1, new[] { 1 }, new[] { 0, 0 }, new[] { 1 }, options);
        if (down.Status != OpStatus.Success) return down;
        var result = Transpose(down.Outputs[0], new[] { 0, 2, 1 }, options, null);
        if (result.Status == OpStatus.Success) result.Op = OpType.FusedConformerConv;
        return result;
    }
}
