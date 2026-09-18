namespace Lokad.Onnx;

using System;
using System.Buffers;
using System.Threading.Tasks;
using static Lokad.Onnx.MathOps;

public abstract partial class Tensor<T>
{
    const int ConvReductionTile = 128;
    const int ConvPositionTile = 128;
    const int ConvFilterTile = 32;
    const int ConvPatchElements = ConvReductionTile * ConvPositionTile;
    const int ConvWeightElements = ConvFilterTile * ConvReductionTile;
    const int ConvScratchElements = ConvPatchElements + ConvWeightElements + ConvFilterTile * ConvPositionTile;

    // Shared Conv planning has already validated geometry and densified operands.
    // One private rent per batch worker, independent of input length and group count.
    static void RunSegmentedConvFloat(Memory<float> input, Memory<float> weights, Memory<float> bias,
        bool hasBias, Memory<float> output, int batches, int groups, int channels, int height, int width,
        int filters, int kernelH, int kernelW, int dilationH, int dilationW, int strideH, int strideW,
        PadInfo pad, int outH, int outW, int inBatch, int outBatch, int dop, TensorExecutionOptions options)
    {
        if (batches == 0 || filters == 0 || outH == 0 || outW == 0) return;
        if (dop > 1)
        {
            Parallel.For(0, batches, new ParallelOptions { MaxDegreeOfParallelism = dop },
                () => RentScratch<float>(ConvScratchElements, options),
                (batch, state, scratch) =>
                {
                    RunSegmentedConvBatchFloat(input, weights, bias, hasBias, output, scratch, batch,
                        groups, channels, height, width, filters, kernelH, kernelW, dilationH, dilationW,
                        strideH, strideW, pad, outH, outW, inBatch, outBatch, options);
                    return scratch;
                },
                scratch => ArrayPool<float>.Shared.Return(scratch));
        }
        else
        {
            var scratch = RentScratch<float>(ConvScratchElements, options);
            try
            {
                for (int batch = 0; batch < batches; batch++)
                    RunSegmentedConvBatchFloat(input, weights, bias, hasBias, output, scratch, batch,
                        groups, channels, height, width, filters, kernelH, kernelW, dilationH, dilationW,
                        strideH, strideW, pad, outH, outW, inBatch, outBatch, options);
            }
            finally { ArrayPool<float>.Shared.Return(scratch); }
        }
    }

    static void RunSegmentedConvBatchFloat(Memory<float> input, Memory<float> weights, Memory<float> bias,
        bool hasBias, Memory<float> output, float[] scratch, int batch, int groups, int channels,
        int height, int width, int filters, int kernelH, int kernelW, int dilationH, int dilationW,
        int strideH, int strideW, PadInfo pad, int outH, int outW, int inBatch, int outBatch,
        TensorExecutionOptions options)
    {
        int positions = outH * outW, groupChannels = channels / groups, groupFilters = filters / groups;
        int kernelSize = kernelH * kernelW, reduction = groupChannels * kernelSize;
        var x = input.Span.Slice(batch * inBatch, inBatch);
        var w = weights.Span;
        var y = output.Span.Slice(batch * outBatch, outBatch);
        var patch = scratch.AsSpan(0, ConvPatchElements);
        var packedWeights = scratch.AsSpan(ConvPatchElements, ConvWeightElements);
        var partial = scratch.AsSpan(ConvPatchElements + ConvWeightElements);
        for (int group = 0; group < groups; group++)
        for (int startN = 0; startN < positions; startN += ConvPositionTile)
        {
            int countN = Math.Min(ConvPositionTile, positions - startN);
            for (int startK = 0; startK < reduction; startK += ConvReductionTile)
            {
                int countK = Math.Min(ConvReductionTile, reduction - startK);
                for (int k = 0; k < countK; k++)
                {
                    int term = startK + k, channel = group * groupChannels + term / kernelSize;
                    int kernelOffset = term % kernelSize;
                    int rowOffset = kernelOffset / kernelW * dilationH - pad.top;
                    int columnOffset = kernelOffset % kernelW * dilationW - pad.left;
                    var row = patch.Slice(k * ConvPositionTile, ConvPositionTile);
                    row.Clear(); // Padding, including unused columns, is always initialized.
                    int position = startN, done = 0;
                    while (done < countN)
                    {
                        int outputRow = position / outW, outputColumn = position % outW;
                        int take = Math.Min(countN - done, outW - outputColumn);
                        int sy = outputRow * strideH + rowOffset;
                        if ((uint)sy < (uint)height)
                        {
                            int sourceRow = (channel * height + sy) * width;
                            int sx = outputColumn * strideW + columnOffset;
                            for (int j = 0; j < take; j++, sx += strideW)
                                if ((uint)sx < (uint)width) row[done + j] = x[sourceRow + sx];
                        }
                        done += take;
                        position += take;
                    }
                }
                for (int startM = 0; startM < groupFilters; startM += ConvFilterTile)
                {
                    int countM = Math.Min(ConvFilterTile, groupFilters - startM);
                    for (int m = 0; m < countM; m++)
                        w.Slice((group * groupFilters + startM + m) * reduction + startK, countK)
                            .CopyTo(packedWeights.Slice(m * countK, countK));
                    partial.Slice(0, countM * ConvPositionTile).Clear();
                    MultiplyConvPartial(countM, countK, packedWeights, patch, partial, options);
                    for (int m = 0; m < countM; m++)
                    {
                        var destination = y.Slice((group * groupFilters + startM + m) * positions + startN, countN);
                        var product = partial.Slice(m * ConvPositionTile, countN);
                        if (startK == 0) product.CopyTo(destination);
                        else
                            for (int n = 0; n < countN; n++) destination[n] += product[n];
                    }
                }
            }
        }
        if (hasBias)
        {
            var b = bias.Span;
            for (int m = 0; m < filters; m++)
            {
                var row = y.Slice(m * positions, positions);
                for (int n = 0; n < positions; n++) row[n] += b[m];
            }
        }
    }

    // Direct raw kernels avoid per-tile panel rents. All 128 columns are real
    // scratch storage, so active tails retain FMA without out-of-range accesses.
    static unsafe void MultiplyConvPartial(int rows, int reduction, Span<float> weights,
        Span<float> patch, Span<float> partial, TensorExecutionOptions options)
    {
        fixed (float* a = weights)
        fixed (float* b = patch)
        fixed (float* c = partial)
        {
            if (options.UseIntrinsics)
            {
                int pairs = rows - rows % 2;
                if (pairs > 0) mm_unsafe_vectorized_intrinsics_2x4tiled(pairs, reduction, ConvPositionTile, a, b, c);
                if (pairs != rows) mm_unsafe_vectorized_intrinsics(1, reduction, ConvPositionTile,
                    a + pairs * reduction, b, c + pairs * ConvPositionTile);
            }
            else if (options.UseSimd) mm_unsafe_vectorized(rows, reduction, ConvPositionTile, a, b, c);
            else mm(rows, reduction, ConvPositionTile, a, b, c);
        }
    }
}
