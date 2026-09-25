namespace Lokad.Onnx;

using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using static Lokad.Onnx.MathOps;

public abstract partial class Tensor<T> where T : unmanaged
{
    // Bypass expanded patches and per-channel matrix dispatch for the two
    // measured nine-tap depthwise geometries. Other options keep their old path.
    static unsafe bool TryConvDirectDepthwise(DenseTensor<float> input, DenseTensor<float> weight,
        DenseTensor<float>? bias, DenseTensor<float> output, int n, int group, int channels,
        int height, int width, int filters, int kh, int kw, int dh, int dw, int sh, int sw,
        PadInfo pad, int outHeight, int outWidth, TensorExecutionOptions options)
    {
        if (n != 1 || channels <= 1 || group != channels || filters != channels
            || !options.UseSimd || !options.UseIntrinsics || options.MaxDegreeOfParallelism != 1
            || !Avx2.IsSupported || !Fma.IsSupported || dh != 1 || dw != 1)
            return false;
        bool line = height == 1 && outHeight == 1 && kh == 1 && kw == 9 && sh == 1 && sw == 1
            && pad.top == 0 && pad.left == 0 && pad.bottom == 0 && pad.right == 0;
        bool spatial = kh == 3 && kw == 3 && sh == 2 && sw == 2
            && pad.top == 1 && pad.left == 1 && pad.bottom == 1 && pad.right == 1;
        if (!line && !spatial) return false;
        fixed (float* x = input.Buffer.Span)
        fixed (float* w = weight.Buffer.Span)
        fixed (float* b = bias is null ? default(Span<float>) : bias.Buffer.Span)
        fixed (float* y = output.Buffer.Span)
        {
            for (int c = 0; c < channels; c++)
            {
                float bi = bias is null ? 0f : b[c];
                if (line)
                    RunDirectDepthwiseLine(x + c * width, w + c * 9, y + c * outWidth,
                        outWidth, bias is not null, bi);
                else
                    RunDirectDepthwiseSpatial(x + c * height * width, w + c * 9,
                        y + c * outHeight * outWidth, height, width, outHeight, outWidth,
                        bias is not null, bi);
            }
        }
        return true;
    }

    static unsafe void RunDirectDepthwiseLine(float* x, float* w, float* y, int count, bool hasBias, float bias)
    {
        int ceiling = count / 8 * 8;
        var w0 = Vector256.Create(w[0]); var w1 = Vector256.Create(w[1]); var w2 = Vector256.Create(w[2]);
        var w3 = Vector256.Create(w[3]); var w4 = Vector256.Create(w[4]); var w5 = Vector256.Create(w[5]);
        var w6 = Vector256.Create(w[6]); var w7 = Vector256.Create(w[7]); var w8 = Vector256.Create(w[8]);
        var bv = Vector256.Create(bias);
        int j = 0;
        for (; j < ceiling; j += 8)
        {
            var sum = Fma.MultiplyAdd(Avx.LoadVector256(x + j), w0, Vector256<float>.Zero);
            sum = Fma.MultiplyAdd(Avx.LoadVector256(x + j + 1), w1, sum);
            sum = Fma.MultiplyAdd(Avx.LoadVector256(x + j + 2), w2, sum);
            sum = Fma.MultiplyAdd(Avx.LoadVector256(x + j + 3), w3, sum);
            sum = Fma.MultiplyAdd(Avx.LoadVector256(x + j + 4), w4, sum);
            sum = Fma.MultiplyAdd(Avx.LoadVector256(x + j + 5), w5, sum);
            sum = Fma.MultiplyAdd(Avx.LoadVector256(x + j + 6), w6, sum);
            sum = Fma.MultiplyAdd(Avx.LoadVector256(x + j + 7), w7, sum);
            sum = Fma.MultiplyAdd(Avx.LoadVector256(x + j + 8), w8, sum);
            Avx.Store(y + j, hasBias ? Avx.Add(sum, bv) : sum);
        }
        for (; j < count; j++)
        {
            float sum = 0f;
            for (int q = 0; q < 9; q++) sum += w[q] * x[j + q];
            y[j] = hasBias ? sum + bias : sum;
        }
    }

    static unsafe void RunDirectDepthwiseSpatial(float* x, float* w, float* y, int height,
        int width, int outHeight, int outWidth, bool hasBias, float bias)
    {
        // The original 32-column panels put only the final flattened remainder
        // on non-FMA arithmetic. Row boundaries must not change that choice.
        int ceiling = outHeight * outWidth / 8 * 8;
        var offsets = Vector256.Create(0, 2, 4, 6, 8, 10, 12, 14);
        var bv = Vector256.Create(bias);
        for (int row = 0; row < outHeight; row++)
        {
            int j = 0, outputBase = row * outWidth;
            bool interiorRow = row > 0 && 2 * row + 1 < height;
            if (interiorRow)
            {
                y[outputBase] = DirectDepthwisePoint(x, w, height, width, row, 0,
                    outputBase < ceiling, hasBias, bias);
                j = 1;
                for (; j + 8 <= width / 2 && outputBase + j + 8 <= ceiling; j += 8)
                {
                    float* start = x + (2 * row - 1) * width + 2 * j - 1;
                    var sum = Vector256<float>.Zero;
                    for (int h = 0; h < 3; h++)
                    for (int k = 0; k < 3; k++)
                        sum = Fma.MultiplyAdd(Avx2.GatherVector256(start + h * width + k, offsets, 4),
                            Vector256.Create(w[h * 3 + k]), sum);
                    Avx.Store(y + outputBase + j, hasBias ? Avx.Add(sum, bv) : sum);
                }
            }
            for (; j < outWidth; j++)
                y[outputBase + j] = DirectDepthwisePoint(x, w, height, width, row, j,
                    outputBase + j < ceiling, hasBias, bias);
        }
    }

    static unsafe float DirectDepthwisePoint(float* x, float* w, int height, int width,
        int row, int column, bool fused, bool hasBias, float bias)
    {
        float sum = 0f;
        for (int h = 0; h < 3; h++)
        for (int k = 0; k < 3; k++)
        {
            int ih = 2 * row - 1 + h, iw = 2 * column - 1 + k;
            float value = (uint)ih < (uint)height && (uint)iw < (uint)width ? x[ih * width + iw] : 0f;
            // Include zero padding in arithmetic: zero times infinity is NaN.
            sum = fused ? MathF.FusedMultiplyAdd(value, w[h * 3 + k], sum) : sum + w[h * 3 + k] * value;
        }
        return hasBias ? sum + bias : sum;
    }
}
