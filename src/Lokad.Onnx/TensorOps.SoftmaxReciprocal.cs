namespace Lokad.Onnx;

using System.Numerics;
using System.Runtime.CompilerServices;

public abstract partial class Tensor<T>
{
    // Opt-in normalization experiment: the exp/sum loops match the pointer
    // reference exactly. A reciprocal multiply changes final rounding only.
    internal static unsafe void SoftmaxMaskedFloatSpanPtrReciprocal(System.Span<float> inputSpan, System.Span<float> maskSpan, System.Span<float> outputSpan, int outer, int block, bool useSimd)
    {
        if (!useSimd || !Vector.IsHardwareAccelerated)
        {
            SoftmaxMaskedFloatSpanPtr(inputSpan, maskSpan, outputSpan, outer, block, useSimd);
            return;
        }

        if (maskSpan.Length < block) throw new ArgumentException(nameof(maskSpan), "Mask row must cover a full block.");
        fixed (float* px = inputSpan, pm = maskSpan, py = outputSpan)
        {
            int w = Vector<float>.Count;
            int pairs = outer / 2;
            for (int p = 0; p < pairs; p++)
            {
                int base0 = (2 * p) * block;
                int base1 = (2 * p + 1) * block;
                float max0 = SoftmaxContiguousMaxMasked(inputSpan, base0, maskSpan, block, useSimd);
                float max1 = SoftmaxContiguousMaxMasked(inputSpan, base1, maskSpan, block, useSimd);
                float sum0 = 0f;
                int expIndex0 = 0;
                if (useSimd && Vector.IsHardwareAccelerated)
                {
                    var vmax0 = new Vector<float>(max0);
                    var vsum0 = Vector<float>.Zero;
                    var xv0 = (Vector<float>*)(px + base0);
                    var mv0 = (Vector<float>*)pm;
                    var yv0 = (Vector<float>*)(py + base0);
                    for (; expIndex0 <= block - w; expIndex0 += w)
                    {
                        var activated0 = MathOps.ExpVectorEstrin((*xv0 + *mv0) - vmax0);
                        *yv0 = activated0;
                        vsum0 += activated0;
                        xv0++;
                        mv0++;
                        yv0++;
                    }
                    sum0 = Vector.Sum(vsum0);
                }
                for (; expIndex0 < block; expIndex0++)
                {
                    float activated0 = MathF.Exp((px[base0 + expIndex0] + pm[expIndex0]) - max0);
                    py[base0 + expIndex0] = activated0;
                    sum0 += activated0;
                }
                float sum1 = 0f;
                int expIndex1 = 0;
                if (useSimd && Vector.IsHardwareAccelerated)
                {
                    var vmax1 = new Vector<float>(max1);
                    var vsum1 = Vector<float>.Zero;
                    var xv1 = (Vector<float>*)(px + base1);
                    var mv1 = (Vector<float>*)pm;
                    var yv1 = (Vector<float>*)(py + base1);
                    for (; expIndex1 <= block - w; expIndex1 += w)
                    {
                        var activated1 = MathOps.ExpVectorEstrin((*xv1 + *mv1) - vmax1);
                        *yv1 = activated1;
                        vsum1 += activated1;
                        xv1++;
                        mv1++;
                        yv1++;
                    }
                    sum1 = Vector.Sum(vsum1);
                }
                for (; expIndex1 < block; expIndex1++)
                {
                    float activated1 = MathF.Exp((px[base1 + expIndex1] + pm[expIndex1]) - max1);
                    py[base1 + expIndex1] = activated1;
                    sum1 += activated1;
                }
                NormalizeSoftmaxReciprocal(py + base0, block, sum0);
                NormalizeSoftmaxReciprocal(py + base1, block, sum1);
            }
            for (int outerIndex = pairs * 2; outerIndex < outer; outerIndex++)
            {
                int baseR = outerIndex * block;
                float max = SoftmaxContiguousMaxMasked(inputSpan, baseR, maskSpan, block, useSimd);
                float sum = 0f;
                int expIndex = 0;
                if (useSimd && Vector.IsHardwareAccelerated)
                {
                    var vmax = new Vector<float>(max);
                    var vsum = Vector<float>.Zero;
                    var xv = (Vector<float>*)(px + baseR);
                    var mv = (Vector<float>*)pm;
                    var yv = (Vector<float>*)(py + baseR);
                    for (; expIndex <= block - w; expIndex += w)
                    {
                        var activated = MathOps.ExpVectorEstrin((*xv + *mv) - vmax);
                        *yv = activated;
                        vsum += activated;
                        xv++;
                        mv++;
                        yv++;
                    }
                    sum = Vector.Sum(vsum);
                }
                for (; expIndex < block; expIndex++)
                {
                    float activated = MathF.Exp((px[baseR + expIndex] + pm[expIndex]) - max);
                    py[baseR + expIndex] = activated;
                    sum += activated;
                }
                NormalizeSoftmaxReciprocal(py + baseR, block, sum);
            }
        }
    }

    internal static unsafe void SoftmaxContiguousFloatSpanPtrReciprocal(System.Span<float> inputSpan, System.Span<float> outputSpan, int outer, int block, bool useSimd)
    {
        if (!useSimd || !Vector.IsHardwareAccelerated)
        {
            SoftmaxContiguousFloatSpanPtr(inputSpan, outputSpan, outer, block, useSimd);
            return;
        }

        fixed (float* px = inputSpan, py = outputSpan)
        {
            int w = Vector<float>.Count;
            int pairs = outer / 2;
            for (int p = 0; p < pairs; p++)
            {
                int base0 = (2 * p) * block;
                int base1 = (2 * p + 1) * block;
                float max0 = SoftmaxContiguousMax(inputSpan, base0, block, useSimd);
                float max1 = SoftmaxContiguousMax(inputSpan, base1, block, useSimd);
                float sum0 = 0f;
                int expIndex0 = 0;
                if (useSimd && Vector.IsHardwareAccelerated)
                {
                    var vmax0 = new Vector<float>(max0);
                    var vsum0 = Vector<float>.Zero;
                    var xv0 = (Vector<float>*)(px + base0);
                    var yv0 = (Vector<float>*)(py + base0);
                    for (; expIndex0 <= block - w; expIndex0 += w)
                    {
                        var activated0 = MathOps.ExpVectorEstrin(*xv0 - vmax0);
                        *yv0 = activated0;
                        vsum0 += activated0;
                        xv0++;
                        yv0++;
                    }
                    sum0 = Vector.Sum(vsum0);
                }
                for (; expIndex0 < block; expIndex0++)
                {
                    float activated0 = MathF.Exp(px[base0 + expIndex0] - max0);
                    py[base0 + expIndex0] = activated0;
                    sum0 += activated0;
                }
                float sum1 = 0f;
                int expIndex1 = 0;
                if (useSimd && Vector.IsHardwareAccelerated)
                {
                    var vmax1 = new Vector<float>(max1);
                    var vsum1 = Vector<float>.Zero;
                    var xv1 = (Vector<float>*)(px + base1);
                    var yv1 = (Vector<float>*)(py + base1);
                    for (; expIndex1 <= block - w; expIndex1 += w)
                    {
                        var activated1 = MathOps.ExpVectorEstrin(*xv1 - vmax1);
                        *yv1 = activated1;
                        vsum1 += activated1;
                        xv1++;
                        yv1++;
                    }
                    sum1 = Vector.Sum(vsum1);
                }
                for (; expIndex1 < block; expIndex1++)
                {
                    float activated1 = MathF.Exp(px[base1 + expIndex1] - max1);
                    py[base1 + expIndex1] = activated1;
                    sum1 += activated1;
                }
                NormalizeSoftmaxReciprocal(py + base0, block, sum0);
                NormalizeSoftmaxReciprocal(py + base1, block, sum1);
            }
            for (int outerIndex = pairs * 2; outerIndex < outer; outerIndex++)
            {
                int baseR = outerIndex * block;
                float max = SoftmaxContiguousMax(inputSpan, baseR, block, useSimd);
                float sum = 0f;
                int expIndex = 0;
                if (useSimd && Vector.IsHardwareAccelerated)
                {
                    var vmax = new Vector<float>(max);
                    var vsum = Vector<float>.Zero;
                    var xv = (Vector<float>*)(px + baseR);
                    var yv = (Vector<float>*)(py + baseR);
                    for (; expIndex <= block - w; expIndex += w)
                    {
                        var activated = MathOps.ExpVectorEstrin(*xv - vmax);
                        *yv = activated;
                        vsum += activated;
                        xv++;
                        yv++;
                    }
                    sum = Vector.Sum(vsum);
                }
                for (; expIndex < block; expIndex++)
                {
                    float activated = MathF.Exp(px[baseR + expIndex] - max);
                    py[baseR + expIndex] = activated;
                    sum += activated;
                }
                NormalizeSoftmaxReciprocal(py + baseR, block, sum);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static unsafe void NormalizeSoftmaxReciprocal(float* output, int length, float sum)
    {
        int width = Vector<float>.Count;
        int i = 0;
        if (float.IsFinite(sum) && sum > 0f)
        {
            float reciprocal = 1f / sum;
            var scale = new Vector<float>(reciprocal);
            for (; i <= length - width; i += width)
                *(Vector<float>*)(output + i) *= scale;
            for (; i < length; i++) output[i] *= reciprocal;
        }
        else
        {
            // Keep the reference's division and NaN propagation for exceptional
            // rows. A valid finite softmax has a row sum of at least one.
            var divisor = new Vector<float>(sum);
            for (; i <= length - width; i += width)
                *(Vector<float>*)(output + i) /= divisor;
            for (; i < length; i++) output[i] /= sum;
        }
    }
}
