using System.Numerics;

namespace Lokad.Onnx;

public abstract partial class Tensor<T>
{
    /// <summary>
    /// Experimental masked softmax with exact all-underflow vector bypasses.
    /// Mask inspection only selects the route: each vector checks its actual
    /// score-plus-mask-minus-maximum arguments before skipping exponentiation.
    /// Maxima, paired rows, accumulation, tails and division match the original.
    /// </summary>
    internal static unsafe void SoftmaxMaskedFloatSpanPtrZeroBlocks(System.Span<float> inputSpan, System.Span<float> maskSpan, System.Span<float> outputSpan, int outer, int block, bool useSimd)
    {
        if (maskSpan.Length < block) throw new ArgumentException(nameof(maskSpan), "Mask row must cover a full block.");
        // Preserve the original routes outside the qualified nonpositive SIMD case.
        if (!AblationSwitches.EnableSoftmaxNonpositive || AblationSwitches.EnableSoftmaxWideExp
            || !useSimd || !Vector.IsHardwareAccelerated || Vector<float>.Count != 8 || block < Vector<float>.Count)
        {
            SoftmaxMaskedFloatSpanPtr(inputSpan, maskSpan, outputSpan, outer, block, useSimd);
            return;
        }
        bool possible = false;
        for (int i = 0; i < block; i++)
        {
            if (maskSpan[i] < -88.722839f) { possible = true; break; }
        }
        if (!possible)
        {
            SoftmaxMaskedFloatSpanPtr(inputSpan, maskSpan, outputSpan, outer, block, useSimd);
            return;
        }
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
                        var activated0 = MathOps.ExpVectorNonpositiveZeroBlocks((*xv0 + *mv0) - vmax0);
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
                        var activated1 = MathOps.ExpVectorNonpositiveZeroBlocks((*xv1 + *mv1) - vmax1);
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
                int normIndex0 = 0;
                if (useSimd && Vector.IsHardwareAccelerated)
                {
                    var vdiv0 = new Vector<float>(sum0);
                    var yv0 = (Vector<float>*)(py + base0);
                    for (; normIndex0 <= block - w; normIndex0 += w)
                    {
                        *yv0 = *yv0 / vdiv0;
                        yv0++;
                    }
                }
                for (; normIndex0 < block; normIndex0++) py[base0 + normIndex0] /= sum0;
                int normIndex1 = 0;
                if (useSimd && Vector.IsHardwareAccelerated)
                {
                    var vdiv1 = new Vector<float>(sum1);
                    var yv1 = (Vector<float>*)(py + base1);
                    for (; normIndex1 <= block - w; normIndex1 += w)
                    {
                        *yv1 = *yv1 / vdiv1;
                        yv1++;
                    }
                }
                for (; normIndex1 < block; normIndex1++) py[base1 + normIndex1] /= sum1;
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
                        var activated = MathOps.ExpVectorNonpositiveZeroBlocks((*xv + *mv) - vmax);
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
                int normIndex = 0;
                if (useSimd && Vector.IsHardwareAccelerated)
                {
                    var vdiv = new Vector<float>(sum);
                    var yv = (Vector<float>*)(py + baseR);
                    for (; normIndex <= block - w; normIndex += w)
                    {
                        *yv = *yv / vdiv;
                        yv++;
                    }
                }
                for (; normIndex < block; normIndex++) py[baseR + normIndex] /= sum;
            }
        }
    }
}
