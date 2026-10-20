using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using static Lokad.Onnx.MathOps;

namespace Lokad.Onnx.Tests.Support;

// Independent reference/experimental kernels shared by tests and benchmark tools.
// Relocated from the core without changing arithmetic or compilation attributes.
internal static class ReferenceSoftmaxKernels
{

    /// <summary>Legacy scalar summation with vectorized exp, independent of the optimized span softmax.</summary>
    internal static void SoftmaxContiguousFloat(System.Span<float> inputSpan, System.Span<float> outputSpan, int outer, int block, bool useSimd)
    {
        for (int outerIndex = 0; outerIndex < outer; outerIndex++)
        {
            float max = SoftmaxContiguousMax(inputSpan, outerIndex * block, block, useSimd);
            float sum = 0f;
            int expIndex = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vmax = new Vector<float>(max);
                for (; expIndex <= block - width; expIndex += width)
                {
                    int baseIndex = outerIndex * block + expIndex;
                    var activated = MathOps.ExpVector(new Vector<float>(inputSpan.Slice(baseIndex, width)) - vmax);
                    activated.CopyTo(outputSpan.Slice(baseIndex, width));
                    for (int j = 0; j < width; j++) sum += outputSpan[baseIndex + j];
                }
            }
            for (; expIndex < block; expIndex++)
            {
                float activated = MathF.Exp(inputSpan[outerIndex * block + expIndex] - max);
                outputSpan[outerIndex * block + expIndex] = activated;
                sum += activated;
            }
            for (int blockIndex = 0; blockIndex < block; blockIndex++) outputSpan[outerIndex * block + blockIndex] /= sum;
        }
    }

    /// <summary>
    /// Row maximum for the span softmax (E62): vector reduction with an
    /// explicit any-NaN rule reproducing the scalar break-on-first-NaN loop.
    /// Maximum is exact and order-free, so non-NaN results equal the scalar
    /// loop (up to signed zero, which downstream exp absorbs bit-identically).
    /// </summary>
    internal static float SoftmaxContiguousMax(System.Span<float> inputSpan, int start, int block, bool useSimd)
    {
        if (useSimd && Vector.IsHardwareAccelerated && block >= Vector<float>.Count)
        {
            int width = Vector<float>.Count;
            var vmax = new Vector<float>(float.NegativeInfinity);
            var vok = new Vector<int>(-1);
            int i = 0;
            for (; i <= block - width; i += width)
            {
                var v = new Vector<float>(inputSpan.Slice(start + i, width));
                vmax = Vector.Max(vmax, v);
                vok = vok & Vector.Equals(v, v);
            }
            float max = vmax[0];
            for (int l = 1; l < width; l++) if (vmax[l] > max) max = vmax[l];
            for (; i < block; i++)
            {
                float candidate = inputSpan[start + i];
                if (float.IsNaN(candidate)) return float.NaN;
                if (candidate > max) max = candidate;
            }
            for (int l = 0; l < width; l++) if (vok[l] == 0) return float.NaN;
            return max;
        }
        float smax = float.NegativeInfinity;
        for (int blockIndex = 0; blockIndex < block; blockIndex++)
        {
            float candidate = inputSpan[start + blockIndex];
            if (float.IsNaN(candidate)) { smax = float.NaN; break; }
            if (candidate > smax) smax = candidate;
        }
        return smax;
    }
    /// <summary>
    /// E82 helper: pointer-addressed row maximum with identical lane order,
    /// NaN rules and reduction sequence to SoftmaxContiguousMax. Shared by
    /// the inline-max twins so the scan is written once.
    /// </summary>
    internal static unsafe float SoftmaxContiguousMaxPtr(float* px, int baseOff, int block, bool useSimd)
    {
        if (useSimd && Vector.IsHardwareAccelerated && block >= Vector<float>.Count)
        {
            int width = Vector<float>.Count;
            var vmax = new Vector<float>(float.NegativeInfinity);
            var vok = new Vector<int>(-1);
            var xv = (Vector<float>*)(px + baseOff);
            int i = 0;
            for (; i <= block - width; i += width)
            {
                var v = *xv;
                vmax = Vector.Max(vmax, v);
                vok = vok & Vector.Equals(v, v);
                xv++;
            }
            float max = vmax[0];
            for (int l = 1; l < width; l++) if (vmax[l] > max) max = vmax[l];
            for (; i < block; i++)
            {
                float candidate = px[baseOff + i];
                if (float.IsNaN(candidate)) return float.NaN;
                if (candidate > max) max = candidate;
            }
            for (int l = 0; l < width; l++) if (vok[l] == 0) return float.NaN;
            return max;
        }
        float smax = float.NegativeInfinity;
        for (int blockIndex = 0; blockIndex < block; blockIndex++)
        {
            float candidate = px[baseOff + blockIndex];
            if (float.IsNaN(candidate)) { smax = float.NaN; break; }
            if (candidate > smax) smax = candidate;
        }
        return smax;
    }

    /// <summary>
    /// E82 helper: pointer-addressed masked row maximum with identical lane
    /// order, NaN rules and reduction sequence to SoftmaxContiguousMaxMasked.
    /// </summary>
    internal static unsafe float SoftmaxContiguousMaxMaskedPtr(float* px, int baseOff, float* pm, int block, bool useSimd)
    {
        if (useSimd && Vector.IsHardwareAccelerated && block >= Vector<float>.Count)
        {
            int width = Vector<float>.Count;
            var vmax = new Vector<float>(float.NegativeInfinity);
            var vok = new Vector<int>(-1);
            var xv = (Vector<float>*)(px + baseOff);
            var mv = (Vector<float>*)pm;
            int i = 0;
            for (; i <= block - width; i += width)
            {
                var v = *xv + *mv;
                vmax = Vector.Max(vmax, v);
                vok = vok & Vector.Equals(v, v);
                xv++;
                mv++;
            }
            float max = vmax[0];
            for (int l = 1; l < width; l++) if (vmax[l] > max) max = vmax[l];
            for (; i < block; i++)
            {
                float candidate = px[baseOff + i] + pm[i];
                if (float.IsNaN(candidate)) return float.NaN;
                if (candidate > max) max = candidate;
            }
            for (int l = 0; l < width; l++) if (vok[l] == 0) return float.NaN;
            return max;
        }
        float smax = float.NegativeInfinity;
        for (int blockIndex = 0; blockIndex < block; blockIndex++)
        {
            float candidate = px[baseOff + blockIndex] + pm[blockIndex];
            if (float.IsNaN(candidate)) { smax = float.NaN; break; }
            if (candidate > smax) smax = candidate;
        }
        return smax;
    }
    internal static void SoftmaxContiguousFloatSpan(System.Span<float> inputSpan, System.Span<float> outputSpan, int outer, int block, bool useSimd)
    {
        for (int outerIndex = 0; outerIndex < outer; outerIndex++)
        {
            float max = SoftmaxContiguousMax(inputSpan, outerIndex * block, block, useSimd);
            float sum = 0f;
            int expIndex = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vmax = new Vector<float>(max);
                var vsum = Vector<float>.Zero;
                for (; expIndex <= block - width; expIndex += width)
                {
                    int baseIndex = outerIndex * block + expIndex;
                    var activated = MathOps.ExpVectorEstrin(new Vector<float>(inputSpan.Slice(baseIndex, width)) - vmax);
                    activated.CopyTo(outputSpan.Slice(baseIndex, width));
                    vsum += activated;
                }
                sum = Vector.Sum(vsum);
            }
            for (; expIndex < block; expIndex++)
            {
                float activated = MathF.Exp(inputSpan[outerIndex * block + expIndex] - max);
                outputSpan[outerIndex * block + expIndex] = activated;
                sum += activated;
            }
            int normIndex = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vdiv = new Vector<float>(sum);
                for (; normIndex <= block - width; normIndex += width)
                {
                    int baseIndex = outerIndex * block + normIndex;
                    (new Vector<float>(outputSpan.Slice(baseIndex, width)) / vdiv).CopyTo(outputSpan.Slice(baseIndex, width));
                }
            }
            for (; normIndex < block; normIndex++) outputSpan[outerIndex * block + normIndex] /= sum;
        }
    }

    /// <summary>
    /// Row maximum over scores plus a shared mask row (E69): the masked twin
    /// of SoftmaxContiguousMax with identical NaN rules; mask lanes address
    /// the row position, shared across all rows by the caller.
    /// </summary>
    internal static float SoftmaxContiguousMaxMasked(System.Span<float> inputSpan, int start, System.Span<float> maskSpan, int block, bool useSimd)
    {
        if (useSimd && Vector.IsHardwareAccelerated && block >= Vector<float>.Count)
        {
            int width = Vector<float>.Count;
            var vmax = new Vector<float>(float.NegativeInfinity);
            var vok = new Vector<int>(-1);
            int i = 0;
            for (; i <= block - width; i += width)
            {
                var v = new Vector<float>(inputSpan.Slice(start + i, width)) + new Vector<float>(maskSpan.Slice(i, width));
                vmax = Vector.Max(vmax, v);
                vok = vok & Vector.Equals(v, v);
            }
            float max = vmax[0];
            for (int l = 1; l < width; l++) if (vmax[l] > max) max = vmax[l];
            for (; i < block; i++)
            {
                float candidate = inputSpan[start + i] + maskSpan[i];
                if (float.IsNaN(candidate)) return float.NaN;
                if (candidate > max) max = candidate;
            }
            for (int l = 0; l < width; l++) if (vok[l] == 0) return float.NaN;
            return max;
        }
        float smax = float.NegativeInfinity;
        for (int blockIndex = 0; blockIndex < block; blockIndex++)
        {
            float candidate = inputSpan[start + blockIndex] + maskSpan[blockIndex];
            if (float.IsNaN(candidate)) { smax = float.NaN; break; }
            if (candidate > smax) smax = candidate;
        }
        return smax;
    }
    /// <summary>
    /// Softmax over contiguous rows with a shared additive mask row (E69):
    /// bit-identical to Add-then-span-softmax with the same mask, because every
    /// lane adds before consuming in the unfused order and only the materialized
    /// intermediate is skipped. Test-reachable only; no call sites yet.
    /// </summary>
    internal static void SoftmaxMaskedFloatSpan(System.Span<float> inputSpan, System.Span<float> maskSpan, System.Span<float> outputSpan, int outer, int block, bool useSimd)
    {
        if (maskSpan.Length < block) throw new ArgumentException(nameof(maskSpan), "Mask row must cover a full block.");
        for (int outerIndex = 0; outerIndex < outer; outerIndex++)
        {
            float max = SoftmaxContiguousMaxMasked(inputSpan, outerIndex * block, maskSpan, block, useSimd);
            float sum = 0f;
            int expIndex = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vmax = new Vector<float>(max);
                var vsum = Vector<float>.Zero;
                for (; expIndex <= block - width; expIndex += width)
                {
                    int baseIndex = outerIndex * block + expIndex;
                    var activated = MathOps.ExpVectorEstrin((new Vector<float>(inputSpan.Slice(baseIndex, width)) + new Vector<float>(maskSpan.Slice(expIndex, width))) - vmax);
                    activated.CopyTo(outputSpan.Slice(baseIndex, width));
                    vsum += activated;
                }
                sum = Vector.Sum(vsum);
            }
            for (; expIndex < block; expIndex++)
            {
                float activated = MathF.Exp((inputSpan[outerIndex * block + expIndex] + maskSpan[expIndex]) - max);
                outputSpan[outerIndex * block + expIndex] = activated;
                sum += activated;
            }
            int normIndex = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vdiv = new Vector<float>(sum);
                for (; normIndex <= block - width; normIndex += width)
                {
                    int baseIndex = outerIndex * block + normIndex;
                    (new Vector<float>(outputSpan.Slice(baseIndex, width)) / vdiv).CopyTo(outputSpan.Slice(baseIndex, width));
                }
            }
            for (; normIndex < block; normIndex++) outputSpan[outerIndex * block + normIndex] /= sum;
        }
    }
    /// <summary>
    /// <summary>
    /// E73 twin: row-pair unrolled masked softmax. Same per-element arithmetic
    /// and order as SoftmaxMaskedFloatSpan, two independent rows per iteration
    /// for instruction-level parallelism. Test-reachable only; no dispatch yet.
    /// </summary>
    internal static void SoftmaxMaskedFloatSpan2x(System.Span<float> inputSpan, System.Span<float> maskSpan, System.Span<float> outputSpan, int outer, int block, bool useSimd)
    {
        if (maskSpan.Length < block) throw new ArgumentException(nameof(maskSpan), "Mask row must cover a full block.");
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
                int width = Vector<float>.Count;
                var vmax0 = new Vector<float>(max0);
                var vsum0 = Vector<float>.Zero;
                for (; expIndex0 <= block - width; expIndex0 += width)
                {
                    int baseIndex0 = base0 + expIndex0;
                    var activated0 = MathOps.ExpVectorEstrin((new Vector<float>(inputSpan.Slice(baseIndex0, width)) + new Vector<float>(maskSpan.Slice(expIndex0, width))) - vmax0);
                    activated0.CopyTo(outputSpan.Slice(baseIndex0, width));
                    vsum0 += activated0;
                }
                sum0 = Vector.Sum(vsum0);
            }
            for (; expIndex0 < block; expIndex0++)
            {
                float activated0 = MathF.Exp((inputSpan[base0 + expIndex0] + maskSpan[expIndex0]) - max0);
                outputSpan[base0 + expIndex0] = activated0;
                sum0 += activated0;
            }
            float sum1 = 0f;
            int expIndex1 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vmax1 = new Vector<float>(max1);
                var vsum1 = Vector<float>.Zero;
                for (; expIndex1 <= block - width; expIndex1 += width)
                {
                    int baseIndex1 = base1 + expIndex1;
                    var activated1 = MathOps.ExpVectorEstrin((new Vector<float>(inputSpan.Slice(baseIndex1, width)) + new Vector<float>(maskSpan.Slice(expIndex1, width))) - vmax1);
                    activated1.CopyTo(outputSpan.Slice(baseIndex1, width));
                    vsum1 += activated1;
                }
                sum1 = Vector.Sum(vsum1);
            }
            for (; expIndex1 < block; expIndex1++)
            {
                float activated1 = MathF.Exp((inputSpan[base1 + expIndex1] + maskSpan[expIndex1]) - max1);
                outputSpan[base1 + expIndex1] = activated1;
                sum1 += activated1;
            }
            int normIndex0 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vdiv0 = new Vector<float>(sum0);
                for (; normIndex0 <= block - width; normIndex0 += width)
                {
                    int baseIndex0 = base0 + normIndex0;
                    (new Vector<float>(outputSpan.Slice(baseIndex0, width)) / vdiv0).CopyTo(outputSpan.Slice(baseIndex0, width));
                }
            }
            for (; normIndex0 < block; normIndex0++) outputSpan[base0 + normIndex0] /= sum0;
            int normIndex1 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vdiv1 = new Vector<float>(sum1);
                for (; normIndex1 <= block - width; normIndex1 += width)
                {
                    int baseIndex1 = base1 + normIndex1;
                    (new Vector<float>(outputSpan.Slice(baseIndex1, width)) / vdiv1).CopyTo(outputSpan.Slice(baseIndex1, width));
                }
            }
            for (; normIndex1 < block; normIndex1++) outputSpan[base1 + normIndex1] /= sum1;
        }
        for (int outerIndex = pairs * 2; outerIndex < outer; outerIndex++)
        {
            float max = SoftmaxContiguousMaxMasked(inputSpan, outerIndex * block, maskSpan, block, useSimd);
            float sum = 0f;
            int expIndex = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vmax = new Vector<float>(max);
                var vsum = Vector<float>.Zero;
                for (; expIndex <= block - width; expIndex += width)
                {
                    int baseIndex = outerIndex * block + expIndex;
                    var activated = MathOps.ExpVectorEstrin((new Vector<float>(inputSpan.Slice(baseIndex, width)) + new Vector<float>(maskSpan.Slice(expIndex, width))) - vmax);
                    activated.CopyTo(outputSpan.Slice(baseIndex, width));
                    vsum += activated;
                }
                sum = Vector.Sum(vsum);
            }
            for (; expIndex < block; expIndex++)
            {
                float activated = MathF.Exp((inputSpan[outerIndex * block + expIndex] + maskSpan[expIndex]) - max);
                outputSpan[outerIndex * block + expIndex] = activated;
                sum += activated;
            }
            int normIndex = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vdiv = new Vector<float>(sum);
                for (; normIndex <= block - width; normIndex += width)
                {
                    int baseIndex = outerIndex * block + normIndex;
                    (new Vector<float>(outputSpan.Slice(baseIndex, width)) / vdiv).CopyTo(outputSpan.Slice(baseIndex, width));
                }
            }
            for (; normIndex < block; normIndex++) outputSpan[outerIndex * block + normIndex] /= sum;
        }
    }

    /// <summary>
    /// E73 twin: row-pair unrolled plain softmax. Same per-element arithmetic
    /// and order as SoftmaxContiguousFloatSpan, two independent rows per
    /// iteration. Test-reachable only; no dispatch yet.
    /// </summary>
    internal static void SoftmaxContiguousFloatSpan2x(System.Span<float> inputSpan, System.Span<float> outputSpan, int outer, int block, bool useSimd)
    {
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
                int width = Vector<float>.Count;
                var vmax0 = new Vector<float>(max0);
                var vsum0 = Vector<float>.Zero;
                for (; expIndex0 <= block - width; expIndex0 += width)
                {
                    int baseIndex0 = base0 + expIndex0;
                    var activated0 = MathOps.ExpVectorEstrin(new Vector<float>(inputSpan.Slice(baseIndex0, width)) - vmax0);
                    activated0.CopyTo(outputSpan.Slice(baseIndex0, width));
                    vsum0 += activated0;
                }
                sum0 = Vector.Sum(vsum0);
            }
            for (; expIndex0 < block; expIndex0++)
            {
                float activated0 = MathF.Exp(inputSpan[base0 + expIndex0] - max0);
                outputSpan[base0 + expIndex0] = activated0;
                sum0 += activated0;
            }
            float sum1 = 0f;
            int expIndex1 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vmax1 = new Vector<float>(max1);
                var vsum1 = Vector<float>.Zero;
                for (; expIndex1 <= block - width; expIndex1 += width)
                {
                    int baseIndex1 = base1 + expIndex1;
                    var activated1 = MathOps.ExpVectorEstrin(new Vector<float>(inputSpan.Slice(baseIndex1, width)) - vmax1);
                    activated1.CopyTo(outputSpan.Slice(baseIndex1, width));
                    vsum1 += activated1;
                }
                sum1 = Vector.Sum(vsum1);
            }
            for (; expIndex1 < block; expIndex1++)
            {
                float activated1 = MathF.Exp(inputSpan[base1 + expIndex1] - max1);
                outputSpan[base1 + expIndex1] = activated1;
                sum1 += activated1;
            }
            int normIndex0 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vdiv0 = new Vector<float>(sum0);
                for (; normIndex0 <= block - width; normIndex0 += width)
                {
                    int baseIndex0 = base0 + normIndex0;
                    (new Vector<float>(outputSpan.Slice(baseIndex0, width)) / vdiv0).CopyTo(outputSpan.Slice(baseIndex0, width));
                }
            }
            for (; normIndex0 < block; normIndex0++) outputSpan[base0 + normIndex0] /= sum0;
            int normIndex1 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vdiv1 = new Vector<float>(sum1);
                for (; normIndex1 <= block - width; normIndex1 += width)
                {
                    int baseIndex1 = base1 + normIndex1;
                    (new Vector<float>(outputSpan.Slice(baseIndex1, width)) / vdiv1).CopyTo(outputSpan.Slice(baseIndex1, width));
                }
            }
            for (; normIndex1 < block; normIndex1++) outputSpan[base1 + normIndex1] /= sum1;
        }
        for (int outerIndex = pairs * 2; outerIndex < outer; outerIndex++)
        {
            float max = SoftmaxContiguousMax(inputSpan, outerIndex * block, block, useSimd);
            float sum = 0f;
            int expIndex = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vmax = new Vector<float>(max);
                var vsum = Vector<float>.Zero;
                for (; expIndex <= block - width; expIndex += width)
                {
                    int baseIndex = outerIndex * block + expIndex;
                    var activated = MathOps.ExpVectorEstrin(new Vector<float>(inputSpan.Slice(baseIndex, width)) - vmax);
                    activated.CopyTo(outputSpan.Slice(baseIndex, width));
                    vsum += activated;
                }
                sum = Vector.Sum(vsum);
            }
            for (; expIndex < block; expIndex++)
            {
                float activated = MathF.Exp(inputSpan[outerIndex * block + expIndex] - max);
                outputSpan[outerIndex * block + expIndex] = activated;
                sum += activated;
            }
            int normIndex = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vdiv = new Vector<float>(sum);
                for (; normIndex <= block - width; normIndex += width)
                {
                    int baseIndex = outerIndex * block + normIndex;
                    (new Vector<float>(outputSpan.Slice(baseIndex, width)) / vdiv).CopyTo(outputSpan.Slice(baseIndex, width));
                }
            }
            for (; normIndex < block; normIndex++) outputSpan[outerIndex * block + normIndex] /= sum;
        }
    }
    /// <summary>
    /// E78 twin: row-quad unrolled masked softmax. Same per-element arithmetic
    /// and order as SoftmaxMaskedFloatSpan, four independent rows per iteration.
    /// Test-reachable only; no dispatch yet.
    /// </summary>
    internal static void SoftmaxMaskedFloatSpan4x(System.Span<float> inputSpan, System.Span<float> maskSpan, System.Span<float> outputSpan, int outer, int block, bool useSimd)
    {
        if (maskSpan.Length < block) throw new ArgumentException(nameof(maskSpan), "Mask row must cover a full block.");
        int quads = outer / 4;
        for (int p = 0; p < quads; p++)
        {
            int base0 = (4 * p) * block;
            int base1 = (4 * p + 1) * block;
            int base2 = (4 * p + 2) * block;
            int base3 = (4 * p + 3) * block;
            float max0 = SoftmaxContiguousMaxMasked(inputSpan, base0, maskSpan, block, useSimd);
            float max1 = SoftmaxContiguousMaxMasked(inputSpan, base1, maskSpan, block, useSimd);
            float max2 = SoftmaxContiguousMaxMasked(inputSpan, base2, maskSpan, block, useSimd);
            float max3 = SoftmaxContiguousMaxMasked(inputSpan, base3, maskSpan, block, useSimd);
            float sum0 = 0f;
            int expIndex0 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vmax0 = new Vector<float>(max0);
                var vsum0 = Vector<float>.Zero;
                for (; expIndex0 <= block - width; expIndex0 += width)
                {
                    int baseIndex0 = base0 + expIndex0;
                    var activated0 = MathOps.ExpVectorEstrin((new Vector<float>(inputSpan.Slice(baseIndex0, width)) + new Vector<float>(maskSpan.Slice(expIndex0, width))) - vmax0);
                    activated0.CopyTo(outputSpan.Slice(baseIndex0, width));
                    vsum0 += activated0;
                }
                sum0 = Vector.Sum(vsum0);
            }
            for (; expIndex0 < block; expIndex0++)
            {
                float activated0 = MathF.Exp((inputSpan[base0 + expIndex0] + maskSpan[expIndex0]) - max0);
                outputSpan[base0 + expIndex0] = activated0;
                sum0 += activated0;
            }
            float sum1 = 0f;
            int expIndex1 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vmax1 = new Vector<float>(max1);
                var vsum1 = Vector<float>.Zero;
                for (; expIndex1 <= block - width; expIndex1 += width)
                {
                    int baseIndex1 = base1 + expIndex1;
                    var activated1 = MathOps.ExpVectorEstrin((new Vector<float>(inputSpan.Slice(baseIndex1, width)) + new Vector<float>(maskSpan.Slice(expIndex1, width))) - vmax1);
                    activated1.CopyTo(outputSpan.Slice(baseIndex1, width));
                    vsum1 += activated1;
                }
                sum1 = Vector.Sum(vsum1);
            }
            for (; expIndex1 < block; expIndex1++)
            {
                float activated1 = MathF.Exp((inputSpan[base1 + expIndex1] + maskSpan[expIndex1]) - max1);
                outputSpan[base1 + expIndex1] = activated1;
                sum1 += activated1;
            }
            float sum2 = 0f;
            int expIndex2 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vmax2 = new Vector<float>(max2);
                var vsum2 = Vector<float>.Zero;
                for (; expIndex2 <= block - width; expIndex2 += width)
                {
                    int baseIndex2 = base2 + expIndex2;
                    var activated2 = MathOps.ExpVectorEstrin((new Vector<float>(inputSpan.Slice(baseIndex2, width)) + new Vector<float>(maskSpan.Slice(expIndex2, width))) - vmax2);
                    activated2.CopyTo(outputSpan.Slice(baseIndex2, width));
                    vsum2 += activated2;
                }
                sum2 = Vector.Sum(vsum2);
            }
            for (; expIndex2 < block; expIndex2++)
            {
                float activated2 = MathF.Exp((inputSpan[base2 + expIndex2] + maskSpan[expIndex2]) - max2);
                outputSpan[base2 + expIndex2] = activated2;
                sum2 += activated2;
            }
            float sum3 = 0f;
            int expIndex3 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vmax3 = new Vector<float>(max3);
                var vsum3 = Vector<float>.Zero;
                for (; expIndex3 <= block - width; expIndex3 += width)
                {
                    int baseIndex3 = base3 + expIndex3;
                    var activated3 = MathOps.ExpVectorEstrin((new Vector<float>(inputSpan.Slice(baseIndex3, width)) + new Vector<float>(maskSpan.Slice(expIndex3, width))) - vmax3);
                    activated3.CopyTo(outputSpan.Slice(baseIndex3, width));
                    vsum3 += activated3;
                }
                sum3 = Vector.Sum(vsum3);
            }
            for (; expIndex3 < block; expIndex3++)
            {
                float activated3 = MathF.Exp((inputSpan[base3 + expIndex3] + maskSpan[expIndex3]) - max3);
                outputSpan[base3 + expIndex3] = activated3;
                sum3 += activated3;
            }
            int normIndex0 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vdiv0 = new Vector<float>(sum0);
                for (; normIndex0 <= block - width; normIndex0 += width)
                {
                    int baseIndex0 = base0 + normIndex0;
                    (new Vector<float>(outputSpan.Slice(baseIndex0, width)) / vdiv0).CopyTo(outputSpan.Slice(baseIndex0, width));
                }
            }
            for (; normIndex0 < block; normIndex0++) outputSpan[base0 + normIndex0] /= sum0;
            int normIndex1 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vdiv1 = new Vector<float>(sum1);
                for (; normIndex1 <= block - width; normIndex1 += width)
                {
                    int baseIndex1 = base1 + normIndex1;
                    (new Vector<float>(outputSpan.Slice(baseIndex1, width)) / vdiv1).CopyTo(outputSpan.Slice(baseIndex1, width));
                }
            }
            for (; normIndex1 < block; normIndex1++) outputSpan[base1 + normIndex1] /= sum1;
            int normIndex2 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vdiv2 = new Vector<float>(sum2);
                for (; normIndex2 <= block - width; normIndex2 += width)
                {
                    int baseIndex2 = base2 + normIndex2;
                    (new Vector<float>(outputSpan.Slice(baseIndex2, width)) / vdiv2).CopyTo(outputSpan.Slice(baseIndex2, width));
                }
            }
            for (; normIndex2 < block; normIndex2++) outputSpan[base2 + normIndex2] /= sum2;
            int normIndex3 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vdiv3 = new Vector<float>(sum3);
                for (; normIndex3 <= block - width; normIndex3 += width)
                {
                    int baseIndex3 = base3 + normIndex3;
                    (new Vector<float>(outputSpan.Slice(baseIndex3, width)) / vdiv3).CopyTo(outputSpan.Slice(baseIndex3, width));
                }
            }
            for (; normIndex3 < block; normIndex3++) outputSpan[base3 + normIndex3] /= sum3;
        }
        for (int outerIndex = quads * 4; outerIndex < outer; outerIndex++)
        {
            float max = SoftmaxContiguousMaxMasked(inputSpan, outerIndex * block, maskSpan, block, useSimd);
            float sum = 0f;
            int expIndex = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vmax = new Vector<float>(max);
                var vsum = Vector<float>.Zero;
                for (; expIndex <= block - width; expIndex += width)
                {
                    int baseIndex = outerIndex * block + expIndex;
                    var activated = MathOps.ExpVectorEstrin((new Vector<float>(inputSpan.Slice(baseIndex, width)) + new Vector<float>(maskSpan.Slice(expIndex, width))) - vmax);
                    activated.CopyTo(outputSpan.Slice(baseIndex, width));
                    vsum += activated;
                }
                sum = Vector.Sum(vsum);
            }
            for (; expIndex < block; expIndex++)
            {
                float activated = MathF.Exp((inputSpan[outerIndex * block + expIndex] + maskSpan[expIndex]) - max);
                outputSpan[outerIndex * block + expIndex] = activated;
                sum += activated;
            }
            int normIndex = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vdiv = new Vector<float>(sum);
                for (; normIndex <= block - width; normIndex += width)
                {
                    int baseIndex = outerIndex * block + normIndex;
                    (new Vector<float>(outputSpan.Slice(baseIndex, width)) / vdiv).CopyTo(outputSpan.Slice(baseIndex, width));
                }
            }
            for (; normIndex < block; normIndex++) outputSpan[outerIndex * block + normIndex] /= sum;
        }
    }
    /// <summary>
    /// E78 twin: row-quad unrolled plain softmax. Same per-element arithmetic
    /// and order as SoftmaxContiguousFloatSpan, four independent rows per
    /// iteration. Test-reachable only; no dispatch yet.
    /// </summary>
    internal static void SoftmaxContiguousFloatSpan4x(System.Span<float> inputSpan, System.Span<float> outputSpan, int outer, int block, bool useSimd)
    {
        int quads = outer / 4;
        for (int p = 0; p < quads; p++)
        {
            int base0 = (4 * p) * block;
            int base1 = (4 * p + 1) * block;
            int base2 = (4 * p + 2) * block;
            int base3 = (4 * p + 3) * block;
            float max0 = SoftmaxContiguousMax(inputSpan, base0, block, useSimd);
            float max1 = SoftmaxContiguousMax(inputSpan, base1, block, useSimd);
            float max2 = SoftmaxContiguousMax(inputSpan, base2, block, useSimd);
            float max3 = SoftmaxContiguousMax(inputSpan, base3, block, useSimd);
            float sum0 = 0f;
            int expIndex0 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vmax0 = new Vector<float>(max0);
                var vsum0 = Vector<float>.Zero;
                for (; expIndex0 <= block - width; expIndex0 += width)
                {
                    int baseIndex0 = base0 + expIndex0;
                    var activated0 = MathOps.ExpVectorEstrin(new Vector<float>(inputSpan.Slice(baseIndex0, width)) - vmax0);
                    activated0.CopyTo(outputSpan.Slice(baseIndex0, width));
                    vsum0 += activated0;
                }
                sum0 = Vector.Sum(vsum0);
            }
            for (; expIndex0 < block; expIndex0++)
            {
                float activated0 = MathF.Exp(inputSpan[base0 + expIndex0] - max0);
                outputSpan[base0 + expIndex0] = activated0;
                sum0 += activated0;
            }
            float sum1 = 0f;
            int expIndex1 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vmax1 = new Vector<float>(max1);
                var vsum1 = Vector<float>.Zero;
                for (; expIndex1 <= block - width; expIndex1 += width)
                {
                    int baseIndex1 = base1 + expIndex1;
                    var activated1 = MathOps.ExpVectorEstrin(new Vector<float>(inputSpan.Slice(baseIndex1, width)) - vmax1);
                    activated1.CopyTo(outputSpan.Slice(baseIndex1, width));
                    vsum1 += activated1;
                }
                sum1 = Vector.Sum(vsum1);
            }
            for (; expIndex1 < block; expIndex1++)
            {
                float activated1 = MathF.Exp(inputSpan[base1 + expIndex1] - max1);
                outputSpan[base1 + expIndex1] = activated1;
                sum1 += activated1;
            }
            float sum2 = 0f;
            int expIndex2 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vmax2 = new Vector<float>(max2);
                var vsum2 = Vector<float>.Zero;
                for (; expIndex2 <= block - width; expIndex2 += width)
                {
                    int baseIndex2 = base2 + expIndex2;
                    var activated2 = MathOps.ExpVectorEstrin(new Vector<float>(inputSpan.Slice(baseIndex2, width)) - vmax2);
                    activated2.CopyTo(outputSpan.Slice(baseIndex2, width));
                    vsum2 += activated2;
                }
                sum2 = Vector.Sum(vsum2);
            }
            for (; expIndex2 < block; expIndex2++)
            {
                float activated2 = MathF.Exp(inputSpan[base2 + expIndex2] - max2);
                outputSpan[base2 + expIndex2] = activated2;
                sum2 += activated2;
            }
            float sum3 = 0f;
            int expIndex3 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vmax3 = new Vector<float>(max3);
                var vsum3 = Vector<float>.Zero;
                for (; expIndex3 <= block - width; expIndex3 += width)
                {
                    int baseIndex3 = base3 + expIndex3;
                    var activated3 = MathOps.ExpVectorEstrin(new Vector<float>(inputSpan.Slice(baseIndex3, width)) - vmax3);
                    activated3.CopyTo(outputSpan.Slice(baseIndex3, width));
                    vsum3 += activated3;
                }
                sum3 = Vector.Sum(vsum3);
            }
            for (; expIndex3 < block; expIndex3++)
            {
                float activated3 = MathF.Exp(inputSpan[base3 + expIndex3] - max3);
                outputSpan[base3 + expIndex3] = activated3;
                sum3 += activated3;
            }
            int normIndex0 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vdiv0 = new Vector<float>(sum0);
                for (; normIndex0 <= block - width; normIndex0 += width)
                {
                    int baseIndex0 = base0 + normIndex0;
                    (new Vector<float>(outputSpan.Slice(baseIndex0, width)) / vdiv0).CopyTo(outputSpan.Slice(baseIndex0, width));
                }
            }
            for (; normIndex0 < block; normIndex0++) outputSpan[base0 + normIndex0] /= sum0;
            int normIndex1 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vdiv1 = new Vector<float>(sum1);
                for (; normIndex1 <= block - width; normIndex1 += width)
                {
                    int baseIndex1 = base1 + normIndex1;
                    (new Vector<float>(outputSpan.Slice(baseIndex1, width)) / vdiv1).CopyTo(outputSpan.Slice(baseIndex1, width));
                }
            }
            for (; normIndex1 < block; normIndex1++) outputSpan[base1 + normIndex1] /= sum1;
            int normIndex2 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vdiv2 = new Vector<float>(sum2);
                for (; normIndex2 <= block - width; normIndex2 += width)
                {
                    int baseIndex2 = base2 + normIndex2;
                    (new Vector<float>(outputSpan.Slice(baseIndex2, width)) / vdiv2).CopyTo(outputSpan.Slice(baseIndex2, width));
                }
            }
            for (; normIndex2 < block; normIndex2++) outputSpan[base2 + normIndex2] /= sum2;
            int normIndex3 = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vdiv3 = new Vector<float>(sum3);
                for (; normIndex3 <= block - width; normIndex3 += width)
                {
                    int baseIndex3 = base3 + normIndex3;
                    (new Vector<float>(outputSpan.Slice(baseIndex3, width)) / vdiv3).CopyTo(outputSpan.Slice(baseIndex3, width));
                }
            }
            for (; normIndex3 < block; normIndex3++) outputSpan[base3 + normIndex3] /= sum3;
        }
        for (int outerIndex = quads * 4; outerIndex < outer; outerIndex++)
        {
            float max = SoftmaxContiguousMax(inputSpan, outerIndex * block, block, useSimd);
            float sum = 0f;
            int expIndex = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vmax = new Vector<float>(max);
                var vsum = Vector<float>.Zero;
                for (; expIndex <= block - width; expIndex += width)
                {
                    int baseIndex = outerIndex * block + expIndex;
                    var activated = MathOps.ExpVectorEstrin(new Vector<float>(inputSpan.Slice(baseIndex, width)) - vmax);
                    activated.CopyTo(outputSpan.Slice(baseIndex, width));
                    vsum += activated;
                }
                sum = Vector.Sum(vsum);
            }
            for (; expIndex < block; expIndex++)
            {
                float activated = MathF.Exp(inputSpan[outerIndex * block + expIndex] - max);
                outputSpan[outerIndex * block + expIndex] = activated;
                sum += activated;
            }
            int normIndex = 0;
            if (useSimd && Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                var vdiv = new Vector<float>(sum);
                for (; normIndex <= block - width; normIndex += width)
                {
                    int baseIndex = outerIndex * block + normIndex;
                    (new Vector<float>(outputSpan.Slice(baseIndex, width)) / vdiv).CopyTo(outputSpan.Slice(baseIndex, width));
                }
            }
            for (; normIndex < block; normIndex++) outputSpan[outerIndex * block + normIndex] /= sum;
        }
    }
    /// E80 twin: inline-max masked softmax (E82): pointer loops with the maximum scan inlined. Same
    /// per-element arithmetic and order as SoftmaxMaskedFloatSpan2x with raw
    /// vector pointers instead of per-vector Slice plus Cast, removing
    /// bounds-check setup from the hot loops and scalar tails.
    /// Test-reachable only; no dispatch yet.
    /// </summary>
    internal static unsafe void SoftmaxMaskedFloatSpanPtrMax(System.Span<float> inputSpan, System.Span<float> maskSpan, System.Span<float> outputSpan, int outer, int block, bool useSimd)
    {
        if (maskSpan.Length < block) throw new ArgumentException(nameof(maskSpan), "Mask row must cover a full block.");
        fixed (float* px = inputSpan, pm = maskSpan, py = outputSpan)
        {
            int w = Vector<float>.Count;
            int pairs = outer / 2;
            for (int p = 0; p < pairs; p++)
            {
                int base0 = (2 * p) * block;
                int base1 = (2 * p + 1) * block;
                float max0 = SoftmaxContiguousMaxMaskedPtr(px, base0, pm, block, useSimd);
                float max1 = SoftmaxContiguousMaxMaskedPtr(px, base1, pm, block, useSimd);
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
                float max = SoftmaxContiguousMaxMaskedPtr(px, baseR, pm, block, useSimd);
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
    /// <summary>
    /// E80 twin: inline-max plain softmax (E82): pointer loops with the maximum scan inlined. Same
    /// per-element arithmetic and order as SoftmaxContiguousFloatSpan2x with
    /// raw vector pointers instead of per-vector Slice plus Cast.
    /// Test-reachable only; no dispatch yet.
    /// </summary>
    internal static unsafe void SoftmaxContiguousFloatSpanPtrMax(System.Span<float> inputSpan, System.Span<float> outputSpan, int outer, int block, bool useSimd)
    {
        fixed (float* px = inputSpan, py = outputSpan)
        {
            int w = Vector<float>.Count;
            int pairs = outer / 2;
            for (int p = 0; p < pairs; p++)
            {
                int base0 = (2 * p) * block;
                int base1 = (2 * p + 1) * block;
                float max0 = SoftmaxContiguousMaxPtr(px, base0, block, useSimd);
                float max1 = SoftmaxContiguousMaxPtr(px, base1, block, useSimd);
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
                float max = SoftmaxContiguousMaxPtr(px, baseR, block, useSimd);
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
