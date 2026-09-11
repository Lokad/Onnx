namespace Lokad.Onnx;

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

using static Lokad.Onnx.MathOps;
using static Lokad.Onnx.Profiler;

public abstract partial class Tensor<T> : TensorBase, IList, IList<T>, IReadOnlyList<T>, IStructuralComparable, IStructuralEquatable, ITensor, INumericTensor
where T : unmanaged
{
    // Shared reduction setup: resolves keepdims and builds the validated plan.
    // The keepdims default differs by operation (ReduceMax-18 keeps by default).
    // One validated plan owns absent/empty axes, normalization, dedupe and
    // keepdims (ORT 1.29: absent/empty + noop is a no-op; dupes reduce once).
    static ReductionPlan PlanReduction(int rank, Tensor<int>? axes, bool? keepDims, bool defaultKeepDims, bool? noopWithEmptyAxes, out bool resolvedKeepDims)
    {
        resolvedKeepDims = keepDims.HasValue ? keepDims.Value : defaultKeepDims;
        return ReductionPlan.Create(rank, axes, resolvedKeepDims, noopWithEmptyAxes.HasValue && noopWithEmptyAxes.Value);
    }

    static Tensor<TElement> ApplyKeepDims<TElement>(Tensor<TElement> output, ReductionPlan plan, bool keepDims) where TElement : unmanaged
        => keepDims ? Tensor<TElement>.Unsqueeze(output, plan.Axes) : output;

    /// <summary>
    /// Shared reduction orchestration: optional transpose to innermost axes,
    /// densified standard input, output shape and inner extent. Typed kernels
    /// accumulate over the returned spans; empty and no-op cases are resolved
    /// by the caller through the shared plan.
    /// </summary>
    static (DenseTensor<TElement> input, int[] outputShape, int inner) PrepareReduction<TElement>(Tensor<TElement> data, int[] axes) where TElement : unmanaged
    {
        var permutation = ArrayUtilities.GetAxesPermutationForReduction(axes, data.Rank);
        Tensor<TElement> pdata;
        int[] paxes;
        if (permutation is not null)
        {
            pdata = Tensor<TElement>.Transpose(data, permutation);
            paxes = ArrayUtilities.GetInnerMostAxes(axes.Length, data.Rank);
        }
        else
        {
            pdata = data;
            paxes = axes;
        }
        var (oshape, rshape) = ArrayUtilities.ComputeShapesForReduction(pdata.dimensions, paxes);
        int r = ArrayUtilities.ComputeOffsetForReduction(rshape, 0);
        DenseTensor<TElement> dense;
        if (pdata is DenseTensor<TElement> d && !d.IsReversedStride && HasStandardStrides(d) && d.Buffer.Length == (int)d.Length)
            dense = d;
        else
            dense = pdata.ToDenseTensor();
        return (dense, oshape, r);
    }

    public static Tensor<int> ReduceSum(Tensor<int> data, Tensor<int>? axes) => ReduceSum(data, axes, null, null);

        public static Tensor<int> ReduceSum(Tensor<int> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
    {
        StartOpStage(OpStage.ValidateArguments);
        var plan = PlanReduction(data.Rank, axes, _keepDims, false, _noOpWithEmptyAxes, out var keepDims);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<int>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            // Integer sums saturate (ORT 1.29), unlike elementwise
            // arithmetic which wraps: accumulate wide, then clamp.
            long sum = 0L;
            for (int j = 0; j < r; ++j) sum += xs[offset + j];
            os[i] = sum > int.MaxValue ? int.MaxValue : sum < int.MinValue ? int.MinValue : (int)sum;
        }
        return ApplyKeepDims(output, plan, keepDims);
    }

    public static Tensor<long> ReduceSum(Tensor<long> data, Tensor<int>? axes) => ReduceSum(data, axes, null, null);

        public static Tensor<long> ReduceSum(Tensor<long> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
    {
        StartOpStage(OpStage.ValidateArguments);
        var plan = PlanReduction(data.Rank, axes, _keepDims, false, _noOpWithEmptyAxes, out var keepDims);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<long>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            // Integer sums saturate (ORT 1.29), and int64 probes show the
            // accumulation itself runs in double: extreme vectors like
            // [Max, Max, Min, Min] yield 0, impossible for running
            // saturation (Min) or wide-then-clamp (-2). In-range double
            // sums of integers are exact, so only the final cast saturates.
            double sum = 0.0;
            for (int j = 0; j < r; ++j) sum += xs[offset + j];
            os[i] = sum >= 9223372036854775808.0 ? long.MaxValue : sum <= -9223372036854775808.0 ? long.MinValue : (long)sum;
        }
        return ApplyKeepDims(output, plan, keepDims);
    }

    public static Tensor<float> ReduceSum(Tensor<float> data, Tensor<int>? axes) => ReduceSum(data, axes, null, null);

        public static Tensor<float> ReduceSum(Tensor<float> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
    {
        StartOpStage(OpStage.ValidateArguments);
        var plan = PlanReduction(data.Rank, axes, _keepDims, false, _noOpWithEmptyAxes, out var keepDims);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<float>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            float sum = 0f;
            for (int j = 0; j < r; ++j) sum += xs[offset + j];
            os[i] = sum;
        }
        return ApplyKeepDims(output, plan, keepDims);
    }

    public static Tensor<double> ReduceSum(Tensor<double> data, Tensor<int>? axes) => ReduceSum(data, axes, null, null);

        public static Tensor<double> ReduceSum(Tensor<double> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
    {
        StartOpStage(OpStage.ValidateArguments);
        var plan = PlanReduction(data.Rank, axes, _keepDims, false, _noOpWithEmptyAxes, out var keepDims);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<double>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            double sum = 0.0;
            for (int j = 0; j < r; ++j) sum += xs[offset + j];
            os[i] = sum;
        }
        return ApplyKeepDims(output, plan, keepDims);
    }

    public static Tensor<int> ReduceMean(Tensor<int> data, Tensor<int>? axes) => ReduceMean(data, axes, null, null);

        public static Tensor<int> ReduceMean(Tensor<int> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
        => ReduceMean(data, axes, _keepDims, _noOpWithEmptyAxes, TensorExecutionOptions.Auto);

    public static Tensor<int> ReduceMean(Tensor<int> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes, TensorExecutionOptions options)
    {
        StartOpStage(OpStage.ValidateArguments);
        var plan = PlanReduction(data.Rank, axes, _keepDims, false, _noOpWithEmptyAxes, out var keepDims);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<int>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            // Empty reductions yield zero; without the guard the quotient
            // below divides by zero (ORT 1.29 agrees on 0 for every dtype).
            if (r == 0)
            {
                os[i] = 0;
                continue;
            }
            long sum = 0L;
            for (int j = 0; j < r; ++j) sum += xs[offset + j];
            // The quotient always fits: |mean| <= max|x|. No clamp needed.
            os[i] = (int)(sum / r);
        }
        return ApplyKeepDims(output, plan, keepDims);
    }

    public static Tensor<float> ReduceMean(Tensor<float> data, Tensor<int>? axes) => ReduceMean(data, axes, null, null);

        public static Tensor<float> ReduceMean(Tensor<float> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
        => ReduceMean(data, axes, _keepDims, _noOpWithEmptyAxes, TensorExecutionOptions.Auto);

    public static Tensor<float> ReduceMean(Tensor<float> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes, TensorExecutionOptions options)
    {
        options.Validate();
        StartOpStage(OpStage.ValidateArguments);
        var plan = PlanReduction(data.Rank, axes, _keepDims, false, _noOpWithEmptyAxes, out var keepDims);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<float>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            // Empty reductions yield zero instead of NaN (ORT 1.29).
            if (r == 0)
            {
                os[i] = 0f;
                continue;
            }
            float sum = 0f;
            for (int j = 0; j < r; ++j) sum += xs[offset + j];
            os[i] = sum / r;
        }
        return ApplyKeepDims(output, plan, keepDims);
    }

    public static Tensor<double> ReduceMean(Tensor<double> data, Tensor<int>? axes) => ReduceMean(data, axes, null, null);

        public static Tensor<double> ReduceMean(Tensor<double> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
        => ReduceMean(data, axes, _keepDims, _noOpWithEmptyAxes, TensorExecutionOptions.Auto);

    public static Tensor<double> ReduceMean(Tensor<double> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes, TensorExecutionOptions options)
    {
        options.Validate();
        StartOpStage(OpStage.ValidateArguments);
        var plan = PlanReduction(data.Rank, axes, _keepDims, false, _noOpWithEmptyAxes, out var keepDims);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<double>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            // Empty reductions yield zero instead of NaN (ORT 1.29).
            if (r == 0)
            {
                os[i] = 0.0;
                continue;
            }
            double sum = 0.0;
            for (int j = 0; j < r; ++j) sum += xs[offset + j];
            os[i] = sum / r;
        }
        return ApplyKeepDims(output, plan, keepDims);
    }

    public static Tensor<float> ReduceMax(Tensor<float> data, Tensor<int>? axes) => ReduceMax(data, axes, null, null);

        public static Tensor<float> ReduceMax(Tensor<float> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
    {
        StartOpStage(OpStage.ValidateArguments);
        var plan = PlanReduction(data.Rank, axes, _keepDims, true, _noOpWithEmptyAxes, out var keepDims);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<float>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            // Empty reductions yield -Infinity; init from the first element so
            // all-negative spans keep their true max and a leading NaN persists (ORT).
            if (r == 0)
            {
                os[i] = float.NegativeInfinity;
                continue;
            }
            float max = xs[offset];
            for (int j = 1; j < r; ++j)
            {
                float v = xs[offset + j];
                if (v > max) max = v;
            }
            os[i] = max;
        }
        return ApplyKeepDims(output, plan, keepDims);
    }

    public static Tensor<int> ReduceMax(Tensor<int> data, Tensor<int>? axes) => ReduceMax(data, axes, null, null);

        public static Tensor<int> ReduceMax(Tensor<int> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
    {
        StartOpStage(OpStage.ValidateArguments);
        var plan = PlanReduction(data.Rank, axes, _keepDims, true, _noOpWithEmptyAxes, out var keepDims);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<int>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            // No integer identity exists: ORT 1.29 fails empty int
            // reductions, unlike float which yields -Infinity, so an empty
            // extent throws instead of writing a seed.
            if (r == 0) throw new ArgumentException(nameof(axes), "ReduceMax over an empty extent has no result for integer types.");
            int max = xs[offset];
            for (int j = 1; j < r; ++j)
            {
                int v = xs[offset + j];
                if (v > max) max = v;
            }
            os[i] = max;
        }
        return ApplyKeepDims(output, plan, keepDims);
    }

    public static Tensor<sbyte> ReduceMax(Tensor<sbyte> data, Tensor<int>? axes) => ReduceMax(data, axes, null, null);
    public static Tensor<sbyte> ReduceMax(Tensor<sbyte> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
    {
        StartOpStage(OpStage.ValidateArguments);
        var plan = PlanReduction(data.Rank, axes, _keepDims, true, _noOpWithEmptyAxes, out var keepDims);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<sbyte>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            // No integer identity exists: ORT 1.29 fails empty int
            // reductions, unlike float which yields -Infinity, so an empty
            // extent throws instead of writing a seed.
            if (r == 0) throw new ArgumentException(nameof(axes), "ReduceMax over an empty extent has no result for integer types.");
            sbyte max = xs[offset];
            for (int j = 1; j < r; ++j)
            {
                sbyte v = xs[offset + j];
                if (v > max) max = v;
            }
            os[i] = max;
        }
        return ApplyKeepDims(output, plan, keepDims);
    }

    public static Tensor<byte> ReduceMax(Tensor<byte> data, Tensor<int>? axes) => ReduceMax(data, axes, null, null);
    public static Tensor<byte> ReduceMax(Tensor<byte> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
    {
        StartOpStage(OpStage.ValidateArguments);
        var plan = PlanReduction(data.Rank, axes, _keepDims, true, _noOpWithEmptyAxes, out var keepDims);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<byte>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            // No integer identity exists: ORT 1.29 fails empty int
            // reductions, unlike float which yields -Infinity, so an empty
            // extent throws instead of writing a seed.
            if (r == 0) throw new ArgumentException(nameof(axes), "ReduceMax over an empty extent has no result for integer types.");
            byte max = xs[offset];
            for (int j = 1; j < r; ++j)
            {
                byte v = xs[offset + j];
                if (v > max) max = v;
            }
            os[i] = max;
        }
        return ApplyKeepDims(output, plan, keepDims);
    }

    public static Tensor<double> ReduceMax(Tensor<double> data, Tensor<int>? axes) => ReduceMax(data, axes, null, null);

        public static Tensor<double> ReduceMax(Tensor<double> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
    {
        StartOpStage(OpStage.ValidateArguments);
        var plan = PlanReduction(data.Rank, axes, _keepDims, true, _noOpWithEmptyAxes, out var keepDims);
        if (plan.IsNoOp) return data.Clone();

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<double>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            if (r == 0)
            {
                os[i] = double.NegativeInfinity;
                continue;
            }
            double max = xs[offset];
            for (int j = 1; j < r; ++j)
            {
                double v = xs[offset + j];
                if (v > max) max = v;
            }
            os[i] = max;
        }
        return ApplyKeepDims(output, plan, keepDims);
    }
}

/// <summary>
/// Validated reduction plan shared by every ReduceSum/Mean/Max copy.
/// </summary>
/// <remarks>
/// Absent or explicitly empty axes are a no-op when flagged, else reduce all
/// axes. Otherwise axes normalize (negative += rank), range-check against
/// [-rank, rank-1], then deduplicate: the native engine reduces duplicated
/// axes once instead of rejecting them (verified against ORT 1.29).
/// </remarks>
internal readonly struct ReductionPlan
{
    public readonly bool IsNoOp;
    public readonly int[] Axes;
    public readonly bool KeepDims;

    private ReductionPlan(bool isNoOp, int[] axes, bool keepDims)
    {
        IsNoOp = isNoOp;
        Axes = axes;
        KeepDims = keepDims;
    }

    public static ReductionPlan Create(int rank, Tensor<int>? axes, bool keepDims, bool noOpWithEmptyAxes)
    {
        var raw = axes is null ? System.Array.Empty<int>() : axes.ToArray();
        if (raw.Length == 0)
        {
            if (noOpWithEmptyAxes) return new ReductionPlan(true, System.Array.Empty<int>(), keepDims);
            var all = new int[rank];
            for (int i = 0; i < rank; i++) all[i] = i;
            return new ReductionPlan(false, all, keepDims);
        }
        var normalized = new int[raw.Length];
        for (int i = 0; i < raw.Length; i++)
        {
            int a = raw[i] < 0 ? raw[i] + rank : raw[i];
            if (a < 0 || a >= rank)
                throw new System.ArgumentException(nameof(axes), $"Axis {raw[i]} is out of range for tensor rank {rank}.");
            normalized[i] = a;
        }
        return new ReductionPlan(false, normalized.Distinct().ToArray(), keepDims);
    }
}
