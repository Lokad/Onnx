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
            dense = pdata.ToRowMajor();
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
            // Match ORT's serial compensated double accumulation and final
            // saturation. Conversion of individual Int64 values can round.
            var sum = new Int64ReductionAccumulator();
            for (int j = 0; j < r; ++j) sum.Add(xs[offset + j]);
            os[i] = sum.Sum;
        }
        return ApplyKeepDims(output, plan, keepDims);
    }

    internal static Tensor<long> ReduceSumSquareInt64(Tensor<long> data, Tensor<int>? axes, bool keepDims, bool noop)
    {
        StartOpStage(OpStage.ValidateArguments);
        var plan = PlanReduction(data.Rank, axes, keepDims, false, noop, out _);
        if (plan.IsNoOp)
        {
            // A reduction no-op still squares each value, without aliasing input.
            var values = data.ToArray();
            StartOpStage(OpStage.Math);
            for (int i = 0; i < values.Length; i++)
            {
                var square = new Int64ReductionAccumulator();
                square.AddSquare(values[i]);
                values[i] = square.Sum;
            }
            return new DenseTensor<long>(values, data.Dimensions);
        }

        StartOpStage(OpStage.CalculateIndices);
        var (pdata, oshape, r) = PrepareReduction(data, plan.Axes);
        var output = DenseTensor<long>.OfShape(oshape);
        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        for (int i = 0; i < os.Length; i++)
        {
            int offset = i * r;
            var sum = new Int64ReductionAccumulator();
            for (int j = 0; j < r; j++) sum.AddSquare(xs[offset + j]);
            os[i] = sum.Sum;
        }
        return ApplyKeepDims(output, plan, keepDims);
    }

    public static Tensor<float> ReduceSum(Tensor<float> data, Tensor<int>? axes) => ReduceSum(data, axes, null, null);

        public static Tensor<float> ReduceSum(Tensor<float> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
    {
        StartOpStage(OpStage.ValidateArguments);
        var plan = PlanReduction(data.Rank, axes, _keepDims, false, _noOpWithEmptyAxes, out var keepDims);
        if (plan.IsNoOp) return data.Clone();

        return ApplyKeepDims(ReduceFloatCore(data, plan, mean: false), plan, keepDims);
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
        options.Validate();
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

    public static Tensor<long> ReduceMean(Tensor<long> data, Tensor<int>? axes) => ReduceMean(data, axes, null, null);

        public static Tensor<long> ReduceMean(Tensor<long> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
        => ReduceMean(data, axes, _keepDims, _noOpWithEmptyAxes, TensorExecutionOptions.Auto);

        public static Tensor<long> ReduceMean(Tensor<long> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes, TensorExecutionOptions options)
    {
        options.Validate();
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
            // Divide the compensated double sum, then truncate/saturate.
            // Conversion still rounds: [2^53+1,1] gives 2^52. Empty means are 0.
            var sum = new Int64ReductionAccumulator();
            for (int j = 0; j < r; ++j) sum.Add(xs[offset + j]);
            os[i] = sum.Mean(r);
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

        return ApplyKeepDims(ReduceFloatCore(data, plan, mean: true), plan, keepDims);
    }

    // Native sum and mean share their accumulation geometry: a fixed inner
    // tree, linear outer sums, or inner block sums followed by outer sums.
    // Mean divides once after accumulation, keeping its established association.
    static DenseTensor<float> ReduceFloatCore(Tensor<float> data, ReductionPlan plan, bool mean)
    {
        StartOpStage(OpStage.CalculateIndices);
        bool innerAxes = ArrayUtilities.AxesAreInnerMostDims(plan.Axes, data.Rank);
        Tensor<float> reductionData = data;
        int[] reductionAxes = plan.Axes;
        int innerExtent = 1;
        if (!innerAxes && plan.Axes[^1] == data.Rank - 1)
        {
            // Native mixed-axis reduction sums contiguous inner blocks first,
            // then accumulates those block sums over the remaining outer axes.
            int split = plan.Axes.Length - 1;
            while (split > 0 && plan.Axes[split - 1] == plan.Axes[split] - 1) split--;
            innerExtent = ArrayUtilities.ComputeOffsetForReduction(data.dimensions, plan.Axes[split]);
            reductionData = ReduceSum(data, DenseTensor<int>.OfValues(plan.Axes[split..]), true, false);
            reductionAxes = plan.Axes[..split];
        }
        var (pdata, oshape, r) = PrepareReduction(reductionData, reductionAxes);
        if (!ReferenceEquals(reductionData, data))
            oshape = ArrayUtilities.ComputeShapesForReduction(data.dimensions, plan.Axes).Item1;
        var output = DenseTensor<float>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var xs = pdata.Buffer.Span;
        var os = output.Buffer.Span;
        float divisor = (float)r * innerExtent;
        for (int i = 0; i < os.Length; ++i)
        {
            int offset = i * r;
            // Empty reductions yield zero instead of NaN (ORT 1.29).
            if (divisor == 0)
            {
                os[i] = 0f;
                continue;
            }
            float sum;
            if (innerAxes)
                sum = FloatReduction.Sum(xs.Slice(offset, r), Math.Min((-offset) & 3, r), 0, false);
            else
            {
                sum = 0f;
                for (int j = 0; j < r; ++j) sum += xs[offset + j];
            }
            os[i] = mean ? sum / divisor : sum;
        }
        return output;
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

    public static Tensor<long> ReduceMax(Tensor<long> data, Tensor<int>? axes) => ReduceMax(data, axes, null, null);

        public static Tensor<long> ReduceMax(Tensor<long> data, Tensor<int>? axes, bool? _keepDims, bool? _noOpWithEmptyAxes)
    {
        StartOpStage(OpStage.ValidateArguments);
        var plan = PlanReduction(data.Rank, axes, _keepDims, true, _noOpWithEmptyAxes, out var keepDims);
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
            // Empty reductions yield the dtype minimum (ORT 1.29 returns
            // Min for int64, probed; float yields -Infinity).
            if (r == 0)
            {
                os[i] = long.MinValue;
                continue;
            }
            long max = xs[offset];
            for (int j = 1; j < r; ++j)
            {
                long v = xs[offset + j];
                if (v > max) max = v;
            }
            os[i] = max;
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
            // Empty reductions yield the dtype minimum (ORT 1.29 returns
            // Min across integer widths, probed; float yields -Infinity).
            if (r == 0)
            {
                os[i] = int.MinValue;
                continue;
            }
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
            // Empty reductions yield the dtype minimum (ORT 1.29 returns
            // Min across integer widths, probed; float yields -Infinity).
            if (r == 0)
            {
                os[i] = sbyte.MinValue;
                continue;
            }
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
            // Empty reductions yield the dtype minimum (ORT 1.29 returns
            // Min across integer widths, probed; float yields -Infinity).
            if (r == 0)
            {
                os[i] = byte.MinValue;
                continue;
            }
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
/// [-rank, rank-1], then sort and deduplicate: axis order does not change
/// native reduction traversal, and duplicated axes reduce once (ORT 1.29).
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
                throw new System.ArgumentException($"Axis {raw[i]} is out of range for tensor rank {rank}.", nameof(axes));
            normalized[i] = a;
        }
        // Equivalent axis sets must use the same traversal and addition order.
        Array.Sort(normalized);
        return new ReductionPlan(false, normalized.Distinct().ToArray(), keepDims);
    }
}
