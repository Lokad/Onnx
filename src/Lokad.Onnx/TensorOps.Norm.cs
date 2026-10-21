namespace Lokad.Onnx;

using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

using static Lokad.Onnx.MathOps;
using static Lokad.Onnx.Profiler;

public abstract partial class Tensor<T> : TensorBase, IList, IList<T>, IReadOnlyList<T>, IStructuralComparable, IStructuralEquatable, ITensor, INumericTensor
where T : unmanaged
{
    /// <summary>
    /// Validates LayerNormalization arguments and derives the normalization block geometry.
    /// The axis indexes the input tensor; negative values count from the back.
    /// </summary>
    static (int Block, int Outer) LayerNormalizationPlan(int rank, int axis, ReadOnlySpan<int> dimensions, long inputLength, long scaleLength, long? biasLength)
    {
        int normalizedAxis = axis < 0 ? axis + rank : axis;
        if (normalizedAxis < 0 || normalizedAxis >= rank) throw new ArgumentException(nameof(axis));
        int block = 1;
        for (int dimension = normalizedAxis; dimension < rank; dimension++) block *= dimensions[dimension];
        if (block <= 0) throw new ArgumentException("LayerNormalization requires the normalized dimensions to have a positive extent.", nameof(axis));
        if (scaleLength != block) throw new ArgumentException("Scale length must match the normalized dimensions.", "scale");
        if (biasLength.HasValue && biasLength.Value != block) throw new ArgumentException("Bias length must match the normalized dimensions.", "bias");
        return (block, (int)(inputLength / block));
    }

    static DenseTensor<TElement> SnapshotOverlappingSource<TElement>(DenseTensor<TElement> source, DenseTensor<TElement> destination) where TElement : unmanaged
        => source.Buffer.Span.Overlaps(destination.Buffer.Span) ? Tensor<TElement>.Snapshot(source) : source;

    static (DenseTensor<float>? Mean, DenseTensor<float>? InvStdDev) LayerNormalizationStatistics(ReadOnlySpan<int> dimensions, int axis, int outputCount)
    {
        if (outputCount == 1) return (null, null);
        var shape = dimensions.ToArray();
        shape.AsSpan(axis < 0 ? axis + shape.Length : axis).Fill(1);
        return (new DenseTensor<float>(shape), outputCount == 3 ? new DenseTensor<float>(shape) : null);
    }

    /// <summary>
    /// Normalizes over the input dimensions from axis to the last one using scale, optional bias, and epsilon.
    /// Statistics accumulate in double precision; the axis indexes the input tensor.
    /// </summary>
    public static Tensor<float> LayerNormalization(Tensor<float> x, Tensor<float> scale, Tensor<float>? bias, int axis, float epsilon)
        => LayerNormalizationCore(x, scale, bias, null, axis, epsilon, 1).Output;

    /// <summary>Shared float layer-normalization kernel used by both entries.</summary>
    static void LayerNormFloatInto(DenseTensor<float> xd, DenseTensor<float> sd, DenseTensor<float>? bd, DenseTensor<float> destination, int block, int outer, float epsilon, Span<float> means, Span<float> inverseStdDevs, bool useSimd)
    {
        var xs = xd.Buffer.Span;
        var ss = sd.Buffer.Span;
        var bs = bd is null ? new Span<float>() : bd.Buffer.Span;
        var os = destination.Buffer.Span;
        if (useSimd && Vector.IsHardwareAccelerated)
        {
            // Vectorized twin of the scalar body below: statistics stay in
            // double vectors and every element keeps the scalar association
            // ((x - mean) * inv) * scale + bias, so only the reduction order
            // of the two sums can move results, within float tolerance.
            int wf = Vector<float>.Count;
            var vzero = Vector<double>.Zero;
            for (int o = 0; o < outer; o++)
            {
                int off = o * block;
                int i = 0;
                int vlen = block - (block % wf);
                var vsum = Vector<double>.Zero;
                for (; i < vlen; i += wf)
                {
                    Vector.Widen(new Vector<float>(xs.Slice(off + i)), out Vector<double> lo, out Vector<double> hi);
                    vsum += lo + hi;
                }
                double mean = Vector.Sum(vsum);
                for (; i < block; i++) mean += xs[off + i];
                mean /= block;
                var vmean = new Vector<double>(mean);
                var vvar = Vector<double>.Zero;
                i = 0;
                for (; i < vlen; i += wf)
                {
                    Vector.Widen(new Vector<float>(xs.Slice(off + i)), out Vector<double> lo, out Vector<double> hi);
                    var d0 = lo - vmean;
                    var d1 = hi - vmean;
                    vvar += d0 * d0 + d1 * d1;
                }
                double variance = Vector.Sum(vvar);
                for (; i < block; i++) { double d = xs[off + i] - mean; variance += d * d; }
                variance /= block;
                double inv = 1.0 / Math.Sqrt(variance + epsilon);
                if (!means.IsEmpty) means[o] = (float)mean;
                if (!inverseStdDevs.IsEmpty) inverseStdDevs[o] = (float)inv;
                var vinv = new Vector<double>(inv);
                i = 0;
                for (; i < vlen; i += wf)
                {
                    Vector.Widen(new Vector<float>(xs.Slice(off + i)), out Vector<double> x0, out Vector<double> x1);
                    Vector.Widen(new Vector<float>(ss.Slice(i)), out Vector<double> s0, out Vector<double> s1);
                    Vector<double> b0 = vzero;
                    Vector<double> b1 = vzero;
                    if (bd is not null)
                    {
                        Vector.Widen(new Vector<float>(bs.Slice(i)), out b0, out b1);
                    }
                    var r0 = (x0 - vmean) * vinv * s0 + b0;
                    var r1 = (x1 - vmean) * vinv * s1 + b1;
                    var rf = Vector.Narrow(r0, r1);
                    rf.CopyTo(os.Slice(off + i));
                }
                for (; i < block; i++) os[off + i] = (float)((xs[off + i] - mean) * inv * ss[i] + (bd is null ? 0f : bs[i]));
            }
            return;
        }
        for (int o = 0; o < outer; o++)
        {
            double mean = 0.0;
            for (int i = 0; i < block; i++) mean += xs[o * block + i];
            mean /= block;
            double variance = 0.0;
            for (int i = 0; i < block; i++) { double d = xs[o * block + i] - mean; variance += d * d; }
            variance /= block;
            double inv = 1.0 / Math.Sqrt(variance + epsilon);
            if (!means.IsEmpty) means[o] = (float)mean;
            if (!inverseStdDevs.IsEmpty) inverseStdDevs[o] = (float)inv;
            for (int i = 0; i < block; i++) os[o * block + i] = (float)((xs[o * block + i] - mean) * inv * ss[i] + (bd is null ? 0f : bs[i]));
        }
    }

    /// <summary>Writes float layer normalization into an existing dense destination.</summary>
    public static Tensor<float> LayerNormalization(Tensor<float> x, Tensor<float> scale, Tensor<float>? bias, DenseTensor<float> destination, int axis, float epsilon)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        return LayerNormalizationCore(x, scale, bias, destination, axis, epsilon, 1).Output;
    }

    internal static (Tensor<float> Output, DenseTensor<float>? Mean, DenseTensor<float>? InvStdDev) LayerNormalizationCore(
        Tensor<float> x, Tensor<float> scale, Tensor<float>? bias, DenseTensor<float>? destination, int axis, float epsilon, int outputCount, TensorExecutionOptions? options = null)
    {
        var resolved = options ?? TensorExecutionOptions.Auto;
        resolved.Validate();
        var xd = x.ToRowMajor();
        var sd = scale.ToRowMajor();
        var bd = bias?.ToRowMajor();
        var plan = LayerNormalizationPlan(x.Rank, axis, xd.Dimensions, xd.Length, sd.Length, bd?.Length);
        if (destination is null) destination = new DenseTensor<float>(xd.Dimensions);
        else
        {
            if (!destination.Dimensions.SequenceEqual(xd.Dimensions)) throw new ArgumentException("Destination shape must match the input shape.", nameof(destination));
            if (!HasStandardStrides(destination)) throw new ArgumentException("Destination must have standard row-major strides.", nameof(destination));
            xd = (DenseTensor<float>)Tensor<float>.PreserveElementwiseSource(xd, destination);
            sd = SnapshotOverlappingSource(sd, destination);
            if (bd is not null) bd = SnapshotOverlappingSource(bd, destination);
        }
        var stats = LayerNormalizationStatistics(xd.Dimensions, axis, outputCount);
        LayerNormFloatInto(xd, sd, bd, destination, plan.Block, plan.Outer, epsilon,
            stats.Mean is null ? default : stats.Mean.Buffer.Span, stats.InvStdDev is null ? default : stats.InvStdDev.Buffer.Span, resolved.UseSimd);
        return (destination, stats.Mean, stats.InvStdDev);
    }

    /// <summary>
    /// Normalizes over the input dimensions from axis to the last one using scale, optional bias, and epsilon.
    /// Statistics accumulate in double precision; the axis indexes the input tensor.
    /// </summary>
    public static Tensor<double> LayerNormalization(Tensor<double> x, Tensor<double> scale, Tensor<double>? bias, int axis, double epsilon)
        => LayerNormalizationCore(x, scale, bias, null, axis, epsilon, 1).Output;

    /// <summary>Writes double layer normalization into an existing dense destination.</summary>
    public static Tensor<double> LayerNormalization(Tensor<double> x, Tensor<double> scale, Tensor<double>? bias, DenseTensor<double> destination, int axis, double epsilon)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        return LayerNormalizationCore(x, scale, bias, destination, axis, epsilon, 1).Output;
    }

    internal static (Tensor<double> Output, DenseTensor<float>? Mean, DenseTensor<float>? InvStdDev) LayerNormalizationCore(
        Tensor<double> x, Tensor<double> scale, Tensor<double>? bias, DenseTensor<double>? destination, int axis, double epsilon, int outputCount)
    {
        var xd = x.ToRowMajor();
        var sd = scale.ToRowMajor();
        var bd = bias?.ToRowMajor();
        var plan = LayerNormalizationPlan(x.Rank, axis, xd.Dimensions, xd.Length, sd.Length, bd?.Length);
        if (destination is null) destination = new DenseTensor<double>(xd.Dimensions);
        else
        {
            if (!destination.Dimensions.SequenceEqual(xd.Dimensions)) throw new ArgumentException("Destination shape must match the input shape.", nameof(destination));
            if (!HasStandardStrides(destination)) throw new ArgumentException("Destination must have standard row-major strides.", nameof(destination));
            xd = (DenseTensor<double>)Tensor<double>.PreserveElementwiseSource(xd, destination);
            sd = SnapshotOverlappingSource(sd, destination);
            if (bd is not null) bd = SnapshotOverlappingSource(bd, destination);
        }
        var stats = LayerNormalizationStatistics(xd.Dimensions, axis, outputCount);
        LayerNormDoubleInto(xd, sd, bd, destination, plan.Block, plan.Outer, epsilon,
            stats.Mean is null ? default : stats.Mean.Buffer.Span, stats.InvStdDev is null ? default : stats.InvStdDev.Buffer.Span);
        return (destination, stats.Mean, stats.InvStdDev);
    }

    /// <summary>Shared double layer-normalization kernel used by both entries.</summary>
    static void LayerNormDoubleInto(DenseTensor<double> xd, DenseTensor<double> sd, DenseTensor<double>? bd, DenseTensor<double> destination, int block, int outer, double epsilon, Span<float> means, Span<float> inverseStdDevs)
    {
        var xs = xd.Buffer.Span;
        var ss = sd.Buffer.Span;
        var bs = bd is null ? new Span<double>() : bd.Buffer.Span;
        var os = destination.Buffer.Span;
        for (int o = 0; o < outer; o++)
        {
            double mean = 0.0;
            for (int i = 0; i < block; i++) mean += xs[o * block + i];
            mean /= block;
            double variance = 0.0;
            for (int i = 0; i < block; i++) { double d = xs[o * block + i] - mean; variance += d * d; }
            variance /= block;
            double inv = 1.0 / Math.Sqrt(variance + epsilon);
            if (!means.IsEmpty) means[o] = (float)mean;
            if (!inverseStdDevs.IsEmpty) inverseStdDevs[o] = (float)inv;
            for (int i = 0; i < block; i++) os[o * block + i] = (xs[o * block + i] - mean) * inv * ss[i] + (bd is null ? 0.0 : bs[i]);
        }
    }

    /// <summary>
    /// Applies rotary position embedding: out = x * cos + rotate_half(x) * sin with
    /// rotate_half(x)[i] = i &lt; span ? -x[i + half] : x[i - span] along the axis,
    /// where span = dim - half (the classic rotation at dim == 2 * half). Cos/sin
    /// follow standard right-aligned broadcast against x. Fused form of the
    /// Slice/Slice/Neg/Concat/Mul/Mul/Add pattern; computes every output element
    /// with the same operations in the same order, so results match it bitwise.
    /// </summary>
    public static Tensor<float> RotaryEmbedding(Tensor<float> x, Tensor<float> cos, Tensor<float> sin, int half, int axis, int concatAxis)
    {
        var output = new DenseTensor<float>(x.Dimensions);
        return RotaryEmbedding(x, cos, sin, output, half, axis, concatAxis);
    }

    /// <summary>Writes the rotary position embedding into an existing dense destination.</summary>
    public static Tensor<float> RotaryEmbedding(Tensor<float> x, Tensor<float> cos, Tensor<float> sin, DenseTensor<float> destination, int half, int axis, int concatAxis)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        StartOpStage(OpStage.ValidateArguments);
        var xd = x.ToRowMajor();
        var cd = cos.ToRowMajor();
        var sd = sin.ToRowMajor();
        if (!destination.Dimensions.SequenceEqual(xd.Dimensions.ToArray())) throw new ArgumentException("Destination shape must match the input shape.", nameof(destination));
        if (!HasStandardStrides(destination)) throw new ArgumentException("Destination must have standard row-major strides.", nameof(destination));
        if (half <= 0) throw new ArgumentException("Half size must be positive.", nameof(half));
        int rank = xd.Rank;
        int a = axis < 0 ? axis + rank : axis;
        if (a < 0 || a >= rank) throw new ArgumentException("Axis is out of range for the input rank.", nameof(axis));
        int ca = concatAxis < 0 ? concatAxis + rank : concatAxis;
        if (ca != a) throw new ArgumentException("Slice and concat axes disagree after rank normalization.", nameof(concatAxis));
        if (!cd.Dimensions.SequenceEqual(sd.Dimensions.ToArray())) throw new ArgumentException("Cos and sin must have identical shapes.", nameof(sin));
        if (cd.Rank > rank) throw new ArgumentException("Cos rank must not exceed input rank.", nameof(cos));
        int inner = xd.Dimensions[a];
        if (inner < half) throw new ArgumentException("Half size must not exceed the axis dimension.", nameof(half));
        int span = inner - half;
        var dims = xd.Dimensions.ToArray();
        var xstrides = new int[rank];
        var cstrides = new int[rank];
        var cshape = new int[rank];
        int stride = 1;
        int cosineStride = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            xstrides[d] = stride;
            int cd2 = d < rank - cd.Rank ? 1 : cd.Dimensions[d - (rank - cd.Rank)];
            if (cd2 != 1 && cd2 != dims[d]) throw new ArgumentException("Cos shape must broadcast against the input shape.", nameof(cos));
            cshape[d] = cd2;
            cstrides[d] = cd2 == 1 ? 0 : cosineStride;
            stride *= dims[d];
            cosineStride *= cd2;
        }
        // Rotation reads other input positions, so even exact in-place input
        // needs preservation. Broadcast parameters can be reused after writes.
        xd = SnapshotOverlappingSource(xd, destination);
        cd = SnapshotOverlappingSource(cd, destination);
        sd = SnapshotOverlappingSource(sd, destination);
        var xs = xd.Buffer.Span;
        var cs = cd.Buffer.Span;
        var ss = sd.Buffer.Span;
        var os = destination.Buffer.Span;
        StartOpStage(OpStage.Math);
        var index = new int[rank];
        int cx = 0;
        int cc = 0;
        int total = (int)xd.Length;
        for (int n = 0; n < total; n++)
        {
            int pos = index[a];
            int rot = pos < span ? cx + half * xstrides[a] : cx - span * xstrides[a];
            float rotated = pos < span ? -xs[rot] : xs[rot];
            os[cx] = xs[cx] * cs[cc] + rotated * ss[cc];
            for (int d = rank - 1; d >= 0; d--)
            {
                index[d]++;
                cx += xstrides[d];
                cc += cstrides[d];
                if (index[d] < dims[d]) break;
                index[d] = 0;
                cx -= xstrides[d] * dims[d];
                cc -= cstrides[d] * cshape[d];
            }
        }
        return destination;
    }
}
