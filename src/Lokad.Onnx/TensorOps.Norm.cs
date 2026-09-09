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
    /// <summary>
    /// Validates LayerNormalization arguments and derives the normalization block geometry.
    /// The axis indexes the input tensor; negative values count from the back.
    /// </summary>
    static (int Block, int Outer) LayerNormalizationPlan(int rank, int axis, int[] dimensions, long inputLength, long scaleLength, long? biasLength)
    {
        int normalizedAxis = axis < 0 ? axis + rank : axis;
        if (normalizedAxis < 0 || normalizedAxis >= rank) throw new ArgumentException(nameof(axis));
        int block = 1;
        for (int dimension = normalizedAxis; dimension < rank; dimension++) block *= dimensions[dimension];
        if (scaleLength != block) throw new ArgumentException("Scale length must match the normalized dimensions.", "scale");
        if (biasLength.HasValue && biasLength.Value != block) throw new ArgumentException("Bias length must match the normalized dimensions.", "bias");
        return (block, (int)(inputLength / block));
    }

    /// <summary>
    /// Normalizes over the input dimensions from axis to the last one using scale, optional bias, and epsilon.
    /// Statistics accumulate in double precision; the axis indexes the input tensor.
    /// </summary>
    public static Tensor<float> LayerNormalization(Tensor<float> x, Tensor<float> scale, Tensor<float>? bias, int axis, float epsilon)
    {
        var xd = x.ToDenseTensor();
        var sd = scale.ToDenseTensor();
        var bd = bias?.ToDenseTensor();
        var plan = LayerNormalizationPlan(x.Rank, axis, xd.Dimensions.ToArray(), xd.Length, sd.Length, bd?.Length);
        var output = new DenseTensor<float>(xd.Dimensions);
        LayerNormFloatInto(xd, sd, bd, output, plan.Block, plan.Outer, epsilon);
        return output;
    }

    /// <summary>Shared float layer-normalization kernel used by both entries.</summary>
    static void LayerNormFloatInto(DenseTensor<float> xd, DenseTensor<float> sd, DenseTensor<float>? bd, DenseTensor<float> destination, int block, int outer, float epsilon)
    {
        var xs = xd.Buffer.Span;
        var ss = sd.Buffer.Span;
        var bs = bd is null ? new Span<float>() : bd.Buffer.Span;
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
            for (int i = 0; i < block; i++) os[o * block + i] = (float)((xs[o * block + i] - mean) * inv * ss[i] + (bd is null ? 0f : bs[i]));
        }
    }

    /// <summary>Writes float layer normalization into an existing dense destination.</summary>
    public static Tensor<float> LayerNormalization(Tensor<float> x, Tensor<float> scale, Tensor<float>? bias, DenseTensor<float> destination, int axis, float epsilon)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        var xd = x.ToDenseTensor();
        var sd = scale.ToDenseTensor();
        var bd = bias?.ToDenseTensor();
        var plan = LayerNormalizationPlan(x.Rank, axis, xd.Dimensions.ToArray(), xd.Length, sd.Length, bd?.Length);
        if (!destination.Dimensions.SequenceEqual(xd.Dimensions.ToArray())) throw new ArgumentException(nameof(destination), "Destination shape must match the input shape.");
        if (!HasStandardStrides(destination)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        LayerNormFloatInto(xd, sd, bd, destination, plan.Block, plan.Outer, epsilon);
        return destination;
    }

    /// <summary>
    /// Normalizes over the input dimensions from axis to the last one using scale, optional bias, and epsilon.
    /// Statistics accumulate in double precision; the axis indexes the input tensor.
    /// </summary>
    public static Tensor<double> LayerNormalization(Tensor<double> x, Tensor<double> scale, Tensor<double>? bias, int axis, double epsilon)
    {
        var xd = x.ToDenseTensor();
        var sd = scale.ToDenseTensor();
        var bd = bias?.ToDenseTensor();
        var plan = LayerNormalizationPlan(x.Rank, axis, xd.Dimensions.ToArray(), xd.Length, sd.Length, bd?.Length);
        var output = new DenseTensor<double>(xd.Dimensions);
        LayerNormDoubleInto(xd, sd, bd, output, plan.Block, plan.Outer, epsilon);
        return output;
    }

    /// <summary>Writes double layer normalization into an existing dense destination.</summary>
    public static Tensor<double> LayerNormalization(Tensor<double> x, Tensor<double> scale, Tensor<double>? bias, DenseTensor<double> destination, int axis, double epsilon)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        var xd = x.ToDenseTensor();
        var sd = scale.ToDenseTensor();
        var bd = bias?.ToDenseTensor();
        var plan = LayerNormalizationPlan(x.Rank, axis, xd.Dimensions.ToArray(), xd.Length, sd.Length, bd?.Length);
        if (!destination.Dimensions.SequenceEqual(xd.Dimensions.ToArray())) throw new ArgumentException(nameof(destination), "Destination shape must match the input shape.");
        if (!HasStandardStrides(destination)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        LayerNormDoubleInto(xd, sd, bd, destination, plan.Block, plan.Outer, epsilon);
        return destination;
    }

    /// <summary>Shared double layer-normalization kernel used by both entries.</summary>
    static void LayerNormDoubleInto(DenseTensor<double> xd, DenseTensor<double> sd, DenseTensor<double>? bd, DenseTensor<double> destination, int block, int outer, double epsilon)
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
        var output = new DenseTensor<float>(x.ToDenseTensor().Dimensions);
        return RotaryEmbedding(x, cos, sin, output, half, axis, -1);
    }

    /// <summary>Writes the rotary position embedding into an existing dense destination.</summary>
    public static Tensor<float> RotaryEmbedding(Tensor<float> x, Tensor<float> cos, Tensor<float> sin, DenseTensor<float> destination, int half, int axis, int concatAxis)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        StartOpStage(OpStage.ValidateArguments);
        var xd = x.ToDenseTensor();
        var cd = cos.ToDenseTensor();
        var sd = sin.ToDenseTensor();
        if (!destination.Dimensions.SequenceEqual(xd.Dimensions.ToArray())) throw new ArgumentException(nameof(destination), "Destination shape must match the input shape.");
        if (!HasStandardStrides(destination)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        if (half <= 0) throw new ArgumentException(nameof(half), "Half size must be positive.");
        int rank = xd.Rank;
        int a = axis < 0 ? axis + rank : axis;
        if (a < 0 || a >= rank) throw new ArgumentException(nameof(axis), "Axis is out of range for the input rank.");
        int ca = concatAxis < 0 ? concatAxis + rank : concatAxis;
        if (ca != a) throw new ArgumentException(nameof(concatAxis), "Slice and concat axes disagree after rank normalization.");
        if (!cd.Dimensions.SequenceEqual(sd.Dimensions.ToArray())) throw new ArgumentException(nameof(sin), "Cos and sin must have identical shapes.");
        if (cd.Rank > rank) throw new ArgumentException(nameof(cos), "Cos rank must not exceed input rank.");
        int inner = xd.Dimensions[a];
        if (inner < half) throw new ArgumentException(nameof(half), "Half size must not exceed the axis dimension.");
        int span = inner - half;
        var dims = xd.Dimensions.ToArray();
        var xstrides = new int[rank];
        var cstrides = new int[rank];
        var cshape = new int[rank];
        int stride = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            xstrides[d] = stride;
            int cd2 = d < rank - cd.Rank ? 1 : cd.Dimensions[d - (rank - cd.Rank)];
            if (cd2 != 1 && cd2 != dims[d]) throw new ArgumentException(nameof(cos), "Cos shape must broadcast against the input shape.");
            cshape[d] = cd2;
            cstrides[d] = cd2 == 1 ? 0 : stride;
            stride *= dims[d];
        }
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
