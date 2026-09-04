namespace Lokad.Onnx;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.CompilerServices;

using static Lokad.Onnx.MathOps;
using static Lokad.Onnx.Profiler;

public abstract partial class Tensor<T> : TensorBase, IList, IList<T>, IReadOnlyList<T>, IStructuralComparable, IStructuralEquatable, ITensor
where T : unmanaged
{
    public virtual void Apply(Func<T, T> op, Tensor<T> destination)
    {
        if (this.Length > destination.Length)
            throw new ArgumentException(nameof(destination), "Destination tensor is too small.");
  
        for (int index = 0; index < Length; index++)
        {
            destination.SetValue(index, op(GetValue(index)));
        }        
    }

    public Tensor<T> Apply(Func<T, T> op)
    {
        var output = CloneEmpty();
        Apply(op, output);
        return output;
    }

    public virtual void VectorizedApply(Func<Vector<T>, Vector<T>> op, Func<T, T> sop, Tensor<T> destination)
    {
        if (this.Length > destination.Length)
            throw new ArgumentException(nameof(destination), "Destination tensor is too small.");

        if (HardwareConfig.UseSimd && this is DenseTensor<T> d1 && destination is DenseTensor<T> d2)
        {
            var vspan1 = MemoryMarshal.Cast<T, Vector<T>>(d1.Buffer.Span);
            var vspan2 = MemoryMarshal.Cast<T, Vector<T>>(d2.Buffer.Span);
            var ceiling = (Convert.ToInt32(this.length) / Vector<T>.Count) * Vector<T>.Count;
            for (int i = 0; i < vspan1.Length; i++)
            {
                vspan2[i] = op(vspan1[i]);
            }
            for (int i = ceiling; i < this.length; i++)
            {
                destination.SetValue(i, sop(GetValue(i)));
            }
        }
        else
        {
            Apply(sop, destination);
        }
    }

    public Tensor<T> VectorizedApply(Func<Vector<T>, Vector<T>> op, Func<T, T> sop)
    {
        var output = CloneEmpty();
        VectorizedApply(op, sop, output);
        return output;
    }

    public virtual void Apply(Func<T, T, T> op, Tensor<T> tensor2, Tensor<T> destination)
    {
        if (this.Length > tensor2.Length)
            throw new ArgumentException(nameof(tensor2), "2nd tensor is too small.");

        if (this.Length > destination.Length)
            throw new ArgumentException(nameof(destination), "Destination tensor is too small.");

        for (int index = 0; index < this.Length; index++)
        {
            destination.SetValue(index, op(GetValue(index), tensor2.GetValue(index)));
        }        
    }

    public Tensor<T> Apply(Func<T, T, T> op, Tensor<T> tensor2)
    {
        var output = CloneEmpty();
        Apply(op, tensor2, output);
        return output;
    }

    public virtual void VectorizedApply(Func<Vector<T>, Vector<T>, Vector<T>> op, Func<T, T, T> sop, Tensor<T> tensor2, Tensor<T> destination)
    {
        if (this.Length > tensor2.Length)
            throw new ArgumentException(nameof(tensor2), "2nd tensor is too small.");

        if (this.Length > destination.Length)
            throw new ArgumentException(nameof(destination), "Destination tensor is too small.");

        if (HardwareConfig.UseSimd && this is DenseTensor<T> d1 && tensor2 is DenseTensor<T> d2 && destination is DenseTensor<T> d3)
        {
            var vspan1 = MemoryMarshal.Cast<T, Vector<T>>(d1.Buffer.Span);
            var vspan2 = MemoryMarshal.Cast<T, Vector<T>>(d2.Buffer.Span);
            var vspan3 = MemoryMarshal.Cast<T, Vector<T>>(d3.Buffer.Span);
            var ceiling = (Convert.ToInt32(this.length) / Vector<T>.Count) * Vector<T>.Count;
            for (int i = 0; i < vspan1.Length; i++)
            {
                vspan3[i] = op(vspan1[i], vspan2[i]);
            }
            for (int i = ceiling; i < this.length; i++)
            {
                destination.SetValue(i, sop(GetValue(i), tensor2.GetValue(i)));
            }
        }
        else
        {
            Apply(sop, tensor2, destination);
        }
    }

    public virtual Tensor<T> VectorizedApply(Func<Vector<T>, Vector<T>, Vector<T>> op, Func<T, T, T> sop, Tensor<T> tensor2)
    {
        var output = CloneEmpty();
        VectorizedApply(op, sop, tensor2, output);
        return output;
    }

    public virtual T Accumulate(Func<T, T, T> op, T state)
    {
        var result = state;
        for (int index = 0; index < Length; index++)
        {
            result = op(result, GetValue(index));
        }
        return result;
    }

    public static Tensor<T>[] Broadcast(Tensor<T> inA, Tensor<T> inB)
    {
        if (inA.dimensions.SequenceEqual(inB.dimensions))
        {
            return [inA, inB];
        }
        else if (inA.Rank == 0 && inB.Rank != 0)
        {
            var _A = inB.CloneEmpty();
            for (int i = 0; i < _A.Length; i++)
            {
                _A.SetValue(i, inA.GetValue(0));
            }
            return [_A, inB ];
        }
        else if (inB.Rank == 0 && inA.Rank != 0)
        {
            var _B = inA.CloneEmpty();
            for (int i = 0; i < _B.Length; i++)
            {
                _B.SetValue(i, inB.GetValue(0));
            }
            return [inA, _B ];
        }

        var broadcastRank = Math.Max(inA.Rank, inB.Rank);
        var outA = inA;
        var outB = inB;
        for (var i = 0; i < broadcastRank; i++)
        {
            var idxA = i - broadcastRank + inA.Rank;
            var idxB = i - broadcastRank + inB.Rank;
            if (i < broadcastRank - inA.Rank)
            {
                outA = outA.InsertDim(i);
                outA = outA.BroadcastDim(i, inB.Dimensions[idxB]);
            }
            else if (i < broadcastRank - inB.Rank)
            {
                outB = outB.InsertDim(i);
                outB = outB.BroadcastDim(i, inA.Dimensions[idxA]);
            }
            else if (inA.Dimensions[idxA] == inB.Dimensions[idxB])
            {
                continue;
            }
            else if (inA.Dimensions[idxA] == 1)
            {
                outA = outA.BroadcastDim(i, inB.Dimensions[idxB]);
            }
            else if (inB.Dimensions[idxB] == 1)
            {
                outB = outB.BroadcastDim(i, inA.Dimensions[idxA]);
            }
            else
            {
                return Array.Empty<Tensor<T>>();
            }
        }
        return [outA, outB ];
    }

    public static bool Broadcast(Tensor<T> x, Tensor<T> y, out Tensor<T> outx, out Tensor<T> outy)
    {
        var b = Broadcast(x, y);
        if (b.Length == 0)
        {
            outx = null;
            outy = null;
            return false;
        }
        else
        {
            outx = b[0];
            outy = b[1];
            return true;
        }
    }

    public static bool Broadcast(Tensor<T> x, ReadOnlySpan<int> y, out Tensor<T> bx) =>
        Broadcast(x, new DenseTensor<T>(y), out bx, out _);

    public static bool BroadcastShape(ReadOnlySpan<int> x, ReadOnlySpan<int> y, out int[] b)
    {
        var tx = new DenseTensor<T>(x, true);
        var ty = new DenseTensor<T>(y, true);
        if (Broadcast(tx, ty, out var bx, out var _) == true)
        {
            b = bx.dimensions;
            return true;
        }
        else
        {
            b = null;
            return false;
        }
    }

    public static bool BroadcastShape(Tensor<T> x, Tensor<T> y, out int[] b) => BroadcastShape(x.Dimensions, y.Dimensions, out b);

    public static Tensor<T> BroadcastTo(Tensor<T> input, int[] targetShape)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (targetShape is null) throw new ArgumentNullException(nameof(targetShape));

        var targetRank = targetShape.Length;
        if (input.Rank > targetRank)
        {
            throw new ArgumentException(nameof(targetShape), "Target shape has fewer dimensions than the input tensor.");
        }

        var paddedInputDims = new int[targetRank];
        var offset = targetRank - input.Rank;
        for (int i = 0; i < targetRank; i++)
        {
            paddedInputDims[i] = i < offset ? 1 : input.Dimensions[i - offset];
        }

        var normalizedShape = new int[targetRank];
        for (int i = 0; i < targetRank; i++)
        {
            var dim = targetShape[i];
            if (dim == -1)
            {
                dim = paddedInputDims[i];
            }
            if (dim < 1)
            {
                throw new ArgumentException(nameof(targetShape), "Target shape dimensions must be positive or -1.");
            }
            normalizedShape[i] = dim;
        }

        StartOpStage(OpStage.CalculateIndices);
        var result = input;
        for (int i = 0; i < offset; i++)
        {
            result = result.InsertDim(0);
        }
        for (int i = 0; i < targetRank; i++)
        {
            var dim = result.Dimensions[i];
            var targetDim = normalizedShape[i];
            if (dim == targetDim)
            {
                continue;
            }
            if (dim == 1)
            {
                result = result.BroadcastDim(i, targetDim);
                continue;
            }
            throw new ArgumentException(nameof(targetShape), $"Cannot broadcast dimension {dim} to {targetDim}.");
        }
        return result;
    }

    /// <summary>
    /// Broadcasts data to the target shape following ONNX broadcast rules.
    /// A target dimension of 1 preserves the input dimension; added dimensions are stride-zero views.
    /// </summary>
    public static Tensor<T> Expand(Tensor<T> data, int[] targetShape)
    {
        // Broadcast is view-based: added dimensions are stride-zero views, so Expand itself allocates no element storage.
        StartOpStage(OpStage.ValidateArguments);
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (targetShape is null) throw new ArgumentNullException(nameof(targetShape));
        var targetRank = targetShape.Length;
        if (data.Rank > targetRank)
        {
            throw new ArgumentException(nameof(targetShape), "Target shape has fewer dimensions than the input tensor.");
        }

        StartOpStage(OpStage.CalculateIndices);
        var result = data;
        var offset = targetRank - data.Rank;
        for (int i = 0; i < offset; i++)
        {
            result = result.InsertDim(0);
        }

        for (int i = 0; i < targetRank; i++)
        {
            var inputDim = result.Dimensions[i];
            var targetDim = targetShape[i];
            if (targetDim == -1)
            {
                targetDim = inputDim;
            }
            else if (targetDim == 1 && inputDim > 1)
            {
                // Keep-dim on target 1: variable-size models (verified on the DINOv2 position-embedding interpolation path at 224 and 518 pixels) emit shape masks that collapse to 1 for dimensions that must be preserved, so 1 means keep here rather than broadcast.
                targetDim = inputDim;
            }

            if (inputDim == targetDim)
            {
                continue;
            }
            if (inputDim == 1)
            {
                result = result.BroadcastDim(i, targetDim);
                continue;
            }
            throw new ArgumentException(nameof(targetShape), $"Cannot broadcast dimension {inputDim} to {targetDim}.");
        }

        return result;
    }

    public static Tensor<bool> Equal(Tensor<T> x, Tensor<T> y)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (!Broadcast(x, y, out var bx, out var by))
        {
            throw new ArgumentException("Inputs are not broadcastable.");
        }
        var output = DenseTensor<bool>.OfShape(bx.Dimensions.ToArray());
        for (int i = 0; i < output.Length; i++)
        {
            output.SetValue(i, EqualityComparer<T>.Default.Equals(bx.GetValue(i), by.GetValue(i)));
        }
        return output;
    }

    public static Tensor<T> Where(Tensor<bool> condition, Tensor<T> x, Tensor<T> y)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (!Broadcast(x, y, out var bx, out var by))
        {
            throw new ArgumentException("Inputs are not broadcastable.");
        }
        var bcond = Tensor<bool>.BroadcastTo(condition, bx.Dimensions.ToArray());
        var output = bx.CloneEmpty();
        for (int i = 0; i < output.Length; i++)
        {
            output.SetValue(i, bcond.GetValue(i) ? bx.GetValue(i) : by.GetValue(i));
        }
        return output;
    }

    public static Tensor<byte> Add(Tensor<byte> x, Tensor<byte> y) => x.VectorizedApply((l, r) => (l + r), (l, r) => (byte) (l + r), y);

    public static Tensor<byte> Add(Tensor<byte> x, byte y) => x.Apply(l => (byte)(l + y));

    public static Tensor<int> Add(Tensor<int> x, Tensor<int> y) => x.VectorizedApply((l, r) => l + r, (l, r) => l + r, y);

    public static Tensor<int> Add(Tensor<int> x, int y) => x.VectorizedApply(l => l + new Vector<int>(y), l => l + y);

    public static Tensor<float> Add(Tensor<float> x, Tensor<float> y) => x.VectorizedApply((l, r) => l + r, (l, r) => l + r, y);

    public static Tensor<float> Add(Tensor<float> x, float y) => x.VectorizedApply(l => l + new Vector<float>(y), l => l + y);

    public static Tensor<double> Add(Tensor<double> x, Tensor<double> y) => x.VectorizedApply((l, r) => l + r, (l, r) => l + r, y);

    public static Tensor<double> Add(Tensor<double> x, double y) => x.VectorizedApply(l => l + new Vector<double>(y), l => l + y);

    public static Tensor<byte> Subtract(Tensor<byte> x, Tensor<byte> y) => x.VectorizedApply((l, r) => (l - r), (l, r) => (byte)(l - r), y);

    public static Tensor<byte> Subtract(Tensor<byte> x, byte y) => x.Apply(l => (byte)(l - y));

    public static Tensor<int> Subtract(Tensor<int> x, Tensor<int> y) => x.VectorizedApply((l, r) => l - r, (l, r) => l - r, y);

    public static Tensor<int> Subtract(Tensor<int> x, int y) => x.VectorizedApply(l => l - new Vector<int>(y), l => l - y);

    public static Tensor<float> Subtract(Tensor<float> x, Tensor<float> y) => x.VectorizedApply((l, r) => l - r, (l, r) => l - r, y);

    public static Tensor<float> Subtract(Tensor<float> x, float y) => x.VectorizedApply(l => l - new Vector<float>(y), l => l - y);

    public static Tensor<double> Subtract(Tensor<double> x, Tensor<double> y) => x.VectorizedApply((l, r) => l - r, (l, r) => l - r, y);

    public static Tensor<double> Subtract(Tensor<double> x, double y) => x.VectorizedApply(l => l - new Vector<double>(y), l => l - y);

    public static Tensor<byte> Multiply(Tensor<byte> x, Tensor<byte> y) => x.VectorizedApply((l, r) => (l * r), (l, r) => (byte)(l * r), y);

    public static Tensor<byte> Multiply(Tensor<byte> x, byte y) => x.Apply(l => (byte)(l * y));

    public static Tensor<int> Multiply(Tensor<int> x, Tensor<int> y) => x.VectorizedApply((l, r) => l * r, (l, r) => l * r, y);

    public static Tensor<int> Multiply(Tensor<int> x, int y) => x.VectorizedApply(l => l * new Vector<int>(y), l => l * y);

    public static Tensor<float> Multiply(Tensor<float> x, Tensor<float> y) => x.VectorizedApply((l, r) => l * r, (l, r) => l * r, y);

    public static Tensor<float> Multiply(Tensor<float> x, float y) => x.VectorizedApply(l => l * new Vector<float>(y), l => l * y);

    public static Tensor<double> Multiply(Tensor<double> x, Tensor<double> y) => x.VectorizedApply((l, r) => l * r, (l, r) => l * r, y);

    public static Tensor<double> Multiply(Tensor<double> x, double y) => x.VectorizedApply(l => l * new Vector<double>(y), l => l * y);

    public static Tensor<byte> Divide(Tensor<byte> x, Tensor<byte> y) => x.VectorizedApply((l, r) => (l / r), (l, r) => (byte)(l / r), y);

    public static Tensor<byte> Divide(Tensor<byte> x, byte y) => x.Apply(l => (byte)(l / y));

    public static Tensor<int> Divide(Tensor<int> x, Tensor<int> y) => x.VectorizedApply((l, r) => l / r, (l, r) => l / r, y);

    public static Tensor<int> Divide(Tensor<int> x, int y) => x.VectorizedApply(l => l / new Vector<int>(y), l => l / y);

    public static Tensor<long> Add(Tensor<long> x, Tensor<long> y) => x.VectorizedApply((l, r) => l + r, (l, r) => l + r, y);

    public static Tensor<long> Subtract(Tensor<long> x, Tensor<long> y) => x.VectorizedApply((l, r) => l - r, (l, r) => l - r, y);

    public static Tensor<long> Multiply(Tensor<long> x, Tensor<long> y) => x.VectorizedApply((l, r) => l * r, (l, r) => l * r, y);

    public static Tensor<long> Divide(Tensor<long> x, Tensor<long> y) => x.Apply((l, r) => l / r, y);

    public static Tensor<long> Divide(Tensor<long> x, long y) => x.Apply(l => l / y);

    public static Tensor<float> Divide(Tensor<float> x, Tensor<float> y) => x.VectorizedApply((l, r) => l / r, (l, r) => l / r, y);

    public static Tensor<float> Divide(Tensor<float> x, float y) => x.VectorizedApply(l => l / new Vector<float>(y), l => l / y);

    public static Tensor<double> Divide(Tensor<double> x, Tensor<double> y) => x.VectorizedApply((l, r) => l / r, (l, r) => l / r, y);

    public static Tensor<double> Divide(Tensor<double> x, double y) => x.VectorizedApply(l => l / new Vector<double>(y), l => l / y);

    public static Tensor<float> Negate(Tensor<float> x) => x.VectorizedApply(Vector.Negate, l => -l);

    public static Tensor<double> Negate(Tensor<double> x) => x.VectorizedApply(Vector.Negate, l => -l);

    public static Tensor<float> Pow(Tensor<float> x, Tensor<float> y) => x.Apply(MathF.Pow, y);

    public static Tensor<double> Pow(Tensor<double> x, Tensor<double> y) => x.Apply(Math.Pow, y);

    public static Tensor<float> Square(Tensor<float> x) => x.Apply(l => l * l);

    public static Tensor<double> Square(Tensor<double> x) => x.Apply(l => l * l);

    public static Tensor<float> Abs(Tensor<float> x) => x.Apply(l => l >= 0.0f ? l : -l);

    public static Tensor<double> Abs(Tensor<double> x) => x.Apply(l => l >= 0.0 ? l : -l);
    public static Tensor<float> Cos(Tensor<float> x) => x.VectorizedApply(Vector.Cos, MathF.Cos);

    public static Tensor<double> Cos(Tensor<double> x) => x.VectorizedApply(Vector.Cos, Math.Cos);

    public static Tensor<float> Sin(Tensor<float> x) => x.VectorizedApply(Vector.Sin, MathF.Sin);

    public static Tensor<double> Sin(Tensor<double> x) => x.VectorizedApply(Vector.Sin, Math.Sin);

    public static Tensor<int> Negate(Tensor<int> x) => x.VectorizedApply(Vector.Negate, l => -l);

    public static Tensor<long> Negate(Tensor<long> x) => x.VectorizedApply(Vector.Negate, l => -l);

    public static Tensor<int> Abs(Tensor<int> x) => x.Apply(l => l >= 0 ? l : -l);

    public static Tensor<long> Abs(Tensor<long> x) => x.Apply(l => l >= 0L ? l : -l);

    /// <summary>
    /// Exact Gaussian error linear unit: 0.5 * x * (1 + erf(x / sqrt(2))).
    /// </summary>
    public static Tensor<float> Gelu(Tensor<float> x) => x.Apply((float v) => 0.5f * v * (1f + MathOps.Erf(v * 0.7071067811865476f)));

    /// <summary>
    /// Exact Gaussian error linear unit: 0.5 * x * (1 + erf(x / sqrt(2))).
    /// </summary>
    public static Tensor<double> Gelu(Tensor<double> x) => x.Apply((double v) => 0.5 * v * (1.0 + MathOps.Erf(v * 0.7071067811865476)));

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
    public static Tensor<float> LayerNormalization(Tensor<float> x, Tensor<float> scale, Tensor<float>? bias, int axis = -1, float epsilon = 1e-5f)
    {
        var xd = x.ToDenseTensor();
        var sd = scale.ToDenseTensor();
        var bd = bias?.ToDenseTensor();
        var plan = LayerNormalizationPlan(x.Rank, axis, xd.Dimensions.ToArray(), xd.Length, sd.Length, bd?.Length);
        int block = plan.Block;
        int outer = plan.Outer;
        var output = new DenseTensor<float>(xd.Dimensions);
        var xs = xd.Buffer.Span;
        var ss = sd.Buffer.Span;
        var os = output.Buffer.Span;
        for (int o = 0; o < outer; o++)
        {
            double mean = 0.0;
            for (int i = 0; i < block; i++) mean += xs[o * block + i];
            mean /= block;
            double variance = 0.0;
            for (int i = 0; i < block; i++) { double d = xs[o * block + i] - mean; variance += d * d; }
            variance /= block;
            double inv = 1.0 / Math.Sqrt(variance + epsilon);
            for (int i = 0; i < block; i++) os[o * block + i] = (float)((xs[o * block + i] - mean) * inv * ss[i] + (bd is null ? 0f : bd.Buffer.Span[i]));
        }
        return output;
    }

    /// <summary>
    /// Normalizes over the input dimensions from axis to the last one using scale, optional bias, and epsilon.
    /// Statistics accumulate in double precision; the axis indexes the input tensor.
    /// </summary>
    public static Tensor<double> LayerNormalization(Tensor<double> x, Tensor<double> scale, Tensor<double>? bias, int axis = -1, double epsilon = 1e-5)
    {
        var xd = x.ToDenseTensor();
        var sd = scale.ToDenseTensor();
        var bd = bias?.ToDenseTensor();
        var plan = LayerNormalizationPlan(x.Rank, axis, xd.Dimensions.ToArray(), xd.Length, sd.Length, bd?.Length);
        int block = plan.Block;
        int outer = plan.Outer;
        var output = new DenseTensor<double>(xd.Dimensions);
        var xs = xd.Buffer.Span;
        var ss = sd.Buffer.Span;
        var os = output.Buffer.Span;
        for (int o = 0; o < outer; o++)
        {
            double mean = 0.0;
            for (int i = 0; i < block; i++) mean += xs[o * block + i];
            mean /= block;
            double variance = 0.0;
            for (int i = 0; i < block; i++) { double d = xs[o * block + i] - mean; variance += d * d; }
            variance /= block;
            double inv = 1.0 / Math.Sqrt(variance + epsilon);
            for (int i = 0; i < block; i++) os[o * block + i] = (xs[o * block + i] - mean) * inv * ss[i] + (bd is null ? 0.0 : bd.Buffer.Span[i]);
        }
        return output;
    }

    /// <summary>
    /// Generates start, start plus delta, and so on, stopping before limit. Zero delta throws.
    /// </summary>
    public static Tensor<float> Range(float start, float limit, float delta)
    {
        if (delta == 0f) throw new ArgumentException(nameof(delta));
        int count = Math.Max((int)Math.Ceiling((limit - start) / delta), 0);
        var output = new DenseTensor<float>(count);
        var span = output.Buffer.Span;
        for (int i = 0; i < count; i++) span[i] = start + i * delta;
        return output;
    }

    /// <summary>
    /// Generates start, start plus delta, and so on, stopping before limit. Zero delta throws.
    /// </summary>
    public static Tensor<double> Range(double start, double limit, double delta)
    {
        if (delta == 0.0) throw new ArgumentException(nameof(delta));
        int count = Math.Max((int)Math.Ceiling((limit - start) / delta), 0);
        var output = new DenseTensor<double>(count);
        var span = output.Buffer.Span;
        for (int i = 0; i < count; i++) span[i] = start + i * delta;
        return output;
    }

    /// <summary>
    /// Generates start, start plus delta, and so on, stopping before limit. Zero delta throws.
    /// </summary>
    public static Tensor<long> Range(long start, long limit, long delta)
    {
        if (delta == 0L) throw new ArgumentException(nameof(delta));
        int count = Math.Max((int)Math.Ceiling(((double)limit - start) / delta), 0);
        var output = new DenseTensor<long>(count);
        var span = output.Buffer.Span;
        for (int i = 0; i < count; i++) span[i] = start + i * delta;
        return output;
    }

    /// <summary>
    /// Generates start, start plus delta, and so on, stopping before limit. Zero delta throws.
    /// </summary>
    public static Tensor<int> Range(int start, int limit, int delta)
    {
        if (delta == 0) throw new ArgumentException(nameof(delta));
        int count = Math.Max((int)Math.Ceiling(((double)limit - start) / delta), 0);
        var output = new DenseTensor<int>(count);
        var span = output.Buffer.Span;
        for (int i = 0; i < count; i++) span[i] = start + i * delta;
        return output;
    }

    static DenseTensor<U> TileCore<U>(Tensor<U> x, int[] repeats) where U : unmanaged
    {
        if (repeats.Length != x.Rank) throw new ArgumentException("Repeats rank must match input rank.", nameof(repeats));
        var xd = x.ToDenseTensor();
        var outDims = new int[x.Rank];
        for (int i = 0; i < x.Rank; i++)
        {
            if (repeats[i] < 0) throw new ArgumentException("Repeats must be non-negative.", nameof(repeats));
            outDims[i] = xd.Dimensions[i] * repeats[i];
        }
        var output = new DenseTensor<U>((ReadOnlySpan<int>)outDims);
        var src = new int[x.Rank];
        foreach (var index in output.GetDimensionsIterator())
        {
            for (int i = 0; i < x.Rank; i++) src[i] = xd.Dimensions[i] == 0 ? 0 : index[i] % xd.Dimensions[i];
            output[index] = xd[src];
        }
        return output;
    }

    /// <summary>
    /// Repeats the input repeats[i] times along each dimension.
    /// The repeats rank must match the input rank and every repeat must be non-negative.
    /// </summary>
    public static Tensor<float> Tile(Tensor<float> x, int[] repeats) => TileCore(x, repeats);

    /// <summary>
    /// Repeats the input repeats[i] times along each dimension.
    /// The repeats rank must match the input rank and every repeat must be non-negative.
    /// </summary>
    public static Tensor<double> Tile(Tensor<double> x, int[] repeats) => TileCore(x, repeats);

    /// <summary>
    /// Repeats the input repeats[i] times along each dimension.
    /// The repeats rank must match the input rank and every repeat must be non-negative.
    /// </summary>
    public static Tensor<int> Tile(Tensor<int> x, int[] repeats) => TileCore(x, repeats);

    /// <summary>
    /// Repeats the input repeats[i] times along each dimension.
    /// The repeats rank must match the input rank and every repeat must be non-negative.
    /// </summary>
    public static Tensor<long> Tile(Tensor<long> x, int[] repeats) => TileCore(x, repeats);

    /// <summary>
    /// Copies the length elements starting at start along the axis.
    /// Negative axes resolve against the input rank; out-of-range chunks throw.
    /// </summary>
    public static DenseTensor<U> ChunkCopy<U>(Tensor<U> x, int axis, int start, int length) where U : unmanaged
    {
        int rank = x.Rank;
        int ax = axis < 0 ? axis + rank : axis;
        if (ax < 0 || ax >= rank) throw new ArgumentException(nameof(axis));
        if (start < 0 || length < 0 || start + length > x.Dimensions[ax]) throw new ArgumentException("Chunk is out of range.");
        var xd = x.ToDenseTensor();
        var outDims = xd.Dimensions.ToArray();
        outDims[ax] = length;
        var output = new DenseTensor<U>((ReadOnlySpan<int>)outDims);
        int inner = 1;
        for (int trailing = ax + 1; trailing < rank; trailing++) inner *= outDims[trailing];
        int outer = 1;
        for (int leading = 0; leading < ax; leading++) outer *= outDims[leading];
        if (HasStandardStrides(xd))
        {
            var outputSpan = output.Buffer.Span;
            var sourceSpan = xd.Buffer.Span;
            int axisLength = xd.Dimensions[ax];
            for (int outerIndex = 0; outerIndex < outer; outerIndex++)
            {
                sourceSpan.Slice((outerIndex * axisLength + start) * inner, length * inner).CopyTo(outputSpan.Slice(outerIndex * length * inner, length * inner));
            }
            return output;
        }
        var src = new int[rank];
        foreach (var index in output.GetDimensionsIterator())
        {
            for (int i = 0; i < rank; i++) src[i] = index[i];
            src[ax] += start;
            output[index] = xd[src];
        }
        return output;
    }

    public static Tensor<float> Sqrt(Tensor<float> x) => x.VectorizedApply(Vector.SquareRoot, MathF.Sqrt);

    public static Tensor<double> Sqrt(Tensor<double> x) => x.VectorizedApply(Vector.SquareRoot, Math.Sqrt);

    public static Tensor<float> Resize(Tensor<float> input, int[] sizes, string mode, string coordinateTransformationMode, string nearestMode, float cubicCoeffA)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (input.Rank != 4) throw new ArgumentException(nameof(input), "Resize currently supports only 4D tensors (NCHW).");
        if (sizes is null || sizes.Length != 4) throw new ArgumentException(nameof(sizes), "Resize sizes must be a 1D array of length 4.");

        var nOut = sizes[0];
        var cOut = sizes[1];
        var hOut = sizes[2];
        var wOut = sizes[3];
        var nIn = input.Dimensions[0];
        var cIn = input.Dimensions[1];
        var hIn = input.Dimensions[2];
        var wIn = input.Dimensions[3];

        if (nOut != nIn || cOut != cIn)
        {
            throw new ArgumentException(nameof(sizes), "Resize currently requires N and C dimensions to remain unchanged.");
        }

        var output = DenseTensor<float>.OfShape(sizes);
        var scaleH = (float)hOut / hIn;
        var scaleW = (float)wOut / wIn;

        float TransformCoordinate(int outIndex, int inSize, int outSize, float scale)
        {
            return coordinateTransformationMode switch
            {
                "half_pixel" => (outIndex + 0.5f) / scale - 0.5f,
                "align_corners" => outSize == 1 ? 0f : outIndex * (inSize - 1f) / (outSize - 1f),
                "asymmetric" => outIndex / scale,
                _ => throw new NotSupportedException($"coordinate_transformation_mode {coordinateTransformationMode} is not supported."),
            };
        }

        int NearestIndex(float coord, int inSize)
        {
            var value = nearestMode switch
            {
                "floor" => (int)MathF.Floor(coord),
                "ceil" => (int)MathF.Ceiling(coord),
                "round_prefer_floor" => (int)MathF.Floor(coord + 0.5f),
                "round_prefer_ceil" => (int)MathF.Ceiling(coord - 0.5f),
                _ => throw new NotSupportedException($"nearest_mode {nearestMode} is not supported."),
            };
            return Math.Clamp(value, 0, inSize - 1);
        }

        static float CubicWeight(float x, float a)
        {
            var t = MathF.Abs(x);
            if (t <= 1f)
            {
                return ((a + 2f) * t * t * t) - ((a + 3f) * t * t) + 1f;
            }
            if (t < 2f)
            {
                return (a * t * t * t) - (5f * a * t * t) + (8f * a * t) - (4f * a);
            }
            return 0f;
        }

        for (int n = 0; n < nOut; n++)
        {
            for (int c = 0; c < cOut; c++)
            {
                for (int oy = 0; oy < hOut; oy++)
                {
                    var inY = TransformCoordinate(oy, hIn, hOut, scaleH);
                    if (mode == "nearest")
                    {
                        var ny = NearestIndex(inY, hIn);
                        for (int ox = 0; ox < wOut; ox++)
                        {
                            var inX = TransformCoordinate(ox, wIn, wOut, scaleW);
                            var nx = NearestIndex(inX, wIn);
                            output[n, c, oy, ox] = input[n, c, ny, nx];
                        }
                    }
                    else if (mode == "linear")
                    {
                        var y0 = MathF.Floor(inY);
                        var y1 = y0 + 1f;
                        var y0i = Math.Clamp((int)y0, 0, hIn - 1);
                        var y1i = Math.Clamp((int)y1, 0, hIn - 1);
                        var ly = inY - y0;
                        var wy0 = 1f - ly;
                        var wy1 = ly;
                        for (int ox = 0; ox < wOut; ox++)
                        {
                            var inX = TransformCoordinate(ox, wIn, wOut, scaleW);
                            var x0 = MathF.Floor(inX);
                            var x1 = x0 + 1f;
                            var x0i = Math.Clamp((int)x0, 0, wIn - 1);
                            var x1i = Math.Clamp((int)x1, 0, wIn - 1);
                            var lx = inX - x0;
                            var wx0 = 1f - lx;
                            var wx1 = lx;
                            var v00 = input[n, c, y0i, x0i];
                            var v01 = input[n, c, y0i, x1i];
                            var v10 = input[n, c, y1i, x0i];
                            var v11 = input[n, c, y1i, x1i];
                            output[n, c, oy, ox] = (v00 * wy0 * wx0) + (v01 * wy0 * wx1) + (v10 * wy1 * wx0) + (v11 * wy1 * wx1);
                        }
                    }
                    else if (mode == "cubic")
                    {
                        var yBase = (int)MathF.Floor(inY);
                        var wy = new float[4];
                        var yIdx = new int[4];
                        for (int i = 0; i < 4; i++)
                        {
                            var yi = yBase - 1 + i;
                            yIdx[i] = Math.Clamp(yi, 0, hIn - 1);
                            wy[i] = CubicWeight(inY - yi, cubicCoeffA);
                        }
                        for (int ox = 0; ox < wOut; ox++)
                        {
                            var inX = TransformCoordinate(ox, wIn, wOut, scaleW);
                            var xBase = (int)MathF.Floor(inX);
                            var wx = new float[4];
                            var xIdx = new int[4];
                            for (int i = 0; i < 4; i++)
                            {
                                var xi = xBase - 1 + i;
                                xIdx[i] = Math.Clamp(xi, 0, wIn - 1);
                                wx[i] = CubicWeight(inX - xi, cubicCoeffA);
                            }
                            var sum = 0f;
                            for (int iy = 0; iy < 4; iy++)
                            {
                                var wyv = wy[iy];
                                for (int ix = 0; ix < 4; ix++)
                                {
                                    sum += wyv * wx[ix] * input[n, c, yIdx[iy], xIdx[ix]];
                                }
                            }
                            output[n, c, oy, ox] = sum;
                        }
                    }
                    else
                    {
                        throw new NotSupportedException($"Resize mode {mode} is not supported.");
                    }
                }
            }
        }

        return output;
    }

    public static Tensor<double> Resize(Tensor<double> input, int[] sizes, string mode, string coordinateTransformationMode, string nearestMode, double cubicCoeffA)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (input.Rank != 4) throw new ArgumentException(nameof(input), "Resize currently supports only 4D tensors (NCHW).");
        if (sizes is null || sizes.Length != 4) throw new ArgumentException(nameof(sizes), "Resize sizes must be a 1D array of length 4.");

        var nOut = sizes[0];
        var cOut = sizes[1];
        var hOut = sizes[2];
        var wOut = sizes[3];
        var nIn = input.Dimensions[0];
        var cIn = input.Dimensions[1];
        var hIn = input.Dimensions[2];
        var wIn = input.Dimensions[3];

        if (nOut != nIn || cOut != cIn)
        {
            throw new ArgumentException(nameof(sizes), "Resize currently requires N and C dimensions to remain unchanged.");
        }

        var output = DenseTensor<double>.OfShape(sizes);
        var scaleH = (double)hOut / hIn;
        var scaleW = (double)wOut / wIn;

        double TransformCoordinate(int outIndex, int inSize, int outSize, double scale)
        {
            return coordinateTransformationMode switch
            {
                "half_pixel" => (outIndex + 0.5) / scale - 0.5,
                "align_corners" => outSize == 1 ? 0d : outIndex * (inSize - 1d) / (outSize - 1d),
                "asymmetric" => outIndex / scale,
                _ => throw new NotSupportedException($"coordinate_transformation_mode {coordinateTransformationMode} is not supported."),
            };
        }

        int NearestIndex(double coord, int inSize)
        {
            var value = nearestMode switch
            {
                "floor" => (int)Math.Floor(coord),
                "ceil" => (int)Math.Ceiling(coord),
                "round_prefer_floor" => (int)Math.Floor(coord + 0.5),
                "round_prefer_ceil" => (int)Math.Ceiling(coord - 0.5),
                _ => throw new NotSupportedException($"nearest_mode {nearestMode} is not supported."),
            };
            return Math.Clamp(value, 0, inSize - 1);
        }

        static double CubicWeight(double x, double a)
        {
            var t = Math.Abs(x);
            if (t <= 1d)
            {
                return ((a + 2d) * t * t * t) - ((a + 3d) * t * t) + 1d;
            }
            if (t < 2d)
            {
                return (a * t * t * t) - (5d * a * t * t) + (8d * a * t) - (4d * a);
            }
            return 0d;
        }

        for (int n = 0; n < nOut; n++)
        {
            for (int c = 0; c < cOut; c++)
            {
                for (int oy = 0; oy < hOut; oy++)
                {
                    var inY = TransformCoordinate(oy, hIn, hOut, scaleH);
                    if (mode == "nearest")
                    {
                        var ny = NearestIndex(inY, hIn);
                        for (int ox = 0; ox < wOut; ox++)
                        {
                            var inX = TransformCoordinate(ox, wIn, wOut, scaleW);
                            var nx = NearestIndex(inX, wIn);
                            output[n, c, oy, ox] = input[n, c, ny, nx];
                        }
                    }
                    else if (mode == "linear")
                    {
                        var y0 = Math.Floor(inY);
                        var y1 = y0 + 1d;
                        var y0i = Math.Clamp((int)y0, 0, hIn - 1);
                        var y1i = Math.Clamp((int)y1, 0, hIn - 1);
                        var ly = inY - y0;
                        var wy0 = 1d - ly;
                        var wy1 = ly;
                        for (int ox = 0; ox < wOut; ox++)
                        {
                            var inX = TransformCoordinate(ox, wIn, wOut, scaleW);
                            var x0 = Math.Floor(inX);
                            var x1 = x0 + 1d;
                            var x0i = Math.Clamp((int)x0, 0, wIn - 1);
                            var x1i = Math.Clamp((int)x1, 0, wIn - 1);
                            var lx = inX - x0;
                            var wx0 = 1d - lx;
                            var wx1 = lx;
                            var v00 = input[n, c, y0i, x0i];
                            var v01 = input[n, c, y0i, x1i];
                            var v10 = input[n, c, y1i, x0i];
                            var v11 = input[n, c, y1i, x1i];
                            output[n, c, oy, ox] = (v00 * wy0 * wx0) + (v01 * wy0 * wx1) + (v10 * wy1 * wx0) + (v11 * wy1 * wx1);
                        }
                    }
                    else if (mode == "cubic")
                    {
                        var yBase = (int)Math.Floor(inY);
                        var wy = new double[4];
                        var yIdx = new int[4];
                        for (int i = 0; i < 4; i++)
                        {
                            var yi = yBase - 1 + i;
                            yIdx[i] = Math.Clamp(yi, 0, hIn - 1);
                            wy[i] = CubicWeight(inY - yi, cubicCoeffA);
                        }
                        for (int ox = 0; ox < wOut; ox++)
                        {
                            var inX = TransformCoordinate(ox, wIn, wOut, scaleW);
                            var xBase = (int)Math.Floor(inX);
                            var wx = new double[4];
                            var xIdx = new int[4];
                            for (int i = 0; i < 4; i++)
                            {
                                var xi = xBase - 1 + i;
                                xIdx[i] = Math.Clamp(xi, 0, wIn - 1);
                                wx[i] = CubicWeight(inX - xi, cubicCoeffA);
                            }
                            var sum = 0d;
                            for (int iy = 0; iy < 4; iy++)
                            {
                                var wyv = wy[iy];
                                for (int ix = 0; ix < 4; ix++)
                                {
                                    sum += wyv * wx[ix] * input[n, c, yIdx[iy], xIdx[ix]];
                                }
                            }
                            output[n, c, oy, ox] = sum;
                        }
                    }
                    else
                    {
                        throw new NotSupportedException($"Resize mode {mode} is not supported.");
                    }
                }
            }
        }

        return output;
    }

    public static Tensor<int> MatMul2D(Tensor<int> x, Tensor<int> y)
    {
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
        var m = x.Dimensions[0];
        var n = x.Dimensions[1];
        var k = y.Dimensions[1];

        var dx = x as DenseTensor<int>;
        var dy = y as DenseTensor<int>;
        var _x = dx is not null && !dx.IsReversedStride ? dx : x.ToDenseTensor();
        var _y = dy is not null && !dy.IsReversedStride ? dy : y.ToDenseTensor();
        var output = DenseTensor<int>.OfShape(new int[] { x.Dimensions[0], y.Dimensions[1] });

        var xh = _x.Buffer.Pin();
        var yh = _y.Buffer.Pin();
        var oh = output.Buffer.Pin();
        if (HardwareConfig.UseSimd)
        {
            unsafe
            {
                mm_unsafe_vectorized(m, n, k, (int*)xh.Pointer, (int*)yh.Pointer, (int*)oh.Pointer);
            }
        }
        else
        {
            unsafe
            {
                mm(m, n, k, (int*)xh.Pointer, (int*)yh.Pointer, (int*)oh.Pointer);
            }
        }

        xh.Dispose();
        yh.Dispose();
        oh.Dispose();
        return output;
    }

    static (DenseTensor<float> x, DenseTensor<float> y) DensifyFloatOperands(Tensor<float> x, Tensor<float> y)
    {
        var dx = x as DenseTensor<float>;
        var dy = y as DenseTensor<float>;
        var needsCopyX = dx is null || dx.IsReversedStride;
        var needsCopyY = dy is null || dy.IsReversedStride;
        if (needsCopyX || needsCopyY)
        {
            StartOpStage(OpStage.Copy);
        }
        return (needsCopyX ? x.ToDenseTensor() : dx!, needsCopyY ? y.ToDenseTensor() : dy!);
    }

    static unsafe void RunFloatMatMulKernel(int m, int n, int k, float* x, float* y, float* output, TensorExecutionOptions options)
    {
        if (options.UseSimd && options.UseIntrinsics && Fma.IsSupported)
        {
            if (m % 2 == 0 && k % 32 == 0)
            {
                mm_unsafe_vectorized_intrinsics_2x4(m, n, k, x, y, output);
            }
            else
            {
                mm_unsafe_vectorized_intrinsics(m, n, k, x, y, output);
            }
        }
        else if (options.UseSimd)
        {
            mm_unsafe_vectorized(m, n, k, x, y, output);
        }
        else
        {
            mm(m, n, k, x, y, output);
        }
    }

    public static Tensor<float> MatMul2D(Tensor<float> x, Tensor<float> y) => MatMul2D(x, y, TensorExecutionOptions.Auto);

    public static Tensor<float> MatMul2D(Tensor<float> x, Tensor<float> y, TensorExecutionOptions options)
    {
        options.Validate();
        StartOpStage(OpStage.ValidateArguments);
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException($"The number of columns in the first matrix ({x.Dimensions[1]}) is not equal to the number of rows in the second matrix ({y.Dimensions[0]}).");
        var m = x.Dimensions[0];
        var n = x.Dimensions[1];
        var k = y.Dimensions[1];

        var (_x, _y) = DensifyFloatOperands(x, y);
        var output = DenseTensor<float>.OfShape(new int[] { x.Dimensions[0], y.Dimensions[1] });

        StartOpStage(OpStage.Math);
        using var xh = _x.Buffer.Pin(); 
        using var yh = _y.Buffer.Pin();
        using var oh = output.Buffer.Pin();
        unsafe
        {
            RunFloatMatMulKernel(m, n, k, (float*)xh.Pointer, (float*)yh.Pointer, (float*)oh.Pointer, options);
        }
        return output;
    }

    public static Tensor<double> MatMul2D(Tensor<double> x, Tensor<double> y)
    {
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
        var m = x.Dimensions[0];
        var n = x.Dimensions[1];
        var k = y.Dimensions[1];


        var dx = x as DenseTensor<double>;
        var dy = y as DenseTensor<double>;
        var _x = dx is not null && !dx.IsReversedStride ? dx : x.ToDenseTensor();
        var _y = dy is not null && !dy.IsReversedStride ? dy : y.ToDenseTensor();
        var output = DenseTensor<double>.OfShape(new int[] { x.Dimensions[0], y.Dimensions[1] });

        var xh = _x.Buffer.Pin();
        var yh = _y.Buffer.Pin();
        var oh = output.Buffer.Pin();
        if (HardwareConfig.UseSimd && HardwareConfig.UseIntrinsics && Fma.IsSupported)
        {
            unsafe
            {
                mm_unsafe_vectorized_intrinsics(m, n, k, (double*)xh.Pointer, (double*)yh.Pointer, (double*)oh.Pointer);
            }
        }
        else if (HardwareConfig.UseSimd)
        {
            unsafe
            {
                mm_unsafe_vectorized(m, n, k, (double*)xh.Pointer, (double*)yh.Pointer, (double*)oh.Pointer);
            }
        }
        else
            unsafe
            {
                mm(m, n, k, (double*)xh.Pointer, (double*)yh.Pointer, (double*)oh.Pointer);
            }
        xh.Dispose();
        yh.Dispose();
        oh.Dispose();
        return output;
    }

    public static Tensor<int> MatMul2D_managed(Tensor<int> x, Tensor<int> y)
    {
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
        int rA = x.Dimensions[0];
        int cA = x.Dimensions[1];
        int cB = y.Dimensions[1];
        var output = DenseTensor<int>.OfShape(new int[] { rA, cB });
        int temp;
       
        for (int i = 0; i < rA; i++)
        {
            for (int j = 0; j < cB; j++)
            {
                temp = 0;
                for (int k = 0; k < cA; k++)
                {
                    temp += x[i, k] * y[k, j];
                }
                output.SetValue((i * rA + j), temp);
            }
        }
        return output;
    }

    public static Tensor<float> MatMul2D_managed(Tensor<float> x, Tensor<float> y)
    {
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
        int rA = x.Dimensions[0];
        int cA = x.Dimensions[1];
        int cB = y.Dimensions[1];
        var output = DenseTensor<float>.OfShape(new int[] { rA, cB });
        float temp;

        for (int i = 0; i < rA; i++)
        {
            for (int j = 0; j < cB; j++)
            {
                temp = 0;
                for (int k = 0; k < cA; k++)
                {
                    temp += x[i, k] * y[k, j];
                }
                output.SetValue((i * rA + j), temp);
            }
        }
        return output;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static Tensor<int> MatMul(Tensor<int> x, Tensor<int> y)
    {
        if (x.Rank == 0 || y.Rank == 0) throw new ArgumentException("The rank of each tensor in matrix multiplication must be greater than 1.");
        if (x.Rank == 2 && y.Rank == 2)
        {
            return Tensor<int>.MatMul2D(x, y);
        }
        else if (x.Rank >= 2 && y.Rank >= 2)
        {
            var xdl = x.Dimensions[^2..];
            var ydl = y.Dimensions[^2..];
            if (xdl[1] != ydl[0])
            {
                throw new ArgumentException($"The number of columns in the first matrix ({xdl[1]}) is not equal to the number of rows in the second matrix ({ydl[0]}).");
            }
            StartOpStage(OpStage.Broadcast);
            if (!BroadcastShape(x.Dimensions[0..^2], y.Dimensions[0..^2], out var bd))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }
            var bdx = bd.Append(xdl[0]).Append(xdl[1]).ToArray();
            var bdy = bd.Append(ydl[0]).Append(ydl[1]).ToArray();
            if (!Tensor<int>.Broadcast(y, bdy, out var by))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }
            if (!Tensor<int>.Broadcast(x, bdx, out var bx))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }
            //StopOpStage();
            var z = DenseTensor<int>.OfShape(bd.Append(xdl[0]).Append(ydl[1]).ToArray());
            var di = bx.GetDimensionsIterator(0..^2);
            using var xh = bx.Storage.Pin();
            using var yh = by.Storage.Pin();
            using var zh = z.Storage.Pin();
            var m = bx.Dimensions[^2];
            var n = bx.Dimensions[^1];
            var k = by.Dimensions[^1];
            StartOpStage(OpStage.Math);
            unsafe
            {
                var xp = (int*)xh.Pointer;
                var yp = (int*)yh.Pointer;
                var zp = (int*)zh.Pointer;
                
                foreach (var idx in di)
                {
                    if (HardwareConfig.UseSimd)
                    {
                        mm_unsafe_vectorized(m, n, k, xp + bx.GetStorageIndex(idx), yp + by.GetStorageIndex(idx), zp + z.GetStorageIndex(idx));
                    }
                    else
                    {
                        mm(m, n, k, xp + bx.GetStorageIndex(idx), yp + by.GetStorageIndex(idx), zp + z.GetStorageIndex(idx));
                    }
                }
            }
            return z;
        }
        else if (x.Rank >= 2 || y.Rank >= 2)
        {
            StartOpStage(OpStage.Broadcast);
            if (!Tensor<int>.Broadcast(x, y, out var bx, out var by))
            {
                throw new ArgumentException($"The shapes {x.PrintShape()} and {y.PrintShape()} are not compatible for broadcasting.");
            }
            else
            {
                //StopOpStage();
                return MatMul(bx, by);
            }
        }
        else //(x.Rank < 2 && y.Rank < 2)
        {
            bool bcast = false;
            if (x.Rank == 1)
            {
                x = x.PadLeft();
                bcast = true;
            }
            if (y.Rank == 1)
            {
                y = y.PadRight();
                bcast = true;
            }
            var c = MatMul2D(x, y);
            if (bcast)
            {
                c.RemoveDim(0);
            }
            return c;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]  
    public static Tensor<float> MatMul(Tensor<float> x, Tensor<float> y) => MatMul(x, y, TensorExecutionOptions.Auto);

    public static Tensor<float> MatMul(Tensor<float> x, Tensor<float> y, TensorExecutionOptions options)
    {
        if (x.Rank == 0 || y.Rank == 0) throw new ArgumentException("The rank of each tensor in matrix multiplication must be greater than 1.");
        if (x.Rank == 2 && y.Rank == 2)
        {
            return Tensor<float>.MatMul2D(x, y, options);
        }
        else if (x.Rank >= 2 && y.Rank >= 2)
        {
            var xdl = x.Dimensions[^2..];
            var ydl = y.Dimensions[^2..];
            if (xdl[1] != ydl[0])
            {
                throw new ArgumentException($"The number of columns in the first matrix ({xdl[1]}) is not equal to the number of rows in the second matrix ({ydl[0]}).");
            }
            StartOpStage(OpStage.Broadcast);
            if (!BroadcastShape(x.Dimensions[0..^2], y.Dimensions[0..^2], out var bd))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }

            var bdx = bd.Append(xdl[0]).Append(xdl[1]).ToArray();
            if (!Tensor<float>.Broadcast(x, bdx, out var bx))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }
            var bdy = bd.Append(ydl[0]).Append(ydl[1]).ToArray();
            if (!Tensor<float>.Broadcast(y, bdy, out var by))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }

            StartOpStage(OpStage.Math);

            var z = DenseTensor<float>.OfShape(bd.Append(xdl[0]).Append(ydl[1]).ToArray());
            var di = bx.GetDimensionsIterator(0..^2);
            using var xh = bx.Storage.Pin();
            using var yh = by.Storage.Pin();
            using var zh = z.Storage.Pin();
            var m = bx.Dimensions[^2];
            var n = bx.Dimensions[^1];
            var k = by.Dimensions[^1];
          
            unsafe
            {
                var xp = (float*)xh.Pointer;
                var yp = (float*)yh.Pointer;
                var zp = (float*)zh.Pointer;
                foreach (var idx in di)
                {
                    RunFloatMatMulKernel(m, n, k, xp + bx.GetStorageIndex(idx), yp + by.GetStorageIndex(idx), zp + z.GetStorageIndex(idx), options);
                }
            }
            return z;
        }
        else if (x.Rank >= 2 || y.Rank >= 2)
        {
            if (!Tensor<float>.Broadcast(x, y, out var bx, out var by))
            {
                throw new ArgumentException($"The shapes {x.PrintShape()} and {y.PrintShape()} are not compatible for broadcasting.");
            }
            else
            {
                return MatMul(bx, by, options);
            }
        }
        else //(x.Rank < 2 && y.Rank < 2)
        {
            bool bcast = false;
            if (x.Rank == 1)
            {
                x = x.PadLeft();
                bcast = true;
            }
            if (y.Rank == 1)
            {
                y = y.PadRight();
                bcast = true;
            }
            var c = MatMul2D(x, y, options);
            if (bcast)
            {
                c.RemoveDim(0);
            }
            return c;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static Tensor<float> MatMul_old(Tensor<float> x, Tensor<float> y)
    {
        if (x.Rank == 0 || y.Rank == 0) throw new ArgumentException("The rank of each tensor in matrix multiplication must be greater than 1.");
        if (x.Rank == 2 && y.Rank == 2)
        {
            return Tensor<float>.MatMul2D(x, y);
        }
        else if (x.Rank >= 2 && y.Rank >= 2)
        {
            var xdl = x.Dimensions[^2..];
            var ydl = y.Dimensions[^2..];
            if (xdl[1] != ydl[0])
            {
                throw new ArgumentException($"The number of columns in the first matrix ({xdl[1]}) is not equal to the number of rows in the second matrix ({ydl[0]}).");
            }

            if (!BroadcastShape(x.Dimensions[0..^2], y.Dimensions[0..^2], out var bd))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }

            var bdx = bd.Append(xdl[0]).Append(xdl[1]).ToArray();
            if (!Tensor<float>.Broadcast(x, bdx, out var bx))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }
            var bdy = bd.Append(ydl[0]).Append(ydl[1]).ToArray();
            if (!Tensor<float>.Broadcast(y, bdy, out var by))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }
            var z = DenseTensor<float>.OfShape(bd.Append(xdl[0]).Append(ydl[1]).ToArray());
            var di = bx.GetDimensionsIterator(0..^2);
            
            foreach (var idx in di)
            {
                var bxe = new TensorFixedDimensionsIterator(idx, bx.dimensions[^2], bx.dimensions[^1]);
                var bxs = DenseTensor<float>.OfShape(bx.dimensions[^2], bx.dimensions[^1]);
                foreach (var idx2 in bxe)
                {
                    bxs[bxe.VariableIndex] = bx[idx2];
                }

                var bye = new TensorFixedDimensionsIterator(idx, by.dimensions[^2], by.dimensions[^1]);
                var bys = DenseTensor<float>.OfShape(by.dimensions[^2], by.dimensions[^1]);
                foreach (var idx2 in bye)
                {
                    bys[bye.VariableIndex] = by[idx2];
                }

                var bzs = Tensor<float>.MatMul2D(bxs, bys);
                var bze = new TensorFixedDimensionsIterator(idx, bx.dimensions[^2], by.dimensions[^1]);
                foreach (var idx2 in bze)
                {
                    z[idx2] = bzs[bze.VariableIndex];
                }
            }
            
            return z;
        }
        else if (x.Rank >= 2 || y.Rank >= 2)
        {
            if (!Tensor<float>.Broadcast(x, y, out var bx, out var by))
            {
                throw new ArgumentException($"The shapes {x.PrintShape()} and {y.PrintShape()} are not compatible for broadcasting.");
            }
            else
            {
                return MatMul(bx, by);
            }
        }
        else //(x.Rank < 2 && y.Rank < 2)
        {
            bool bcast = false;
            if (x.Rank == 1)
            {
                x = x.PadLeft();
                bcast = true;
            }
            if (y.Rank == 1)
            {
                y = y.PadRight();
                bcast = true;
            }
            var c = MatMul2D(x, y);
            if (bcast)
            {
                c.RemoveDim(0);
            }
            return c;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static Tensor<double> MatMul(Tensor<double> x, Tensor<double> y)
    {
        if (x.Rank == 0 || y.Rank == 0) throw new ArgumentException("The rank of each tensor in matrix multiplication must be greater than 1.");
        if (x.Rank == 2 && y.Rank == 2)
        {
            return Tensor<double>.MatMul2D(x, y);
        }
        else if (x.Rank >= 2 && y.Rank >= 2)
        {
            var xdl = x.Dimensions[^2..];
            var ydl = y.Dimensions[^2..];
            if (xdl[1] != ydl[0])
            {
                throw new ArgumentException($"The number of columns in the first matrix ({xdl[1]}) is not equal to the number of rows in the second matrix ({ydl[0]}).");
            }
            StartOpStage(OpStage.Broadcast);
            if (!BroadcastShape(x.Dimensions[0..^2], y.Dimensions[0..^2], out var bd))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }

            var bdx = bd.Append(xdl[0]).Append(xdl[1]).ToArray();
            if (!Tensor<double>.Broadcast(x, bdx, out var bx))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }
            var bdy = bd.Append(ydl[0]).Append(ydl[1]).ToArray();
            if (!Tensor<double>.Broadcast(y, bdy, out var by))
            {
                throw new ArgumentException("The tensor shapes are not compatible for broadcasting.");
            }

            StartOpStage(OpStage.Math);
            var z = DenseTensor<double>.OfShape(bd.Append(xdl[0]).Append(ydl[1]).ToArray());
            var di = bx.GetDimensionsIterator(0..^2);
            using var xh = bx.Storage.Pin();
            using var yh = by.Storage.Pin();
            using var zh = z.Storage.Pin();
            var m = bx.Dimensions[^2];
            var n = bx.Dimensions[^1];
            var k = by.Dimensions[^1];
            unsafe
            {
                var xp = (double*)xh.Pointer;
                var yp = (double*)yh.Pointer;
                var zp = (double*)zh.Pointer;
                foreach (var idx in di)
                {
                    if (HardwareConfig.UseSimd && HardwareConfig.UseIntrinsics && Fma.IsSupported)
                    {
                        mm_unsafe_vectorized_intrinsics(m, n, k, xp + bx.GetStorageIndex(idx), yp + by.GetStorageIndex(idx), zp + z.GetStorageIndex(idx));
                        
                    }
                    else if (HardwareConfig.UseSimd)
                    {

                        mm_unsafe_vectorized(m, n, k, xp + bx.GetStorageIndex(idx), yp + by.GetStorageIndex(idx), zp + z.GetStorageIndex(idx));

                    }
                    else
                    {
                        mm(m, n, k, xp + bx.GetStorageIndex(idx), yp + by.GetStorageIndex(idx), zp + z.GetStorageIndex(idx));
                    }
                }
            }
            return z;
        }
        else if (x.Rank >= 2 || y.Rank >= 2)
        {
            StartOpStage(OpStage.Broadcast);
            if (!Tensor<double>.Broadcast(x, y, out var bx, out var by))
            {
                throw new ArgumentException($"The shapes {x.PrintShape()} and {y.PrintShape()} are not compatible for broadcasting.");
            }
            else
            {
                return MatMul(bx, by);
            }
        }
        else //(x.Rank < 2 && y.Rank < 2)
        {
            StartOpStage(OpStage.Broadcast);
            bool bcast = false;
            if (x.Rank == 1)
            {
                x = x.PadLeft();
                bcast = true;
            }
            if (y.Rank == 1)
            {
                y = y.PadRight();
                bcast = true;
            }
            var c = MatMul2D(x, y);
            if (bcast)
            {
                c.RemoveDim(0);
            }
            return c;
        }
    }

    public static Tensor<float> Conv2D(Tensor<float> input, Tensor<float> weight, int group, PadType padtype = PadType.Valid, int? padvalue = null, Tensor<float> bias = null, int[] kernelshape = null, int[] strides = null, int[] dilations = null)
    {
        if (input.Rank != 4)
        {
            throw new ArgumentException(nameof(input), "Input tensors must be of rank 4 with the layout NxCxHxW.");
        }
        if (weight.Rank != 4)
        {
            throw new ArgumentException(nameof(weight), "Weight tensors must be of rank 4 with the layout M x C/group x kH x kW.");
        }
        if (strides == null)
        {
            strides = new int[2] { 1, 1 };
        }
        if (dilations == null)
        {
            dilations = new int[2] { 1, 1 };
        }
        var N = input.Dimensions[0];
        var C = input.Dimensions[1];
        var H = input.Dimensions[2];
        var W = input.Dimensions[3];
        var M = weight.Dimensions[0];
        var kH = kernelshape == null ? weight.Dimensions[2] : kernelshape[0];
        var kW = kernelshape == null ? weight.Dimensions[3] : kernelshape[1];
        var info = GetConv2DOutputInfo(padtype, H, W, strides[0], strides[1], GetConv2DEffectiveFilterSize(kH, dilations[0]), GetConv2DEffectiveFilterSize(kW, dilations[1]), padvalue);
        var output = new DenseTensor<float>((ReadOnlySpan<int>)new int[] { N, M, info.Shape[0], info.Shape[1] });

        unsafe
        {
            if (bias != null)
            {
                fixed (
                    float* inputp = input.ToDenseTensor().Buffer.Span,
                    outputp = output.Buffer.Span,
                    weightp = weight.ToDenseTensor().Buffer.Span,
                    biasp = bias.ToDenseTensor().Buffer.Span
                    )
                {
                    MathOps.Conv2D(inputp, N, C, H, W, kH, kW, dilations[0], dilations[1], strides[0], strides[1], info.PadInfo.left, info.PadInfo.top, info.PadInfo.right, info.PadInfo.bottom, group, weightp, outputp, M, biasp);
                }
            }
            else
            {
                {
                    fixed (
                        float* inputp = input.ToDenseTensor().Buffer.Span,
                        outputp = output.Buffer.Span,
                        weightp = weight.ToDenseTensor().Buffer.Span
                        )
                    {
                        MathOps.Conv2D(inputp, N, C, H, W, kH, kW, dilations[0], dilations[1], strides[0], strides[1], info.PadInfo.left, info.PadInfo.top, info.PadInfo.right, info.PadInfo.bottom, group, weightp, outputp, M);
                    }
                }
            }
        }

        return output;

    }

    public static Tensor<double> Conv2D(Tensor<double> input, Tensor<double> weight, int group, PadType padtype = PadType.Valid, int? padvalue = null, Tensor<double> bias = null, int[] kernelshape = null, int[] strides = null, int[] dilations = null)
    {
        if (input.Rank != 4)
        {
            throw new ArgumentException(nameof(input), "Input tensors must be of rank 4 with the layout NxCxHxW.");
        }
        if (weight.Rank != 4)
        {
            throw new ArgumentException(nameof(weight), "Weight tensors must be of rank 4 with the layout M x C/group x kH x kW.");
        }
        if (strides == null)
        {
            strides = new int[2] { 1, 1 };
        }
        if (dilations == null)
        {
            dilations = new int[2] { 1, 1 };
        }
        var N = input.Dimensions[0];
        var C = input.Dimensions[1];
        var H = input.Dimensions[2];
        var W = input.Dimensions[3];
        var M = weight.Dimensions[0];
        var kH = kernelshape == null ? weight.Dimensions[2] : kernelshape[0];
        var kW = kernelshape == null ? weight.Dimensions[3] : kernelshape[1];
        var info = GetConv2DOutputInfo(padtype, H, W, strides[0], strides[1], GetConv2DEffectiveFilterSize(kH, dilations[0]), GetConv2DEffectiveFilterSize(kW, dilations[1]), padvalue);
        var output = new DenseTensor<double>((ReadOnlySpan<int>)new int[] { N, M, info.Shape[0], info.Shape[1] });

        unsafe
        {
            if (bias != null)
            {
                fixed (
                    double* inputp = input.ToDenseTensor().Buffer.Span,
                    outputp = output.Buffer.Span,
                    weightp = weight.ToDenseTensor().Buffer.Span,
                    biasp = bias.ToDenseTensor().Buffer.Span
                    )
                {
                    MathOps.Conv2D(inputp, N, C, H, W, kH, kW, dilations[0], dilations[1], strides[0], strides[1], info.PadInfo.left, info.PadInfo.top, info.PadInfo.right, info.PadInfo.bottom, group, weightp, biasp, outputp, M);
                }
            }
            else
            {
                {
                    fixed (
                        double* inputp = input.ToDenseTensor().Buffer.Span,
                        outputp = output.Buffer.Span,
                        weightp = weight.ToDenseTensor().Buffer.Span
                        )
                    {
                        MathOps.Conv2D(inputp, N, C, H, W, kH, kW, dilations[0], dilations[1], strides[0], strides[1], info.PadInfo.left, info.PadInfo.top, info.PadInfo.right, info.PadInfo.bottom, group, weightp, null, outputp, M);
                    }
                }
            }
        }

        return output;

    }

    public static Tensor<float> MaxPool2D(Tensor<float> input, int[] kernelshape, PadType padtype = PadType.Valid, int? padvalue = null, int[] strides = null, int[] dilations = null)
    {
        if (kernelshape is null)
        {
            throw new ArgumentNullException("kernelshape");
        }
        if (input.Rank != 4)
        {
            throw new ArgumentException("Input tensors must be of rank 4 with the layout NxCxHxW.");
        }
        if (kernelshape.Rank != 1 || kernelshape.Length != 2)
        {
            throw new ArgumentException("The kernel must have shape m x n.");
        }

        if (strides == null)
        {
            strides = kernelshape;
        }
        if (dilations == null)
        {
            dilations = new int[] { 1, 1 };
        }

        var N = input.Dimensions[0];
        var C = input.Dimensions[1];
        var H = input.Dimensions[2];
        var W = input.Dimensions[3];
        var kH = kernelshape[0];
        var kW = kernelshape[1];
        var strideHeight = strides[0];
        var strideWidth = strides[1];
        var info = GetConv2DOutputInfo(padtype, H, W, strides[0], strides[1], GetConv2DEffectiveFilterSize(kH, dilations[0]), GetConv2DEffectiveFilterSize(kW, dilations[1]), padvalue);
        var Y = DenseTensor<float>.OfShape(N, C, info.Shape[0], info.Shape[1]);

        for (var n = 0; n < N; ++n)
        {
            for (var d = 0; d < C; ++d)
            {
                for (var yR = 0; yR < info.Shape[0]; ++yR)
                {
                    var xRCorner = yR * strideHeight - info.PadInfo.top;
                    var xRMin = Math.Max(0, xRCorner);
                    var xRMax = Math.Min(H, kH + xRCorner);
                    for (var yC = 0; yC < info.Shape[1]; ++yC)
                    {
                        var xCCorner = yC * strideWidth - info.PadInfo.left;
                        var xCMin = Math.Max(0, xCCorner);
                        var xCMax = Math.Min(W, kW + xCCorner);

                        var maxValue = float.NegativeInfinity;

                        for (var xR = xRMin; xR < xRMax; ++xR)
                        {
                            for (var xC = xCMin; xC < xCMax; ++xC)
                            {
                                var v = input[n, d, xR, xC];

                                if (v > maxValue)
                                {
                                    maxValue = v;
                                }
                            }
                            if (maxValue == float.NegativeInfinity)
                            {
                                break;
                            }
                        }
                        Y[n, d, yR, yC] = maxValue;
                    }
                }
            }
        }
        return Y;
    }

    public static Tensor<double> MaxPool2D(Tensor<double> input, int[] kernelshape, PadType padtype = PadType.Valid, int? padvalue = null, int[] strides = null, int[] dilations = null)
    {
        if (kernelshape is null)
        {
            throw new ArgumentNullException("kernelshape");
        }
        if (input.Rank != 4)
        {
            throw new ArgumentException("Input tensors must be of rank 4 with the layout NxCxHxW.");
        }
        if (kernelshape.Rank != 1 || kernelshape.Length != 2)
        {
            throw new ArgumentException("The kernel must have shape m x n.");
        }

        if (strides is null)
        {
            strides = kernelshape;
        }
        if (dilations is null)
        {
            dilations = new int[] { 1, 1 };
        }

        var N = input.Dimensions[0];
        var C = input.Dimensions[1];
        var H = input.Dimensions[2];
        var W = input.Dimensions[3];
        var kH = kernelshape[0];
        var kW = kernelshape[1];
        var strideHeight = strides[0];
        var strideWidth = strides[1];
        var info = GetConv2DOutputInfo(padtype, H, W, strides[0], strides[1], GetConv2DEffectiveFilterSize(kH, dilations[0]), GetConv2DEffectiveFilterSize(kW, dilations[1]), padvalue);
        var Y = DenseTensor<double>.OfShape(N, C, info.Shape[0], info.Shape[1]);

        for (var n = 0; n < N; ++n)
        {
            for (var d = 0; d < C; ++d)
            {
                for (var yR = 0; yR < info.Shape[0]; ++yR)
                {
                    var xRCorner = yR * strideHeight - info.PadInfo.top;
                    var xRMin = Math.Max(0, xRCorner);
                    var xRMax = Math.Min(H, kH + xRCorner);
                    for (var yC = 0; yC < info.Shape[1]; ++yC)
                    {
                        var xCCorner = yC * strideWidth - info.PadInfo.left;
                        var xCMin = Math.Max(0, xCCorner);
                        var xCMax = Math.Min(W, kW + xCCorner);

                        var maxValue = double.NegativeInfinity;

                        for (var xR = xRMin; xR < xRMax; ++xR)
                        {
                            for (var xC = xCMin; xC < xCMax; ++xC)
                            {
                                var v = input[n, d, xR, xC];

                                if (v > maxValue)
                                {
                                    maxValue = v;
                                }
                            }
                            if (maxValue == double.NegativeInfinity)
                            {
                                break;
                            }
                        }
                        Y[n, d, yR, yC] = maxValue;
                    }
                }
            }
        }
        return Y;
    }

    public static Tensor<int> MaxPool2D(Tensor<int> input, int[] kernelshape, PadType padtype = PadType.Valid, int? padvalue = null, int[] strides = null, int[] dilations = null)
    {
        if (kernelshape == null)
        {
            throw new ArgumentNullException("kernelshape");
        }
        if (input.Rank != 4)
        {
            throw new ArgumentException("Input tensors must be of rank 4 with the layout NxCxHxW.");
        }
        if (kernelshape.Rank != 1 || kernelshape.Length != 2)
        {
            throw new ArgumentException("The kernel must have shape m x n.");
        }

        if (strides == null)
        {
            strides = kernelshape;
        }
        if (dilations == null)
        {
            dilations = new int[] { 1, 1 };
        }

        var N = input.Dimensions[0];
        var C = input.Dimensions[1];
        var H = input.Dimensions[2];
        var W = input.Dimensions[3];
        var kH = kernelshape[0];
        var kW = kernelshape[1];
        var strideHeight = strides[0];
        var strideWidth = strides[1];
        var info = GetConv2DOutputInfo(padtype, H, W, strides[0], strides[1], GetConv2DEffectiveFilterSize(kH, dilations[0]), GetConv2DEffectiveFilterSize(kW, dilations[1]), padvalue);
        var Y = DenseTensor<int>.OfShape(N, C, info.Shape[0], info.Shape[1]);

        for (var n = 0; n < N; ++n)
        {
            for (var d = 0; d < C; ++d)
            {
                for (var yR = 0; yR < info.Shape[0]; ++yR)
                {
                    var xRCorner = yR * strideHeight - info.PadInfo.top;
                    var xRMin = Math.Max(0, xRCorner);
                    var xRMax = Math.Min(H, kH + xRCorner);
                    for (var yC = 0; yC < info.Shape[1]; ++yC)
                    {
                        var xCCorner = yC * strideWidth - info.PadInfo.left;
                        var xCMin = Math.Max(0, xCCorner);
                        var xCMax = Math.Min(W, kW + xCCorner);

                        var maxValue = 0;

                        for (var xR = xRMin; xR < xRMax; ++xR)
                        {
                            for (var xC = xCMin; xC < xCMax; ++xC)
                            {
                                var v = input[n, d, xR, xC];

                                if (v > maxValue)
                                {
                                    maxValue = v;
                                }
                            }
                            if (maxValue == 0)
                            {
                                break;
                            }
                        }
                        Y[n, d, yR, yC] = maxValue;
                    }
                }
            }
        }
        return Y;
    }
    public static Tensor<float> Relu(Tensor<float> x) => x.Apply(l => l > 0.0f ? l : 0.0f);

    public static Tensor<double> Relu(Tensor<double> x) => x.Apply(l => l > 0.0 ? l : 0.0);

    public static Tensor<T> Reshape(Tensor<T> input, Tensor<long> shape, bool allowZero = false)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (shape.Rank != 1)
        {
            throw new ArgumentException(nameof(shape), "Shape tensors must be of rank 1.");
        }
        if (shape.Any(v => v < -1))
        {
            throw new ArgumentException(nameof(shape), $"A shape dimension cannot be < -1, got {shape.First(v => v < -1)}.");
        }
        if (shape.Count(v => v == -1) > 1)
        {
            throw new ArgumentException(nameof(shape), $"At most 1 shape dimension can be -1.");
        }

        StartOpStage(OpStage.CalculateIndices);
        int unknownDim = -1;
        List<int> newShapeDims = new List<int>();
        int newSize = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] == -1)
            {
                unknownDim = i;
                newShapeDims.Add(-1);
            }
            else if (shape[i] == 0 && !allowZero)
            {
                newShapeDims.Add(input.Dimensions[i]);
                newSize *= input.Dimensions[i];
            }
            else if (shape[i] == 0 && allowZero)
            {
                newShapeDims.Add(0);
            }
            else
            {
                newShapeDims.Add(Convert.ToInt32(shape[i]));
                newSize *= Convert.ToInt32(shape[i]);
            }
        }
        if (unknownDim != -1)
        {
            newShapeDims[unknownDim] = Convert.ToInt32(input.Length / newSize);
            newSize *= newShapeDims[unknownDim];
        }

        if (newSize != input.Length)
        {
            throw new ArgumentException(nameof(shape), $"The input tensor cannot be reshaped to the requested shape. Input shape:{input.PrintShape()}, requested shape:{newShapeDims.Print()}");
        }

        return input.Reshape(newShapeDims.ToArray());
    }

    public static Tensor<float> Softmax(Tensor<float> x, int axis = -1) 
    {
        StartOpStage(OpStage.ValidateArguments);
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(x.Rank, axis);
        if (axis >= x.Rank) throw new ArgumentException(nameof(axis), "The specified axis must be less than the rank of the tensor.");      
        var max = Tensor<float>.ReduceMax(x, (new int[] { axis }).ToTensor<int>(), true);
        if (!Tensor<float>.Broadcast(x, max, out var bx, out var bmax))
        {
            throw new InvalidOperationException("Could not broadcast result of Max op with original tensor.");
        }
        var sub = Tensor<float>.Subtract(x, bmax);
        var t = sub.Apply(MathF.Exp);
        var s = Tensor<float>.ReduceSum(t, (new int[] { axis }).ToTensor<int>(), true);
        if (!Tensor<float>.Broadcast(t, s, out var bt, out var bs))
        {
            throw new InvalidOperationException("Could not broadcast results of ReduceSum and Exp ops.");
        }
        return Tensor<float>.Divide(bt, bs);
    }

    public static Tensor<double> Softmax(Tensor<double> x, int axis = -1)
    {
        StartOpStage(OpStage.ValidateArguments);
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(x.Rank, axis);
        if (axis >= x.Rank) throw new ArgumentException(nameof(axis), "The specified axis must be less than the rank of the tensor.");

        var max = Tensor<double>.ReduceMax(x, (new int[] { axis }).ToTensor<int>(), true);
        if (!Tensor<double>.Broadcast(x, max, out var bx, out var bmax))
        {
            throw new InvalidOperationException("Could not broadcast result of Max op with original tensor.");
        }
        var sub = Tensor<double>.Subtract(x, bmax);
        var t = sub.Apply(Math.Exp);
        var s = Tensor<double>.ReduceSum(t, (new int[] { axis }).ToTensor<int>(), true);
        if (!Tensor<double>.Broadcast(t, s, out var bt, out var bs))
        {
            throw new InvalidOperationException("Could not broadcast results of ReduceSum and Exp ops.");
        }
        return Tensor<double>.Divide(bt, bs);
    }

    public static Tensor<float> Erf(Tensor<float> x) => x.Apply(MathOps.Erf);

    public static Tensor<double> Erf(Tensor<double> x) => x.Apply(MathOps.Erf);

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<T> Transpose(Tensor<T> data, int[] perm = null)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (perm is not null)
        {
            if (perm.Length != data.Rank)
            {
                throw new ArgumentException(nameof(perm), $"The size of the permutation array must be the rank of the tensor: {data.Rank}.");
            }
            if (!perm.All(p => p < data.Rank))
            {
                throw new ArgumentException(nameof(perm), $"The permuted dimension {perm.First(p => p >= data.Rank)} exceeds the number of dimensions in the tensor.");
            }
            if (!ArrayUtilities.CheckNoRepeatedDims(perm))
            {
                throw new ArgumentException(nameof(perm), "The permutation array has a repeated dimension.");
            }
            for (int i = 0; i < perm.Length; i++)
            {
                perm[i] = ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, perm[i]);
            }
        }
        else
        {
            perm = Enumerable.Range(0, data.Rank).Reverse().ToArray();
        }
        if (data.Rank <= 1)
        {
            return data;
        }

        StartOpStage(OpStage.Copy);
        var shape = new int[data.Rank];
        for (int i =0; i < perm.Length; i++)
        {
            shape[i] = data.dimensions[perm[i]];
        }
        var r = DenseTensor<T>.OfShape(shape);
        var di = data.GetDimensionsIterator();
        foreach (var index in di)
        {
            int permindex = 0;
            for (int i = 0; i < perm.Length; i++)
            {
                permindex += index[perm[i]] * r.strides[i];
            }
            r.SetValue(permindex, data[index]);
        }
        return r;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]  
    public unsafe static Tensor<T> Gather(Tensor<T> data, Tensor<int> indices, int? _axis = null)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (data.Rank == 0) throw new ArgumentException(nameof (data), "Cannot gather from a tensor of rank 0.");

        StartOpStage(OpStage.CalculateIndices);
        var axis = _axis.HasValue ? ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, _axis.Value) : 0;    
        if (axis > data.Rank - 1)
        {
            throw new ArgumentException(nameof(axis), $"The specified axis {_axis} exceeds the number of dimensions in the tensor.");
        }
        int* p = stackalloc int[data.Rank - 1 + indices.Rank];
        UnsafeFixedSizeList<int> shape = new UnsafeFixedSizeList<int>(p, data.Rank - 1 + indices.Rank);
        for (int i = 0; i < axis; i++)
        {
            shape.Add(data.dimensions[i]);
        }
        for (int i = 0; i < indices.Rank; i++)
        {
            shape.Add(indices.dimensions[i]);
        }
        for (int i = axis + 1; i < data.Rank; i++)
        {
            shape.Add(data.dimensions[i]);
        }
        var output = DenseTensor<T>.OfShape(shape.ToArray());
        StartOpStage(OpStage.Copy);
        foreach (var di in output.GetDimensionsIterator())
        {
            var a = di[0..axis];
            var k = ArrayUtilities.HandleNegativeAxisOrIndex(data.dimensions[axis], indices[di[axis..(axis + indices.Rank)]]);
            var b = di[(axis + (indices.Rank == 0 ? 1 : indices.Rank))..];
            var oloc = a.Append(k).Concat(b).ToArray();
            output[di] = data[oloc];
        }
        return output;  
    }

    static bool HasStandardStrides<T>(DenseTensor<T> tensor) where T : unmanaged
    {
        if (tensor.IsReversedStride) return false;
        return tensor.strides.SequenceEqual(ArrayUtilities.GetStrides(tensor.dimensions));
    }

    public static Tensor<T> Concat(Tensor<T> x, Tensor<T> y, int axis)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (x.Rank != y.Rank) throw new ArgumentException(nameof(y), "The rank of each tensor in a concat operation must be the same.");
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(x.Rank, axis);
        for (int i = 0; i < x.Rank; i++)
        {
            if (i == axis) continue;
            if (x.dimensions[i] != y.dimensions[i])
            {
                throw new ArgumentException(nameof(y), "The dimensions of each tensor in a concat operation must be the same, with the exception of the axis dimension.");
            }
        }
        var shape = x.dimensions.Copy();
        shape[axis] += y.dimensions[axis];

        StartOpStage(OpStage.Copy);
        var output = DenseTensor<T>.OfShape(shape);
        var di = output.GetDimensionsIterator();    
        foreach (var index in di)
        {
            if (index[axis] < x.dimensions[axis])
            {
                output[index] = x[index];
            }
            else
            {
                var loc = index.Copy();
                loc[axis] -= x.dimensions[axis];
                output[index] = y[loc];
            }
        }
        return output;
    }
    public static Tensor<T> Concat(Tensor<T>[] inputs, int axis)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (inputs.Length < 2) throw new ArgumentException(nameof(inputs), "At least two tensors must be specified for the concat operation.");
        if (!inputs.All(i => i.Rank == inputs[0].Rank)) throw new ArgumentException(nameof(inputs), $"Each input tensor in a concat operation must be of the same rank.");
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(inputs[0].Rank, axis);
        if (!inputs.All(i => i.dimensions.Select((d, n) => n == axis ? 0 : d - inputs[0].dimensions[n]).All(s => s == 0)))
            throw new ArgumentException(nameof(inputs), "The dimensions of each tensor in a concat operation must be the same, with the exception of the axis dimension.");
        var shape = inputs[0].dimensions.Copy();
        shape[axis] = 0;
        foreach (var input in inputs) shape[axis] += input.dimensions[axis];

        StartOpStage(OpStage.Copy);
        var output = DenseTensor<T>.OfShape(shape);
        int inner = 1;
        for (int trailing = axis + 1; trailing < output.Rank; trailing++) inner *= shape[trailing];
        int outer = 1;
        for (int leading = 0; leading < axis; leading++) outer *= shape[leading];
        if (inputs.All(i => i is DenseTensor<T> dense && HasStandardStrides(dense)))
        {
            var outputSpan = output.Buffer.Span;
            int destinationAxisOffset = 0;
            foreach (var input in inputs)
            {
                var sourceSpan = ((DenseTensor<T>)input).Buffer.Span;
                int axisLength = input.dimensions[axis];
                for (int outerIndex = 0; outerIndex < outer; outerIndex++)
                {
                    sourceSpan.Slice(outerIndex * axisLength * inner, axisLength * inner).CopyTo(outputSpan.Slice((outerIndex * shape[axis] + destinationAxisOffset) * inner, axisLength * inner));
                }
                destinationAxisOffset += axisLength;
            }
            return output;
        }
        var axisOffsets = new int[inputs.Length];
        int runningAxisOffset = 0;
        for (int inputIndex = 0; inputIndex < inputs.Length; inputIndex++)
        {
            axisOffsets[inputIndex] = runningAxisOffset;
            runningAxisOffset += inputs[inputIndex].dimensions[axis];
        }
        var iterator = output.GetDimensionsIterator();
        foreach (var index in iterator)
        {
            int source = 0;
            while (source + 1 < inputs.Length && index[axis] >= axisOffsets[source + 1]) source++;
            var location = index.Copy();
            location[axis] -= axisOffsets[source];
            output[index] = inputs[source][location];
        }
        return output;
    }

    public static Tensor<T> Slice(Tensor<T> data, Tensor<int> start, Tensor<int> ends, Tensor<int> axes = null, Tensor<int> steps = null)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (data.Rank == 0) throw new ArgumentException(nameof(data), "Cannot slice a tensor of rank 0.");
        if (start.Rank != 1) throw new ArgumentException(nameof(start), "The rank of the start tensor must be 1.");
        if (start.Length > data.Rank) throw new ArgumentException(nameof(start), "The length of the start tensor must be less-than or equal to the rank of the data tensor.");
        if (ends.Rank != 1) throw new ArgumentException(nameof(start), "The rank of the end tensor must be 1");
        if (start.Length != ends.Length) throw new ArgumentException(nameof(ends), "The end tensor must be the same length as the start tensor.");
        if (axes is not null && (axes.Rank != 1 || axes.Length != start.Length)) throw new ArgumentException(nameof(axes), "The axes tensor must be a rank 1 tensor with the same length as the start tensor.");
        if (steps is not null && (steps.Rank != 1 || steps.Length != start.Length)) throw new ArgumentException(nameof(steps), "The steps tensor must be a rank 1 tensor with the same length as the start tensor.");
        
        StartOpStage(OpStage.CalculateIndices);
        int length = Convert.ToInt32(start.Length);
        if (axes is null)
        {
            axes = Enumerable.Range(0, length).ToArray().ToTensor<int>();
        }
        else
        {
            axes = axes.Select(a => ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, a)).ToArray().ToTensor<int>();
        }
        if (steps is null)
        {
            steps = Tensor<int>.Ones(length);
        }
       
        start = start.Select((s, i) => ArrayUtilities.Clamp(ArrayUtilities.HandleNegativeAxisOrIndex(data.Dimensions[axes[i]], s), 0, data.Dimensions[axes[i]])).ToArray().ToTensor<int>();
        ends = ends.Select((s, i) => ArrayUtilities.Clamp(ArrayUtilities.HandleNegativeAxisOrIndex(data.Dimensions[axes[i]], s), 0, data.Dimensions[axes[i]])).ToArray().ToTensor<int>();

        SliceIndex[] indices = new SliceIndex[data.Rank];
        for (int i = 0; i < data.Rank; i++) 
        {
            indices[i] = axes.Contains(i) ? new SliceIndex(start[axes.IndexOf(i)], ends[axes.IndexOf(i)], steps[axes.IndexOf(i)]) : new SliceIndex(0, data.dimensions[i]);
        }
        return data.Slice(indices); 
    }

    public static Tensor<T> Unsqueeze(Tensor<T> data, int[] axes)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (!ArrayUtilities.CheckNoRepeatedDims(axes)) throw new ArgumentException(nameof(axes), "axes contains a repeated dimension.");
        
        StartOpStage(OpStage.CalculateIndices);
        for (int i = 0; i < axes.Length; i++)
        {
            axes[i] = ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, axes[i]);
        }
        if (!Array.TrueForAll(axes, (a => a <= (data.Rank + axes.Length) - 1))) throw new ArgumentException(nameof(axes), $"Each specified axis must be less than the rank of the output tensor. Got {axes.First(a => a > data.Rank - 1)}");
        var newshape = new int[axes.Length + data.Rank];
        for (int i = 0; i < axes.Length; i++)
        {
            newshape[axes[i]] = 1;
        }
        var e = data.Dimensions.GetEnumerator();
        for (int i = 0; i < newshape.Length; i++)
        {
            if (newshape[i] == 0)
            {
                if (!e.MoveNext()) throw new InvalidOperationException("Out of dimensions.");
                newshape[i] = e.Current;
            }
        }
        return data.Reshape(newshape);
    }
    public static Tensor<int> ReduceSum(Tensor<int> data, Tensor<int> axes = null, bool? _keepDims = null, bool? _noOpWithEmptyAxes = null)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (axes is not null && axes.Length > data.Rank) throw new ArgumentException(nameof(axes), "The number of axes specified must be less than the tensor rank.");
        if (axes is not null && !axes.All(a =>  a < data.Rank)) throw new ArgumentException(nameof(axes), $"Each axis specified must be less than the rank of the tensor.");
        if (axes is not null && !ArrayUtilities.CheckNoRepeatedDims(axes.ToArray())) throw new ArgumentException(nameof(axes), "axes contains a repeated dimension.");

        StartOpStage(OpStage.CalculateIndices);
        var keepDims = _keepDims.HasValue ? _keepDims.Value : false;
        var noOpWithEmptyAxes = _noOpWithEmptyAxes.HasValue ? _noOpWithEmptyAxes.Value : false;
        var _axes = axes is null || axes.Length == 0 ? Enumerable.Range(0, data.Rank).ToArray() : axes.Select(a => ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, a)).ToArray();
        var permutation = ArrayUtilities.GetAxesPermutationForReduction(_axes, data.Rank);
        Tensor<int> pdata;
        int[] paxes;
        if (permutation is not null)
        {
            pdata = Tensor<int>.Transpose(data, permutation);
            paxes = ArrayUtilities.GetInnerMostAxes(_axes.Length, data.Rank);
        }
        else
        {
            pdata = data;
            paxes = _axes;
        }
        var (oshape, rshape) = ArrayUtilities.ComputeShapesForReduction(pdata.dimensions, paxes);
        var output = DenseTensor<int>.OfShape(oshape);
        var r = ArrayUtilities.ComputeOffsetForReduction(rshape);

        StartOpStage(OpStage.Math);
        for (var i = 0; i < output.Length; ++i)
        {
            var offset = i * r;
            var sum = 0;
            for (int j = 0; j < r; ++j)
            {
                sum += pdata.GetValue(offset + j);
            }
            output.SetValue(i, sum);
        }
        if (keepDims)
        {
            return Tensor<int>.Unsqueeze(output, _axes);
        }
        else
        {
            return output;
        }
    }

    public static Tensor<float> ReduceSum(Tensor<float> data, Tensor<int> axes = null, bool? _keepDims = null, bool? _noOpWithEmptyAxes = null)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (axes is not null && axes.Length > data.Rank) throw new ArgumentException(nameof(axes), "The number of axes specified must be less than the tensor rank.");
        if (axes is not null && !axes.All(a => a < data.Rank)) throw new ArgumentException(nameof(axes), $"Each axis specified must be less than the rank of the tensor.");
        if (axes is not null && !ArrayUtilities.CheckNoRepeatedDims(axes.ToArray())) throw new ArgumentException(nameof(axes), "axes contains a repeated dimension.");

        StartOpStage(OpStage.ValidateArguments);
        var keepDims = _keepDims.HasValue ? _keepDims.Value : false;
        var noOpWithEmptyAxes = _noOpWithEmptyAxes.HasValue ? _noOpWithEmptyAxes.Value : false;
        var _axes = axes is null || axes.Length == 0 ? Enumerable.Range(0, data.Rank).ToArray() : axes.Select(a => ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, a)).ToArray();
        var permutation = ArrayUtilities.GetAxesPermutationForReduction(_axes, data.Rank);
        Tensor<float> pdata;
        int[] paxes;
        if (permutation is not null)
        {
            pdata = Tensor<float>.Transpose(data, permutation);
            paxes = ArrayUtilities.GetInnerMostAxes(_axes.Length, data.Rank);
        }
        else
        {
            pdata = data;
            paxes = _axes;
        }
        var (oshape, rshape) = ArrayUtilities.ComputeShapesForReduction(pdata.dimensions, paxes);
        var output = DenseTensor<float>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var r = ArrayUtilities.ComputeOffsetForReduction(rshape);
        for (var i = 0; i < output.Length; ++i)
        {
            var offset = i * r;
            var sum = 0.0f;
            for (int j = 0; j < r; ++j)
            {
                sum += pdata.GetValue(offset + j);
            }
            output.SetValue(i, sum);
        }
        if (keepDims)
        {
            return Tensor<float>.Unsqueeze(output, _axes);
        }
        else
        {
            return output;
        }
    }

    public static Tensor<double> ReduceSum(Tensor<double> data, Tensor<int> axes = null, bool? _keepDims = null, bool? _noOpWithEmptyAxes = null)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (axes is not null && axes.Length > data.Rank) throw new ArgumentException(nameof(axes), "The number of axes specified must be less than the tensor rank.");
        if (axes is not null && !axes.All(a => a < data.Rank)) throw new ArgumentException(nameof(axes), $"Each axis specified must be less than the rank of the tensor.");
        if (axes is not null && !ArrayUtilities.CheckNoRepeatedDims(axes.ToArray())) throw new ArgumentException(nameof(axes), "axes contains a repeated dimension.");

        StartOpStage(OpStage.CalculateIndices);
        var keepDims = _keepDims.HasValue ? _keepDims.Value : false;
        var noOpWithEmptyAxes = _noOpWithEmptyAxes.HasValue ? _noOpWithEmptyAxes.Value : false;
        var _axes = axes is null || axes.Length == 0 ? Enumerable.Range(0, data.Rank).ToArray() : axes.Select(a => ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, a)).ToArray();
        var permutation = ArrayUtilities.GetAxesPermutationForReduction(_axes, data.Rank);
        Tensor<double> pdata;
        int[] paxes;
        if (permutation is not null)
        {
            pdata = Tensor<double>.Transpose(data, permutation);
            paxes = ArrayUtilities.GetInnerMostAxes(_axes.Length, data.Rank);
        }
        else
        {
            pdata = data;
            paxes = _axes;
        }

        var (oshape, rshape) = ArrayUtilities.ComputeShapesForReduction(pdata.dimensions, paxes);
        var output = DenseTensor<double>.OfShape(oshape);

        StartOpStage(OpStage.Math);
        var r = ArrayUtilities.ComputeOffsetForReduction(rshape);
        for (var i = 0; i < output.Length; ++i)
        {
            var offset = i * r;
            var sum = 0.0;
            for (int j = 0; j < r; ++j)
            {
                sum += pdata.GetValue(offset + j);
            }
            output.SetValue(i, sum);
        }
        if (keepDims)
        {
            return Tensor<double>.Unsqueeze(output, _axes);
        }
        else
        {
            return output;
        }
    }

    public static Tensor<int> ReduceMean(Tensor<int> data, Tensor<int> axes = null, bool? _keepDims = null, bool? _noOpWithEmptyAxes = null)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (axes is not null && axes.Length > data.Rank) throw new ArgumentException(nameof(axes), "The number of axes specified must be less than the tensor rank.");
        if (axes is not null && !axes.All(a => a < data.Rank)) throw new ArgumentException(nameof(axes), $"Each axis specified must be less than the rank of the tensor.");
        if (axes is not null && !ArrayUtilities.CheckNoRepeatedDims(axes.ToArray())) throw new ArgumentException(nameof(axes), "axes contains a repeated dimension.");

        StartOpStage(OpStage.CalculateIndices);
        var keepDims = _keepDims.HasValue ? _keepDims.Value : false;
        var noOpWithEmptyAxes = _noOpWithEmptyAxes.HasValue ? _noOpWithEmptyAxes.Value : false;
        var _axes = axes is null || axes.Length == 0 ? Enumerable.Range(0, data.Rank).ToArray() : axes.Select(a => ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, a)).ToArray();
        var (oshape, rshape) = ArrayUtilities.ComputeShapesForReduction(data.dimensions, _axes);
        var r = ArrayUtilities.ComputeOffsetForReduction(rshape);
        //Tensor<int> output = DenseTensor<int>.OfShape(oshape);
        Tensor<int> output = Tensor<int>.Divide(data, r);
        output = Tensor<int>.ReduceSum(output, _axes.ToTensor<int>());
        if (keepDims)
        {
            return Tensor<int>.Unsqueeze(output, _axes);
        }
        else
        {
            return output;
        }
    }

    public static Tensor<float> ReduceMean(Tensor<float> data, Tensor<int> axes = null, bool? _keepDims = null, bool? _noOpWithEmptyAxes = null)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (axes is not null && axes.Length > data.Rank) throw new ArgumentException(nameof(axes), "The number of axes specified must be less than the tensor rank.");
        if (axes is not null && !axes.All(a => a < data.Rank)) throw new ArgumentException(nameof(axes), $"Each axis specified must be less than the rank of the tensor.");
        if (axes is not null && !ArrayUtilities.CheckNoRepeatedDims(axes.ToArray())) throw new ArgumentException(nameof(axes), "axes contains a repeated dimension.");

        StartOpStage(OpStage.CalculateIndices);
        var keepDims = _keepDims.HasValue ? _keepDims.Value : false;
        var noOpWithEmptyAxes = _noOpWithEmptyAxes.HasValue ? _noOpWithEmptyAxes.Value : false;
        var _axes = axes is null || axes.Length == 0 ? Enumerable.Range(0, data.Rank).ToArray() : axes.Select(a => ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, a)).ToArray();
        var rshape = ArrayUtilities.ComputeReducedShape(data.dimensions, _axes);
        var r = Convert.ToSingle(ArrayUtilities.ComputeOffsetForReduction(rshape)); 
        
        StartOpStage(OpStage.Math);
        Tensor<float> output = Tensor<float>.ReduceSum(data, _axes.ToTensor<int>());
        output = Tensor<float>.Divide(output, r);
        if (keepDims)
        {
            return Tensor<float>.Unsqueeze(output, _axes);
        }
        else
        {
            return output;
        }
    }

    public static Tensor<double> ReduceMean(Tensor<double> data, Tensor<int> axes = null, bool? _keepDims = null, bool? _noOpWithEmptyAxes = null)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (axes is not null && axes.Length > data.Rank) throw new ArgumentException(nameof(axes), "The number of axes specified must be less than the tensor rank.");
        if (axes is not null && !axes.All(a => a < data.Rank)) throw new ArgumentException(nameof(axes), $"Each axis specified must be less than the rank of the tensor.");
        if (axes is not null && !ArrayUtilities.CheckNoRepeatedDims(axes.ToArray())) throw new ArgumentException(nameof(axes), "axes contains a repeated dimension.");

        StartOpStage(OpStage.CalculateIndices);
        var keepDims = _keepDims.HasValue ? _keepDims.Value : false;
        var noOpWithEmptyAxes = _noOpWithEmptyAxes.HasValue ? _noOpWithEmptyAxes.Value : false;
        var _axes = axes is null || axes.Length == 0 ? Enumerable.Range(0, data.Rank).ToArray() : axes.Select(a => ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, a)).ToArray();
        var rshape = ArrayUtilities.ComputeReducedShape(data.dimensions, _axes);
        var r = Convert.ToDouble(ArrayUtilities.ComputeOffsetForReduction(rshape));
        
        StartOpStage(OpStage.Math);
        Tensor<double> output = Tensor<double>.Divide(data, r);
        output = Tensor<double>.ReduceSum(output, _axes.ToTensor<int>());
        if (keepDims)
        {
            return Tensor<double>.Unsqueeze(output, _axes);
        }
        else
        {
            return output;
        }
    }

    public static Tensor<float> ReduceMax(Tensor<float> data, Tensor<int> axes = null, bool? _keepDims = null)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (axes is not null && axes.Length > data.Rank) throw new ArgumentException(nameof(axes), "The number of axes specified must be less than the tensor rank.");
        if (axes is not null && !axes.All(a => a < data.Rank)) throw new ArgumentException(nameof(axes), $"Each axis specified must be less than the rank of the tensor.");
        if (axes is not null && !ArrayUtilities.CheckNoRepeatedDims(axes.ToArray())) throw new ArgumentException(nameof(axes), "axes contains a repeated dimension.");

        StartOpStage(OpStage.CalculateIndices);
        var _axes = axes is null || axes.Length == 0 ? Enumerable.Range(0, data.Rank).ToArray() : axes.Select(a => ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, a)).ToArray();
        var keepDims = _keepDims.HasValue ? _keepDims.Value : true;
        var permutation = ArrayUtilities.GetAxesPermutationForReduction(_axes, data.Rank);
        Tensor<float> pdata;
        int[] paxes;
        if (permutation is not null)
        {
            pdata = Tensor<float>.Transpose(data, permutation);
            paxes = ArrayUtilities.GetInnerMostAxes(_axes.Length, data.Rank);
        }
        else
        {
            pdata = data;
            paxes = _axes;
        }
        var (oshape, rshape) = ArrayUtilities.ComputeShapesForReduction(pdata.dimensions, paxes);
        var output = DenseTensor<float>.OfShape(oshape);
        var r = ArrayUtilities.ComputeOffsetForReduction(rshape);

        StartOpStage(OpStage.Math);
        for (var i = 0; i < output.Length; ++i)
        {
            var offset = i * r;
            var max = 0.0f;
            for (int j = 0; j < r; ++j)
            {
                var v = pdata.GetValue(offset + j);
                if (v > max)
                {
                    max = v;
                }
            }
            output.SetValue(i, max);
        }
        if (keepDims)
        {
            return Tensor<float>.Unsqueeze(output, _axes);
        }
        else
        {
            return output;
        }
    }

    public static Tensor<double> ReduceMax(Tensor<double> data, Tensor<int> axes = null, bool? _keepDims = null)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (axes is not null && axes.Length > data.Rank) throw new ArgumentException(nameof(axes), "The number of axes specified must be less than the tensor rank.");
        if (axes is not null && !axes.All(a => a < data.Rank)) throw new ArgumentException(nameof(axes), $"Each axis specified must be less than the rank of the tensor.");
        if (axes is not null && !ArrayUtilities.CheckNoRepeatedDims(axes.ToArray())) throw new ArgumentException(nameof(axes), "axes contains a repeated dimension.");

        StartOpStage(OpStage.CalculateIndices);
        var _axes = axes is null || axes.Length == 0 ? Enumerable.Range(0, data.Rank).ToArray() : axes.Select(a => ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, a)).ToArray();
        var keepDims = _keepDims.HasValue ? _keepDims.Value : true;
        var permutation = ArrayUtilities.GetAxesPermutationForReduction(_axes, data.Rank);
        Tensor<double> pdata;
        int[] paxes;
        if (permutation is not null)
        {
            pdata = Tensor<double>.Transpose(data, permutation);
            paxes = ArrayUtilities.GetInnerMostAxes(_axes.Length, data.Rank);
        }
        else
        {
            pdata = data;
            paxes = _axes;
        }
        var (oshape, rshape) = ArrayUtilities.ComputeShapesForReduction(pdata.dimensions, paxes);
        var output = DenseTensor<double>.OfShape(oshape);
        var r = ArrayUtilities.ComputeOffsetForReduction(rshape);

        StartOpStage(OpStage.Math);
        for (var i = 0; i < output.Length; ++i)
        {
            var offset = i * r;
            var max = 0.0;
            for (int j = 0; j < r; ++j)
            {
                var v = pdata.GetValue(offset + j);
                if (v > max)
                {
                    max = v;
                }
            }
            output.SetValue(i, max);
        }
        if (keepDims)
        {
            return Tensor<double>.Unsqueeze(output, _axes);
        }
        else
        {
            return output;
        }
    }
}
