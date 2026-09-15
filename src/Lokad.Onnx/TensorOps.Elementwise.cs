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
    /// Checks an elementwise destination: it must exist and hold exactly one
    /// slot per source element. The destination is fully overwritten; it may
    /// alias the sources because every write pairs flat index to flat index.
    /// </summary>
    void CheckDestination(Tensor<T> destination)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        if (destination.Length != Length) throw new ArgumentException("Destination length must match the source length.", nameof(destination));
    }

    void CheckBinaryDestination(Tensor<T> tensor2, Tensor<T> destination)
    {
        if (tensor2 is null) throw new ArgumentNullException(nameof(tensor2));
        if (tensor2.Length != Length) throw new ArgumentException("Second operand length must match the source length.", nameof(tensor2));
        CheckDestination(destination);
    }

    /// <summary>
    /// True when the whole logical contents are addressable as one flat span,
    /// which is what the vector fast paths require. Dense backing always matches
    /// the logical length by construction; anything else uses the index path.
    /// </summary>
    static bool IsDirectSpanCompatible(Tensor<T> tensor) =>
        tensor is DenseTensor<T> dense && dense.Buffer.Length == dense.Length;

    /// <summary>
    /// Writes op(element) for every element, overwriting the destination.
    /// The destination may alias this tensor; pairing is by flat index.
    /// </summary>
    public virtual void Apply(Func<T, T> op, Tensor<T> destination)
    {
        CheckDestination(destination);
  
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

    /// <summary>
    /// Writes op(element) for every element, overwriting the destination, with automatic
    /// execution options. The destination starts uninitialized and is fully overwritten; it
    /// may alias this tensor because pairing is by flat index. The vector path runs only
    /// on directly spannable storage, otherwise the scalar path runs. An empty tensor
    /// writes nothing but the destination must still match its length.
    /// </summary>
    /// <param name="op">Vector operation; must compute the lane-wise image of <paramref name="sop"/>.</param>
    /// <param name="sop">Scalar fallback operation; must be non-null.</param>
    /// <param name="destination">Overwrite-only destination holding exactly one slot per element.</param>
    /// <exception cref="ArgumentNullException"><paramref name="destination"/> is null.</exception>
    /// <exception cref="ArgumentException">The destination length differs from the source length.</exception>
    public virtual void VectorizedApply(Func<Vector<T>, Vector<T>> op, Func<T, T> sop, Tensor<T> destination)
        => VectorizedApply(op, sop, destination, TensorExecutionOptions.Auto);

    /// <summary>
    /// Writes op(element) for every element, overwriting the destination.
    /// The destination may alias this tensor; pairing is by flat index.
    /// The vector path runs only on directly spannable storage.
    /// </summary>
    public virtual void VectorizedApply(Func<Vector<T>, Vector<T>> op, Func<T, T> sop, Tensor<T> destination, TensorExecutionOptions options)
    {
        CheckDestination(destination);

        if (options.UseSimd && this is DenseTensor<T> d1 && destination is DenseTensor<T> d2
            && IsDirectSpanCompatible(this) && IsDirectSpanCompatible(destination))
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

    public Tensor<T> VectorizedApply(Func<Vector<T>, Vector<T>> op, Func<T, T> sop, TensorExecutionOptions options)
    {
        var output = CloneEmpty();
        VectorizedApply(op, sop, output, options);
        return output;
    }

    /// <summary>
    /// Writes op(left, right) for every element pair, overwriting the destination.
    /// The destination may alias either source; pairing is by flat index.
    /// </summary>
    public virtual void Apply(Func<T, T, T> op, Tensor<T> tensor2, Tensor<T> destination)
    {
        CheckBinaryDestination(tensor2, destination);

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

    /// <summary>
    /// Writes op(left, right) for every element pair, overwriting the destination, with
    /// automatic execution options. The destination starts uninitialized and is fully
    /// overwritten; it may alias either source because pairing is by flat index. The vector
    /// path runs only when all three tensors are directly spannable, otherwise the scalar
    /// path runs. Empty tensors write nothing but lengths must still match.
    /// </summary>
    /// <param name="op">Vector operation; must compute the lane-wise image of <paramref name="sop"/>.</param>
    /// <param name="sop">Scalar fallback operation; must be non-null.</param>
    /// <param name="tensor2">Second operand with exactly one element per source element.</param>
    /// <param name="destination">Overwrite-only destination holding exactly one slot per element.</param>
    /// <exception cref="ArgumentNullException"><paramref name="tensor2"/> or <paramref name="destination"/> is null.</exception>
    /// <exception cref="ArgumentException">An operand length differs from the source length.</exception>
    public virtual void VectorizedApply(Func<Vector<T>, Vector<T>, Vector<T>> op, Func<T, T, T> sop, Tensor<T> tensor2, Tensor<T> destination)
        => VectorizedApply(op, sop, tensor2, destination, TensorExecutionOptions.Auto);

    /// <summary>
    /// Writes op(left, right) for every element pair, overwriting the destination.
    /// The destination may alias either source; pairing is by flat index.
    /// The vector path runs only on directly spannable storage.
    /// </summary>
    public virtual void VectorizedApply(Func<Vector<T>, Vector<T>, Vector<T>> op, Func<T, T, T> sop, Tensor<T> tensor2, Tensor<T> destination, TensorExecutionOptions options)
    {
        CheckBinaryDestination(tensor2, destination);

        if (options.UseSimd && this is DenseTensor<T> d1 && tensor2 is DenseTensor<T> d2 && destination is DenseTensor<T> d3
            && IsDirectSpanCompatible(this) && IsDirectSpanCompatible(tensor2) && IsDirectSpanCompatible(destination))
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

    /// <summary>
    /// Allocates a fresh destination with CloneEmpty and writes op(left, right) for every
    /// element pair, with automatic execution options. Neither source is modified; the
    /// destination starts uninitialized and is fully overwritten.
    /// </summary>
    /// <param name="op">Vector operation; must compute the lane-wise image of <paramref name="sop"/>.</param>
    /// <param name="sop">Scalar fallback operation; must be non-null.</param>
    /// <param name="tensor2">Second operand with exactly one element per source element.</param>
    /// <returns>A new tensor holding the pairwise results.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="tensor2"/> is null.</exception>
    public virtual Tensor<T> VectorizedApply(Func<Vector<T>, Vector<T>, Vector<T>> op, Func<T, T, T> sop, Tensor<T> tensor2)
    {
        var output = CloneEmpty();
        VectorizedApply(op, sop, tensor2, output);
        return output;
    }

    /// <summary>
    /// Allocates a fresh destination with CloneEmpty and writes op(left, right) for every
    /// element pair under the given execution options. Neither source is modified; the
    /// destination starts uninitialized and is fully overwritten.
    /// </summary>
    /// <param name="op">Vector operation; must compute the lane-wise image of <paramref name="sop"/>.</param>
    /// <param name="sop">Scalar fallback operation; must be non-null.</param>
    /// <param name="tensor2">Second operand with exactly one element per source element.</param>
    /// <param name="options">Execution options selecting the SIMD and threading behavior.</param>
    /// <returns>A new tensor holding the pairwise results.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="tensor2"/> is null.</exception>
    public virtual Tensor<T> VectorizedApply(Func<Vector<T>, Vector<T>, Vector<T>> op, Func<T, T, T> sop, Tensor<T> tensor2, TensorExecutionOptions options)
    {
        var output = CloneEmpty();
        VectorizedApply(op, sop, tensor2, output, options);
        return output;
    }

    /// <summary>
    /// Folds op(state, element) left to right in flat index order starting from
    /// <paramref name="state"/>. An empty tensor returns <paramref name="state"/> unchanged.
    /// </summary>
    /// <param name="op">Fold operation; must be non-null.</param>
    /// <param name="state">Initial accumulator value.</param>
    /// <returns>The final accumulator value.</returns>
    public virtual T Accumulate(Func<T, T, T> op, T state)
    {
        var result = state;
        for (int index = 0; index < Length; index++)
        {
            result = op(result, GetValue(index));
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
        // ONNX numeric semantics: NaN equals nothing, including itself. The generic
        // EqualityComparer treats NaN as equal to NaN, so floats use == (verified
        // against ORT 1.29: NaN==NaN is false, +0==-0 is true).
        if (typeof(T) == typeof(float))
        {
            var fx = (Tensor<float>)(object)bx;
            var fy = (Tensor<float>)(object)by;
            for (int i = 0; i < output.Length; i++)
            {
                output.SetValue(i, fx.GetValue(i) == fy.GetValue(i));
            }
            return output;
        }
        if (typeof(T) == typeof(double))
        {
            var dx = (Tensor<double>)(object)bx;
            var dy = (Tensor<double>)(object)by;
            for (int i = 0; i < output.Length; i++)
            {
                output.SetValue(i, dx.GetValue(i) == dy.GetValue(i));
            }
            return output;
        }
        for (int i = 0; i < output.Length; i++)
        {
            output.SetValue(i, EqualityComparer<T>.Default.Equals(bx.GetValue(i), by.GetValue(i)));
        }
        return output;
    }

    public static Tensor<bool> Less(Tensor<T> x, Tensor<T> y)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (!Broadcast(x, y, out var bx, out var by))
        {
            throw new ArgumentException("Inputs are not broadcastable.");
        }
        var output = DenseTensor<bool>.OfShape(bx.Dimensions.ToArray());
        // ONNX numeric semantics: ordered comparisons with NaN are always false.
        // Comparer<T> orders NaN below every value, so floats use < (verified
        // against ORT 1.29: NaN<0 and 0<NaN are both false).
        if (typeof(T) == typeof(float))
        {
            var fx = (Tensor<float>)(object)bx;
            var fy = (Tensor<float>)(object)by;
            for (int i = 0; i < output.Length; i++)
            {
                output.SetValue(i, fx.GetValue(i) < fy.GetValue(i));
            }
            return output;
        }
        if (typeof(T) == typeof(double))
        {
            var dx = (Tensor<double>)(object)bx;
            var dy = (Tensor<double>)(object)by;
            for (int i = 0; i < output.Length; i++)
            {
                output.SetValue(i, dx.GetValue(i) < dy.GetValue(i));
            }
            return output;
        }
        for (int i = 0; i < output.Length; i++)
        {
            output.SetValue(i, Comparer<T>.Default.Compare(bx.GetValue(i), by.GetValue(i)) < 0);
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
        // ONNX broadcasts all three inputs multidirectionally: the condition
        // may outrank x/y (e.g. [2,2] over [2]), which the two-way broadcast
        // above cannot see.
        if (!BroadcastShape(bx.Dimensions.ToArray(), condition.Dimensions.ToArray(), out var shape))
        {
            throw new ArgumentException("Inputs are not broadcastable.");
        }
        if (!Broadcast(bx, shape, out var fx) || !Broadcast(by, shape, out var fy))
        {
            throw new ArgumentException("Inputs are not broadcastable.");
        }
        var bcond = Tensor<bool>.BroadcastTo(condition, shape);
        var output = fx.CloneEmpty();
        for (int i = 0; i < output.Length; i++)
        {
            output.SetValue(i, bcond.GetValue(i) ? fx.GetValue(i) : fy.GetValue(i));
        }
        return output;
    }

    /// <summary>Elementwise select with a pooled span fast path (M6).</summary>
    /// <remarks>When condition, x, and y are row-major contiguous dense tensors, the
    /// broadcast plan maps to per-input strides (zero where an axis broadcasts) and one
    /// flat odometer loop over pointers fills a pooled destination; anything else keeps
    /// the view-based implementation bit-identically, including its error behavior.</remarks>
    public static Tensor<T> Where(Tensor<bool> condition, Tensor<T> x, Tensor<T> y, TensorExecutionOptions options, TensorBufferPool? pool)
    {
        options.Validate();
        if (TryWhereFast(condition, x, y, pool, out var fast) && fast is not null) return fast;
        return Where(condition, x, y);
    }

    static bool IsRowMajorDense<TElement>(DenseTensor<TElement> t) where TElement : unmanaged
    {
        if (t.IsReversedStride) return false;
        if (t.Buffer.Length != t.Length) return false;
        return t.Strides.SequenceEqual(ArrayUtilities.GetStrides(t.Dimensions));
    }

    static int[] ExpandWhereStrides(ReadOnlySpan<int> dims, ReadOnlySpan<int> strides, int[] shape)
    {
        int rank = shape.Length;
        var expanded = new int[rank];
        int off = rank - dims.Length;
        for (int d = 0; d < rank; d++)
        {
            int idim = d < off ? 1 : dims[d - off];
            if (idim == 1)
            {
                expanded[d] = 0;
            }
            else
            {
                if (idim != shape[d]) return new int[0];
                expanded[d] = strides[d - off];
            }
        }
        return expanded;
    }

    static bool TryWhereFast(Tensor<bool> condition, Tensor<T> x, Tensor<T> y, TensorBufferPool? pool, out Tensor<T>? fast)
    {
        fast = null;
        if (condition is not DenseTensor<bool> dc || x is not DenseTensor<T> dx || y is not DenseTensor<T> dy) return false;
        if (!BroadcastShape(dx.Dimensions, dy.Dimensions, out var s1) || s1 is null) return false;
        if (!BroadcastShape(s1, dc.Dimensions, out var shape) || shape is null) return false;
        if (!IsRowMajorDense(dc) || !IsRowMajorDense(dx) || !IsRowMajorDense(dy)) return false;
        var cs = ExpandWhereStrides(dc.Dimensions, dc.Strides, shape);
        var xs = ExpandWhereStrides(dx.Dimensions, dx.Strides, shape);
        var ys = ExpandWhereStrides(dy.Dimensions, dy.Strides, shape);
        if (cs.Length == 0 || xs.Length == 0 || ys.Length == 0) return false;
        long total = 1;
        foreach (var e in shape) total = checked(total * e);
        int len = checked((int)total);
        T[] buf = pool is null ? new T[len] : pool.Rent<T>(len);
        var output = new DenseTensor<T>(new Memory<T>(buf, 0, len), shape);
        int rank = shape.Length;
        var cSpan = dc.Buffer.Span;
        var xSpan = dx.Buffer.Span;
        var ySpan = dy.Buffer.Span;
        var dSpan = output.Buffer.Span;
        unsafe
        {
            fixed (bool* cp = cSpan)
            fixed (T* xp = xSpan, yp = ySpan, dp = dSpan)
            {
                if (rank == 0)
                {
                    if (len > 0) dp[0] = cp[0] ? xp[0] : yp[0];
                    fast = output;
                    return true;
                }
                var coord = new int[rank];
                int co = 0, xo = 0, yo = 0;
                for (int o = 0; o < len; o++)
                {
                    dp[o] = cp[co] ? xp[xo] : yp[yo];
                    for (int d = rank - 1; d >= 0; d--)
                    {
                        coord[d]++;
                        co += cs[d];
                        xo += xs[d];
                        yo += ys[d];
                        if (coord[d] < shape[d]) break;
                        coord[d] = 0;
                        co -= cs[d] * shape[d];
                        xo -= xs[d] * shape[d];
                        yo -= ys[d] * shape[d];
                    }
                }
            }
        }
        fast = output;
        return true;
    }

    public static Tensor<byte> Add(Tensor<byte> x, Tensor<byte> y) => Add(x, y, TensorExecutionOptions.Auto);
    public static Tensor<byte> Add(Tensor<byte> x, Tensor<byte> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => (l + r), (l, r) => (byte) (l + r), y, options);

    public static Tensor<byte> Add(Tensor<byte> x, byte y) => x.Apply(l => (byte)(l + y));

    public static Tensor<int> Add(Tensor<int> x, Tensor<int> y) => Add(x, y, TensorExecutionOptions.Auto);
    public static Tensor<int> Add(Tensor<int> x, Tensor<int> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l + r, (l, r) => l + r, y, options);

    public static Tensor<int> Add(Tensor<int> x, int y) => Add(x, y, TensorExecutionOptions.Auto);
    public static Tensor<int> Add(Tensor<int> x, int y, TensorExecutionOptions options) => x.VectorizedApply(l => l + new Vector<int>(y), l => l + y, options);

    public static Tensor<float> Add(Tensor<float> x, Tensor<float> y) => Add(x, y, TensorExecutionOptions.Auto);
    public static Tensor<float> Add(Tensor<float> x, Tensor<float> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l + r, (l, r) => l + r, y, options);

    /// <summary>Writes the float sum into an existing destination tensor.</summary>
    public static Tensor<float> Add(Tensor<float> x, Tensor<float> y, Tensor<float> destination, TensorExecutionOptions options)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        return x.BroadcastApply<AddBroadcast<float>>(y, destination, options);
    }

    public static Tensor<float> Add(Tensor<float> x, float y) => Add(x, y, TensorExecutionOptions.Auto);
    public static Tensor<float> Add(Tensor<float> x, float y, TensorExecutionOptions options) => x.VectorizedApply(l => l + new Vector<float>(y), l => l + y, options);

    public static Tensor<double> Add(Tensor<double> x, Tensor<double> y) => Add(x, y, TensorExecutionOptions.Auto);
    public static Tensor<double> Add(Tensor<double> x, Tensor<double> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l + r, (l, r) => l + r, y, options);

    public static Tensor<double> Add(Tensor<double> x, double y) => Add(x, y, TensorExecutionOptions.Auto);
    public static Tensor<double> Add(Tensor<double> x, double y, TensorExecutionOptions options) => x.VectorizedApply(l => l + new Vector<double>(y), l => l + y, options);

    public static Tensor<byte> Subtract(Tensor<byte> x, Tensor<byte> y) => Subtract(x, y, TensorExecutionOptions.Auto);
    public static Tensor<byte> Subtract(Tensor<byte> x, Tensor<byte> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => (l - r), (l, r) => (byte)(l - r), y, options);

    public static Tensor<byte> Subtract(Tensor<byte> x, byte y) => x.Apply(l => (byte)(l - y));

    public static Tensor<int> Subtract(Tensor<int> x, Tensor<int> y) => Subtract(x, y, TensorExecutionOptions.Auto);
    public static Tensor<int> Subtract(Tensor<int> x, Tensor<int> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l - r, (l, r) => l - r, y, options);

    public static Tensor<int> Subtract(Tensor<int> x, int y) => Subtract(x, y, TensorExecutionOptions.Auto);
    public static Tensor<int> Subtract(Tensor<int> x, int y, TensorExecutionOptions options) => x.VectorizedApply(l => l - new Vector<int>(y), l => l - y, options);

    public static Tensor<float> Subtract(Tensor<float> x, Tensor<float> y) => Subtract(x, y, TensorExecutionOptions.Auto);
    public static Tensor<float> Subtract(Tensor<float> x, Tensor<float> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l - r, (l, r) => l - r, y, options);

    public static Tensor<float> Subtract(Tensor<float> x, float y) => Subtract(x, y, TensorExecutionOptions.Auto);
    public static Tensor<float> Subtract(Tensor<float> x, float y, TensorExecutionOptions options) => x.VectorizedApply(l => l - new Vector<float>(y), l => l - y, options);

    public static Tensor<double> Subtract(Tensor<double> x, Tensor<double> y) => Subtract(x, y, TensorExecutionOptions.Auto);
    public static Tensor<double> Subtract(Tensor<double> x, Tensor<double> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l - r, (l, r) => l - r, y, options);

    public static Tensor<double> Subtract(Tensor<double> x, double y) => Subtract(x, y, TensorExecutionOptions.Auto);
    public static Tensor<double> Subtract(Tensor<double> x, double y, TensorExecutionOptions options) => x.VectorizedApply(l => l - new Vector<double>(y), l => l - y, options);

    public static Tensor<byte> Multiply(Tensor<byte> x, Tensor<byte> y) => Multiply(x, y, TensorExecutionOptions.Auto);
    public static Tensor<byte> Multiply(Tensor<byte> x, Tensor<byte> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => (l * r), (l, r) => (byte)(l * r), y, options);

    public static Tensor<byte> Multiply(Tensor<byte> x, byte y) => x.Apply(l => (byte)(l * y));

    public static Tensor<int> Multiply(Tensor<int> x, Tensor<int> y) => Multiply(x, y, TensorExecutionOptions.Auto);
    public static Tensor<int> Multiply(Tensor<int> x, Tensor<int> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l * r, (l, r) => l * r, y, options);

    public static Tensor<int> Multiply(Tensor<int> x, int y) => Multiply(x, y, TensorExecutionOptions.Auto);
    public static Tensor<int> Multiply(Tensor<int> x, int y, TensorExecutionOptions options) => x.VectorizedApply(l => l * new Vector<int>(y), l => l * y, options);

    public static Tensor<float> Multiply(Tensor<float> x, Tensor<float> y) => Multiply(x, y, TensorExecutionOptions.Auto);
    public static Tensor<float> Multiply(Tensor<float> x, Tensor<float> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l * r, (l, r) => l * r, y, options);

    /// <summary>Writes the float product into an existing destination tensor.</summary>
    public static Tensor<float> Multiply(Tensor<float> x, Tensor<float> y, Tensor<float> destination, TensorExecutionOptions options)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        return x.BroadcastApply<MultiplyBroadcast<float>>(y, destination, options);
    }

    public static Tensor<float> Multiply(Tensor<float> x, float y) => Multiply(x, y, TensorExecutionOptions.Auto);
    public static Tensor<float> Multiply(Tensor<float> x, float y, TensorExecutionOptions options) => x.VectorizedApply(l => l * new Vector<float>(y), l => l * y, options);

    public static Tensor<double> Multiply(Tensor<double> x, Tensor<double> y) => Multiply(x, y, TensorExecutionOptions.Auto);
    public static Tensor<double> Multiply(Tensor<double> x, Tensor<double> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l * r, (l, r) => l * r, y, options);

    public static Tensor<double> Multiply(Tensor<double> x, double y) => Multiply(x, y, TensorExecutionOptions.Auto);
    public static Tensor<double> Multiply(Tensor<double> x, double y, TensorExecutionOptions options) => x.VectorizedApply(l => l * new Vector<double>(y), l => l * y, options);

    public static Tensor<byte> Divide(Tensor<byte> x, Tensor<byte> y) => Divide(x, y, TensorExecutionOptions.Auto);
    public static Tensor<byte> Divide(Tensor<byte> x, Tensor<byte> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => (l / r), (l, r) => (byte)(l / r), y, options);

    public static Tensor<byte> Divide(Tensor<byte> x, byte y) => x.Apply(l => (byte)(l / y));

    public static Tensor<int> Divide(Tensor<int> x, Tensor<int> y) => Divide(x, y, TensorExecutionOptions.Auto);
    public static Tensor<int> Divide(Tensor<int> x, Tensor<int> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l / r, (l, r) => l / r, y, options);

    public static Tensor<int> Divide(Tensor<int> x, int y) => Divide(x, y, TensorExecutionOptions.Auto);
    public static Tensor<int> Divide(Tensor<int> x, int y, TensorExecutionOptions options) => x.VectorizedApply(l => l / new Vector<int>(y), l => l / y, options);

    public static Tensor<long> Add(Tensor<long> x, Tensor<long> y) => Add(x, y, TensorExecutionOptions.Auto);
    public static Tensor<long> Add(Tensor<long> x, Tensor<long> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l + r, (l, r) => l + r, y, options);

    public static Tensor<long> Subtract(Tensor<long> x, Tensor<long> y) => Subtract(x, y, TensorExecutionOptions.Auto);
    public static Tensor<long> Subtract(Tensor<long> x, Tensor<long> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l - r, (l, r) => l - r, y, options);

    public static Tensor<long> Multiply(Tensor<long> x, Tensor<long> y) => Multiply(x, y, TensorExecutionOptions.Auto);
    public static Tensor<long> Multiply(Tensor<long> x, Tensor<long> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l * r, (l, r) => l * r, y, options);

    public static Tensor<long> Divide(Tensor<long> x, Tensor<long> y) => x.Apply((l, r) => l / r, y);

    public static Tensor<long> Divide(Tensor<long> x, long y) => x.Apply(l => l / y);

    public static Tensor<float> Divide(Tensor<float> x, Tensor<float> y) => Divide(x, y, TensorExecutionOptions.Auto);
    public static Tensor<float> Divide(Tensor<float> x, Tensor<float> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l / r, (l, r) => l / r, y, options);

    /// <summary>Writes the float quotient into an existing destination tensor.</summary>
    public static Tensor<float> Divide(Tensor<float> x, Tensor<float> y, Tensor<float> destination, TensorExecutionOptions options)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        return x.BroadcastApply<DivideBroadcast<float>>(y, destination, options);
    }

    public static Tensor<float> Divide(Tensor<float> x, float y) => Divide(x, y, TensorExecutionOptions.Auto);
    public static Tensor<float> Divide(Tensor<float> x, float y, TensorExecutionOptions options) => x.VectorizedApply(l => l / new Vector<float>(y), l => l / y, options);

    public static Tensor<double> Divide(Tensor<double> x, Tensor<double> y) => Divide(x, y, TensorExecutionOptions.Auto);
    public static Tensor<double> Divide(Tensor<double> x, Tensor<double> y, TensorExecutionOptions options) => x.VectorizedApply((l, r) => l / r, (l, r) => l / r, y, options);

    public static Tensor<double> Divide(Tensor<double> x, double y) => Divide(x, y, TensorExecutionOptions.Auto);
    public static Tensor<double> Divide(Tensor<double> x, double y, TensorExecutionOptions options) => x.VectorizedApply(l => l / new Vector<double>(y), l => l / y, options);

    public static Tensor<float> Negate(Tensor<float> x) => Negate(x, TensorExecutionOptions.Auto);
    public static Tensor<float> Negate(Tensor<float> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.Negate, l => -l, options);

    public static Tensor<double> Negate(Tensor<double> x) => Negate(x, TensorExecutionOptions.Auto);
    public static Tensor<double> Negate(Tensor<double> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.Negate, l => -l, options);

    public static Tensor<float> Pow(Tensor<float> x, Tensor<float> y) => x.Apply(MathF.Pow, y);

    public static Tensor<double> Pow(Tensor<double> x, Tensor<double> y) => x.Apply(Math.Pow, y);

    public static Tensor<float> Square(Tensor<float> x) => x.Apply(l => l * l);

    public static Tensor<double> Square(Tensor<double> x) => x.Apply(l => l * l);

    // MathF.Abs clears the NaN sign bit, matching ORT (Abs(NaN) is +NaN).
    public static Tensor<float> Abs(Tensor<float> x) => x.Apply(MathF.Abs);
    /// <summary>Float Abs honoring execution options: dense standard inputs run the span fast path on AVX hardware, everything else keeps the scalar contract.</summary>
    public static Tensor<float> Abs(Tensor<float> x, TensorExecutionOptions options)
    {
        var dx = x.ToDenseTensor();
        if (options.UseSimd && Avx.IsSupported
            && dx is { IsReversedStride: false } own && HasStandardStrides(own) && own.Buffer.Length == (int)own.Length)
        {
            var output = DenseTensor<float>.OfShape(own.Dimensions.ToArray());
            AbsSpanFloat(own.Buffer.Span, output.Buffer.Span);
            return output;
        }
        return Abs(x);
    }

    /// <summary>Vectorized float Abs with exact edge semantics: sign bit cleared, matching scalar MathF.Abs bit for bit (including NaN).</summary>
    static unsafe void AbsSpanFloat(System.Span<float> xs, System.Span<float> ys)
    {
        int n = xs.Length;
        int w = Vector256<float>.Count;
        fixed (float* s = xs, d = ys)
        {
            int i = 0;
            for (; i <= n - w; i += w)
                *(Vector256<float>*)(d + i) = Vector256.Abs(*(Vector256<float>*)(s + i));
            for (; i < n; i++) d[i] = MathF.Abs(s[i]);
        }
    }

    public static Tensor<double> Abs(Tensor<double> x) => x.Apply(Math.Abs);
    /// <summary>Double Abs honoring execution options, mirroring the float fast path.</summary>
    public static Tensor<double> Abs(Tensor<double> x, TensorExecutionOptions options)
    {
        var dx = x.ToDenseTensor();
        if (options.UseSimd && Avx.IsSupported
            && dx is { IsReversedStride: false } own && HasStandardStrides(own) && own.Buffer.Length == (int)own.Length)
        {
            var output = DenseTensor<double>.OfShape(own.Dimensions.ToArray());
            AbsSpanDouble(own.Buffer.Span, output.Buffer.Span);
            return output;
        }
        return Abs(x);
    }

    /// <summary>Vectorized double Abs with exact edge semantics.</summary>
    static unsafe void AbsSpanDouble(System.Span<double> xs, System.Span<double> ys)
    {
        int n = xs.Length;
        int w = Vector256<double>.Count;
        fixed (double* s = xs, d = ys)
        {
            int i = 0;
            for (; i <= n - w; i += w)
                *(Vector256<double>*)(d + i) = Vector256.Abs(*(Vector256<double>*)(s + i));
            for (; i < n; i++) d[i] = Math.Abs(s[i]);
        }
    }
    public static Tensor<float> Cos(Tensor<float> x) => Cos(x, TensorExecutionOptions.Auto);
    public static Tensor<float> Cos(Tensor<float> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.Cos, MathF.Cos, options);

    public static Tensor<double> Cos(Tensor<double> x) => Cos(x, TensorExecutionOptions.Auto);
    public static Tensor<double> Cos(Tensor<double> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.Cos, Math.Cos, options);

    public static Tensor<float> Sin(Tensor<float> x) => Sin(x, TensorExecutionOptions.Auto);
    public static Tensor<float> Sin(Tensor<float> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.Sin, MathF.Sin, options);

    public static Tensor<double> Sin(Tensor<double> x) => Sin(x, TensorExecutionOptions.Auto);
    public static Tensor<double> Sin(Tensor<double> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.Sin, Math.Sin, options);

    public static Tensor<int> Negate(Tensor<int> x) => Negate(x, TensorExecutionOptions.Auto);
    public static Tensor<int> Negate(Tensor<int> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.Negate, l => -l, options);

    public static Tensor<long> Negate(Tensor<long> x) => Negate(x, TensorExecutionOptions.Auto);
    public static Tensor<long> Negate(Tensor<long> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.Negate, l => -l, options);

    public static Tensor<int> Abs(Tensor<int> x) => x.Apply(l => l >= 0 ? l : -l);

    public static Tensor<long> Abs(Tensor<long> x) => x.Apply(l => l >= 0L ? l : -l);

    public static Tensor<uint> Abs(Tensor<uint> x) => x.Apply(v => v);

    public static Tensor<ulong> Abs(Tensor<ulong> x) => x.Apply(v => v);
    public static Tensor<sbyte> Negate(Tensor<sbyte> x) => Negate(x, TensorExecutionOptions.Auto);
    public static Tensor<sbyte> Negate(Tensor<sbyte> x, TensorExecutionOptions options) =>
        x.VectorizedApply(Vector.Negate, l => (sbyte)-l, options);
    public static Tensor<short> Negate(Tensor<short> x) => Negate(x, TensorExecutionOptions.Auto);
    public static Tensor<short> Negate(Tensor<short> x, TensorExecutionOptions options) =>
        x.VectorizedApply(Vector.Negate, l => (short)-l, options);
    public static Tensor<sbyte> Abs(Tensor<sbyte> x) => x.Apply(l => l >= 0 ? l : (sbyte)-l);
    public static Tensor<byte> Abs(Tensor<byte> x) => x.Apply(v => v);
    public static Tensor<short> Abs(Tensor<short> x) => x.Apply(l => l >= 0 ? l : (short)-l);
    public static Tensor<ushort> Abs(Tensor<ushort> x) => x.Apply(v => v);

    /// <summary>
    /// Exact Gaussian error linear unit: 0.5 * x * (1 + erf(x / sqrt(2))).
    /// </summary>
    public static Tensor<float> Gelu(Tensor<float> x) => Gelu(x, TensorExecutionOptions.Auto);

    public static Tensor<float> Gelu(Tensor<float> x, TensorExecutionOptions options) =>
        x.VectorizedApply((Vector<float> v) => new Vector<float>(0.5f) * v * (Vector<float>.One + ErfVector(new Vector<float>(0.7071067811865476f) * v)), (float v) => 0.5f * v * (1f + MathOps.Erf(v * 0.7071067811865476f)), options);

    public static Tensor<float> Gelu(Tensor<float> x, Tensor<float> destination) => Gelu(x, destination, TensorExecutionOptions.Auto);

    public static Tensor<float> Gelu(Tensor<float> x, Tensor<float> destination, TensorExecutionOptions options)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        x.VectorizedApply((Vector<float> v) => new Vector<float>(0.5f) * v * (Vector<float>.One + ErfVector(new Vector<float>(0.7071067811865476f) * v)), (float v) => 0.5f * v * (1f + MathOps.Erf(v * 0.7071067811865476f)), destination, options);
        return destination;
    }

    /// <summary>
    /// Exact Gaussian error linear unit: 0.5 * x * (1 + erf(x / sqrt(2))).
    /// </summary>
    public static Tensor<double> Gelu(Tensor<double> x) => Gelu(x, TensorExecutionOptions.Auto);

    /// <summary>Double-precision GELU runs a fixed scalar path; the options are validated but select no kernel variant.</summary>
    public static Tensor<double> Gelu(Tensor<double> x, TensorExecutionOptions options)
    {
        options.Validate();
        return x.Apply((double v) => 0.5 * v * (1.0 + MathOps.Erf(v * 0.7071067811865476)));
    }

    /// <summary>Tanh-approximate Gaussian error linear unit in one pass: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).</summary>
    /// <remarks>Matches the Mul/Pow/Mul/Add/Mul/Tanh/Add/Mul chain element order (cube via multiplies), so outputs agree with the unfused chain within 1e-6.</remarks>
    public static Tensor<float> GeluTanh(Tensor<float> x, TensorExecutionOptions? options)
    {
        (options ?? TensorExecutionOptions.Auto).Validate();
        var dx = x.ToDenseTensor();
        var output = DenseTensor<float>.OfShape(dx.Dimensions.ToArray());
        GeluTanhInto(dx.Buffer.Span, output.Buffer.Span);
        return output;
    }

    public static Tensor<float> GeluTanh(Tensor<float> x, DenseTensor<float> destination, TensorExecutionOptions? options)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        (options ?? TensorExecutionOptions.Auto).Validate();
        var dx = x.ToDenseTensor();
        if (!destination.Dimensions.SequenceEqual(dx.Dimensions.ToArray())) throw new ArgumentException(nameof(destination), "Destination shape must match the input shape.");
        GeluTanhInto(dx.Buffer.Span, destination.Buffer.Span);
        return destination;
    }

    static void GeluTanhInto(System.Span<float> xs, System.Span<float> ys)
    {
        for (int i = 0; i < xs.Length; i++)
        {
            float v = xs[i];
            float t1 = 0.5f * v;
            float t3 = 0.044715f * v * v * v;
            float t5 = 0.7978846f * (v + t3);
            ys[i] = t1 * (MathF.Tanh(t5) + 1f);
        }
    }

    /// <summary>Double-precision tanh-approximate GELU, same single-pass order as the float kernel.</summary>
    public static Tensor<double> GeluTanh(Tensor<double> x, TensorExecutionOptions? options)
    {
        (options ?? TensorExecutionOptions.Auto).Validate();
        var dx = x.ToDenseTensor();
        var output = DenseTensor<double>.OfShape(dx.Dimensions.ToArray());
        var xs = dx.Buffer.Span;
        var ys = output.Buffer.Span;
        for (int i = 0; i < xs.Length; i++)
        {
            double v = xs[i];
            double t1 = 0.5 * v;
            double t3 = 0.044715 * v * v * v;
            double t5 = 0.79788456 * (v + t3);
            ys[i] = t1 * (Math.Tanh(t5) + 1.0);
        }
        return output;
    }

    public static Tensor<float> Sqrt(Tensor<float> x) => Sqrt(x, TensorExecutionOptions.Auto);
    public static Tensor<float> Sqrt(Tensor<float> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.SquareRoot, MathF.Sqrt, options);

    public static Tensor<double> Sqrt(Tensor<double> x) => Sqrt(x, TensorExecutionOptions.Auto);
    public static Tensor<double> Sqrt(Tensor<double> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.SquareRoot, Math.Sqrt, options);

    // NaN propagates and signed zero is preserved (ORT: Relu(NaN)=NaN, Relu(-0)=-0); negatives map to +0.
    public static Tensor<float> Relu(Tensor<float> x) => x.Apply(l => l <= 0.0f ? (l == 0.0f ? l : 0.0f) : l);

    /// <summary>Float Relu honoring execution options: dense standard inputs run the span fast path on AVX hardware, everything else keeps the scalar contract.</summary>
    public static Tensor<float> Relu(Tensor<float> x, TensorExecutionOptions? options)
    {
        var dx = x.ToDenseTensor();
        if ((options?.UseSimd ?? true) && Avx.IsSupported
            && dx is { IsReversedStride: false } own && HasStandardStrides(own) && own.Buffer.Length == (int)own.Length)
        {
            var output = DenseTensor<float>.OfShape(own.Dimensions.ToArray());
            ReluSpanFloat(own.Buffer.Span, output.Buffer.Span);
            return output;
        }
        return Relu(x);
    }

    /// <summary>Vectorized float Relu with exact edge semantics: keep where positive, keep signed zero, keep NaN, zero the rest.</summary>
    static unsafe void ReluSpanFloat(System.Span<float> xs, System.Span<float> ys)
    {
        int n = xs.Length;
        int w = Vector256<float>.Count;
        var zero = Vector256<float>.Zero;
        fixed (float* s = xs, d = ys)
        {
            int i = 0;
            for (; i <= n - w; i += w)
            {
                var v = *(Vector256<float>*)(s + i);
                var keep = Vector256.GreaterThan(v, zero) | Vector256.Equals(v, zero) | ~Vector256.Equals(v, v);
                *(Vector256<float>*)(d + i) = Vector256.ConditionalSelect(keep, v, zero);
            }
            for (; i < n; i++) d[i] = s[i] <= 0.0f ? (s[i] == 0.0f ? s[i] : 0.0f) : s[i];
        }
    }

    public static Tensor<double> Relu(Tensor<double> x) => x.Apply(l => l <= 0.0 ? (l == 0.0 ? l : 0.0) : l);

    /// <summary>Double Relu honoring execution options, mirroring the float fast path.</summary>
    public static Tensor<double> Relu(Tensor<double> x, TensorExecutionOptions? options)
    {
        var dx = x.ToDenseTensor();
        if ((options?.UseSimd ?? true) && Avx.IsSupported
            && dx is { IsReversedStride: false } own && HasStandardStrides(own) && own.Buffer.Length == (int)own.Length)
        {
            var output = DenseTensor<double>.OfShape(own.Dimensions.ToArray());
            ReluSpanDouble(own.Buffer.Span, output.Buffer.Span);
            return output;
        }
        return Relu(x);
    }

    /// <summary>Vectorized double Relu with exact edge semantics.</summary>
    static unsafe void ReluSpanDouble(System.Span<double> xs, System.Span<double> ys)
    {
        int n = xs.Length;
        int w = Vector256<double>.Count;
        var zero = Vector256<double>.Zero;
        fixed (double* s = xs, d = ys)
        {
            int i = 0;
            for (; i <= n - w; i += w)
            {
                var v = *(Vector256<double>*)(s + i);
                var keep = Vector256.GreaterThan(v, zero) | Vector256.Equals(v, zero) | ~Vector256.Equals(v, v);
                *(Vector256<double>*)(d + i) = Vector256.ConditionalSelect(keep, v, zero);
            }
            for (; i < n; i++) d[i] = s[i] <= 0.0 ? (s[i] == 0.0 ? s[i] : 0.0) : s[i];
        }
    }
    // Scalar logistic sigmoid: 1/(1+exp(-v)); NaN propagates and tails saturate like ORT.
    public static Tensor<float> Sigmoid(Tensor<float> x) => x.Apply(v => 1f / (1f + MathF.Exp(-v)));
    /// <summary>Float Sigmoid honoring execution options: dense standard inputs run the shared vector-exp span path when SIMD with x86 intrinsics is enabled, everything else keeps the scalar contract.</summary>
    public static Tensor<float> Sigmoid(Tensor<float> x, TensorExecutionOptions options)
    {
        var dx = x.ToDenseTensor();
        if (options.UseSimd && options.UseIntrinsics && Avx.IsSupported && Avx2.IsSupported && Fma.IsSupported
            && dx is { IsReversedStride: false } own && HasStandardStrides(own) && own.Buffer.Length == (int)own.Length)
        {
            var output = DenseTensor<float>.OfShape(own.Dimensions.ToArray());
            MathOps.SigmoidSpan(own.Buffer.Span, output.Buffer.Span);
            return output;
        }
        return Sigmoid(x);
    }

    // Scalar hyperbolic tangent; matches MathF.Tanh on every input including NaN and infinities.
    public static Tensor<float> Tanh(Tensor<float> x) => x.Apply(MathF.Tanh);
    /// <summary>Float Tanh honoring execution options: dense standard inputs run the shared vector-exp span path when SIMD with x86 intrinsics is enabled, everything else keeps the scalar contract.</summary>
    public static Tensor<float> Tanh(Tensor<float> x, TensorExecutionOptions options)
    {
        var dx = x.ToDenseTensor();
        if (options.UseSimd && options.UseIntrinsics && Avx.IsSupported && Avx2.IsSupported && Fma.IsSupported
            && dx is { IsReversedStride: false } own && HasStandardStrides(own) && own.Buffer.Length == (int)own.Length)
        {
            var output = DenseTensor<float>.OfShape(own.Dimensions.ToArray());
            MathOps.TanhSpan(own.Buffer.Span, output.Buffer.Span);
            return output;
        }
        return Tanh(x);
    }

    // Scalar gated multiply: other * sigmoid(gated side). NaN propagates and
    // infinities follow the libm chain exactly (0 * Inf yields NaN on both).
    static Tensor<float> SigmoidMulScalar(Tensor<float> a, Tensor<float> b, int sigmoidInput)
    {
        var aa = a.ToArray();
        var bb = b.ToArray();
        if (aa.Length != bb.Length) throw new ArgumentException("SigmoidMul scalar fallback requires matching lengths.", nameof(b));
        var output = new float[aa.Length];
        if (sigmoidInput == 1)
        {
            for (int i = 0; i < output.Length; i++) output[i] = aa[i] * (1f / (1f + MathF.Exp(-bb[i])));
        }
        else
        {
            for (int i = 0; i < output.Length; i++) output[i] = (1f / (1f + MathF.Exp(-aa[i]))) * bb[i];
        }
        return new DenseTensor<float>(output, a.Dimensions.ToArray());
    }

    /// <summary>Float gated multiply honoring execution options: y = a*sigmoid(b) (sigmoidInput 1) or sigmoid(a)*b (sigmoidInput 0). Dense standard same-length inputs run one fused span pass when SIMD with x86 intrinsics is enabled; broadcast or exotic layouts route the sigmoid through a temp with the proven broadcast multiply; everything else keeps the scalar contract.</summary>
    public static Tensor<float> SigmoidMul(Tensor<float> a, Tensor<float> b, int sigmoidInput, TensorExecutionOptions options)
    {
        if (sigmoidInput != 0 && sigmoidInput != 1) throw new ArgumentOutOfRangeException(nameof(sigmoidInput), "SigmoidMul selects input 0 or 1 for the sigmoid leg.");
        if (!Tensor<float>.BroadcastShape(a.Dimensions, b.Dimensions, out _)) throw new ArgumentException("SigmoidMul inputs must broadcast.", nameof(b));
        var da = a.ToDenseTensor();
        var db = b.ToDenseTensor();
        // Equal lengths alone do not imply pairable elements ([2,1] against
        // [1,2] broadcasts to [2,2]); only identical shapes run the flat
        // span or scalar lanes, everything else takes the broadcast lane.
        bool sameShape = da.Dimensions.SequenceEqual(db.Dimensions);
        if (sameShape && options.UseSimd && options.UseIntrinsics && Avx.IsSupported && Avx2.IsSupported && Fma.IsSupported
            && da is { IsReversedStride: false } owna && HasStandardStrides(owna) && owna.Buffer.Length == (int)owna.Length
            && db is { IsReversedStride: false } ownb && HasStandardStrides(ownb) && ownb.Buffer.Length == (int)ownb.Length)
        {
            var output = DenseTensor<float>.OfShape(owna.Dimensions.ToArray());
            if (ReferenceEquals(da, db)) MathOps.SwishSpan(owna.Buffer.Span, output.Buffer.Span);
            else if (sigmoidInput == 1) MathOps.SigmoidMulSpan(owna.Buffer.Span, ownb.Buffer.Span, output.Buffer.Span);
            else MathOps.SigmoidMulSpan(ownb.Buffer.Span, owna.Buffer.Span, output.Buffer.Span);
            return output;
        }
        if (sameShape) return SigmoidMulScalar(a, b, sigmoidInput);
        // Buffer spans of densified views need not run in logical order
        // (reversed strides survive ToDenseTensor), so the sigmoid temp is
        // built from the logical-order copy while the proven broadcast
        // multiply pairs the originals.
        var gated = sigmoidInput == 1 ? b : a;
        var gb = gated.ToArray();
        var tt = new float[gb.Length];
        MathOps.SigmoidSpan(gb, tt);
        var temp = new DenseTensor<float>(tt, gated.Dimensions.ToArray());
        return sigmoidInput == 1 ? a.BroadcastApply<MultiplyBroadcast<float>>(temp, options) : temp.BroadcastApply<MultiplyBroadcast<float>>(b, options);
    }

    /// <summary>Float gated multiply into a caller-provided destination honoring execution options, mirroring the rented convention.</summary>
    public static Tensor<float> SigmoidMul(Tensor<float> a, Tensor<float> b, int sigmoidInput, TensorExecutionOptions options, DenseTensor<float> destination)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        if (sigmoidInput != 0 && sigmoidInput != 1) throw new ArgumentOutOfRangeException(nameof(sigmoidInput), "SigmoidMul selects input 0 or 1 for the sigmoid leg.");
        if (!Tensor<float>.BroadcastShape(a.Dimensions, b.Dimensions, out var shape)) throw new ArgumentException("SigmoidMul inputs must broadcast.", nameof(b));
        if (!destination.Dimensions.SequenceEqual(shape)) throw new ArgumentException("Destination shape must match the broadcast shape.", nameof(destination));
        var r = SigmoidMul(a, b, sigmoidInput, options);
        r.ToDenseTensor().Buffer.Span.CopyTo(destination.Buffer.Span);
        return destination;
    }

    public static Tensor<sbyte> Relu(Tensor<sbyte> x) => x.Apply(l => l >= 0 ? l : (sbyte)0);
    public static Tensor<int> Relu(Tensor<int> x) => x.Apply(l => l >= 0 ? l : 0);

    public static Tensor<float> Softmax(Tensor<float> x, int axis, TensorExecutionOptions? options, int opsetVersion)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (x.Rank < 1) throw new ArgumentException(nameof(x), "Softmax requires a tensor of rank 1 or more.");
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(x.Rank, axis);
        if (axis < 0 || axis >= x.Rank) throw new ArgumentException(nameof(axis), "The specified axis must be a dimension of the tensor.");
        var denseInput = x.ToDenseTensor();
        var output = new DenseTensor<float>(denseInput.Dimensions);
        StartOpStage(OpStage.Math);
        SoftmaxFloatInto(denseInput, output, axis, options ?? TensorExecutionOptions.Auto, opsetVersion);
        return output;
    }

    static void SoftmaxFloatInto(DenseTensor<float> input, DenseTensor<float> destination, int axis, TensorExecutionOptions options, int opsetVersion)
    {
        var dims = input.Dimensions.ToArray();
        var inputSpan = input.Buffer.Span;
        var outputSpan = destination.Buffer.Span;
        if (opsetVersion < 13)
        {
            int block = 1;
            for (int dimension = axis; dimension < dims.Length; dimension++) block *= dims[dimension];
            int outer = block == 0 ? 0 : (int)(input.Length / block);
            SoftmaxContiguousFloat(inputSpan, outputSpan, outer, block, options.UseSimd);
            return;
        }
        int outerCount = 1;
        for (int d = 0; d < axis; d++) outerCount *= dims[d];
        int dimLen = dims[axis];
        int inner = 1;
        for (int d = axis + 1; d < dims.Length; d++) inner *= dims[d];
        if (inner == 1)
        {
            int outer = dimLen == 0 ? 0 : (int)(input.Length / dimLen);
            SoftmaxContiguousFloat(inputSpan, outputSpan, outer, dimLen, options.UseSimd);
            return;
        }
        for (int o = 0; o < outerCount; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                float max = float.NegativeInfinity;
                for (int a = 0; a < dimLen; a++)
                {
                    float candidate = inputSpan[(o * dimLen + a) * inner + i];
                    if (float.IsNaN(candidate)) { max = float.NaN; break; }
                    if (candidate > max) max = candidate;
                }
                float sum = 0f;
                for (int a = 0; a < dimLen; a++)
                {
                    float activated = MathF.Exp(inputSpan[(o * dimLen + a) * inner + i] - max);
                    outputSpan[(o * dimLen + a) * inner + i] = activated;
                    sum += activated;
                }
                for (int a = 0; a < dimLen; a++) outputSpan[(o * dimLen + a) * inner + i] /= sum;
            }
        }
    }

    static void SoftmaxContiguousFloat(System.Span<float> inputSpan, System.Span<float> outputSpan, int outer, int block, bool useSimd)
    {
        for (int outerIndex = 0; outerIndex < outer; outerIndex++)
        {
            float max = float.NegativeInfinity;
            for (int blockIndex = 0; blockIndex < block; blockIndex++)
            {
                float candidate = inputSpan[outerIndex * block + blockIndex];
                if (float.IsNaN(candidate)) { max = float.NaN; break; }
                if (candidate > max) max = candidate;
            }
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

    /// <summary>Writes the float softmax into an existing dense destination.</summary>
    public static Tensor<float> Softmax(Tensor<float> x, DenseTensor<float> destination, int axis, TensorExecutionOptions? options, int opsetVersion)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        StartOpStage(OpStage.ValidateArguments);
        if (x.Rank < 1) throw new ArgumentException(nameof(x), "Softmax requires a tensor of rank 1 or more.");
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(x.Rank, axis);
        if (axis < 0 || axis >= x.Rank) throw new ArgumentException(nameof(axis), "The specified axis must be a dimension of the tensor.");
        var denseInput = x.ToDenseTensor();
        if (!destination.Dimensions.SequenceEqual(denseInput.Dimensions.ToArray())) throw new ArgumentException(nameof(destination), "Destination shape must match the input shape.");
        if (!HasStandardStrides(destination)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        StartOpStage(OpStage.Math);
        SoftmaxFloatInto(denseInput, destination, axis, options ?? TensorExecutionOptions.Auto, opsetVersion);
        return destination;
    }

    public static Tensor<double> Softmax(Tensor<double> x, int axis, TensorExecutionOptions? options, int opsetVersion)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (x.Rank < 1) throw new ArgumentException(nameof(x), "Softmax requires a tensor of rank 1 or more.");
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(x.Rank, axis);
        if (axis < 0 || axis >= x.Rank) throw new ArgumentException(nameof(axis), "The specified axis must be a dimension of the tensor.");
        var denseInput = x.ToDenseTensor();
        var output = new DenseTensor<double>(denseInput.Dimensions);
        StartOpStage(OpStage.Math);
        SoftmaxDoubleInto(denseInput, output, axis, opsetVersion);
        return output;
    }

    static void SoftmaxDoubleInto(DenseTensor<double> input, DenseTensor<double> destination, int axis, int opsetVersion)
    {
        var dims = input.Dimensions.ToArray();
        var inputSpan = input.Buffer.Span;
        var outputSpan = destination.Buffer.Span;
        if (opsetVersion < 13)
        {
            int block = 1;
            for (int dimension = axis; dimension < dims.Length; dimension++) block *= dims[dimension];
            int outer = block == 0 ? 0 : (int)(input.Length / block);
            SoftmaxContiguousDouble(inputSpan, outputSpan, outer, block);
            return;
        }
        int outerCount = 1;
        for (int d = 0; d < axis; d++) outerCount *= dims[d];
        int dimLen = dims[axis];
        int inner = 1;
        for (int d = axis + 1; d < dims.Length; d++) inner *= dims[d];
        if (inner == 1)
        {
            int outer = dimLen == 0 ? 0 : (int)(input.Length / dimLen);
            SoftmaxContiguousDouble(inputSpan, outputSpan, outer, dimLen);
            return;
        }
        for (int o = 0; o < outerCount; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                double max = double.NegativeInfinity;
                for (int a = 0; a < dimLen; a++)
                {
                    double candidate = inputSpan[(o * dimLen + a) * inner + i];
                    if (double.IsNaN(candidate)) { max = double.NaN; break; }
                    if (candidate > max) max = candidate;
                }
                double sum = 0d;
                for (int a = 0; a < dimLen; a++)
                {
                    double activated = Math.Exp(inputSpan[(o * dimLen + a) * inner + i] - max);
                    outputSpan[(o * dimLen + a) * inner + i] = activated;
                    sum += activated;
                }
                for (int a = 0; a < dimLen; a++) outputSpan[(o * dimLen + a) * inner + i] /= sum;
            }
        }
    }

    static void SoftmaxContiguousDouble(System.Span<double> inputSpan, System.Span<double> outputSpan, int outer, int block)
    {
        for (int outerIndex = 0; outerIndex < outer; outerIndex++)
        {
            double max = double.NegativeInfinity;
            for (int blockIndex = 0; blockIndex < block; blockIndex++)
            {
                double candidate = inputSpan[outerIndex * block + blockIndex];
                if (double.IsNaN(candidate)) { max = double.NaN; break; }
                if (candidate > max) max = candidate;
            }
            double sum = 0d;
            for (int blockIndex = 0; blockIndex < block; blockIndex++)
            {
                double activated = Math.Exp(inputSpan[outerIndex * block + blockIndex] - max);
                outputSpan[outerIndex * block + blockIndex] = activated;
                sum += activated;
            }
            for (int blockIndex = 0; blockIndex < block; blockIndex++) outputSpan[outerIndex * block + blockIndex] /= sum;
        }
    }

    /// <summary>
    /// Numerically stable log-softmax: each row emits (x - max) - log(sum)
    /// rather than log(softmax(x)), so very negative logits stay finite.
    /// Axis and pre-13 flattening mirror Softmax; the math runs a fixed
    /// scalar path.
    /// </summary>
    public static Tensor<float> LogSoftmax(Tensor<float> x, int axis, TensorExecutionOptions? options, int opsetVersion)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (x.Rank < 1) throw new ArgumentException(nameof(x), "LogSoftmax requires a tensor of rank 1 or more.");
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(x.Rank, axis);
        if (axis < 0 || axis >= x.Rank) throw new ArgumentException(nameof(axis), "The specified axis must be a dimension of the tensor.");
        var denseInput = x.ToDenseTensor();
        var output = new DenseTensor<float>(denseInput.Dimensions);
        StartOpStage(OpStage.Math);
        LogSoftmaxFloatInto(denseInput, output, axis, opsetVersion);
        return output;
    }

    static void LogSoftmaxFloatInto(DenseTensor<float> input, DenseTensor<float> destination, int axis, int opsetVersion)
    {
        var dims = input.Dimensions.ToArray();
        var inputSpan = input.Buffer.Span;
        var outputSpan = destination.Buffer.Span;
        if (opsetVersion < 13)
        {
            int block = 1;
            for (int dimension = axis; dimension < dims.Length; dimension++) block *= dims[dimension];
            int outer = block == 0 ? 0 : (int)(input.Length / block);
            LogSoftmaxContiguousFloat(inputSpan, outputSpan, outer, block);
            return;
        }
        int outerCount = 1;
        for (int d = 0; d < axis; d++) outerCount *= dims[d];
        int dimLen = dims[axis];
        int inner = 1;
        for (int d = axis + 1; d < dims.Length; d++) inner *= dims[d];
        if (inner == 1)
        {
            int outer = dimLen == 0 ? 0 : (int)(input.Length / dimLen);
            LogSoftmaxContiguousFloat(inputSpan, outputSpan, outer, dimLen);
            return;
        }
        for (int o = 0; o < outerCount; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                float max = float.NegativeInfinity;
                for (int a = 0; a < dimLen; a++)
                {
                    float candidate = inputSpan[(o * dimLen + a) * inner + i];
                    if (float.IsNaN(candidate)) { max = float.NaN; break; }
                    if (candidate > max) max = candidate;
                }
                float sum = 0f;
                for (int a = 0; a < dimLen; a++) sum += MathF.Exp(inputSpan[(o * dimLen + a) * inner + i] - max);
                float norm = MathF.Log(sum);
                for (int a = 0; a < dimLen; a++) outputSpan[(o * dimLen + a) * inner + i] = inputSpan[(o * dimLen + a) * inner + i] - max - norm;
            }
        }
    }

    static void LogSoftmaxContiguousFloat(System.Span<float> inputSpan, System.Span<float> outputSpan, int outer, int block)
    {
        for (int outerIndex = 0; outerIndex < outer; outerIndex++)
        {
            float max = float.NegativeInfinity;
            for (int blockIndex = 0; blockIndex < block; blockIndex++)
            {
                float candidate = inputSpan[outerIndex * block + blockIndex];
                if (float.IsNaN(candidate)) { max = float.NaN; break; }
                if (candidate > max) max = candidate;
            }
            float sum = 0f;
            for (int blockIndex = 0; blockIndex < block; blockIndex++) sum += MathF.Exp(inputSpan[outerIndex * block + blockIndex] - max);
            float norm = MathF.Log(sum);
            for (int blockIndex = 0; blockIndex < block; blockIndex++) outputSpan[outerIndex * block + blockIndex] = inputSpan[outerIndex * block + blockIndex] - max - norm;
        }
    }

    /// <summary>Writes the float log-softmax into an existing dense destination.</summary>
    public static Tensor<float> LogSoftmax(Tensor<float> x, DenseTensor<float> destination, int axis, TensorExecutionOptions? options, int opsetVersion)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        StartOpStage(OpStage.ValidateArguments);
        if (x.Rank < 1) throw new ArgumentException(nameof(x), "LogSoftmax requires a tensor of rank 1 or more.");
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(x.Rank, axis);
        if (axis < 0 || axis >= x.Rank) throw new ArgumentException(nameof(axis), "The specified axis must be a dimension of the tensor.");
        var denseInput = x.ToDenseTensor();
        if (!destination.Dimensions.SequenceEqual(denseInput.Dimensions.ToArray())) throw new ArgumentException(nameof(destination), "Destination shape must match the input shape.");
        if (!HasStandardStrides(destination)) throw new ArgumentException(nameof(destination), "Destination must have standard row-major strides.");
        StartOpStage(OpStage.Math);
        LogSoftmaxFloatInto(denseInput, destination, axis, opsetVersion);
        return destination;
    }

    public static Tensor<double> LogSoftmax(Tensor<double> x, int axis, TensorExecutionOptions? options, int opsetVersion)
    {
        StartOpStage(OpStage.ValidateArguments);
        if (x.Rank < 1) throw new ArgumentException(nameof(x), "LogSoftmax requires a tensor of rank 1 or more.");
        axis = ArrayUtilities.HandleNegativeAxisOrIndex(x.Rank, axis);
        if (axis < 0 || axis >= x.Rank) throw new ArgumentException(nameof(axis), "The specified axis must be a dimension of the tensor.");
        var denseInput = x.ToDenseTensor();
        var output = new DenseTensor<double>(denseInput.Dimensions);
        StartOpStage(OpStage.Math);
        LogSoftmaxDoubleInto(denseInput, output, axis, opsetVersion);
        return output;
    }

    static void LogSoftmaxDoubleInto(DenseTensor<double> input, DenseTensor<double> destination, int axis, int opsetVersion)
    {
        var dims = input.Dimensions.ToArray();
        var inputSpan = input.Buffer.Span;
        var outputSpan = destination.Buffer.Span;
        if (opsetVersion < 13)
        {
            int block = 1;
            for (int dimension = axis; dimension < dims.Length; dimension++) block *= dims[dimension];
            int outer = block == 0 ? 0 : (int)(input.Length / block);
            LogSoftmaxContiguousDouble(inputSpan, outputSpan, outer, block);
            return;
        }
        int outerCount = 1;
        for (int d = 0; d < axis; d++) outerCount *= dims[d];
        int dimLen = dims[axis];
        int inner = 1;
        for (int d = axis + 1; d < dims.Length; d++) inner *= dims[d];
        if (inner == 1)
        {
            int outer = dimLen == 0 ? 0 : (int)(input.Length / dimLen);
            LogSoftmaxContiguousDouble(inputSpan, outputSpan, outer, dimLen);
            return;
        }
        for (int o = 0; o < outerCount; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                double max = double.NegativeInfinity;
                for (int a = 0; a < dimLen; a++)
                {
                    double candidate = inputSpan[(o * dimLen + a) * inner + i];
                    if (double.IsNaN(candidate)) { max = double.NaN; break; }
                    if (candidate > max) max = candidate;
                }
                double sum = 0d;
                for (int a = 0; a < dimLen; a++) sum += Math.Exp(inputSpan[(o * dimLen + a) * inner + i] - max);
                double norm = Math.Log(sum);
                for (int a = 0; a < dimLen; a++) outputSpan[(o * dimLen + a) * inner + i] = inputSpan[(o * dimLen + a) * inner + i] - max - norm;
            }
        }
    }

    static void LogSoftmaxContiguousDouble(System.Span<double> inputSpan, System.Span<double> outputSpan, int outer, int block)
    {
        for (int outerIndex = 0; outerIndex < outer; outerIndex++)
        {
            double max = double.NegativeInfinity;
            for (int blockIndex = 0; blockIndex < block; blockIndex++)
            {
                double candidate = inputSpan[outerIndex * block + blockIndex];
                if (double.IsNaN(candidate)) { max = double.NaN; break; }
                if (candidate > max) max = candidate;
            }
            double sum = 0d;
            for (int blockIndex = 0; blockIndex < block; blockIndex++) sum += Math.Exp(inputSpan[outerIndex * block + blockIndex] - max);
            double norm = Math.Log(sum);
            for (int blockIndex = 0; blockIndex < block; blockIndex++) outputSpan[outerIndex * block + blockIndex] = inputSpan[outerIndex * block + blockIndex] - max - norm;
        }
    }

    public static Tensor<float> Erf(Tensor<float> x) => Erf(x, TensorExecutionOptions.Auto);

    public static Tensor<float> Erf(Tensor<float> x, TensorExecutionOptions options) => x.VectorizedApply(MathOps.ErfVector, MathOps.Erf, options);

    /// <summary>Writes the float error function into an existing destination tensor.</summary>
    public static Tensor<float> Erf(Tensor<float> x, Tensor<float> destination) => Erf(x, destination, TensorExecutionOptions.Auto);

    public static Tensor<float> Erf(Tensor<float> x, Tensor<float> destination, TensorExecutionOptions options)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        x.VectorizedApply(MathOps.ErfVector, MathOps.Erf, destination, options);
        return destination;
    }

    public static Tensor<double> Erf(Tensor<double> x) => Erf(x, TensorExecutionOptions.Auto);

    /// <summary>Double-precision erf runs a fixed scalar path; the options are validated but select no kernel variant.</summary>
    public static Tensor<double> Erf(Tensor<double> x, TensorExecutionOptions options)
    {
        options.Validate();
        return x.Apply(MathOps.Erf);
    }
}
