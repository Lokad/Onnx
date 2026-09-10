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

    public static Tensor<double> Abs(Tensor<double> x) => x.Apply(Math.Abs);
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

    public static Tensor<float> Sqrt(Tensor<float> x) => Sqrt(x, TensorExecutionOptions.Auto);
    public static Tensor<float> Sqrt(Tensor<float> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.SquareRoot, MathF.Sqrt, options);

    public static Tensor<double> Sqrt(Tensor<double> x) => Sqrt(x, TensorExecutionOptions.Auto);
    public static Tensor<double> Sqrt(Tensor<double> x, TensorExecutionOptions options) => x.VectorizedApply(Vector.SquareRoot, Math.Sqrt, options);

    // NaN propagates and signed zero is preserved (ORT: Relu(NaN)=NaN, Relu(-0)=-0); negatives map to +0.
    public static Tensor<float> Relu(Tensor<float> x) => x.Apply(l => l <= 0.0f ? (l == 0.0f ? l : 0.0f) : l);

    public static Tensor<double> Relu(Tensor<double> x) => x.Apply(l => l <= 0.0 ? (l == 0.0 ? l : 0.0) : l);

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
