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

    static readonly Func<Vector<float>, Vector<float>> GeluVecOp = v => new Vector<float>(0.5f) * v * (Vector<float>.One + ErfVector(new Vector<float>(0.7071067811865476f) * v));

    static float ScalarGelu(float v) => 0.5f * v * (1f + MathOps.Erf(v * 0.7071067811865476f));

    /// <summary>
    /// Single-pass exact GELU over spans: identical arithmetic to the composed
    /// path (same ErfVector call and combine order per element, same scalar
    /// tail), without per-chunk delegates or virtual calls. Safe in place:
    /// each output depends only on its own input element.
    /// </summary>
    internal static void GeluSpanFloat(ReadOnlySpan<float> xs, Span<float> ys)
    {
        var xvec = MemoryMarshal.Cast<float, Vector<float>>(xs);
        var yvec = MemoryMarshal.Cast<float, Vector<float>>(ys);
        var half = new Vector<float>(0.5f);
        var one = Vector<float>.One;
        var scale = new Vector<float>(0.7071067811865476f);
        for (int i = 0; i < yvec.Length; i++)
        {
            var v = xvec[i];
            yvec[i] = half * v * (one + ErfVector(scale * v));
        }
        for (int i = yvec.Length * Vector<float>.Count; i < xs.Length; i++)
            ys[i] = ScalarGelu(xs[i]);
    }

    public static Tensor<float> Gelu(Tensor<float> x, TensorExecutionOptions options)
    {
        if (options.UseSimd && x is DenseTensor<float> xd && xd.Buffer.Length == xd.Length)
        {
            var output = DenseTensor<float>.OfShape(xd.Dimensions.ToArray());
            GeluSpanFloat(xd.Buffer.Span, output.Buffer.Span);
            return output;
        }
        return x.Apply(ScalarGelu);
    }

    public static Tensor<float> Gelu(Tensor<float> x, Tensor<float> destination) => Gelu(x, destination, TensorExecutionOptions.Auto);

    /// <summary>
    /// Fused bias-add plus exact GELU over spans: y[i] = gelu(x[i] + b[i % M])
    /// with the same per-element arithmetic as the two-node form (identical
    /// add, then the identical erf call and combine order). Rank-one bias of
    /// length M takes the vector path when M is a multiple of the vector
    /// width; anything else runs the scalar tail loop. Safe in place.
    /// </summary>
    /// <summary>
    /// BiasGelu pointer fast path: identical arithmetic and order to BiasGeluSpanFloat
    /// with raw vector pointers instead of a per-vector Slice plus Cast, removing
    /// bounds-check setup from the hot loop.
    /// </summary>
    internal static unsafe void BiasGeluSpanFloatPtr(ReadOnlySpan<float> xs, ReadOnlySpan<float> bias, Span<float> ys)
    {
        int w = Vector<float>.Count;
        var half = new Vector<float>(0.5f);
        var one = Vector<float>.One;
        var scale = new Vector<float>(0.7071067811865476f);
        int M = bias.Length;
        if (M <= 1 || M % w != 0 || xs.Length != ys.Length)
        {
            int soff = 0;
            for (int i = 0; i < xs.Length; i++)
            {
                ys[i] = ScalarGelu(xs[i] + bias[soff]);
                soff++;
                if (soff >= M) soff = 0;
            }
            return;
        }
        fixed (float* px = xs, py = ys, pb = bias)
        {
            var xvec = (Vector<float>*)px;
            var yvec = (Vector<float>*)py;
            var bvec = (Vector<float>*)pb;
            int nvec = xs.Length / w;
            int bvecs = M / w;
            int boff = 0;
            for (int i = 0; i < nvec; i++)
            {
                var tv = xvec[i] + bvec[boff];
                yvec[i] = half * tv * (one + ErfVector(scale * tv));
                if (++boff >= bvecs) boff = 0;
            }
            int tail = nvec * w;
            int toff = tail % M;
            for (int i = tail; i < xs.Length; i++)
            {
                ys[i] = ScalarGelu(xs[i] + bias[toff]);
                toff++;
                if (toff >= M) toff = 0;
            }
        }
    }

    /// <summary>
    /// E72 twin: two-way unrolled BiasGelu pointer loop. Same per-lane arithmetic
    /// and order as BiasGeluSpanFloatPtr, two independent vectors per iteration
    /// for instruction-level parallelism. Test-reachable only; no dispatch yet.
    /// </summary>
    internal static unsafe void BiasGeluSpanFloatPtr2x(ReadOnlySpan<float> xs, ReadOnlySpan<float> bias, Span<float> ys)
    {
        int w = Vector<float>.Count;
        var half = new Vector<float>(0.5f);
        var one = Vector<float>.One;
        var scale = new Vector<float>(0.7071067811865476f);
        int M = bias.Length;
        if (M <= 1 || M % w != 0 || xs.Length != ys.Length)
        {
            int soff = 0;
            for (int i = 0; i < xs.Length; i++)
            {
                ys[i] = ScalarGelu(xs[i] + bias[soff]);
                soff++;
                if (soff >= M) soff = 0;
            }
            return;
        }
        fixed (float* px = xs, py = ys, pb = bias)
        {
            var xvec = (Vector<float>*)px;
            var yvec = (Vector<float>*)py;
            var bvec = (Vector<float>*)pb;
            int nvec = xs.Length / w;
            int bvecs = M / w;
            int boff = 0;
            int pairs = nvec / 2;
            for (int i = 0; i < pairs; i++)
            {
                int b0 = boff;
                int b1 = b0 + 1;
                if (b1 >= bvecs) b1 = 0;
                var tv0 = xvec[2 * i] + bvec[b0];
                var tv1 = xvec[2 * i + 1] + bvec[b1];
                yvec[2 * i] = half * tv0 * (one + ErfVector(scale * tv0));
                yvec[2 * i + 1] = half * tv1 * (one + ErfVector(scale * tv1));
                boff = b1 + 1;
                if (boff >= bvecs) boff = 0;
            }
            for (int i = pairs * 2; i < nvec; i++)
            {
                var tv = xvec[i] + bvec[boff];
                yvec[i] = half * tv * (one + ErfVector(scale * tv));
                if (++boff >= bvecs) boff = 0;
            }
            int tail = nvec * w;
            int toff = tail % M;
            for (int i = tail; i < xs.Length; i++)
            {
                ys[i] = ScalarGelu(xs[i] + bias[toff]);
                toff++;
                if (toff >= M) toff = 0;
            }
        }
    }

    /// <summary>
    /// E72 twin: four-way unrolled BiasGelu pointer loop. Same per-lane arithmetic
    /// and order as BiasGeluSpanFloatPtr, four independent vectors per iteration.
    /// Test-reachable only; no dispatch yet.
    /// </summary>
    internal static unsafe void BiasGeluSpanFloatPtr4x(ReadOnlySpan<float> xs, ReadOnlySpan<float> bias, Span<float> ys)
    {
        int w = Vector<float>.Count;
        var half = new Vector<float>(0.5f);
        var one = Vector<float>.One;
        var scale = new Vector<float>(0.7071067811865476f);
        int M = bias.Length;
        if (M <= 1 || M % w != 0 || xs.Length != ys.Length)
        {
            int soff = 0;
            for (int i = 0; i < xs.Length; i++)
            {
                ys[i] = ScalarGelu(xs[i] + bias[soff]);
                soff++;
                if (soff >= M) soff = 0;
            }
            return;
        }
        fixed (float* px = xs, py = ys, pb = bias)
        {
            var xvec = (Vector<float>*)px;
            var yvec = (Vector<float>*)py;
            var bvec = (Vector<float>*)pb;
            int nvec = xs.Length / w;
            int bvecs = M / w;
            int boff = 0;
            int quads = nvec / 4;
            for (int i = 0; i < quads; i++)
            {
                int b0 = boff;
                int b1 = b0 + 1;
                if (b1 >= bvecs) b1 = 0;
                int b2 = b1 + 1;
                if (b2 >= bvecs) b2 = 0;
                int b3 = b2 + 1;
                if (b3 >= bvecs) b3 = 0;
                var tv0 = xvec[4 * i] + bvec[b0];
                var tv1 = xvec[4 * i + 1] + bvec[b1];
                var tv2 = xvec[4 * i + 2] + bvec[b2];
                var tv3 = xvec[4 * i + 3] + bvec[b3];
                yvec[4 * i] = half * tv0 * (one + ErfVector(scale * tv0));
                yvec[4 * i + 1] = half * tv1 * (one + ErfVector(scale * tv1));
                yvec[4 * i + 2] = half * tv2 * (one + ErfVector(scale * tv2));
                yvec[4 * i + 3] = half * tv3 * (one + ErfVector(scale * tv3));
                boff = b3 + 1;
                if (boff >= bvecs) boff = 0;
            }
            for (int i = quads * 4; i < nvec; i++)
            {
                var tv = xvec[i] + bvec[boff];
                yvec[i] = half * tv * (one + ErfVector(scale * tv));
                if (++boff >= bvecs) boff = 0;
            }
            int tail = nvec * w;
            int toff = tail % M;
            for (int i = tail; i < xs.Length; i++)
            {
                ys[i] = ScalarGelu(xs[i] + bias[toff]);
                toff++;
                if (toff >= M) toff = 0;
            }
        }
    }
    internal static void BiasGeluSpanFloat(ReadOnlySpan<float> xs, ReadOnlySpan<float> bias, Span<float> ys)
    {
        int w = Vector<float>.Count;
        var half = new Vector<float>(0.5f);
        var one = Vector<float>.One;
        var scale = new Vector<float>(0.7071067811865476f);
        int M = bias.Length;
        if (M > 1 && M % w == 0 && xs.Length == ys.Length)
        {
            var xvec = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yvec = MemoryMarshal.Cast<float, Vector<float>>(ys);
            int boff = 0;
            for (int i = 0; i < yvec.Length; i++)
            {
                var bv = MemoryMarshal.Cast<float, Vector<float>>(bias.Slice(boff, w))[0];
                var tv = xvec[i] + bv;
                yvec[i] = half * tv * (one + ErfVector(scale * tv));
                boff += w;
                if (boff >= M) boff -= M;
            }
            int tail = yvec.Length * w;
            int toff = tail % M;
            for (int i = tail; i < xs.Length; i++)
            {
                ys[i] = ScalarGelu(xs[i] + bias[toff]);
                toff++;
                if (toff >= M) toff = 0;
            }
            return;
        }
        int soff = 0;
        for (int i = 0; i < xs.Length; i++)
        {
            ys[i] = ScalarGelu(xs[i] + bias[soff]);
            soff++;
            if (soff >= M) soff = 0;
        }
    }

    /// <summary>
    /// Bias-add fused with tanh-approximate GELU over spans: y[i] = tanhgelu(x[i] + b[i % M]).
    /// Vector fast path mirrors BiasGeluSpanFloat with the GeluTanhSpanFloat formula per lane;
    /// scalar tail uses GeluTanhScalar. Cyclic bias offsets match BiasGeluSpanFloat exactly.
    /// </summary>
    internal static void BiasGeluTanhSpanFloat(ReadOnlySpan<float> xs, ReadOnlySpan<float> bias, Span<float> ys)
    {
        int w = Vector<float>.Count;
        var half = new Vector<float>(0.5f);
        var one = Vector<float>.One;
        var two = new Vector<float>(2f);
        var c3 = new Vector<float>(0.044715f);
        var c5 = new Vector<float>(0.7978846f);
        int M = bias.Length;
        if (M > 1 && M % w == 0 && xs.Length == ys.Length)
        {
            var xvec = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yvec = MemoryMarshal.Cast<float, Vector<float>>(ys);
            int boff = 0;
            for (int i = 0; i < yvec.Length; i++)
            {
                var bv = MemoryMarshal.Cast<float, Vector<float>>(bias.Slice(boff, w))[0];
                var tv = xvec[i] + bv;
                var t3 = c3 * tv * tv * tv;
                var t5 = c5 * (tv + t3);
                var e = ExpVector(t5 + t5);
                var t = one - two / (e + one);
                yvec[i] = half * tv * (t + one);
                boff += w;
                if (boff >= M) boff -= M;
            }
            int tail = yvec.Length * w;
            int toff = tail % M;
            for (int i = tail; i < xs.Length; i++)
            {
                ys[i] = GeluTanhScalar(xs[i] + bias[toff]);
                toff++;
                if (toff >= M) toff = 0;
            }
            return;
        }
        int soff = 0;
        for (int i = 0; i < xs.Length; i++)
        {
            ys[i] = GeluTanhScalar(xs[i] + bias[soff]);
            soff++;
            if (soff >= M) soff = 0;
        }
    }

    public static Tensor<float> Gelu(Tensor<float> x, Tensor<float> destination, TensorExecutionOptions options)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        if (options.UseSimd && x is DenseTensor<float> xd && destination is DenseTensor<float> dd
            && xd.Buffer.Length == xd.Length && dd.Buffer.Length == dd.Length && xd.Buffer.Length == dd.Buffer.Length)
        {
            GeluSpanFloat(xd.Buffer.Span, dd.Buffer.Span);
            return destination;
        }
        x.VectorizedApply(GeluVecOp, ScalarGelu, destination, options);
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
        GeluTanhSpanFloat(dx.Buffer.Span, output.Buffer.Span);
        return output;
    }

    public static Tensor<float> GeluTanh(Tensor<float> x, DenseTensor<float> destination, TensorExecutionOptions? options)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        (options ?? TensorExecutionOptions.Auto).Validate();
        var dx = x.ToDenseTensor();
        if (!destination.Dimensions.SequenceEqual(dx.Dimensions.ToArray())) throw new ArgumentException(nameof(destination), "Destination shape must match the input shape.");
        GeluTanhSpanFloat(dx.Buffer.Span, destination.Buffer.Span);
        return destination;
    }

    static float GeluTanhScalar(float v)
    {
        float t1 = 0.5f * v;
        float t3 = 0.044715f * v * v * v;
        float t5 = 0.7978846f * (v + t3);
        return t1 * (MathF.Tanh(t5) + 1f);
    }

    /// <summary>
    /// Vectorized tanh-approximate GELU over spans: identical formula and
    /// per-element order to GeluTanhScalar, with tanh evaluated through the
    /// shared ExpVector block as 1 - 2 / (exp(2t) + 1) plus a scalar tail
    /// GeluTanhScalar for remainders. Safe in place: each output
    /// depends only on its own input element.
    /// </summary>
    internal static void GeluTanhSpanFloat(ReadOnlySpan<float> xs, Span<float> ys)
    {
        var xvec = MemoryMarshal.Cast<float, Vector<float>>(xs);
        var yvec = MemoryMarshal.Cast<float, Vector<float>>(ys);
        var half = new Vector<float>(0.5f);
        var one = Vector<float>.One;
        var two = new Vector<float>(2f);
        var c3 = new Vector<float>(0.044715f);
        var c5 = new Vector<float>(0.7978846f);
        for (int i = 0; i < yvec.Length; i++)
        {
            var v = xvec[i];
            var t1 = half * v;
            var t3 = c3 * v * v * v;
            var t5 = c5 * (v + t3);
            var e = ExpVector(t5 + t5);
            var t = one - two / (e + one);
            yvec[i] = t1 * (t + one);
        }
        for (int i = yvec.Length * Vector<float>.Count; i < xs.Length; i++)
            ys[i] = GeluTanhScalar(xs[i]);
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
    internal static unsafe void ReluSpanFloat(System.Span<float> xs, System.Span<float> ys)
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
    internal static unsafe void ReluSpanDouble(System.Span<double> xs, System.Span<double> ys)
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
            SoftmaxContiguousFloatRoute(inputSpan, outputSpan, outer, block, options.UseSimd);
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
            SoftmaxContiguousFloatRoute(inputSpan, outputSpan, outer, dimLen, options.UseSimd);
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


    /// <summary>
    /// Default float softmax over contiguous rows: fused exp-summation and vectorized
    /// normalization. Max semantics (including NaN propagation) and exp calls match
    /// the legacy path call for call; only the summation groups vector lanes first,
    /// so finite values agree within 1e-6 while NaN rows stay NaN. Deterministic per
    /// shape: fixed-mode repeats are bit-identical run to run. Scalar (non-SIMD) mode
    /// is bitwise identical to the legacy kernel.
    /// </summary>
    /// <summary>
    /// Ablation router: the default span kernel, or the preserved legacy kernel
    /// when the measurement-only ForceLegacySoftmax switch is set (which also
    /// records the LegacySoftmaxUsed latch).
    /// </summary>
    static void SoftmaxContiguousFloatRoute(System.Span<float> inputSpan, System.Span<float> outputSpan, int outer, int block, bool useSimd)
    {
        if (AblationSwitches.ForceLegacySoftmax)
        {
            AblationSwitches.LegacySoftmaxUsed = true;
            SoftmaxContiguousFloat(inputSpan, outputSpan, outer, block, useSimd);
        }
        else SoftmaxContiguousFloatSpan2x(inputSpan, outputSpan, outer, block, useSimd);
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
    /// Legacy float softmax over contiguous rows, preserved as the tested reference
    /// for the default span kernel. Scalar summation order; vectorized exp only.
    /// </summary>
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


