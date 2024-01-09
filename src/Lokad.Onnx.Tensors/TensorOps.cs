namespace Lokad.Onnx;

using System;
using System.Collections;
using System.Collections.Generic;

public abstract partial class Tensor<T> : TensorBase, IList, IList<T>, IReadOnlyList<T>, IStructuralComparable, IStructuralEquatable, ITensor
where T :  struct
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

    public Tensor<T> Apply(Func<T, T> op)
    {
        var output = CloneEmpty();
        Apply(op, output);
        return output;
    }

    public Tensor<T> Apply(Func<T, T, T> op, Tensor<T> tensor2)
    {
        var output = CloneEmpty();
        Apply(op, tensor2, output);
        return output;
    }

    public static Tensor<U> Add<U>(Tensor<U> x, Tensor<U> y) where U : struct, IAdditionOperators<U, U, U> 
        => x.Apply((l, r) => l + r, y);

    public static Tensor<U> Subtract<U>(Tensor<U> x, Tensor<U> y) where U : struct, ISubtractionOperators<U, U, U>
        => x.Apply((l, r) => l - r, y);

    public static Tensor<U> Multiply<U>(Tensor<U> x, Tensor<U> y) where U : struct, IMultiplyOperators<U, U, U> 
        => x.Apply((l, r) => l * r, y);

    public static Tensor<U> Divide<U>(Tensor<U> x, Tensor<U> y) where U : struct, IDivisionOperators<U, U, U>
        => x.Apply((l, r) => l / r, y);

    public static Tensor<U> Negate<U>(Tensor<U> x) where U : struct, IUnaryNegationOperators<U, U>
        => x.Apply(l => -l);

    public static Tensor<U> Square<U>(Tensor<U> x) where U : struct, IMultiplyOperators<U, U, U>
        => x.Apply(l => l * l);
}

