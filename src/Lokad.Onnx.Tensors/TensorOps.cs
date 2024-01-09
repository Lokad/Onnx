namespace Lokad.Onnx;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.Versioning;

[RequiresPreviewFeatures]
public interface IUnaryOperator<T> where T : struct
{
    static abstract T Invoke(T x);
}

[RequiresPreviewFeatures]
public interface IBinaryOperator<T> where T : struct
{
    static abstract T Invoke(T x, T y);

    //static abstract Vector<T> Invoke(Vector<T> x, Vector<T> y);
}

[RequiresPreviewFeatures]
public interface ITernaryOperator<T> where T : struct
{
    static abstract T Invoke(T x, T y, T z);
    //static abstract Vector<T> Invoke(Vector<T> x, Vector<T> y, Vector<T> z);
}

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

    public Tensor<U> Add<U>(Tensor<U> x) where U : struct, IAdditionOperators<U, U, U> 
        => x.Apply(AddOperator<U>.Invoke, x);
}

