namespace Lokad.Onnx;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;

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

    public Tensor<T> Apply(Func<T, T> op)
    {
        var output = CloneEmpty();
        Apply(op, output);
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

    public static Tensor<U> Add<U>(Tensor<U> x, Tensor<U> y) where U : struct, IAdditionOperators<U, U, U> 
        => x.Apply((l, r) => l + r, y);

    public static Tensor<U> Add<U>(Tensor<U> x, U y) where U : struct, IAdditionOperators<U, U, U>
      => x.Apply(l => l + y);

    public static Tensor<U> Subtract<U>(Tensor<U> x, Tensor<U> y) where U : struct, ISubtractionOperators<U, U, U>
        => x.Apply((l, r) => l - r, y);

    public static Tensor<U> Subtract<U>(Tensor<U> x, U y) where U : struct, ISubtractionOperators<U, U, U>
      => x.Apply(l => l - y);

    public static Tensor<U> Multiply<U>(Tensor<U> x, Tensor<U> y) where U : struct, IMultiplyOperators<U, U, U> 
        => x.Apply((l, r) => l * r, y);

    public static Tensor<U> Multiply<U>(Tensor<U> x, U y) where U : struct, IMultiplyOperators<U, U, U>
      => x.Apply(l => l * y);

    public static Tensor<U> Divide<U>(Tensor<U> x, Tensor<U> y) where U : struct, IDivisionOperators<U, U, U>
        => x.Apply((l, r) => l / r, y);

    public static Tensor<U> Divide<U>(Tensor<U> x, U y) where U : struct, IDivisionOperators<U, U, U>
      => x.Apply(l => l / y);

    public static Tensor<U> Negate<U>(Tensor<U> x) where U : struct, IUnaryNegationOperators<U, U>
        => x.Apply(l => -l);

    public static Tensor<U> Square<U>(Tensor<U> x) where U : struct, IMultiplyOperators<U, U, U>
        => x.Apply(l => l * l);

    public static Tensor<U> Abs<U>(Tensor<U> x) where U : struct, IAdditiveIdentity<U, U>, IUnaryNegationOperators<U, U>, IComparisonOperators<U, U>
       => x.Apply(l => l >= U.AdditiveIdentity ? l : -l);

    public static Tensor<float> Sqrt(Tensor<float> x) => x.Apply(MathF.Sqrt);

    public static Tensor<double> Sqrt(Tensor<double> x) => x.Apply(Math.Sqrt);

    public static Tensor<U> MatMul2D<U>(Tensor<U> x, Tensor<U> y) where U : struct, IAdditiveIdentity<U, U>, IAdditionOperators<U, U, U>, IMultiplyOperators<U, U, U>
    {
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException("The number of columns in the first tensor is not equal to the num");
        var n = x.Dimensions[0];
        var m = x.Dimensions[1];
        var r = y.Dimensions[1];    
        var output = new DenseTensor<U>((ReadOnlySpan<int>) new int[] { x.Dimensions[0], y.Dimensions[1] });
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < r; j++)
            {
                var sum = U.AdditiveIdentity;
                for (int k = 0; k < m; k++)
                    sum += x[i, k] * y[k, j];
                output[i, j] = sum;
            }
            
        }
        return output;
    }
}

