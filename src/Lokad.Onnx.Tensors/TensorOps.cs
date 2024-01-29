namespace Lokad.Onnx;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

public abstract partial class Tensor<T> : TensorBase, IList, IList<T>, IReadOnlyList<T>, IStructuralComparable, IStructuralEquatable, ITensor
where T : struct
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

    public static Tensor<U>[] Broadcast<U>(Tensor<U> inA, Tensor<U> inB) where U: struct
    {
        var broadcastRank = Math.Max(inA.Rank, inB.Rank);
        var outA = inA.Clone();
        var outB = inB.Clone();
        for (var i = 0; i < broadcastRank; i++)
        {
            var idxA = i - broadcastRank + inA.Rank;
            var idxB = i - broadcastRank + inB.Rank;
            if (i < broadcastRank - inA.Rank)
            {
                outA = outA.PadLeft();
                outA = outA.BroadcastDim(0, inB.Dimensions[idxB]);
            }
            else if (i < broadcastRank - inB.Rank)
            {
                outB = outB.PadLeft();
                outB = outB.BroadcastDim(0, inA.Dimensions[idxA]);
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
                return Array.Empty<Tensor<U>>();
            }
        }
        return new[] { outA, outB };
    }

    public static bool Broadcast<U>(Tensor<U> x, Tensor<U> y, out Tensor<U> outx, out Tensor<U> outy) where U: struct
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

    public static bool BroadcastShape(ReadOnlySpan<int> x, ReadOnlySpan<int> y, out int[] b)
    {
        var tx = new DenseTensor<byte>(x, true);
        var ty = new DenseTensor<byte>(y, true);
        if (Broadcast(tx, ty, out var bx, out var _))
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

    public static bool BroadcastShape<U>(Tensor<U> x, Tensor<U> y, out int[] b) where U: struct => BroadcastShape(x.Dimensions, y.Dimensions, out b);   

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
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
        var n = x.Dimensions[0];
        var m = x.Dimensions[1];
        var r = y.Dimensions[1];
        var output = new DenseTensor<U>((ReadOnlySpan<int>)new int[] { x.Dimensions[0], y.Dimensions[1] });
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

    public static Tensor<U> MatMul<U>(Tensor<U> x, Tensor<U> y) where U : struct, IAdditiveIdentity<U, U>, IAdditionOperators<U, U, U>, IMultiplyOperators<U, U, U>
    {
        if (x.Rank == 0 || y.Rank == 0) throw new ArgumentException("The rank of each tensor in matrix multiplication must be greater than 1.");
        if (x.Rank == 2 && y.Rank == 2)
        {
            return Tensor<U>.MatMul2D(x, y);
        }
        else if (x.Rank >= 2 && y.Rank >= 2)
        {
            var xdl = x.Dimensions[^2..];
            var ydl = y.Dimensions[^2..];
            if (xdl[1] != ydl[0])
            {
                throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
            }

            if (!BroadcastShape(x.Dimensions[0..^2], y.Dimensions[0..^2], out var bd))
            {
                throw new ArgumentException("The tensor shapes are not compatble for broadcasting.");
            }
            
            var d = (ReadOnlySpan<int>) bd.Append(xdl[0]).Append(ydl[1]).ToArray();
            var c = new DenseTensor<U>(d);
            var di = x.GetDimensionsIterator(0..^2);
            foreach (var _ in di)
            {
                c[di[..]] = Tensor<U>.MatMul2D(x[di[..]], y[di[..]]);
            }
            return c;
        }
        else if (x.Rank >= 2 || y.Rank >= 2)
        {
            var b = Tensor<U>.Broadcast(x, y);
            if (!b.Any())
            {
                return null;
            }
            else
            {
                x = b[0];
                y = b[1];
                var c = x.CloneEmpty();
                var di = x.GetDimensionsIterator(0..^2);
                foreach (var _ in di)
                {
                    c[di[..]] = Tensor<U>.MatMul2D(x[di[..]], y[di[..]]);
                }
                return c;
            }
        }
        else
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
}

