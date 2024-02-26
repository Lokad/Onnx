namespace Lokad.Onnx;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;

using static Lokad.Onnx.MathOps;

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

    public static Tensor<U>[] Broadcast<U>(Tensor<U> inA, Tensor<U> inB) where U : struct
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

    public static bool Broadcast<U>(Tensor<U> x, Tensor<U> y, out Tensor<U> outx, out Tensor<U> outy) where U : struct
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

    public static bool Broadcast<U>(Tensor<U> x, ReadOnlySpan<int> y, out Tensor<U> bx) where U : struct =>
        Broadcast(x, new DenseTensor<U>(y), out bx, out _);

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

    public static bool BroadcastShape<U>(Tensor<U> x, Tensor<U> y, out int[] b) where U : struct => BroadcastShape(x.Dimensions, y.Dimensions, out b);

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
        var output = DenseTensor<U>.OfShape(new int[] { x.Dimensions[0], y.Dimensions[1] });
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

            var bdx = bd.Append(xdl[0]).Append(xdl[1]).ToArray();
            if (!Tensor<U>.Broadcast(x, bdx, out var bx))
            {
                throw new ArgumentException("The tensor shapes are not compatble for broadcasting.");
            }
            var bdy = bd.Append(ydl[0]).Append(ydl[1]).ToArray();
            if (!Tensor<U>.Broadcast(y, bdy, out var by))
            {
                throw new ArgumentException("The tensor shapes are not compatble for broadcasting.");
            }
            var z =  DenseTensor<U>.OfShape(bd.Append(xdl[0]).Append(ydl[1]).ToArray());

            var di = bx.GetDimensionsIterator(0..^2);
            foreach (var _ in di)
            {
                z[di[..]] = Tensor<U>.MatMul2D(bx[di[..]], by[di[..]]);
            }
            return z;
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
                    MathOps.Conv2DD(inputp, N, C, H, W, kH, kW, dilations[0], dilations[1], strides[0], strides[1], info.PadInfo.left, info.PadInfo.top, info.PadInfo.right, info.PadInfo.bottom, group, weightp, null, outputp, Convert.ToInt32(output.Length));
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
                        MathOps.Conv2DD(inputp, N, C, H, W, kH, kW, dilations[0], dilations[1], strides[0], strides[1], info.PadInfo.left, info.PadInfo.top, info.PadInfo.right, info.PadInfo.bottom, group, weightp, null, outputp, Convert.ToInt32(output.Length));
                    }
                }
            }
        }

        return output;

    }

    public static Tensor<U> MaxPool2D<U>(Tensor<U> input, int[] kernelshape, PadType padtype = PadType.Valid, int? padvalue = null, int[] strides = null, int[] dilations = null)
        where U : struct, IComparisonOperators<U, U>, IMinMaxValue<U>
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
        var Y = DenseTensor<U>.OfShape(N, C, info.Shape[0], info.Shape[1]);

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

                        var maxValue = U.MinValue;
                          
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
                            if (maxValue == U.MinValue)
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

    public static Tensor<U> ReLu<U>(Tensor<U> x) where U : struct, IComparisonOperators<U, U>, IAdditiveIdentity<U, U>
       => x.Apply(l => l > U.AdditiveIdentity ? l : U.AdditiveIdentity);


    public static Tensor<T> Reshape(Tensor<T> input,  Tensor<long> shape, bool allowZero = false)
    {
        if (shape.Rank != 1)
        {
            throw new ArgumentException(nameof(shape), "Shape tensors must be of rank 1.");
        }
        if (shape.Any(v => v <-1))
        {
            throw new ArgumentException(nameof(shape), $"A shape dimension cannot be < -1, got {shape.First(v => v < -1)}.");
        }
        if (shape.Count(v => v == -1) > 1)
        {
            throw new ArgumentException(nameof(shape), $"At most 1 shape dimension can be -1.");
        }

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
}
