namespace Lokad.Onnx;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

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
        var broadcastRank = Math.Max(inA.Rank, inB.Rank);
        var outA = inA.Clone();
        var outB = inB.Clone();
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
        return new[] { outA, outB };
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

    public static Tensor<byte> Add(Tensor<byte> x, Tensor<byte> y) => x.Apply((l, r) => (byte) (l + r), y);

    public static Tensor<byte> Add(Tensor<byte> x, byte y) => x.Apply(l => (byte)(l + y));

    public static Tensor<int> Add(Tensor<int> x, Tensor<int> y) => x.Apply((l, r) => l + r, y);

    public static Tensor<int> Add(Tensor<int> x, int y) => x.Apply(l => l + y);

    public static Tensor<float> Add(Tensor<float> x, Tensor<float> y) => x.Apply((l, r) => l + r, y);

    public static Tensor<float> Add(Tensor<float> x, float y) => x.Apply(l => l + y);

    public static Tensor<double> Add(Tensor<double> x, Tensor<double> y) => x.Apply((l, r) => l + r, y);

    public static Tensor<double> Add(Tensor<double> x, double y) => x.Apply(l => l + y);

    public static Tensor<byte> Subtract(Tensor<byte> x, Tensor<byte> y) => x.Apply((l, r) => (byte)(l - r), y);

    public static Tensor<byte> Subtract(Tensor<byte> x, byte y) => x.Apply(l => (byte)(l - y));

    public static Tensor<float> Subtract(Tensor<float> x, Tensor<float> y) => x.Apply((l, r) => l - r, y);

    public static Tensor<float> Subtract(Tensor<float> x, float y) => x.Apply(l => l - y);

    public static Tensor<double> Subtract(Tensor<double> x, Tensor<double> y) => x.Apply((l, r) => l - r, y);

    public static Tensor<double> Subtract(Tensor<double> x, double y) => x.Apply(l => l - y);

    public static Tensor<int> Subtract(Tensor<int> x, Tensor<int> y) => x.Apply((l, r) => l - r, y);

    public static Tensor<int> Subtract(Tensor<int> x, int y) => x.Apply(l => l - y);

    public static Tensor<byte> Multiply(Tensor<byte> x, Tensor<byte> y) => x.Apply((l, r) => (byte)(l * r), y);

    public static Tensor<byte> Multiply(Tensor<byte> x, byte y) => x.Apply(l => (byte)(l * y));

    public static Tensor<int> Multiply(Tensor<int> x, Tensor<int> y) => x.Apply((l, r) => l * r, y);

    public static Tensor<int> Multiply(Tensor<int> x, int y) => x.Apply(l => l * y);

    public static Tensor<float> Multiply(Tensor<float> x, Tensor<float> y) => x.Apply((l, r) => l * r, y);

    public static Tensor<float> Multiply(Tensor<float> x, float y) => x.Apply(l => l * y);

    public static Tensor<double> Multiply(Tensor<double> x, Tensor<double> y) => x.Apply((l, r) => l * r, y);

    public static Tensor<double> Multiply(Tensor<double> x, double y) => x.Apply(l => l * y);

    public static Tensor<byte> Divide(Tensor<byte> x, Tensor<byte> y) => x.Apply((l, r) => (byte)(l / r), y);

    public static Tensor<byte> Divide(Tensor<byte> x, byte y) => x.Apply(l => (byte)(l / y));

    public static Tensor<float> Divide(Tensor<float> x, Tensor<float> y) => x.Apply((l, r) => l / r, y);

    public static Tensor<float> Divide(Tensor<float> x, float y) => x.Apply(l => l / y);

    public static Tensor<double> Divide(Tensor<double> x, Tensor<double> y) => x.Apply((l, r) => l / r, y);

    public static Tensor<double> Divide(Tensor<double> x, double y) => x.Apply(l => l / y);

    public static Tensor<float> Negate(Tensor<float> x) => x.Apply(l => -l);

    public static Tensor<double> Negate(Tensor<double> x) => x.Apply(l => -l);

    public static Tensor<float> Square(Tensor<float> x) => x.Apply(l => l * l);

    public static Tensor<double> Square(Tensor<double> x) => x.Apply(l => l * l);

    public static Tensor<float> Abs(Tensor<float> x) => x.Apply(l => l >= 0.0f ? l : -l);

    public static Tensor<double> Abs(Tensor<double> x) => x.Apply(l => l >= 0.0 ? l : -l);

    public static Tensor<float> Sqrt(Tensor<float> x) => x.Apply(MathF.Sqrt);

    public static Tensor<double> Sqrt(Tensor<double> x) => x.Apply(Math.Sqrt);

    public static Tensor<float> MatMul2D(Tensor<float> x, Tensor<float> y)
    {
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
        var n = x.Dimensions[0];
        var m = x.Dimensions[1];
        var r = y.Dimensions[1];
        var output = DenseTensor<float>.OfShape(new int[] { x.Dimensions[0], y.Dimensions[1] });
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < r; j++)
            {
                var sum = 0.0f;
                for (int k = 0; k < m; k++)
                    sum += x[i, k] * y[k, j];
                output[i, j] = sum;
            }

        }
        return output;
    }

    public static Tensor<double> MatMul2D(Tensor<double> x, Tensor<double> y)
    {
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
        var n = x.Dimensions[0];
        var m = x.Dimensions[1];
        var r = y.Dimensions[1];
        var output = DenseTensor<double>.OfShape(new int[] { x.Dimensions[0], y.Dimensions[1] });
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < r; j++)
            {
                var sum = 0.0;
                for (int k = 0; k < m; k++)
                    sum += x[i, k] * y[k, j];
                output[i, j] = sum;
            }

        }
        return output;
    }

    public static Tensor<int> MatMul2D(Tensor<int> x, Tensor<int> y)
    {
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
        var n = x.Dimensions[0];
        var m = x.Dimensions[1];
        var r = y.Dimensions[1];
        var output = DenseTensor<int>.OfShape(new int[] { x.Dimensions[0], y.Dimensions[1] });
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < r; j++)
            {
                var sum = 0;
                for (int k = 0; k < m; k++)
                    sum += x[i, k] * y[k, j];
                output[i, j] = sum;
            }

        }
        return output;
    }

    public static Tensor<float> MatMul(Tensor<float> x, Tensor<float> y)
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
                throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
            }

            if (!BroadcastShape(x.Dimensions[0..^2], y.Dimensions[0..^2], out var bd))
            {
                throw new ArgumentException("The tensor shapes are not compatble for broadcasting.");
            }

            var bdx = bd.Append(xdl[0]).Append(xdl[1]).ToArray();
            if (!Tensor<float>.Broadcast(x, bdx, out var bx))
            {
                throw new ArgumentException("The tensor shapes are not compatble for broadcasting.");
            }
            var bdy = bd.Append(ydl[0]).Append(ydl[1]).ToArray();
            if (!Tensor<float>.Broadcast(y, bdy, out var by))
            {
                throw new ArgumentException("The tensor shapes are not compatble for broadcasting.");
            }
            var z = DenseTensor<float>.OfShape(bd.Append(xdl[0]).Append(ydl[1]).ToArray());

            var di = bx.GetDimensionsIterator(0..^2);
            foreach (var _ in di)
            {
                z[di[..]] = Tensor<float>.MatMul2D(bx[di[..]], by[di[..]]);
            }
            return z;
        }
        else if (x.Rank >= 2 || y.Rank >= 2)
        {
            var b = Tensor<float>.Broadcast(x, y);
            if (!b.Any())
            {
                throw new ArgumentException($"The shapes {x.PrintShape()} and {y.PrintShape()} are not compatible for broadcasting.");
            }
            else
            {
                x = b[0];
                y = b[1];
                var c = x.CloneEmpty();
                var di = x.GetDimensionsIterator(0..^2);
                foreach (var _ in di)
                {
                    c[di[..]] = Tensor<float>.MatMul2D(x[di[..]], y[di[..]]);
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
                throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
            }

            if (!BroadcastShape(x.Dimensions[0..^2], y.Dimensions[0..^2], out var bd))
            {
                throw new ArgumentException("The tensor shapes are not compatble for broadcasting.");
            }

            var bdx = bd.Append(xdl[0]).Append(xdl[1]).ToArray();
            if (!Tensor<double>.Broadcast(x, bdx, out var bx))
            {
                throw new ArgumentException("The tensor shapes are not compatble for broadcasting.");
            }
            var bdy = bd.Append(ydl[0]).Append(ydl[1]).ToArray();
            if (!Tensor<double>.Broadcast(y, bdy, out var by))
            {
                throw new ArgumentException("The tensor shapes are not compatble for broadcasting.");
            }
            var z = DenseTensor<double>.OfShape(bd.Append(xdl[0]).Append(ydl[1]).ToArray());

            var di = bx.GetDimensionsIterator(0..^2);
            foreach (var _ in di)
            {
                z[di[..]] = Tensor<double>.MatMul2D(bx[di[..]], by[di[..]]);
            }
            return z;
        }
        else if (x.Rank >= 2 || y.Rank >= 2)
        {
            var b = Tensor<double>.Broadcast(x, y);
            if (!b.Any())
            {
                throw new ArgumentException($"The shapes {x.PrintShape()} and {y.PrintShape()} are not compatible for broadcasting.");
            }
            else
            {
                x = b[0];
                y = b[1];
                var c = x.CloneEmpty();
                var di = x.GetDimensionsIterator(0..^2);
                foreach (var _ in di)
                {
                    c[di[..]] = Tensor<double>.MatMul2D(x[di[..]], y[di[..]]);
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
                throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
            }

            if (!BroadcastShape(x.Dimensions[0..^2], y.Dimensions[0..^2], out var bd))
            {
                throw new ArgumentException("The tensor shapes are not compatble for broadcasting.");
            }

            var bdx = bd.Append(xdl[0]).Append(xdl[1]).ToArray();
            if (!Tensor<int>.Broadcast(x, bdx, out var bx))
            {
                throw new ArgumentException("The tensor shapes are not compatble for broadcasting.");
            }
            var bdy = bd.Append(ydl[0]).Append(ydl[1]).ToArray();
            if (!Tensor<int>.Broadcast(y, bdy, out var by))
            {
                throw new ArgumentException("The tensor shapes are not compatble for broadcasting.");
            }
            var z = DenseTensor<int>.OfShape(bd.Append(xdl[0]).Append(ydl[1]).ToArray());

            var di = bx.GetDimensionsIterator(0..^2);
            foreach (var _ in di)
            {
                z[di[..]] = Tensor<int>.MatMul2D(bx[di[..]], by[di[..]]);
            }
            return z;
        }
        else if (x.Rank >= 2 || y.Rank >= 2)
        {
            var b = Tensor<int>.Broadcast(x, y);
            if (!b.Any())
            {
                throw new ArgumentException($"The shapes {x.PrintShape()} and {y.PrintShape()} are not compatible for broadcasting.");
            }
            else
            {
                x = b[0];
                y = b[1];
                var c = x.CloneEmpty();
                var di = x.GetDimensionsIterator(0..^2);
                foreach (var _ in di)
                {
                    c[di[..]] = Tensor<int>.MatMul2D(x[di[..]], y[di[..]]);
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

    public static Tensor<float> Softmax(Tensor<float> x) 
    {
        var t = x.Apply(MathF.Exp).Accumulate((l, r) => l + r, 0.0f);
        return x.Apply(i => MathF.Exp(i) / t);
    }

    public static Tensor<double> Softmax(Tensor<double> x)
    {
        var t = x.Apply(Math.Exp).Accumulate((l, r) => l + r, 0.0);
        return x.Apply(i => Math.Exp(i) / t);
    }

    public static Tensor<float> Erf(Tensor<float> x) => x.Apply(MathOps.Erf);

    public static Tensor<double> Erf(Tensor<double> x) => x.Apply(MathOps.Erf);

    public static Tensor<T> Transpose(Tensor<T> data, int[] perm = null)
    {
        if (perm != null && perm.Length != data.Rank)
        {
            throw new ArgumentException(nameof(perm), $"The size of the permutation array must be the rank of the tensor: {data.Rank}.");
        }
        if (perm != null && !perm.All(p => p < data.Rank))
        {
            throw new ArgumentException(nameof(perm), $"The permuted dimension {perm.First(p => p >= data.Rank)} exceeds the number of dimensions in the tensor.");
        }
        if (perm != null && !ArrayUtilities.CheckNoRepeatedDims(perm))
        {
            throw new ArgumentException(nameof(perm), "The permutation array has a repeated dimension.");
        }

        if (perm is null)
        {
            perm = data.dimensions.Select((_, n) => n).Reverse().ToArray();
        }
        else
        {
            perm = perm.Select(p => ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, p)).ToArray();
        }

        if (data.Rank <= 1)
        {
            return data.Clone();
        }
        var shape = perm.Select((i, _) => data.dimensions[i]).ToArray();
        var r = DenseTensor<T>.OfShape(shape);
        var di = data.GetDimensionsIterator();
        foreach (var index in di)
        {
            var permindex = index.Select((_,n) => index[perm[n]]).ToArray();
            r[permindex] = data[index];
        }
        return r;
    }

    public static Tensor<T> Gather(Tensor<T> data, Tensor<int> indices, int? _axis = null)
    {
        if (data.Rank == 0) throw new ArgumentException(nameof (data), "Cannot gather from a tensor of rank 0.");
        var axis = _axis.HasValue ? ArrayUtilities.HandleNegativeAxisOrIndex(data.Rank, _axis.Value) : 0;    
        if (axis > data.Rank - 1)
        {
            throw new ArgumentException(nameof(axis), $"The specified axis {_axis} exceeds the number of dimensions in the tensor.");
        }

        List<int> shape = new List<int>(data.Rank - 1 + indices.Rank);
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
        foreach (var di in output.GetDimensionsIterator())
        {
            var a = di[0..axis];
            var k = ArrayUtilities.HandleNegativeAxisOrIndex(data.dimensions[axis], indices[di[axis..(axis + indices.Rank)]]);
            var b = di[(axis + indices.Rank)..].ToArray();
            var oloc = a.Append(k).Concat(b).ToArray();
            output[di] = data[oloc];
        }
        return output;  
    }

    public static Tensor<T> Concat(Tensor<T> x, Tensor<T> y, int axis)
    {
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
        if (inputs.Length < 2) throw new ArgumentException(nameof(inputs), "At least two tensors must be specified for the concat operation.");
        if (!inputs.All(i => i.Rank == inputs[0].Rank)) throw new ArgumentException(nameof(inputs), $"Each input tensor in a concat operation must be of the same rank.");
        if (!inputs.All(i => i.dimensions.Select((d, n) => n == axis ? 0 : d - inputs[0].dimensions[n]).All(s => s == 0)))
            throw new ArgumentException(nameof(inputs), "The dimensions of each tensor in a concat operation must be the same, with the exception of the axis dimension.");
        Tensor<T> output = inputs[0];
        for (int i = 1; i < inputs.Length; i++) 
        {
            output = Concat(output, inputs[i], axis);
        }
        return output;
    }

    public static Tensor<T> Slice(Tensor<T> data, Tensor<int> start, Tensor<int> ends, Tensor<int> axes = null, Tensor<int> steps = null)
    {
        if (data.Rank == 0) throw new ArgumentException(nameof(data), "Cannot slice a tensor of rank 0.");
        if (start.Rank != 1) throw new ArgumentException(nameof(start), "The rank of the start tensor must be 1.");
        if (start.Length > data.Rank) throw new ArgumentException(nameof(start), "The length of the start tensor must be less-than or equal to the rank of the data tensor.");
        if (ends.Rank != 1) throw new ArgumentException(nameof(start), "The rank of the end tensor must be 1.");
        if (start.Length != ends.Length) throw new ArgumentException(nameof(ends), "The end tensor must be the same length as the start tensor.");
        if (axes is not null && (axes.Rank != 1 || axes.Length != start.Length)) throw new ArgumentException(nameof(axes), "The axes tensor must be a rank 1 tensor with the same length as the start tensor.");
        if (steps is not null && (steps.Rank != 1 || steps.Length != start.Length)) throw new ArgumentException(nameof(steps), "The steps tensor must be a rank 1 tensor with the same length as the start tensor.");

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
            indices[i] = axes.Contains(i) ? new SliceIndex(start[i], ends[i], steps[i]) : ..;
        }
        return data.Slice(indices); 
    }
}
