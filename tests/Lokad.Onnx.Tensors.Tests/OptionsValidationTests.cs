using System;
using System.Numerics;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Tensors.Tests;

public class OptionsValidationTests
{
    static TensorExecutionOptions Invalid(int kind) => kind switch
    {
        0 => new(true, false, 0),
        1 => new(false, true, 1),
        _ => TensorExecutionOptions.Intrinsics
    };

    static void Reject(Action operation, int kind)
    {
        var error = Record.Exception(operation);
        Assert.NotNull(error);
        if (kind == 0) Assert.IsType<ArgumentOutOfRangeException>(error);
        else if (kind == 1) Assert.IsType<ArgumentException>(error);
        else Assert.IsType<InvalidOperationException>(error);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void MatMul_AllRanksAndTypesRejectBeforeWritesOrRentals(int kind)
    {
        if (kind == 2 && Fma.IsSupported) return;
        var options = Invalid(kind);
        foreach (int rank in new[] { 1, 2, 3 })
        {
            int[] shape = rank == 1 ? new[] { 2 } : rank == 2 ? new[] { 2, 2 } : new[] { 2, 2, 2 };
            var f = DenseTensor<float>.Ones(shape);
            var d = DenseTensor<double>.Ones(shape);
            var n = DenseTensor<int>.Ones(shape);
            Reject(() => Tensor<float>.MatMul(f, f, options), kind);
            Reject(() => Tensor<double>.MatMul(d, d, options), kind);
            Reject(() => Tensor<int>.MatMul(n, n, options), kind);
            int[] outputShape = rank == 1 ? Array.Empty<int>() : shape;
            var destination = DenseTensor<float>.OfShape(outputShape);
            destination.Fill(731);
            Reject(() => Tensor<float>.MatMul(f, f, destination, options), kind);
            Assert.All(destination.ToArray(), value => Assert.Equal(731, value));
            var pool = new TensorBufferPool();
            Reject(() => Tensor<float>.MatMul(f, f, options, pool), kind);
            Assert.Equal(0, pool.AllocatedNew);
            if (rank != 2) continue;
            Reject(() => Tensor<float>.MatMul2D(f, f, options), kind);
            Reject(() => Tensor<double>.MatMul2D(d, d, options), kind);
            Reject(() => Tensor<int>.MatMul2D(n, n, options), kind);
            Reject(() => Tensor<float>.MatMul2D(f, f, options, pool), kind);
            Assert.Equal(0, pool.AllocatedNew);
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void Elementwise_AllLayoutsRejectBeforeWrites(int kind)
    {
        if (kind == 2 && Fma.IsSupported) return;
        var options = Invalid(kind);
        foreach (int layout in new[] { 0, 1, 2 })
        {
            var dense = new DenseTensor<float>(new[] { 2, 8 }, layout == 1);
            dense.Fill(4);
            Tensor<float> input = layout == 2 ? dense.Slice(SliceIndex.All, new SliceIndex(1, 7)) : dense;
            var output = DenseTensor<float>.OfShape(input.Dimensions.ToArray());
            output.Fill(731);
            Reject(() => input.VectorizedApply(Vector.SquareRoot, MathF.Sqrt, output, options), kind);
            Assert.All(output.ToArray(), value => Assert.Equal(731, value));
            Reject(() => input.VectorizedApply(Vector.Add, static (a, b) => a + b, input, output, options), kind);
            Assert.All(output.ToArray(), value => Assert.Equal(731, value));
            Reject(() => Tensor<float>.Sqrt(input, options), kind);
            Reject(() => input.VectorizedApply(Vector.Add, static (a, b) => a + b, input, options), kind);
        }
    }
}
