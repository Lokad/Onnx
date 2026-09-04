using System;
using System.Linq;

namespace Lokad.Onnx.Tensors.Tests;

public class TensorCoreOptionsTests
{
    static Tensor<float> SmallFloat(int rows, int cols, float start = 1f)
    {
        var t = Tensor<float>.Zeros(rows, cols);
        float v = start;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t[i, j] = v++;
        return t;
    }

    static Tensor<int> SmallInt(int rows, int cols, int start = 1)
    {
        var t = Tensor<int>.Zeros(rows, cols);
        int v = start;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t[i, j] = v++;
        return t;
    }

    static Tensor<double> SmallDouble(int rows, int cols, double start = 1.0)
    {
        var t = Tensor<double>.Zeros(rows, cols);
        double v = start;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t[i, j] = v++;
        return t;
    }

    static void AssertFloatEqual(Tensor<float> expected, Tensor<float> actual, int precision = 5)
    {
        Assert.Equal(expected.Dimensions.ToArray(), actual.Dimensions.ToArray());
        var e = expected.ToArray();
        var a = actual.ToArray();
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++)
            Assert.Equal(e[i], a[i], precision);
    }

    static void AssertDoubleEqual(Tensor<double> expected, Tensor<double> actual, int precision = 10)
    {
        Assert.Equal(expected.Dimensions.ToArray(), actual.Dimensions.ToArray());
        var e = expected.ToArray();
        var a = actual.ToArray();
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++)
            Assert.Equal(e[i], a[i], precision);
    }

    static void AssertIntEqual(Tensor<int> expected, Tensor<int> actual)
    {
        Assert.Equal(expected.Dimensions.ToArray(), actual.Dimensions.ToArray());
        Assert.Equal(expected.ToArray(), actual.ToArray());
    }

    [Fact]
    public void ElementwiseCore_ExplicitOptionsMatchDefault()
    {
        var x = SmallFloat(2, 3);
        var y = SmallFloat(2, 3, 10f);
        var scalar = Tensor<float>.Zeros(2, 3);
        var simd = Tensor<float>.Zeros(2, 3);
        x.VectorizedApply((l, r) => l + r, (l, r) => l + r, y, scalar, TensorExecutionOptions.Scalar);
        x.VectorizedApply((l, r) => l + r, (l, r) => l + r, y, simd, TensorExecutionOptions.Simd);
        var @default = Tensor<float>.Add(x, y);
        AssertFloatEqual(@default, scalar);
        AssertFloatEqual(@default, simd);
    }

    [Fact]
    public void IntMatMul2D_ExplicitOptionsMatchDefault()
    {
        var a = SmallInt(2, 3);
        var b = SmallInt(3, 2, 10);
        var expected = Tensor<int>.MatMul2D(a, b);
        AssertIntEqual(expected, Tensor<int>.MatMul2D(a, b, TensorExecutionOptions.Scalar));
        AssertIntEqual(expected, Tensor<int>.MatMul2D(a, b, TensorExecutionOptions.Simd));
    }

    [Fact]
    public void DoubleMatMul2D_ExplicitOptionsMatchDefault()
    {
        var a = SmallDouble(2, 3);
        var b = SmallDouble(3, 2, 0.5);
        var expected = Tensor<double>.MatMul2D(a, b);
        AssertDoubleEqual(expected, Tensor<double>.MatMul2D(a, b, TensorExecutionOptions.Scalar));
        AssertDoubleEqual(expected, Tensor<double>.MatMul2D(a, b, TensorExecutionOptions.Simd));
    }

    [Fact]
    public void IntNDMatMul_ExplicitOptionsMatchDefault()
    {
        var a = Tensor<int>.Zeros(2, 2, 2);
        var b = Tensor<int>.Zeros(1, 2, 2);
        for (int i = 0; i < 8; i++) { a[i / 4, (i / 2) % 2, i % 2] = i + 1; b[0, (i / 2) % 2, i % 2] = i + 1; }
        var expected = Tensor<int>.MatMul(a, b);
        AssertIntEqual(expected, Tensor<int>.MatMul(a, b, TensorExecutionOptions.Scalar));
        AssertIntEqual(expected, Tensor<int>.MatMul(a, b, TensorExecutionOptions.Simd));
    }

    [Fact]
    public void ReduceMean_ExplicitOptionsMatchDefault()
    {
        var data = SmallFloat(2, 3);
        var axes = new int[] { 1 }.ToTensor<int>();
        foreach (bool? keepDims in new bool?[] { true, false, null })
        {
            var expected = Tensor<float>.ReduceMean(data, axes, keepDims, false);
            AssertFloatEqual(expected, Tensor<float>.ReduceMean(data, axes, keepDims, false, TensorExecutionOptions.Scalar));
            AssertFloatEqual(expected, Tensor<float>.ReduceMean(data, axes, keepDims, false, TensorExecutionOptions.Simd));
        }
        var idata = SmallInt(2, 3);
        var iexpected = Tensor<int>.ReduceMean(idata, axes, true, false);
        AssertIntEqual(iexpected, Tensor<int>.ReduceMean(idata, axes, true, false, TensorExecutionOptions.Scalar));
    }

    [Fact]
    public void DoubleNDMatMul_ExplicitOptionsMatchDefault()
    {
        var a = Tensor<double>.Zeros(2, 2, 2);
        var b = Tensor<double>.Zeros(2, 2, 2);
        for (int i = 0; i < 8; i++) { a[i / 4, (i / 2) % 2, i % 2] = i + 1; b[i / 4, (i / 2) % 2, i % 2] = 0.5 * (i + 1); }
        var expected = Tensor<double>.MatMul(a, b);
        AssertDoubleEqual(expected, Tensor<double>.MatMul(a, b, TensorExecutionOptions.Scalar));
        AssertDoubleEqual(expected, Tensor<double>.MatMul(a, b, TensorExecutionOptions.Simd));
    }
}
