using System;
using System.Linq;

namespace Lokad.Onnx.Tensors.Tests;

public class TensorOpsReductionTests
{
    [Fact]
    public void ReduceSum_Mean_Max_Int()
    {
        var data = DenseTensor<int>.OfValues(new int[2, 2] { { 1, 2 }, { 3, 4 } });
        var axes = new int[] { 0 }.ToTensor<int>();

        var sum = Tensor<int>.ReduceSum(data, axes);
        var mean = Tensor<int>.ReduceMean(data, axes);

        Assert.Equal(new[] { 2 }, sum.Dimensions.ToArray());
        Assert.Equal(4, sum[0]);
        Assert.Equal(6, sum[1]);
        // Integer means sum before dividing (verified against ORT 1.29:
        // column sums [4,6] average to [2,3], not divide-first [1,3]).
        Assert.Equal(2, mean[0]);
        Assert.Equal(3, mean[1]);
    }

    [Fact]
    public void ReduceSum_Mean_Max_Float()
    {
        var data = DenseTensor<float>.OfValues(new float[2, 3] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        var axes = new int[] { 1 }.ToTensor<int>();

        var sum = Tensor<float>.ReduceSum(data, axes);
        var mean = Tensor<float>.ReduceMean(data, axes);
        var max = Tensor<float>.ReduceMax(data, axes);

        Assert.Equal(new[] { 2 }, sum.Dimensions.ToArray());
        Assert.Equal(6f, sum[0], 5);
        Assert.Equal(15f, sum[1], 5);
        Assert.Equal(2f, mean[0], 5);
        Assert.Equal(5f, mean[1], 5);
        Assert.Equal(3f, max[0], 5);
        Assert.Equal(6f, max[1], 5);
    }

    [Fact]
    public void ReduceSum_Mean_Max_Double()
    {
        var data = DenseTensor<double>.OfValues(new double[2, 2] { { 1d, 2d }, { 3d, 4d } });
        var axes = new int[] { 1 }.ToTensor<int>();

        var sum = Tensor<double>.ReduceSum(data, axes);
        var mean = Tensor<double>.ReduceMean(data, axes);
        var max = Tensor<double>.ReduceMax(data, axes);

        Assert.Equal(new[] { 2 }, sum.Dimensions.ToArray());
        Assert.Equal(3d, sum[0], 10);
        Assert.Equal(7d, sum[1], 10);
        Assert.Equal(1.5d, mean[0], 10);
        Assert.Equal(3.5d, mean[1], 10);
        Assert.Equal(2d, max[0], 10);
        Assert.Equal(4d, max[1], 10);
    }

    [Fact]
    public void ExpVector_MatchesScalarExp()
    {
        var rnd = new System.Random(20260905);
        var values = new float[2048];
        for (int i = 0; i < values.Length; i++) values[i] = (float)(rnd.NextDouble() * 200.0 - 100.0);
        int width = System.Numerics.Vector<float>.Count;
        Span<float> buf = stackalloc float[System.Numerics.Vector<float>.Count];
        for (int i = 0; i + width <= values.Length; i += width)
        {
            MathOps.ExpVector(new System.Numerics.Vector<float>(values, i)).CopyTo(buf);
            var got = buf;
            for (int j = 0; j < width; j++)
            {
                float expected = MathF.Exp(values[i + j]);
                if (float.IsNaN(expected)) { Assert.True(float.IsNaN(got[j])); continue; }
                if (float.IsInfinity(expected)) { Assert.Equal(expected, got[j]); continue; }
                if (System.Math.Abs(expected) > 1e-30f) Assert.True(System.Math.Abs(got[j] - expected) <= 1e-6f * System.Math.Abs(expected));
                else Assert.True(System.Math.Abs(got[j] - expected) <= 1e-36f);
            }
        }
        Assert.True(float.IsNaN(System.Numerics.Vector.GetElement(MathOps.ExpVector(new System.Numerics.Vector<float>(float.NaN)), 0)));
        Assert.Equal(float.PositiveInfinity, System.Numerics.Vector.GetElement(MathOps.ExpVector(new System.Numerics.Vector<float>(float.PositiveInfinity)), 0));
        Assert.Equal(0f, System.Numerics.Vector.GetElement(MathOps.ExpVector(new System.Numerics.Vector<float>(float.NegativeInfinity)), 0));
        Assert.Equal(1f, System.Numerics.Vector.GetElement(MathOps.ExpVector(System.Numerics.Vector<float>.Zero), 0));
    }

    [Fact]
    public void Softmax_MatchesScalarReference_OnWideRows()
    {
        var rnd = new System.Random(20260905);
        var data = new float[3, 257];
        for (int i = 0; i < 3; i++) for (int j = 0; j < 257; j++) data[i, j] = (float)(rnd.NextDouble() * 20.0 - 10.0);
        var output = Tensor<float>.Softmax(DenseTensor<float>.OfValues(data), 1, null, 13);
        for (int i = 0; i < 3; i++)
        {
            float max = float.NegativeInfinity;
            for (int j = 0; j < 257; j++) max = System.Math.Max(max, data[i, j]);
            float sum = 0f;
            for (int j = 0; j < 257; j++) sum += MathF.Exp(data[i, j] - max);
            for (int j = 0; j < 257; j++) Assert.Equal(MathF.Exp(data[i, j] - max) / sum, output[i, j], 6);
        }
        var nan = Tensor<float>.Softmax(DenseTensor<float>.OfValues(new float[1, 10] { { 0f, 1f, float.NaN, 3f, 4f, 5f, 6f, 7f, 8f, 9f } }), -1, null, 13);
        for (int j = 0; j < 10; j++) Assert.True(float.IsNaN(nan[0, j]));
    }

    [Fact]
    public void Softmax_NormalizesAlongAxis()
    {
        var data = DenseTensor<float>.OfValues(new float[2, 2] { { 0f, 1f }, { -1f, 1f } });
        var output = Tensor<float>.Softmax(data, 1, null, 13);

        var expected0 = MathF.Exp(0f) / (MathF.Exp(0f) + MathF.Exp(1f));
        var expected1 = MathF.Exp(1f) / (MathF.Exp(0f) + MathF.Exp(1f));
        var expected2 = MathF.Exp(-1f) / (MathF.Exp(-1f) + MathF.Exp(1f));
        var expected3 = MathF.Exp(1f) / (MathF.Exp(-1f) + MathF.Exp(1f));

        Assert.Equal(expected0, output[0, 0], 5);
        Assert.Equal(expected1, output[0, 1], 5);
        Assert.Equal(expected2, output[1, 0], 5);
        Assert.Equal(expected3, output[1, 1], 5);
        Assert.Equal(1f, output[0, 0] + output[0, 1], 5);
        Assert.Equal(1f, output[1, 0] + output[1, 1], 5);
    }

    [Fact]
    public void Softmax_NonLastAxis_MatchesOrt()
    {
        // ORT 1.29 on arange(24) shaped [2, 3, 4]: strided reduction paths.
        var data = new float[2, 3, 4];
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) for (int k = 0; k < 4; k++) data[i, j, k] = i * 12 + j * 4 + k;
        var x = DenseTensor<float>.OfValues(data);
        var e0 = new float[24];
        for (int i = 0; i < 12; i++) e0[i] = 6.1441742E-06f;
        for (int i = 12; i < 24; i++) e0[i] = 0.9999938f;
        AssertStage(Tensor<float>.Softmax(x, 0, null, 13).ToArray(), e0);
        var e1 = new float[24];
        for (int b = 0; b < 2; b++)
        {
            for (int j = 0; j < 4; j++) e1[b * 12 + j] = 3.2932041E-04f;
            for (int j = 4; j < 8; j++) e1[b * 12 + j] = 0.017980287f;
            for (int j = 8; j < 12; j++) e1[b * 12 + j] = 0.98169035f;
        }
        AssertStage(Tensor<float>.Softmax(x, 1, null, 13).ToArray(), e1);
        AssertStage(Tensor<float>.Softmax(x, -2, null, 13).ToArray(), e1);
    }

    static void AssertStage(float[] got, float[] expected)
    {
        Assert.Equal(expected.Length, got.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], got[i], 6);
    }
}
