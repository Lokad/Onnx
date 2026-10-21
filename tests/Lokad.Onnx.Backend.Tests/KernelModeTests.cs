using System;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class KernelModeTests
{
    static float TanhGelu(float v)
    {
        float a = .5f * v;
        float b = .044715f * v * v * v;
        float c = .7978846f * (v + b);
        return a * (MathF.Tanh(c) + 1f);
    }

    static void Bits(float[] expected, Tensor<float> actual) =>
        Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), actual.ToArray().Select(BitConverter.SingleToInt32Bits));

    [Fact]
    public void TanhGelu_PortableModeAndScalarExceptionalValuesKeepTheirKernels()
    {
        var values = Enumerable.Range(0, 128).Select(i => (i - 64) / 17f).ToArray();
        var portable = new float[values.Length];
        Tensor<float>.GeluTanhSpanFloatReference(values, portable);
        Bits(portable, Tensor<float>.GeluTanh(new DenseTensor<float>(values, new[] { 128 }), TensorExecutionOptions.Simd));
        float[] special = { -0f, 0f, float.NaN, float.NegativeInfinity, float.PositiveInfinity, -1e-30f, 1e-30f, float.MaxValue };
        var actual = Tensor<float>.GeluTanh(new DenseTensor<float>(special, new[] { 8 }), TensorExecutionOptions.Scalar).ToArray();
        for (int i = 0; i < special.Length; i++)
        {
            float expected = TanhGelu(special[i]);
            if (float.IsNaN(expected)) Assert.True(float.IsNaN(actual[i]));
            else Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(actual[i]));
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ScalarFusedRelu_PreservesFloatingValuesAndRentsOneOutput(bool pooled)
    {
        float[] values = { -0f, 0f, float.NaN, float.NegativeInfinity, float.PositiveInfinity, -1e-30f, 1e-30f, -2 };
        var input = new DenseTensor<float>(values, new[] { 8 });
        var bias = DenseTensor<float>.Scalar(-0f);
        var added = CPUExecutionProvider.Add(input, bias, ExecutionOptions.Scalar, null);
        var expected = (Tensor<float>)CPUExecutionProvider.Relu(added.Outputs[0], ExecutionOptions.Scalar).Outputs[0];
        var pool = pooled ? new TensorBufferPool() : null;
        var actual = CPUExecutionProvider.AddRelu(input, bias, ExecutionOptions.Scalar, pool);
        Bits(expected.ToArray(), (Tensor<float>)actual.Outputs[0]);
        if (pool is not null) Assert.Equal(1, pool.AllocatedNew);
    }

    [Theory]
    [InlineData(7)]
    [InlineData(32)]
    [InlineData(129)]
    public void ScalarTanhGelu_UsesScalarFormulaAcrossLayoutsAndOverlap(int length)
    {
        var values = Enumerable.Range(0, length * 2).Select(i => (i - length) / 17f).ToArray();
        var expected = values.Select(TanhGelu).ToArray();
        foreach (bool reversed in new[] { false, true })
        {
            var root = new DenseTensor<float>(new[] { 2, length }, reversed);
            int index = 0;
            foreach (var coordinate in root.GetDimensionsIterator()) root[coordinate] = values[index++];
            foreach (Tensor<float> input in new Tensor<float>[] { root, root.Slice(SliceIndex.All, SliceIndex.All) })
            {
                Bits(expected, Tensor<float>.GeluTanh(input, TensorExecutionOptions.Scalar));
                Bits(expected, Tensor<float>.GeluTanh(input, new DenseTensor<float>(root.Dimensions, reversed), TensorExecutionOptions.Scalar));
                Bits(expected, (Tensor<float>)CPUExecutionProvider.Gelu(input, "tanh", ExecutionOptions.Scalar, new TensorBufferPool()).Outputs[0]);
            }
        }
        var store = new float[values.Length + 1];
        values.CopyTo(store, 0);
        var source = new DenseTensor<float>(store.AsMemory(0, values.Length), new[] { 2, length });
        var destination = new DenseTensor<float>(store.AsMemory(1, values.Length), source.Dimensions);
        Bits(expected, Tensor<float>.GeluTanh(source, destination, TensorExecutionOptions.Scalar));
    }

    [Fact]
    public void ScalarLayerNorm_UsesSequentialStatisticsWithOrWithoutPool()
    {
        var values = Enumerable.Repeat(1f, 32).ToArray();
        values[0] = 1e20f; values[2] = -1e20f;
        double mean = 0;
        foreach (float value in values) mean += value;
        mean /= values.Length;
        double variance = 0;
        foreach (float value in values) { double delta = value - mean; variance += delta * delta; }
        variance /= values.Length;
        double inverse = 1.0 / Math.Sqrt(variance + 1e-5f);
        var expected = values.Select(value => (float)((value - mean) * inverse)).ToArray();
        foreach (bool pooled in new[] { false, true })
        foreach (int outputCount in new[] { 1, 2, 3 })
        {
            var result = CPUExecutionProvider.LayerNormalization(new DenseTensor<float>(values, new[] { 1, 32 }),
                DenseTensor<float>.Ones(32), null, -1, 1e-5f, 1, outputCount, ExecutionOptions.Scalar, pooled ? new TensorBufferPool() : null);
            Assert.Equal(OpStatus.Success, result.Status);
            Bits(expected, (Tensor<float>)result.Outputs[0]);
            if (outputCount >= 2) Assert.Equal((float)mean, result.Outputs[1].GetValue(0));
            if (outputCount >= 3) Assert.Equal((float)inverse, result.Outputs[2].GetValue(0));
        }
    }

    [Theory]
    [InlineData("none")]
    [InlineData("tanh")]
    public void ScalarGemmGelu_MatchesSeparateScalarOperations(string approximation)
    {
        var a = new DenseTensor<float>(new float[] { 1, 0, 0, 1 }, new[] { 2, 2 });
        var b = new DenseTensor<float>(Enumerable.Range(0, 64).Select(i => (i - 32) / 17f).ToArray(), new[] { 2, 32 });
        var gemm = CPUExecutionProvider.Gemm(a, b, null, 1, 0, ExecutionOptions.Scalar, 0, 0);
        var expected = (Tensor<float>)CPUExecutionProvider.Gelu(gemm.Outputs[0], approximation, ExecutionOptions.Scalar, null).Outputs[0];
        // The independent scalar formula also catches the shared GELU-tanh dispatch defect.
        if (approximation == "tanh") expected = new DenseTensor<float>(((Tensor<float>)gemm.Outputs[0]).ToArray().Select(TanhGelu).ToArray(), b.Dimensions);
        var fused = CPUExecutionProvider.GemmGelu(a, b, null, 1, 0, ExecutionOptions.Scalar, 0, 0, approximation);
        Bits(expected.ToArray(), (Tensor<float>)fused.Outputs[0]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ScalarBiasGelu_MatchesSeparateScalarOperations(bool pooled)
    {
        var x = new DenseTensor<float>(Enumerable.Range(0, 128).Select(i => (i - 64) / 17f).ToArray(), new[] { 4, 32 });
        var bias = new DenseTensor<float>(Enumerable.Range(0, 32).Select(i => i / 13f).ToArray(), new[] { 32 });
        var add = CPUExecutionProvider.Add(x, bias, ExecutionOptions.Scalar, null);
        var expected = (Tensor<float>)CPUExecutionProvider.Gelu(add.Outputs[0], "none", ExecutionOptions.Scalar, null).Outputs[0];
        var fused = CPUExecutionProvider.BiasGelu(x, bias, ExecutionOptions.Scalar, pooled ? new TensorBufferPool() : null);
        Bits(expected.ToArray(), (Tensor<float>)fused.Outputs[0]);
    }
}
