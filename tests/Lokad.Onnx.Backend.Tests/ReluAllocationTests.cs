using System;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class ReluAllocationTests
{
    static long Allocation(Func<ITensor> operation)
    {
        for (int i = 0; i < 32; i++) operation();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10; i++) GC.KeepAlive(operation());
        return (GC.GetAllocatedBytesForCurrentThread() - before) / 10;
    }

    [Theory]
    [InlineData(16, false)]
    [InlineData(128, false)]
    [InlineData(16, true)]
    [InlineData(128, true)]
    public void ScalarTensorAndProvider_AllocateOnlyOutput(int side, bool useDouble)
    {
        var floats = DenseTensor<float>.Ones(side, side + 1).Slice(SliceIndex.All, new SliceIndex(0, side));
        var doubles = DenseTensor<double>.Ones(side, side + 1).Slice(SliceIndex.All, new SliceIndex(0, side));
        ITensor input = useDouble ? doubles : floats;
        long baseline = Allocation(() => useDouble ? Tensor<double>.Relu(doubles) : Tensor<float>.Relu(floats));
        long scalar = Allocation(() => useDouble ? Tensor<double>.Relu(doubles, TensorExecutionOptions.Scalar) : Tensor<float>.Relu(floats, TensorExecutionOptions.Scalar));
        long provider = Allocation(() => CPUExecutionProvider.Relu(input, ExecutionOptions.Scalar).Outputs[0]);
        Assert.True(scalar - baseline < 8192, $"Scalar tensor extra allocation: {scalar - baseline} bytes.");
        Assert.True(provider - baseline < 8192, $"Scalar provider extra allocation: {provider - baseline} bytes.");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LayoutsAndModes_PreserveExceptionalValuesAndSignedZero(bool reversed)
    {
        float[] floats = { -0f, float.NaN, -1, 1, float.NegativeInfinity, float.PositiveInfinity, 1e-30f, -1e-30f };
        double[] doubles = { -0.0, double.NaN, -1, 1, double.NegativeInfinity, double.PositiveInfinity, 1e-300, -1e-300 };
        var fx = new DenseTensor<float>(new[] { 2, 5 }, reversed).Slice(SliceIndex.All, new SliceIndex(0, 4));
        var dx = new DenseTensor<double>(new[] { 2, 5 }, reversed).Slice(SliceIndex.All, new SliceIndex(0, 4));
        int i = 0;
        foreach (var coordinate in fx.GetDimensionsIterator()) { fx[coordinate] = floats[i]; dx[coordinate] = doubles[i++]; }
        var expectedF = new float[] { -0f, float.NaN, 0, 1, 0, float.PositiveInfinity, 1e-30f, 0 }.Select(BitConverter.SingleToInt32Bits).ToArray();
        var expectedD = new double[] { -0.0, double.NaN, 0, 1, 0, double.PositiveInfinity, 1e-300, 0 }.Select(BitConverter.DoubleToInt64Bits).ToArray();
        foreach (var options in new[] { ExecutionOptions.Default, ExecutionOptions.Memory, ExecutionOptions.Scalar })
        {
            Assert.Equal(expectedF, ((Tensor<float>)CPUExecutionProvider.Relu(fx, options).Outputs[0]).ToArray().Select(BitConverter.SingleToInt32Bits));
            Assert.Equal(expectedD, ((Tensor<double>)CPUExecutionProvider.Relu(dx, options).Outputs[0]).ToArray().Select(BitConverter.DoubleToInt64Bits));
            Assert.Equal(expectedF, Tensor<float>.Relu(fx, options.Tensor).ToArray().Select(BitConverter.SingleToInt32Bits));
            Assert.Equal(expectedD, Tensor<double>.Relu(dx, options.Tensor).ToArray().Select(BitConverter.DoubleToInt64Bits));
        }
    }
}
