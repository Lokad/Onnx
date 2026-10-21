using System;
using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class FlatProviderLayoutTests
{
    static Tensor<T> View<T>(T[] values, int layout) where T : unmanaged
    {
        var root = new DenseTensor<T>(new[] { 2, layout == 3 ? 5 : 3 }, layout != 0);
        Tensor<T> result = layout switch
        {
            2 => root.Slice(SliceIndex.All, SliceIndex.All),
            3 => root.Slice(SliceIndex.All, new SliceIndex(1, 4)),
            4 => root.Slice(SliceIndex.All, new SliceIndex(2, null, -1)),
            _ => root
        };
        int position = 0;
        foreach (var coordinate in result.GetDimensionsIterator()) result[coordinate] = values[position++];
        return result;
    }

    static void Same(OpResult expected, OpResult actual)
    {
        Assert.Equal(OpStatus.Success, expected.Status);
        Assert.Equal(OpStatus.Success, actual.Status);
        Assert.Equal(expected.Outputs[0].Dims, actual.Outputs[0].Dims);
        Assert.Equal(expected.Outputs[0].ToArray().Cast<object>(), actual.Outputs[0].ToArray().Cast<object>());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void UnaryAndBooleanConsumers_PreserveLogicalPositions(int layout)
    {
        foreach (var options in new[] { ExecutionOptions.Default, ExecutionOptions.Memory, ExecutionOptions.Scalar })
        foreach (bool useDouble in new[] { false, true })
        {
            ITensor Input(int order) => useDouble ? View(new double[] { -3.5, -1.5, 0, 1.5, 2.5, 4.5 }, order) : View(new float[] { -3.5f, -1.5f, 0, 1.5f, 2.5f, 4.5f }, order);
            foreach (var op in new Func<ITensor, OpResult>[]
            {
                x => CPUExecutionProvider.Tanh(x, options),
                x => CPUExecutionProvider.Sigmoid(x, options),
                x => CPUExecutionProvider.Floor(x, options),
                x => CPUExecutionProvider.Clip(x, null, null, null, null, options, 13)
            }) Same(op(Input(0)), op(Input(layout)));
            var bools = new[] { true, false, true, false, true, false };
            Same(CPUExecutionProvider.Not(View(bools, 0), options), CPUExecutionProvider.Not(View(bools, layout), options));
            Same(CPUExecutionProvider.And(View(bools, 0), View(Enumerable.Repeat(true, 6).ToArray(), 0), options), CPUExecutionProvider.And(View(bools, layout), View(Enumerable.Repeat(true, 6).ToArray(), 0), options));
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void PadAndSoftmaxConsumers_PreserveLogicalPositions(int layout)
    {
        var values = new float[] { -3, -1, 0, 1, 2, 4 };
        foreach (var options in new[] { ExecutionOptions.Default, ExecutionOptions.Memory, ExecutionOptions.Scalar })
        {
            foreach (string mode in new[] { "constant", "reflect" })
            {
                var pads = DenseTensor<long>.OfValues(new long[] { 0, 1, 0, 1 });
                Same(CPUExecutionProvider.Pad(View(values, 0), pads, null, mode, null, null, options), CPUExecutionProvider.Pad(View(values, layout), pads, null, mode, null, null, options));
            }
            foreach (var pool in new TensorBufferPool?[] { null, new() })
            {
                Same(CPUExecutionProvider.Softmax(View(values, 0), -1, options, pool, 13), CPUExecutionProvider.Softmax(View(values, layout), -1, options, pool, 13));
                var mask = DenseTensor<float>.OfValues(new float[] { 0, -1, -2 });
                Same(CPUExecutionProvider.MaskedSoftmax(View(values, 0), mask, -1, options, pool, 13), CPUExecutionProvider.MaskedSoftmax(View(values, layout), mask, -1, options, pool, 13));
            }
        }
    }
}
