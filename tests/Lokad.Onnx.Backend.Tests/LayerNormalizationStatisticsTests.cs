using System;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class LayerNormalizationStatisticsTests
{
    sealed class CountingSlice<T> : TensorSlice<T> where T : unmanaged
    {
        public int Materializations;
        public CountingSlice(Tensor<T> parent) : base(parent, new[] { SliceIndex.All, new SliceIndex(0, 4) }) { }
        public override DenseTensor<T> ToDenseTensor() { Materializations++; return base.ToDenseTensor(); }
    }

    [Theory]
    [InlineData(1, false)]
    [InlineData(2, false)]
    [InlineData(3, false)]
    [InlineData(1, true)]
    [InlineData(2, true)]
    [InlineData(3, true)]
    public void RequestedOutputs_MaterializeInputOnce(int outputCount, bool useDouble)
    {
        var floats = new CountingSlice<float>(new DenseTensor<float>(new float[] { 1, 2, 3, 4, 99, 5, 6, 7, 8, 99 }, new[] { 2, 5 }));
        var doubles = new CountingSlice<double>(new DenseTensor<double>(new double[] { 1, 2, 3, 4, 99, 5, 6, 7, 8, 99 }, new[] { 2, 5 }));
        ITensor input = useDouble ? doubles : floats;
        ITensor scale = useDouble ? DenseTensor<double>.Ones(4) : DenseTensor<float>.Ones(4);
        var result = CPUExecutionProvider.LayerNormalization(input, scale, null, 1, 1e-5f, 1, outputCount, null, new TensorBufferPool());
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(outputCount, result.Outputs.Length);
        Assert.Equal(1, useDouble ? doubles.Materializations : floats.Materializations);
        Assert.Equal(new[] { 2, 4 }, result.Outputs[0].Dims);
        if (outputCount >= 2)
        {
            Assert.Equal(new[] { 2, 1 }, result.Outputs[1].Dims);
            Assert.Equal(new float[] { 2.5f, 6.5f }, ((Tensor<float>)result.Outputs[1]).ToArray());
        }
        if (outputCount == 3)
        {
            Assert.Equal(new[] { 2, 1 }, result.Outputs[2].Dims);
            float inv = (float)(1 / Math.Sqrt(1.25 + 1e-5f));
            Assert.Equal(new[] { inv, inv }, ((Tensor<float>)result.Outputs[2]).ToArray());
        }
    }

    [Theory]
    [InlineData(false, 0)]
    [InlineData(false, 1)]
    [InlineData(true, 0)]
    [InlineData(true, 1)]
    public void Statistics_AgreeAcrossOutputCountsForNonLastAxes(bool useDouble, int axis)
    {
        ITensor input = useDouble
            ? new DenseTensor<double>(Enumerable.Range(1, 24).Select(i => (double)i).ToArray(), new[] { 2, 3, 4 })
            : new DenseTensor<float>(Enumerable.Range(1, 24).Select(i => (float)i).ToArray(), new[] { 2, 3, 4 });
        int block = axis == 0 ? 24 : 12;
        ITensor scale = useDouble ? DenseTensor<double>.Ones(block) : DenseTensor<float>.Ones(block);
        ITensor bias = useDouble ? DenseTensor<double>.Ones(block) : DenseTensor<float>.Ones(block);
        Array? first = null;
        foreach (int count in new[] { 1, 2, 3 })
        {
            var result = CPUExecutionProvider.LayerNormalization(input, scale, bias, axis - 3, 1e-5f, 1, count, null, null);
            Assert.Equal(OpStatus.Success, result.Status);
            var values = result.Outputs[0].ToArray();
            if (first is not null) Assert.Equal(first.Cast<object>(), values.Cast<object>());
            first = values;
            if (count == 1) continue;
            Assert.Equal(axis == 0 ? new[] { 1, 1, 1 } : new[] { 2, 1, 1 }, result.Outputs[1].Dims);
            var means = ((Tensor<float>)result.Outputs[1]).ToArray();
            for (int row = 0; row < 24 / block; row++)
            {
                double mean = row * block + (block + 1) / 2.0;
                Assert.Equal((float)mean, means[row]);
                if (count == 3)
                {
                    double variance = Enumerable.Range(1, block).Average(i => Math.Pow(i + row * block - mean, 2));
                    Assert.Equal((float)(1 / Math.Sqrt(variance + 1e-5f)), ((Tensor<float>)result.Outputs[2]).ToArray()[row]);
                }
            }
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EmptyOuterDimensions_HaveEmptyStatistics(bool useDouble)
    {
        ITensor input = useDouble ? DenseTensor<double>.OfShape(0, 4) : DenseTensor<float>.OfShape(0, 4);
        ITensor scale = useDouble ? DenseTensor<double>.Ones(4) : DenseTensor<float>.Ones(4);
        var result = CPUExecutionProvider.LayerNormalization(input, scale, null, 1, 1e-5f, 1, 3, null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new[] { 0, 1 }, result.Outputs[1].Dims);
        Assert.Equal(new[] { 0, 1 }, result.Outputs[2].Dims);
        Assert.All(result.Outputs, output => Assert.Equal(0, output.Length));
    }
}
