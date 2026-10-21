namespace Lokad.Onnx.Backend.Tests;

public class GatherProviderLayoutTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void Int64Indices_PreserveLogicalOrderAndCheckedBounds(int layout)
    {
        var storage = new DenseTensor<long>(new[] { 2, layout == 1 ? 3 : 2 }, layout == 2);
        Tensor<long> indices = layout == 1
            ? storage.Slice(new SliceIndex(0, 2), new SliceIndex(0, 2)) : storage;
        long[] values = { 0, 2, 1, 3 };
        int position = 0;
        foreach (var coordinate in indices.GetDimensionsIterator()) indices[coordinate] = values[position++];
        var data = DenseTensor<float>.OfValues(new float[] { 10, 20, 30, 40 });
        var equivalent = new DenseTensor<int>(new[] { 0, 2, 1, 3 }, new[] { 2, 2 });
        foreach (ITensor input in new ITensor[] { equivalent, indices })
        {
            var result = CPUExecutionProvider.Gather(data, input, 0, null);
            Assert.Equal(OpStatus.Success, result.Status);
            Assert.Equal(new[] { 2, 2 }, result.Outputs[0].Dims);
            Assert.Equal(new float[] { 10, 30, 20, 40 }, ((Tensor<float>)result.Outputs[0]).ToArray());
        }
        Assert.Equal(values, indices.ToArray());
        Assert.Equal(new float[] { 10, 20, 30, 40 }, data.ToArray());
        indices[0, 0] = long.MaxValue;
        Assert.Throws<OverflowException>(() => CPUExecutionProvider.Gather(data, indices, 0, null));
    }

    [Fact]
    public void Int64ScalarViewAndEmptyIndices_PreserveShape()
    {
        var data = DenseTensor<float>.OfValues(new float[] { 10, 20, 30 });
        var scalar = DenseTensor<long>.OfValues(new long[] { -99, -1 }).Slice(SliceIndex.Index(1));
        var result = CPUExecutionProvider.Gather(data, scalar, 0, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Empty(result.Outputs[0].Dims);
        Assert.Equal(new float[] { 30 }, ((Tensor<float>)result.Outputs[0]).ToArray());
        var empty = CPUExecutionProvider.Gather(data, new DenseTensor<long>(new[] { 0, 2 }, true), 0, null);
        Assert.Equal(OpStatus.Success, empty.Status);
        Assert.Equal(new[] { 0, 2 }, empty.Outputs[0].Dims);
        Assert.Empty(((Tensor<float>)empty.Outputs[0]).ToArray());
    }
}
