namespace Lokad.Onnx.Backend.Tests;

public class UnsqueezeEntryPointTests
{
    [Fact]
    public void ProviderOverloads_InsertAxesAtOutputPositionsWithoutChangingInputs()
    {
        var input = new DenseTensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var axes = new[] { -1, 0 };
        var attributeResult = CPUExecutionProvider.Unsqueeze(input, axes, null);
        var tensorAxes = DenseTensor<long>.OfValues(new long[] { -1, 0 });
        var tensorResult = CPUExecutionProvider.Unsqueeze(input, tensorAxes, null);
        foreach (var result in new[] { attributeResult, tensorResult })
        {
            Assert.Equal(OpStatus.Success, result.Status);
            Assert.Equal(new[] { 1, 2, 3, 1 }, result.Outputs[0].Dims);
            Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6 }, ((Tensor<float>)result.Outputs[0]).ToArray());
        }
        Assert.Equal(new[] { -1, 0 }, axes);
        Assert.Equal(new long[] { -1, 0 }, tensorAxes.ToArray());
    }
}
