namespace Lokad.Onnx.Backend.Tests;

public class ArrayDiagnosticTests
{
    [Fact]
    public void EmptyKernelShape_ReturnsFailureWithFormattedValues()
    {
        var input = DenseTensor<float>.OfShape(1, 1, 3);
        var result = CPUExecutionProvider.MaxPool(input, null, null, null,
            Array.Empty<int>(), null, null, null, null);
        Assert.Equal(OpStatus.Failure, result.Status);
        Assert.Contains("kernel_shape", result.Message);
        Assert.Contains("[]", result.Message);
        Assert.Contains("must hold one value", result.Message);
    }
}
