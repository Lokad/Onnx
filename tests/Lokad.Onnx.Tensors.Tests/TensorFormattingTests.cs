using System.Globalization;

namespace Lokad.Onnx.Tensors.Tests;

public class TensorFormattingTests
{
    [Theory]
    [InlineData("en-US", TensorElementType.Int32, "int32")]
    [InlineData("en-US", TensorElementType.UInt64, "uint64")]
    [InlineData("en-US", TensorElementType.Float, "float")]
    [InlineData("tr-TR", TensorElementType.Int32, "int32")]
    [InlineData("tr-TR", TensorElementType.UInt64, "uint64")]
    [InlineData("tr-TR", TensorElementType.Float, "float")]
    public void RuntimeDtypeLabels_AreInvariant(string culture, TensorElementType type, string dtype)
    {
        var previous = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo(culture);
            ITensor tensor = TensorBase.CreateDenseTensor(type, TensorBase.CreateElementArray(type, 1), new[] { 1 });
            tensor.Name = "value";
            Assert.Equal("value:" + dtype + ":1", tensor.TensorNameDesc());
        }
        finally { CultureInfo.CurrentCulture = previous; }
    }
}
