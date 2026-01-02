using System;
using System.IO;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class TensorProtoConversionTests
{
    [Fact]
    public void CanConvertMnistInitializers()
    {
        var modelPath = Path.Combine(AppContext.BaseDirectory, "models", "mnist-8.onnx");
        byte[] buffer;
        using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
        {
            buffer = new byte[stream.Length];
            stream.ReadExactly(buffer);
        }
        var model = Model.Parse(buffer);
        Assert.NotNull(model);

        var candidates = model!.Graph.Initializer
            .Where(tp => tp.DataType == (int)TensorElementType.Float
                || tp.DataType == (int)TensorElementType.Double
                || tp.DataType == (int)TensorElementType.Int32
                || tp.DataType == (int)TensorElementType.Int64)
            .ToArray();
        Assert.NotEmpty(candidates);

        foreach (var tp in candidates)
        {
            var tensor = tp.ToTensor();
            Assert.Equal(tp.Name, tensor.Name);
            Assert.Equal(tp.Dims.Select(d => (int)d).ToArray(), tensor.Dims);
            Assert.Equal((TensorElementType)tp.DataType, tensor.ElementType);

            var data = (Array)tp.GetTensorData();
            Assert.Equal(data.Length, (int)tensor.Length);
            if (data.Length == 0)
            {
                continue;
            }

            switch ((TensorElementType)tp.DataType)
            {
                case TensorElementType.Float:
                {
                    var arr = (float[])data;
                    var first = (float)tensor.GetValue(0);
                    var last = (float)tensor.GetValue(arr.Length - 1);
                    Assert.Equal(arr[0], first, 5);
                    Assert.Equal(arr[^1], last, 5);
                    break;
                }
                case TensorElementType.Double:
                {
                    var arr = (double[])data;
                    var first = (double)tensor.GetValue(0);
                    var last = (double)tensor.GetValue(arr.Length - 1);
                    Assert.Equal(arr[0], first, 10);
                    Assert.Equal(arr[^1], last, 10);
                    break;
                }
                case TensorElementType.Int32:
                {
                    var arr = (int[])data;
                    var first = (int)tensor.GetValue(0);
                    var last = (int)tensor.GetValue(arr.Length - 1);
                    Assert.Equal(arr[0], first);
                    Assert.Equal(arr[^1], last);
                    break;
                }
                case TensorElementType.Int64:
                {
                    var arr = (long[])data;
                    var first = (long)tensor.GetValue(0);
                    var last = (long)tensor.GetValue(arr.Length - 1);
                    Assert.Equal(arr[0], first);
                    Assert.Equal(arr[^1], last);
                    break;
                }
            }
        }
    }
}
