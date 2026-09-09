namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins the eligible-1x1 direct path: both entries, groups, bias, modes,
/// and fallback geometries agree with hand-computed values.
/// </summary>
public class ConvDirectTests
{
    static DenseTensor<float> Input() =>
        DenseTensor<float>.OfValues(new float[,,,] { { { { 1f, 2f }, { 3f, 4f } }, { { 5f, 6f }, { 7f, 8f } } } });

    static DenseTensor<float> ChannelPicker() =>
        DenseTensor<float>.OfValues(new float[,,,] { { { { 1f } }, { { 0f } } }, { { { 0f } }, { { 1f } } } });

    [Fact]
    public void PadTypeEntry_MapsChannels()
    {
        var y = Tensor<float>.Conv2D(Input(), ChannelPicker(), 1, MathOps.PadType.Valid, null, null, null, new int[] { 1, 1 }, null);
        Assert.Equal(new int[] { 1, 2, 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, y.ToArray());
    }

    [Fact]
    public void ExplicitEntry_WithBias()
    {
        var bias = DenseTensor<float>.OfValues(new float[] { 10f, 20f });
        var y = Tensor<float>.Conv2D(Input(), ChannelPicker(), 1, new int[] { 0, 0, 0, 0 }, bias, null, new int[] { 1, 1 }, null);
        Assert.Equal(new float[] { 11f, 12f, 13f, 14f, 25f, 26f, 27f, 28f }, y.ToArray());
    }

    [Fact]
    public void ScalarMode_MapsChannels()
    {
        var y = Tensor<float>.Conv2D(Input(), ChannelPicker(), 1, MathOps.PadType.Valid, null, null, null, new int[] { 1, 1 }, null, TensorExecutionOptions.Scalar);
        Assert.Equal(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, y.ToArray());
    }

    [Fact]
    public void Grouped_MapsChannelsWithinGroups()
    {
        var x = DenseTensor<float>.OfValues(new float[,,,] { { { { 1f, 2f } }, { { 3f, 4f } }, { { 5f, 6f } }, { { 7f, 8f } } } });
        var w = DenseTensor<float>.OfValues(new float[,,,] { { { { 1f } }, { { 0f } } }, { { { 0f } }, { { 1f } } }, { { { 1f } }, { { 0f } } }, { { { 0f } }, { { 1f } } } });
        var y = Tensor<float>.Conv2D(x, w, 2, new int[] { 0, 0, 0, 0 }, null, null, new int[] { 1, 1 }, null);
        Assert.Equal(new int[] { 1, 4, 1, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, y.ToArray());
    }

    [Fact]
    public void Strided_FallsBack()
    {
        var x = DenseTensor<float>.OfValues(new float[,,,] { { { { 1f, 2f }, { 3f, 4f } } } });
        var w = DenseTensor<float>.OfValues(new float[,,,] { { { { 3f } } } });
        var y = Tensor<float>.Conv2D(x, w, 1, new int[] { 0, 0, 0, 0 }, null, null, new int[] { 2, 2 }, null);
        Assert.Equal(new int[] { 1, 1, 1, 1 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 3f }, y.ToArray());
    }

    [Fact]
    public void Padded_FallsBack()
    {
        var x = DenseTensor<float>.OfValues(new float[,,,] { { { { 7f } } } });
        var w = DenseTensor<float>.OfValues(new float[,,,] { { { { 3f } } } });
        var y = Tensor<float>.Conv2D(x, w, 1, new int[] { 1, 0, 0, 0 }, null, null, new int[] { 1, 1 }, null);
        Assert.Equal(new int[] { 1, 1, 2, 1 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 0f, 21f }, y.ToArray());
    }
}
