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
    public void MismatchedBiasDtype_RejectedCleanly()
    {
        // ORT 1.29 refuses mismatched bias at load; every other
        // Conv input is dtype-checked, bias must be too.
        var x = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });
        var w = DenseTensor<float>.OfValues(new float[1, 1, 1, 1] { { { { 1f } } } });
        var b = DenseTensor<long>.OfValues(new long[] { 1L });
        var r = CPUExecutionProvider.Conv(x, w, b, null, null, null, null, null, null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

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

    [Fact]
    public void EmptyBatch_ReturnsEmpty()
    {
        // ORT 1.29: shape (0, 2, 3, 3).
        var x = DenseTensor<float>.OfShape(0, 2, 5, 5);
        var w = DenseTensor<float>.OfValues(new float[2, 2, 3, 3]);
        var y = Tensor<float>.Conv2D(x, w, 1, new int[] { 0, 0, 0, 0 }, null, null, new int[] { 1, 1 }, null);
        Assert.Equal(new int[] { 0, 2, 3, 3 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
    }

    [Fact]
    public void EmptyChannels_ReturnsZeros()
    {
        // ORT 1.29: shape (1, 2, 3, 3); empty reduction sums to zero.
        var x = DenseTensor<float>.OfShape(1, 0, 5, 5);
        var w = DenseTensor<float>.OfShape(2, 0, 3, 3);
        var y = Tensor<float>.Conv2D(x, w, 1, new int[] { 0, 0, 0, 0 }, null, null, new int[] { 1, 1 }, null);
        Assert.Equal(new int[] { 1, 2, 3, 3 }, y.Dimensions.ToArray());
        Assert.Equal(new float[18], y.ToArray());
    }

    [Fact]
    public void NaNInput_YieldsNaN()
    {
        // ORT 1.29: NaN poisons the multiply-accumulate, 1x1 or otherwise.
        var x = DenseTensor<float>.OfValues(new float[1, 1, 1, 1] { { { { float.NaN } } } });
        var w = DenseTensor<float>.OfValues(new float[1, 1, 1, 1] { { { { 1f } } } });
        var r = CPUExecutionProvider.Conv(x, w, null, null, null, null, null, null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.True(float.IsNaN(((Tensor<float>)r.Outputs[0])[0, 0, 0, 0]));
    }

    [Fact]
    public void InfInput_Propagates()
    {
        // ORT 1.29: inf*0 is NaN on both the eligible-1x1 direct path and
        // the padded fallback (padding zeros times an infinite weight are
        // NaN, not 0); opposing infinities cancel across the accumulation,
        // and an infinite bias propagates through the epilogue.
        var r1 = CPUExecutionProvider.Conv(
            DenseTensor<float>.OfValues(new float[1, 1, 1, 1] { { { { float.PositiveInfinity } } } }),
            DenseTensor<float>.OfValues(new float[1, 1, 1, 1] { { { { 0f } } } }),
            null, null, null, null, null, null, null, null);
        Assert.Equal(OpStatus.Success, r1.Status);
        Assert.True(float.IsNaN(((Tensor<float>)r1.Outputs[0])[0, 0, 0, 0]));
        var r2 = CPUExecutionProvider.Conv(
            DenseTensor<float>.OfValues(new float[1, 1, 1, 1] { { { { float.PositiveInfinity } } } }),
            DenseTensor<float>.OfValues(new float[1, 1, 1, 1] { { { { 2f } } } }),
            null, null, null, null, null, null, null, null);
        Assert.Equal(OpStatus.Success, r2.Status);
        Assert.Equal(float.PositiveInfinity, ((Tensor<float>)r2.Outputs[0])[0, 0, 0, 0]);
        var r3 = CPUExecutionProvider.Conv(
            DenseTensor<float>.OfValues(new float[1, 1, 1, 2] { { { { float.PositiveInfinity, 5f } } } }),
            DenseTensor<float>.OfValues(new float[1, 1, 1, 2] { { { { 1f, float.NegativeInfinity } } } }),
            null, null, null, null, null, null, null, null);
        Assert.Equal(OpStatus.Success, r3.Status);
        Assert.True(float.IsNaN(((Tensor<float>)r3.Outputs[0])[0, 0, 0, 0]));
        var r4 = CPUExecutionProvider.Conv(
            DenseTensor<float>.OfValues(new float[1, 1, 1, 1] { { { { 1f } } } }),
            DenseTensor<float>.OfValues(new float[1, 1, 1, 1] { { { { float.PositiveInfinity } } } }),
            null, null, null, null, null, new[] { 0, 0, 1, 1 }, null, null);
        Assert.Equal(OpStatus.Success, r4.Status);
        var y4 = ((Tensor<float>)r4.Outputs[0]).ToArray();
        Assert.Equal(float.PositiveInfinity, y4[0]);
        Assert.True(float.IsNaN(y4[1]));
        Assert.True(float.IsNaN(y4[2]));
        Assert.True(float.IsNaN(y4[3]));
        var r5 = CPUExecutionProvider.Conv(
            DenseTensor<float>.OfValues(new float[1, 1, 1, 1] { { { { float.PositiveInfinity } } } }),
            DenseTensor<float>.OfValues(new float[1, 1, 1, 1] { { { { 1f } } } }),
            DenseTensor<float>.OfValues(new float[] { float.NegativeInfinity }),
            null, null, null, null, null, null, null);
        Assert.Equal(OpStatus.Success, r5.Status);
        Assert.True(float.IsNaN(((Tensor<float>)r5.Outputs[0])[0, 0, 0, 0]));
    }

    [Fact]
    public void Int32_RejectedCleanly()
    {
        // ORT 1.29 refuses int32 Conv at load (Conv-13 is float-only);
        // the provider fails descriptively instead.
        var x = DenseTensor<int>.OfShape(1, 1, 2, 2);
        var w = DenseTensor<int>.OfValues(new int[1, 1, 1, 1] { { { { 1 } } } });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Conv(x, w, null, null, null, null, null, null, null, null).Status);
    }

    [Fact]
    public void ZeroSpatial_Throws()
    {
        // ORT 1.29 fails the run for zero spatial extents (batch-0 is
        // already pinned flowing to empty); the planner rejects
        // non-positive output dims up front, mirroring MaxPool.
        var x = DenseTensor<float>.OfShape(1, 1, 0, 3);
        var w = DenseTensor<float>.OfValues(new float[1, 1, 1, 1] { { { { 1f } } } });
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.Conv(x, w, null, null, null, null, null, null, null, null));
    }

    [Fact]
    public void MismatchedWeightsDtype_RejectedCleanly()
    {
        // Mismatched bias already pins its guard; ORT refuses mismatched
        // weights at load too, so the provider must fail descriptively.
        var x = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });
        var w = DenseTensor<int>.OfValues(new int[1, 1, 1, 1] { { { { 1 } } } });
        var r = CPUExecutionProvider.Conv(x, w, null, null, null, null, null, null, null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }
}
