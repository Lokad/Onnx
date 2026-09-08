namespace Lokad.Onnx.Backend.Tests;

public class ExpandResizeShapeTests
{
    [Fact]
    public void Expand_LowerRankShape_RightAligns()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        var r = Tensor<float>.Expand(x, new int[] { 3 });
        Assert.Equal(new int[] { 2, 3 }, r.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, r.ToArray());
        var rc = CPUExecutionProvider.Expand(x, new int[] { 3 }.ToTensor<int>(), null);
        Assert.Equal(OpStatus.Success, rc.Status);
        Assert.Equal(new int[] { 2, 3 }, ((Tensor<float>)rc.Outputs[0]).Dimensions.ToArray());
    }

    [Fact]
    public void Expand_EmptyShape_Identity()
    {
        var x = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        var r = Tensor<float>.Expand(x, new int[0]);
        Assert.Equal(new int[] { 2 }, r.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f }, r.ToArray());
    }

    [Fact]
    public void Expand_ZeroExtents()
    {
        var x = DenseTensor<float>.OfValues(new float[0]);
        var r = Tensor<float>.Expand(x, new int[] { 0 });
        Assert.Equal(new int[] { 0 }, r.Dimensions.ToArray());
        var y = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f } });
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Expand(y, new int[] { 2, 0 }));
    }

    [Fact]
    public void Expand_TargetOne_KeepsInputDim()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        var r = Tensor<float>.Expand(x, new int[] { 2, 1 });
        Assert.Equal(new int[] { 2, 3 }, r.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, r.ToArray());
    }

    [Fact]
    public void Expand_MismatchedShape_Fails()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Expand(x, new int[] { 4 }));
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Expand(x, new int[] { 3, 2 }));
    }

    static DenseTensor<float> Img2x2() =>
        DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });

    static ITensor Scales(params float[] s) => DenseTensor<float>.OfValues(s);

    [Fact]
    public void Resize_ScalesFloor_Nonintegral()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 5, 1] { { { { 1f }, { 2f }, { 3f }, { 4f }, { 5f } } } });
        var r = CPUExecutionProvider.Resize(x, null, Scales(1f, 1f, 1.5f, 1f), null, "nearest", "half_pixel", "round_prefer_floor", -0.75f, 0f, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs[0];
        Assert.Equal(new int[] { 1, 1, 7, 1 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 1f, 2f, 3f, 3f, 4f, 5f }, y.ToArray());
    }

    [Fact]
    public void Resize_Nearest_RoundHalvesDown()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 4, 1] { { { { 1f }, { 2f }, { 3f }, { 4f } } } });
        var r = CPUExecutionProvider.Resize(x, null, Scales(1f, 1f, 1.5f, 1f), null, "nearest", "half_pixel", "round_prefer_floor", -0.75f, 0f, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 1f, 1f, 2f, 3f, 3f, 4f }, ((Tensor<float>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void Resize_Nearest_RoundPreferCeil_HalvesUp()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 4, 1] { { { { 1f }, { 2f }, { 3f }, { 4f } } } });
        var r = CPUExecutionProvider.Resize(x, null, Scales(1f, 1f, 1.5f, 1f), null, "nearest", "half_pixel", "round_prefer_ceil", -0.75f, 0f, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 1f, 2f, 2f, 3f, 4f, 4f }, ((Tensor<float>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void Resize_Nearest_Downscale()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 1, 4] { { { { 1f, 2f, 3f, 4f } } } });
        var r = CPUExecutionProvider.Resize(x, null, Scales(1f, 1f, 1f, 0.5f), null, "nearest", "half_pixel", "round_prefer_floor", -0.75f, 0f, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs[0];
        Assert.Equal(new int[] { 1, 1, 1, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 3f }, y.ToArray());
    }

    [Fact]
    public void Resize_Linear_Values()
    {
        var r = CPUExecutionProvider.Resize(Img2x2(), null, Scales(1f, 1f, 2f, 2f), null, "linear", "half_pixel", "round_prefer_floor", -0.75f, 0f, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs[0];
        Assert.Equal(new int[] { 1, 1, 4, 4 }, y.Dimensions.ToArray());
        var a = y.ToArray();
        Assert.Equal(1f, a[0], 4);
        Assert.Equal(1.25f, a[1], 4);
        Assert.Equal(1.75f, a[2], 4);
        Assert.Equal(2f, a[3], 4);
        Assert.Equal(2.5f, a[8], 4);
        Assert.Equal(4f, a[15], 4);
    }

    [Fact]
    public void Resize_Cubic_Values()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 4, 1] { { { { 1f }, { 2f }, { 3f }, { 4f } } } });
        var r = CPUExecutionProvider.Resize(x, null, Scales(1f, 1f, 1.5f, 1f), null, "cubic", "half_pixel", "round_prefer_floor", -0.75f, 0f, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var a = ((Tensor<float>)r.Outputs[0]).ToArray();
        Assert.Equal(6, a.Length);
        Assert.Equal(0.9131949f, a[0], 4);
        Assert.Equal(1.40625f, a[1], 4);
        Assert.Equal(2.2129645f, a[2], 4);
        Assert.Equal(2.7870378f, a[3], 4);
        Assert.Equal(3.59375f, a[4], 4);
        Assert.Equal(4.0868058f, a[5], 4);
    }

    [Fact]
    public void Resize_Double_Parity()
    {
        var x = DenseTensor<double>.OfValues(new double[1, 1, 2, 2] { { { { 1.0, 2.0 }, { 3.0, 4.0 } } } });
        var scales = DenseTensor<float>.OfValues(new float[] { 1f, 1f, 2f, 2f });
        var r = CPUExecutionProvider.Resize(x, null, scales, null, "nearest", "half_pixel", "round_prefer_floor", -0.75f, 0f, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<double>)r.Outputs[0];
        Assert.Equal(new int[] { 1, 1, 4, 4 }, y.Dimensions.ToArray());
        var a = y.ToArray();
        Assert.Equal(1.0, a[0], 9);
        Assert.Equal(1.0, a[1], 9);
        Assert.Equal(2.0, a[2], 9);
        Assert.Equal(4.0, a[15], 9);
        var rl = CPUExecutionProvider.Resize(x, null, scales, null, "linear", "half_pixel", "round_prefer_floor", -0.75f, 0f, null);
        Assert.Equal(OpStatus.Success, rl.Status);
        var al = ((Tensor<double>)rl.Outputs[0]).ToArray();
        Assert.Equal(1.25, al[1], 9);
        var xd = DenseTensor<double>.OfValues(new double[1, 1, 4, 1] { { { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 } } } });
        var scalesH = DenseTensor<float>.OfValues(new float[] { 1f, 1f, 2f, 1f });
        var rd = CPUExecutionProvider.Resize(xd, null, scalesH, null, "nearest", "half_pixel", "round_prefer_floor", -0.75f, 0f, null);
        Assert.Equal(OpStatus.Success, rd.Status);
        Assert.Equal(new double[] { 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0 }, ((Tensor<double>)rd.Outputs[0]).ToArray());
    }

    [Fact]
    public void Resize_UnsupportedMode_FailsAtBoundary()
    {
        var x = Img2x2();
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Resize(x, null, Scales(1f, 1f, 2f, 2f), null, "lanczos", "half_pixel", "round_prefer_floor", -0.75f, 0f, null).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Resize(x, null, Scales(1f, 1f, 2f, 2f), null, "nearest", "pytorch_half_pixel", "round_prefer_floor", -0.75f, 0f, null).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Resize(x, null, Scales(1f, 1f, 2f, 2f), null, "nearest", "half_pixel", "bogus", -0.75f, 0f, null).Status);
    }

    [Fact]
    public void Resize_Non4DOrBatchChange_FailsAtBoundary()
    {
        var x3 = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Resize(x3, null, Scales(1f, 2f), null, "nearest", "half_pixel", "round_prefer_floor", -0.75f, 0f, null).Status);
        var x = Img2x2();
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Resize(x, null, Scales(2f, 1f, 1f, 1f), null, "nearest", "half_pixel", "round_prefer_floor", -0.75f, 0f, null).Status);
    }
}
