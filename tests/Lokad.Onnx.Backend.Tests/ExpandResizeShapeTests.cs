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
    public void Expand_NegativeDim_FailsCleanly()
    {
        // ORT 1.29 fails the run (-1 is not a keep marker in Expand,
        // unlike Reshape); previously Lokad silently kept the dim.
        var x = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Expand(x, new int[] { -1, 2 }));
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 13 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = x;
        graph.Inputs["s"] = DenseTensor<long>.OfValues(new long[] { -1L, 2L });
        var node = new Node
        {
            Name = "n", Op = OpType.Expand, OpTypeName = OpType.Expand.ToString(), Domain = "",
            OpsetVersion = 13, IsFused = false,
            Inputs = new[] { "x", "s" }, Outputs = new[] { "z" },
            Attributes = new Dictionary<string, object>(),
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
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
    public void Expand_EmptyInput_PreservesEmpty()
    {
        // ORT 1.29: [2,0] expanded to [2,0], and [7] expanded to [0],
        // both yield empties rather than failing.
        var x = DenseTensor<float>.OfShape(2, 0);
        var r = CPUExecutionProvider.Expand(x, DenseTensor<long>.OfValues(new long[] { 2L, 0L }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs[0];
        Assert.Equal(new int[] { 2, 0 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
        var s = DenseTensor<float>.OfValues(new float[] { 7f });
        var r2 = CPUExecutionProvider.Expand(s, DenseTensor<long>.OfValues(new long[] { 0L }), null);
        Assert.Equal(OpStatus.Success, r2.Status);
        var y2 = (Tensor<float>)r2.Outputs[0];
        Assert.Equal(new int[] { 0 }, y2.Dimensions.ToArray());
        Assert.Empty(y2.ToArray());
    }

    [Fact]
    public void Resize_EmptyInput_PreservesEmpty()
    {
        // ORT 1.29: unit scales over [1,1,0,3] yield [1,1,0,3], empty.
        var x = DenseTensor<float>.OfShape(1, 1, 0, 3);
        var r = CPUExecutionProvider.Resize(x, null, Scales(1f, 1f, 1f, 1f), null, "nearest", "half_pixel", "round_prefer_floor", -0.75f, 0f, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs[0];
        Assert.Equal(new int[] { 1, 1, 0, 3 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
    }

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

    [Fact]
    public void Resize_UnsupportedAttributes_FailLoudly()
    {
        var x = Img2x2();
        var aa = CPUExecutionProvider.Resize(x, null, Scales(1f, 1f, 2f, 2f), null, "nearest", "half_pixel", "round_prefer_floor", -0.75f, 0f, null, 1, null, null, null);
        Assert.Equal(OpStatus.Failure, aa.Status);
        Assert.Contains("antialias", aa.Message ?? "");
        var eo = CPUExecutionProvider.Resize(x, null, Scales(1f, 1f, 2f, 2f), null, "nearest", "half_pixel", "round_prefer_floor", -0.75f, 0f, null, null, null, 1, null);
        Assert.Equal(OpStatus.Failure, eo.Status);
        Assert.Contains("excludeOutside", eo.Message ?? "");
        var kar = CPUExecutionProvider.Resize(x, null, Scales(1f, 1f, 2f, 2f), null, "nearest", "half_pixel", "round_prefer_floor", -0.75f, 0f, null, null, null, null, "not_larger");
        Assert.Equal(OpStatus.Failure, kar.Status);
        Assert.Contains("keepAspectRatioPolicy", kar.Message ?? "");
        var ax = CPUExecutionProvider.Resize(x, null, Scales(1f, 1f, 2f, 2f), null, "nearest", "half_pixel", "round_prefer_floor", -0.75f, 0f, null, null, new int[] { 2, 3 }, null, null);
        Assert.Equal(OpStatus.Failure, ax.Status);
        Assert.Contains("axes", ax.Message ?? "");
    }

    [Fact]
    public void Resize_DefaultAttributes_BehaveAsBefore()
    {
        var x = Img2x2();
        var plain = (Tensor<float>)CPUExecutionProvider.Resize(x, null, Scales(1f, 1f, 2f, 2f), null, "nearest", "half_pixel", "round_prefer_floor", -0.75f, 0f, null).Outputs[0];
        var full = (Tensor<float>)CPUExecutionProvider.Resize(x, null, Scales(1f, 1f, 2f, 2f), null, "nearest", "half_pixel", "round_prefer_floor", -0.75f, 0f, null, 0, new int[] { 0, 1, 2, 3 }, 0, "stretch").Outputs[0];
        Assert.Equal(plain.ToArray(), full.ToArray());
        Assert.Equal(new int[] { 1, 1, 4, 4 }, full.Dimensions.ToArray());
    }

    [Fact]
    public void Resize_AntialiasAttribute_FailsThroughDispatch()
    {
        var mp = new OnnxModel { Name = "resize-aa" };
        mp.Opset[""] = 18;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 1, 1, 2, 2 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "z", ElementType = TensorElementType.Float, Dims = new[] { 1, 1, 4, 4 } });
        mp.Initializers.Add(new OnnxTensor { Name = "scales", ElementType = TensorElementType.Float, Dims = new[] { 4 }, Data = new float[] { 1f, 1f, 2f, 2f } });
        mp.Nodes.Add(new OnnxNode
        {
            Name = "r", OpType = "Resize", Domain = "",
            Inputs = new[] { "x", "", "scales" }, Outputs = new[] { "z" },
            Attributes = new Dictionary<string, object> { { "antialias", 1L } },
        });
        var graph = Model.Load(mp)!;
        var inputs = new Dictionary<string, ITensor> { { "x", Img2x2() } };
        Assert.False(graph.Execute(inputs, true));
        Assert.Contains("antialias", graph.LastErrorMessage ?? "");
    }

    [Fact]
    public void IndexDtypeGuards_RejectWrongTypes()
    {
        // ORT refuses wrong index/scale dtypes at load (the reduction
        // axes batch pins the same contract); the provider must fail
        // descriptively instead of falling through to InvalidCast.
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f } });
        var fShape = DenseTensor<float>.OfValues(new float[] { 2f, 2f });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Expand(x, fShape, null).Status);
        var xr = Img2x2();
        var fSizes = DenseTensor<float>.OfValues(new float[] { 1f, 1f, 4f, 4f });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Resize(xr, null, null, fSizes, "nearest", "half_pixel", "round_prefer_floor", -0.75f, 0f, null).Status);
        var iScales = DenseTensor<int>.OfValues(new int[] { 1, 1, 2, 2 });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Resize(xr, null, iScales, null, "nearest", "half_pixel", "round_prefer_floor", -0.75f, 0f, null).Status);
    }

    [Fact]
    public void ResizeLinearExceptional_MatchesOrt()
    {
        // ORT 1.29 asymmetric linear x4 on a width-2 row: interior blends
        // keep any infinite endpoint; the left-edge extrapolation is NaN
        // for a finite-left/infinite-right edge but inf for the mirror;
        // opposing infinities cancel across the whole row. Guards the
        // interpolation evaluation order and edge-tap selection.
        var rows = new (float[] x, float[] expected)[]
        {
            (new float[] { float.PositiveInfinity, 1f },
             new float[] { float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity, 1f, 1f, 1f, 1f }),
            (new float[] { 1f, float.PositiveInfinity },
             new float[] { float.NaN, float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity }),
            (new float[] { float.PositiveInfinity, float.NegativeInfinity },
             new float[] { float.NaN, float.NaN, float.NaN, float.NaN, float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity }),
            (new float[] { float.NaN, 1f },
             new float[] { float.NaN, float.NaN, float.NaN, float.NaN, 1f, 1f, 1f, 1f }),
        };
        foreach (var (x, expected) in rows)
        {
            var r = CPUExecutionProvider.Resize(
                DenseTensor<float>.OfValues(new float[1, 1, 1, 2] { { { { x[0], x[1] } } } }),
                null, Scales(1f, 1f, 1f, 4f), null, "linear", "asymmetric", null, null, 0f, null);
            Assert.Equal(OpStatus.Success, r.Status);
            var y = ((Tensor<float>)r.Outputs[0]).ToArray();
            Assert.Equal(8, y.Length);
            for (int i = 0; i < 8; i++)
            {
                if (float.IsNaN(expected[i])) Assert.True(float.IsNaN(y[i]));
                else Assert.Equal(expected[i], y[i]);
            }
        }
    }

    [Fact]
    public void ResizeNearestTie_MatchesOrt()
    {
        // ORT 1.29 with explicit nearest_mode: round_prefer_floor rounds
        // exact halves down and round_prefer_ceil rounds them up, under
        // both asymmetric ([10,20] x4: 0.5 -> 0/1) and align_corners
        // ([10,20,30] x5/3: 0.5/1.5 -> 0/1 and 1/2). Finite regression for
        // the earlier half-up spelling under non-half_pixel modes.
        var r = CPUExecutionProvider.Resize(
            DenseTensor<float>.OfValues(new float[1, 1, 1, 2] { { { { 10f, 20f } } } }),
            null, Scales(1f, 1f, 1f, 4f), null, "nearest", "asymmetric", "round_prefer_floor", null, 0f, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 10f, 10f, 10f, 20f, 20f, 20f, 20f, 20f }, ((Tensor<float>)r.Outputs[0]).ToArray());
        var rc = CPUExecutionProvider.Resize(
            DenseTensor<float>.OfValues(new float[1, 1, 1, 2] { { { { 10f, 20f } } } }),
            null, Scales(1f, 1f, 1f, 4f), null, "nearest", "asymmetric", "round_prefer_ceil", null, 0f, null);
        Assert.Equal(OpStatus.Success, rc.Status);
        Assert.Equal(new float[] { 10f, 10f, 20f, 20f, 20f, 20f, 20f, 20f }, ((Tensor<float>)rc.Outputs[0]).ToArray());
        var sizes5 = DenseTensor<int>.OfValues(new int[] { 1, 1, 1, 5 });
        var ra = CPUExecutionProvider.Resize(
            DenseTensor<float>.OfValues(new float[1, 1, 1, 3] { { { { 10f, 20f, 30f } } } }),
            null, null, sizes5, "nearest", "align_corners", "round_prefer_floor", null, 0f, null);
        Assert.Equal(OpStatus.Success, ra.Status);
        Assert.Equal(new float[] { 10f, 10f, 20f, 20f, 30f }, ((Tensor<float>)ra.Outputs[0]).ToArray());
        var rac = CPUExecutionProvider.Resize(
            DenseTensor<float>.OfValues(new float[1, 1, 1, 3] { { { { 10f, 20f, 30f } } } }),
            null, null, sizes5, "nearest", "align_corners", "round_prefer_ceil", null, 0f, null);
        Assert.Equal(OpStatus.Success, rac.Status);
        Assert.Equal(new float[] { 10f, 20f, 20f, 30f, 30f }, ((Tensor<float>)rac.Outputs[0]).ToArray());
    }

    [Fact]
    public void ResizeNearestExceptional_MatchesOrt()
    {
        // ORT 1.29 asymmetric round_prefer_floor x4: pure replication with
        // no arithmetic, so infinities copy verbatim (exact halves round
        // down: o=2 samples index 0).
        var r = CPUExecutionProvider.Resize(
            DenseTensor<float>.OfValues(new float[1, 1, 1, 2] { { { { float.PositiveInfinity, 1f } } } }),
            null, Scales(1f, 1f, 1f, 4f), null, "nearest", "asymmetric", "round_prefer_floor", null, 0f, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = ((Tensor<float>)r.Outputs[0]).ToArray();
        Assert.Equal(new float[] { float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity, 1f, 1f, 1f, 1f, 1f }, y);
    }

}
