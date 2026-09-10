using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class ConvPoolPaddingTests
{
    static DenseTensor<float> F4(float[,,,] v) => DenseTensor<float>.OfValues(v);

    [Fact]
    public void Conv_OmittedPads_MeansZeros()
    {
        var x = F4(new float[1, 1, 3, 3] { { { { 1f, 2f, 3f }, { 4f, 5f, 6f }, { 7f, 8f, 9f } } } });
        var w = F4(new float[1, 1, 2, 2] { { { { 1f, 1f }, { 1f, 1f } } } });
        var r = CPUExecutionProvider.Conv(x, w, null, null, null, 1, null, null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs[0];
        Assert.Equal(new[] { 1, 1, 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 12f, 16f, 24f, 28f }, y.ToArray());
    }

    [Fact]
    public void Conv_OddSameUpperAndLower_DifferCorrectly()
    {
        var x = F4(new float[1, 1, 1, 5] { { { { 1f, 2f, 3f, 4f, 5f } } } });
        var w = F4(new float[1, 1, 1, 2] { { { { 1f, 1f } } } });
        var ru = CPUExecutionProvider.Conv(x, w, null, "SAME_UPPER", null, 1, null, null, new[] { 1, 2 }, null);
        var rl = CPUExecutionProvider.Conv(x, w, null, "SAME_LOWER", null, 1, null, null, new[] { 1, 2 }, null);
        Assert.Equal(OpStatus.Success, ru.Status);
        Assert.Equal(OpStatus.Success, rl.Status);
        Assert.Equal(new float[] { 3f, 7f, 5f }, ((Tensor<float>)ru.Outputs[0]).ToArray());
        Assert.Equal(new float[] { 1f, 5f, 9f }, ((Tensor<float>)rl.Outputs[0]).ToArray());
    }

    [Fact]
    public void Conv_Same_UnequalAxes_HandledPerAxis()
    {
        var x = DenseTensor<float>.OfShape(1, 1, 4, 6);
        for (int i = 0; i < x.Length; i++) x.SetValue(i, i);
        var w = DenseTensor<float>.OfShape(1, 1, 3, 3);
        w.Fill(1f);
        var ru = CPUExecutionProvider.Conv(x, w, null, "SAME_UPPER", null, 1, null, null, new[] { 2, 2 }, null);
        var rl = CPUExecutionProvider.Conv(x, w, null, "SAME_LOWER", null, 1, null, null, new[] { 2, 2 }, null);
        Assert.Equal(OpStatus.Success, ru.Status);
        Assert.Equal(OpStatus.Success, rl.Status);
        Assert.Equal(new[] { 1, 1, 2, 3 }, ((Tensor<float>)ru.Outputs[0]).Dimensions.ToArray());
        Assert.Equal(new[] { 1, 1, 2, 3 }, ((Tensor<float>)rl.Outputs[0]).Dimensions.ToArray());
        Assert.False(((Tensor<float>)ru.Outputs[0]).ToArray().SequenceEqual(((Tensor<float>)rl.Outputs[0]).ToArray()));
    }

    [Fact]
    public void Conv_AsymmetricExplicitPads_ShapeAndValues()
    {
        var x = F4(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });
        var w = F4(new float[1, 1, 2, 2] { { { { 1f, 1f }, { 1f, 1f } } } });
        var r = CPUExecutionProvider.Conv(x, w, null, "NOTSET", null, 1, null, new[] { 1, 0, 0, 0 }, new[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs[0];
        Assert.Equal(new[] { 1, 1, 2, 1 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 3f, 10f }, y.ToArray());
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Conv(x, w, null, "NOTSET", null, 1, null, new[] { 1, 0, 0 }, new[] { 1, 1 }, null).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Conv(x, w, null, "BOGUS", null, 1, null, null, null, null).Status);
    }

    [Fact]
    public void MaxPool_OmittedPads_MeansZeros()
    {
        var x = F4(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });
        var r = CPUExecutionProvider.MaxPool(x, null, 0, null, new[] { 2, 2 }, null, 0, new[] { 2, 2 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 4f }, ((Tensor<float>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void MaxPool_Dilation_ReadsDilatedTaps()
    {
        var x = F4(new float[1, 1, 1, 5] { { { { 1f, 2f, 3f, 4f, 5f } } } });
        var r = CPUExecutionProvider.MaxPool(x, "VALID", 0, new[] { 1, 2 }, new[] { 1, 2 }, null, 0, new[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs[0];
        Assert.Equal(new[] { 1, 1, 1, 3 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 3f, 4f, 5f }, y.ToArray());
    }

    [Fact]
    public void MaxPool_CeilMode_GrowsEdgeShape()
    {
        var x = F4(new float[1, 1, 1, 5] { { { { 1f, 2f, 3f, 4f, 5f } } } });
        var rf = CPUExecutionProvider.MaxPool(x, "VALID", 0, null, new[] { 1, 2 }, null, 0, new[] { 1, 2 }, null);
        var rc = CPUExecutionProvider.MaxPool(x, "VALID", 1, null, new[] { 1, 2 }, null, 0, new[] { 1, 2 }, null);
        Assert.Equal(OpStatus.Success, rf.Status);
        Assert.Equal(OpStatus.Success, rc.Status);
        Assert.Equal(new float[] { 2f, 4f }, ((Tensor<float>)rf.Outputs[0]).ToArray());
        Assert.Equal(new float[] { 2f, 4f, 5f }, ((Tensor<float>)rc.Outputs[0]).ToArray());
    }

    [Fact]
    public void MaxPool_Values_MatchHandComputation()
    {
        var x = F4(new float[1, 1, 3, 3] { { { { 1f, 2f, 3f }, { 4f, 5f, 6f }, { 7f, 8f, 9f } } } });
        var r = CPUExecutionProvider.MaxPool(x, "VALID", 0, null, new[] { 2, 2 }, null, 0, new[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs[0];
        Assert.Equal(new[] { 1, 1, 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 5f, 6f, 8f, 9f }, y.ToArray());
    }

    [Fact]
    public void MaxPool_NaNWindows_YieldNegativeFloatMax()
    {
        // Probed ORT 1.29 behavior for scalar windows; multi-NaN full-vector
        // windows differ on the native side by SIMD width, which no
        // deterministic contract can match, so ours stays stable instead.
        var all = F4(new float[1, 1, 2, 2] { { { { float.NaN, float.NaN }, { float.NaN, float.NaN } } } });
        var rall = CPUExecutionProvider.MaxPool(all, "VALID", 0, null, new[] { 2, 2 }, null, 0, new[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, rall.Status);
        Assert.Equal(new float[] { -float.MaxValue }, ((Tensor<float>)rall.Outputs[0]).ToArray());
        var mixed = F4(new float[1, 1, 2, 2] { { { { float.NaN, 2f }, { 3f, 4f } } } });
        var rmixed = CPUExecutionProvider.MaxPool(mixed, "VALID", 0, null, new[] { 2, 2 }, null, 0, new[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, rmixed.Status);
        Assert.Equal(new float[] { 4f }, ((Tensor<float>)rmixed.Outputs[0]).ToArray());
        var single = F4(new float[1, 1, 1, 1] { { { { float.NegativeInfinity } } } });
        var rsingle = CPUExecutionProvider.MaxPool(single, "VALID", 0, null, new[] { 1, 1 }, null, 0, new[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, rsingle.Status);
        Assert.Equal(new float[] { -float.MaxValue }, ((Tensor<float>)rsingle.Outputs[0]).ToArray());
    }

    [Fact]
    public void MaxPool_UnsupportedVariants_FailExplicitly()
    {
        var x = F4(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.MaxPool(x, null, 0, null, new[] { 2, 2 }, null, 1, new[] { 2, 2 }, null).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.MaxPool(x, "BOGUS", 0, null, new[] { 2, 2 }, null, 0, new[] { 2, 2 }, null).Status);
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 11 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = x;
        var node = new Node
        {
            Name = "pool", Op = OpType.MaxPool, Inputs = new[] { "x" }, Outputs = new[] { "y", "i" },
            Attributes = new Dictionary<string, object>
            {
                ["auto_pad"] = "VALID", ["kernel_shape"] = new long[] { 2, 2 }, ["strides"] = new long[] { 2, 2 },
            },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Indices", r.Message ?? "");
    }

    [Fact]
    public void ConvGeometry_MatchesSpecFormula()
    {
        // Explicit pads: floor((input + pad - kernel) / stride) + 1 per axis.
        Assert.Equal(new[] { 5, 5 }, MathOps.GetConv2DOutputShape(new[] { 5, 5 }, 3, 3, 1, 1, 2, 2));
        Assert.Equal(new[] { 4, 7 }, MathOps.GetConv2DOutputShape(new[] { 7, 8 }, 3, 2, 2, 1, 2, 0));
        Assert.Equal(new[] { 6, 6 }, MathOps.GetConv2DOutputShape(new[] { 10, 10 }, 5, 5, 1, 1, 0, 0));
        Assert.Equal(new[] { 0, 0 }, MathOps.GetConv2DOutputShape(new[] { 2, 2 }, 3, 3, 2, 2, 0, 0));
        // Effective kernel spans include dilation gaps.
        Assert.Equal(3, MathOps.GetConv2DEffectiveFilterSize(3, 1));
        Assert.Equal(5, MathOps.GetConv2DEffectiveFilterSize(3, 2));
        Assert.Equal(4, MathOps.GetConv2DEffectiveFilterSize(2, 3));
        Assert.Equal(5, MathOps.GetConv2DEffectiveFilterSize(5, 1));
        // VALID pads nothing.
        var valid = MathOps.GetConv2DOutputInfo(MathOps.PadType.Valid, 5, 5, 1, 1, 3, 3, null);
        Assert.Equal(new[] { 3, 3 }, valid.Shape);
        Assert.Equal(new[] { 0, 0, 0, 0 }, new[] { valid.PadInfo.top, valid.PadInfo.bottom, valid.PadInfo.left, valid.PadInfo.right });
        var validStrided = MathOps.GetConv2DOutputInfo(MathOps.PadType.Valid, 7, 8, 2, 1, 3, 2, null);
        Assert.Equal(new[] { 3, 7 }, validStrided.Shape);
        // SAME_UPPER keeps ceil(input/stride) outputs with the odd cell at the end.
        var upper = MathOps.GetConv2DOutputInfo(MathOps.PadType.SameUpper, 5, 5, 1, 1, 3, 3, null);
        Assert.Equal(new[] { 5, 5 }, upper.Shape);
        Assert.Equal(new[] { 1, 1, 1, 1 }, new[] { upper.PadInfo.top, upper.PadInfo.bottom, upper.PadInfo.left, upper.PadInfo.right });
        Assert.Equal(2, upper.PadInfo.h);
        var upperStrided = MathOps.GetConv2DOutputInfo(MathOps.PadType.SameUpper, 7, 8, 2, 2, 3, 2, null);
        Assert.Equal(new[] { 4, 4 }, upperStrided.Shape);
        Assert.Equal(new[] { 1, 1, 0, 0 }, new[] { upperStrided.PadInfo.top, upperStrided.PadInfo.bottom, upperStrided.PadInfo.left, upperStrided.PadInfo.right });
        // SAME_LOWER puts the odd cell at the start.
        var lower = MathOps.GetConv2DOutputInfo(MathOps.PadType.SameLower, 6, 6, 1, 1, 4, 4, null);
        Assert.Equal(new[] { 6, 6 }, lower.Shape);
        Assert.Equal(new[] { 2, 1, 2, 1 }, new[] { lower.PadInfo.top, lower.PadInfo.bottom, lower.PadInfo.left, lower.PadInfo.right });
        // VALUE pads uniformly and sizes explicitly.
        var valued = MathOps.GetConv2DOutputInfo(MathOps.PadType.Value, 5, 5, 1, 1, 3, 3, 2);
        Assert.Equal(new[] { 7, 7 }, valued.Shape);
        Assert.Equal(new[] { 2, 2, 2, 2 }, new[] { valued.PadInfo.top, valued.PadInfo.bottom, valued.PadInfo.left, valued.PadInfo.right });
        Assert.Equal(4, valued.PadInfo.h);
        Assert.Throws<System.ArgumentNullException>(() => MathOps.GetConv2DOutputInfo(MathOps.PadType.Value, 5, 5, 1, 1, 3, 3, null));
    }

    [Fact]
    public void MaxPool_StorageOrder_RejectsNonzero()
    {
        // Only row-major is supported because the optional Indices output is not.
        var x = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });
        var r = CPUExecutionProvider.MaxPool(x, "VALID", 0, null, new int[] { 2, 2 }, null, 1, new int[] { 2, 2 }, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("storage_order", r.Message ?? "");
    }

    [Fact]
    public void MaxPool_CeilDilation_PaddedCombos()
    {
        // ORT 1.29: combined ceil, dilation, and asymmetric-pad geometries.
        var x = DenseTensor<float>.OfValues(new float[1, 1, 7, 7]);
        for (int i = 0; i < 49; i++) x.Buffer.Span[i] = i;
        var cd = CPUExecutionProvider.MaxPool(x, "NOTSET", 1, new int[] { 2, 2 }, new int[] { 3, 3 }, new int[] { 1, 1, 1, 1 }, null, new int[] { 2, 2 }, null);
        Assert.Equal(OpStatus.Success, cd.Status);
        Assert.Equal(new float[] { 24f, 26f, 26f, 38f, 40f, 40f, 38f, 40f, 40f }, ((Tensor<float>)cd.Outputs[0]).ToArray());
        var ac = CPUExecutionProvider.MaxPool(x, "NOTSET", 1, null, new int[] { 3, 3 }, new int[] { 0, 1, 0, 0 }, null, new int[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, ac.Status);
        Assert.Equal(new int[] { 1, 1, 5, 6 }, ((Tensor<float>)ac.Outputs[0]).Dimensions.ToArray());
        Assert.Equal(new float[] { 15f, 16f, 17f, 18f, 19f, 20f, 22f, 23f, 24f, 25f, 26f, 27f, 29f, 30f, 31f, 32f, 33f, 34f, 36f, 37f, 38f, 39f, 40f, 41f, 43f, 44f, 45f, 46f, 47f, 48f }, ((Tensor<float>)ac.Outputs[0]).ToArray());
        var d3 = CPUExecutionProvider.MaxPool(x, "NOTSET", 0, new int[] { 3, 3 }, new int[] { 2, 2 }, new int[] { 0, 0, 0, 0 }, null, new int[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, d3.Status);
        Assert.Equal(new float[] { 24f, 25f, 26f, 27f, 31f, 32f, 33f, 34f, 38f, 39f, 40f, 41f, 45f, 46f, 47f, 48f }, ((Tensor<float>)d3.Outputs[0]).ToArray());
    }

    [Fact]
    public void MaxPool_EmptyBatch_ReturnsEmpty()
    {
        // ORT 1.29: shape (0, 2, 3, 3).
        var x = DenseTensor<float>.OfShape(0, 2, 4, 4);
        var r = CPUExecutionProvider.MaxPool(x, "NOTSET", 0, null, new int[] { 2, 2 }, new int[] { 0, 0, 0, 0 }, null, new int[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new int[] { 0, 2, 3, 3 }, ((Tensor<float>)r.Outputs[0]).Dimensions.ToArray());
    }

    [Fact]
    public void GroupedDilated_MatchesOrt()
    {
        // ORT 1.29: group 2, dilation 2, pads 1 over arange inputs.
        var x = DenseTensor<float>.OfShape(1, 4, 6, 6);
        for (int i = 0; i < 144; i++) x.Buffer.Span[i] = (i % 7) - 3f;
        var w = DenseTensor<float>.OfShape(4, 2, 3, 3);
        for (int i = 0; i < 72; i++) w.Buffer.Span[i] = (i % 5) - 2f;
        var y = Tensor<float>.Conv2D(x, w, 2, new int[] { 1, 1, 1, 1 }, null, null, new int[] { 1, 1 }, new int[] { 2, 2 });
        Assert.Equal(new int[] { 1, 4, 4, 4 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { -12f, -4f, -10f, 3f, 9f, -11f, -7f, -14f, 17f, -1f, -11f, -4f, -4f, 18f, -3f, -3f, 10f, 0f, 18f, 8f, 3f, 0f, 1f, 19f, -18f, -1f, 0f, -5f, 6f, -12f, -4f, -11f, -8f, -9f, 8f, 4f, -1f, -17f, -17f, -1f, -2f, 4f, -17f, -7f, -7f, 18f, 8f, -7f, 10f, -1f, 5f, 1f, -3f, 17f, 9f, -4f, -14f, -3f, 17f, -4f, -2f, -19f, -9f, 1f }, y.ToArray());
    }

    [Fact]
    public void MaxPoolIntDtypes_RejectedCleanly()
    {
        // ORT 1.29 refuses int32 MaxPool at load (agreement), while int8
        // computes there (documented gap: sub-32 kernels are out of scope);
        // both fail descriptively here instead of reaching a kernel cast.
        var i32 = DenseTensor<int>.OfShape(1, 1, 2, 2);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.MaxPool(i32, null, null, null, new int[] { 1, 1 }, null, null, null, null).Status);
        var i8 = DenseTensor<sbyte>.OfShape(1, 1, 2, 2);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.MaxPool(i8, null, null, null, new int[] { 1, 1 }, null, null, null, null).Status);
    }

    [Fact]
    public void MaxPoolNullStrides_DefaultToOne()
    {
        // ORT 1.29: omitted strides default to 1 along each axis (not the
        // kernel); [1,1,3,3] with kernel 2 yields [1,1,2,2].
        var x = F4(new float[1, 1, 3, 3] { { { { 1f, 2f, 3f }, { 4f, 5f, 6f }, { 7f, 8f, 9f } } } });
        var r = CPUExecutionProvider.MaxPool(x, null, null, null, new int[] { 2, 2 }, null, null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 1, 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 5f, 6f, 8f, 9f }, y.ToArray());
    }

    [Fact]
    public void MaxPoolEmptyBatch_YieldsEmpty()
    {
        // ORT 1.29 allows batch-0 MaxPool (only N may be zero): [0,1,2,2]
        // with kernel 1 yields [0,1,2,2], no elements.
        var x = DenseTensor<float>.OfShape(0, 1, 2, 2);
        var r = CPUExecutionProvider.MaxPool(x, null, null, null, new int[] { 1, 1 }, null, null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new int[] { 0, 1, 2, 2 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
    }

    [Fact]
    public void MaxPoolZeroSpatial_Throws()
    {
        // ORT 1.29 fails the run for zero spatial extents (only N may be
        // zero); the kernel rejects non-positive output dims up front.
        var x = DenseTensor<float>.OfShape(1, 1, 0, 3);
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.MaxPool(x, null, null, null, new int[] { 1, 1 }, null, null, null, null));
    }

    [Fact]
    public void NonPositiveKernelDilationsStrides_FailsCleanly()
    {
        // ORT 1.29 refuses zero/negative kernels, dilations and strides at
        // load (shape inference); the planner must fail descriptively.
        var x = F4(new float[1, 1, 3, 3] { { { { 1f, 2f, 3f }, { 4f, 5f, 6f }, { 7f, 8f, 9f } } } });
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.MaxPool(x, null, null, null, new int[] { 0, 2 }, null, null, new int[] { 1, 1 }, null));
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.MaxPool(x, null, null, new int[] { 0, 1 }, new int[] { 2, 2 }, null, null, new int[] { 1, 1 }, null));
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.MaxPool(x, null, null, null, new int[] { 2, 2 }, null, null, new int[] { 0, 1 }, null));
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.MaxPool(x, null, null, null, new int[] { 2, 2 }, null, null, new int[] { -1, 1 }, null));
    }

    [Fact]
    public void MaxPoolInf_MatchesOrt()
    {
        // ORT 1.29 float: infinity wins its window; an all -inf window
        // stays at the -FLT_MAX seed (NOT -inf); pads contribute nothing,
        // so a padded -inf input also yields the seed.
        var k = new[] { 2, 2 };
        var s = new[] { 1, 1 };
        var plain = new (float[] v, float expected)[]
        {
            (new float[] { float.PositiveInfinity, 1f, 2f, 3f }, float.PositiveInfinity),
            (new float[] { float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity }, -float.MaxValue),
            (new float[] { float.NegativeInfinity, 5f, 6f, 7f }, 7f),
        };
        foreach (var (v, expected) in plain)
        {
            var x = F4(new float[1, 1, 2, 2] { { { { v[0], v[1] }, { v[2], v[3] } } } });
            var r = CPUExecutionProvider.MaxPool(x, null, null, null, k, null, null, s, null);
            Assert.Equal(OpStatus.Success, r.Status);
            Assert.Equal(expected, ((Tensor<float>)r.Outputs[0]).ToArray()[0]);
        }
        var padded = new (float v, float expected)[]
        {
            (float.NegativeInfinity, -float.MaxValue),
            (float.PositiveInfinity, float.PositiveInfinity),
            (5f, 5f),
        };
        foreach (var (v, expected) in padded)
        {
            var x = F4(new float[1, 1, 1, 1] { { { { v } } } });
            var r = CPUExecutionProvider.MaxPool(x, null, null, null, k, new[] { 0, 0, 1, 1 }, null, s, null);
            Assert.Equal(OpStatus.Success, r.Status);
            Assert.Equal(expected, ((Tensor<float>)r.Outputs[0]).ToArray()[0]);
        }
    }

    [Fact]
    public void NegativePads_FailsCleanly()
    {
        // ORT 1.29 refuses negative pads at load (shape inference); the
        // planners must fail descriptively instead of computing garbage.
        var x = F4(new float[1, 1, 3, 3] { { { { 1f, 2f, 3f }, { 4f, 5f, 6f }, { 7f, 8f, 9f } } } });
        var w = F4(new float[1, 1, 2, 2] { { { { 1f, 1f }, { 1f, 1f } } } });
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.Conv(x, w, null, null, null, null, null, new int[] { -1, 0, 0, 0 }, null, null));
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.MaxPool(x, null, null, null, new int[] { 2, 2 }, new int[] { -1, 0, 0, 0 }, null, new int[] { 1, 1 }, null));
    }

    [Fact]
    public void ConvBadGeometry_FailsCleanly()
    {
        // ORT 1.29 refuses degenerate Conv geometry at load (zero kernel,
        // strides and dilations) and group/weight/bias mismatches at run;
        // the planner throws descriptively instead of miscomputing.
        var x = F4(new float[1, 1, 4, 4] { { { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f }, { 9f, 10f, 11f, 12f }, { 13f, 14f, 15f, 16f } } } });
        var w = F4(new float[1, 1, 3, 3] { { { { 1f, 0f, 0f }, { 0f, 1f, 0f }, { 0f, 0f, 1f } } } });
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.Conv(x, w, null, null, null, 1, new int[] { 0, 3 }, null, null, null));
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.Conv(x, w, null, null, null, 1, null, null, new int[] { 0, 1 }, null));
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.Conv(x, w, null, null, new int[] { -1, 1 }, 1, null, null, null, null));
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.Conv(x, w, null, null, null, 3, null, null, null, null));
        var w2 = F4(new float[1, 1, 2, 2] { { { { 1f, 1f }, { 1f, 1f } } } });
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.Conv(x, w2, null, null, null, 1, new int[] { 3, 3 }, null, null, null));
        var b2 = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.Conv(x, w, b2, null, null, 1, null, null, null, null));
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 14 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = x;
        graph.Inputs["w"] = w;
        var node = new Node
        {
            Name = "n", Op = OpType.Conv, OpTypeName = OpType.Conv.ToString(), Domain = "",
            OpsetVersion = 14, IsFused = false,
            Inputs = new[] { "x", "w" }, Outputs = new[] { "z" },
            Attributes = new Dictionary<string, object> { ["strides"] = new long[] { 0L, 1L } },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void ConvZeroPadsWithAutoPad_FailsCleanly()
    {
        // ORT 1.29 load-fails a Conv carrying both pads and auto_pad, even
        // when every pad is zero (conv_attributes.h); the planner must fail
        // descriptively instead of running the auto_pad path.
        var x = F4(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });
        var w = F4(new float[1, 1, 2, 2] { { { { 1f, 0f }, { 0f, 1f } } } });
        var r = CPUExecutionProvider.Conv(x, w, null, "SAME_UPPER", null, 1, new int[] { 2, 2 }, new int[] { 0, 0, 0, 0 }, null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("pads", r.Message ?? "");
    }

    [Fact]
    public void MaxPoolAutoPadIgnoresPads_MatchesOrt()
    {
        // ORT 1.29 ignores explicit pads when auto_pad is set on MaxPool:
        // pads [1,1,0,0] + SAME_UPPER over [[1,2],[3,4]] yields [4,4,4,4],
        // identical to auto_pad alone (probed values).
        var x = F4(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });
        var r = CPUExecutionProvider.MaxPool(x, "SAME_UPPER", null, null, new int[] { 2, 2 }, new int[] { 1, 1, 0, 0 }, null, new int[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 4f, 4f, 4f, 4f }, ((Tensor<float>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void MaxPoolCeilDropsWindowsStartingInPostPadding()
    {
        // ORT 1.29 ceil rule: after the ceiling size, drop trailing windows
        // whose start reaches input extent plus begin-pad
        // ((out-1)*stride >= dim+padBegin). Found by differential fuzz
        // (maxpoolA_0): [1,2,1,7] k2 s2 pads1 ceil is (1,2,1,4) on ORT,
        // not the naive ceiling (1,2,2,5).
        var x = F4(new float[1, 2, 1, 7] { { { { 1f, 2f, 3f, 4f, 5f, 6f, 7f } }, { { 8f, 9f, 10f, 11f, 12f, 13f, 14f } } } });
        var r = CPUExecutionProvider.MaxPool(x, null, 1, null, new int[] { 2, 2 }, new int[] { 1, 1, 1, 1 }, null, new int[] { 2, 2 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new int[] { 1, 2, 1, 4 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 3f, 5f, 7f, 8f, 10f, 12f, 14f }, y.ToArray());
        // Asymmetric pads use the begin pad only: end-heavy pads still drop
        // (ORT (1,1,2,2) [11,12,15,16] for k3 s2 over 1..16).
        var q = F4(new float[1, 1, 4, 4] { { { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f }, { 9f, 10f, 11f, 12f }, { 13f, 14f, 15f, 16f } } } });
        var rq = CPUExecutionProvider.MaxPool(q, null, 1, null, new int[] { 3, 3 }, new int[] { 0, 0, 2, 2 }, null, new int[] { 2, 2 }, null);
        Assert.Equal(OpStatus.Success, rq.Status);
        Assert.Equal(new int[] { 1, 1, 2, 2 }, ((Tensor<float>)rq.Outputs![0]).Dimensions.ToArray());
        Assert.Equal(new float[] { 11f, 12f, 15f, 16f }, ((Tensor<float>)rq.Outputs![0]).ToArray());
        // VALID auto_pad trims the same way (ORT (1,1,1,1) [5] for k2 s3).
        var v = F4(new float[1, 1, 3, 3] { { { { 1f, 2f, 3f }, { 4f, 5f, 6f }, { 7f, 8f, 9f } } } });
        var rv = CPUExecutionProvider.MaxPool(v, "VALID", 1, null, new int[] { 2, 2 }, null, null, new int[] { 3, 3 }, null);
        Assert.Equal(OpStatus.Success, rv.Status);
        Assert.Equal(new int[] { 1, 1, 1, 1 }, ((Tensor<float>)rv.Outputs![0]).Dimensions.ToArray());
        Assert.Equal(new float[] { 5f }, ((Tensor<float>)rv.Outputs![0]).ToArray());
    }

    [Fact]
    public void MaxPoolIntegerDtypesFailCleanly()
    {
        // Documented scope boundary, not parity: ORT 1.29 runs int8/uint8
        // MaxPool (probed; int16/int32 stay refused on both sides), but the
        // pooling cores are float/double only here â€” the unreachable int
        // core below seeds max at 0, wrong for all-negative windows â€” so
        // integer inputs fail descriptively (verified end to end via OpDump).
        var u8 = DenseTensor<byte>.OfValues(new byte[1, 1, 2, 2] { { { { 1, 5 }, { 3, 2 } } } });
        var bad8 = CPUExecutionProvider.MaxPool(u8, null, null, null, new int[] { 2, 2 }, null, null, new int[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Failure, bad8.Status);
        Assert.Contains("UInt8", bad8.Message ?? "");
        var s8 = DenseTensor<sbyte>.OfValues(new sbyte[1, 1, 2, 2] { { { { 1, 5 }, { 3, 2 } } } });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.MaxPool(s8, null, null, null, new int[] { 2, 2 }, null, null, new int[] { 1, 1 }, null).Status);
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 14 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = u8;
        var node = new Node
        {
            Name = "n", Op = OpType.MaxPool, OpTypeName = OpType.MaxPool.ToString(), Domain = "",
            OpsetVersion = 14, IsFused = false,
            Inputs = new[] { "x" }, Outputs = new[] { "z" },
            Attributes = new Dictionary<string, object> { ["kernel_shape"] = new long[] { 2L, 2L } },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("UInt8", r.Message ?? "");
    }

    [Fact]
    public void MaxPoolDegenerate_MatchesOrt()
    {
        // ORT 1.29: a zero computed extent yields an empty output while a
        // negative one fails; the floor formula uses truncating division
        // ((2-3)/2+1 is 1, verified across explicit, padded, VALID and ceil
        // geometries). Conv with an oversized kernel fails on both sides.
        var x = F4(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });
        var e0 = CPUExecutionProvider.MaxPool(x, null, null, null, new int[] { 3, 3 }, null, null, new int[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, e0.Status);
        Assert.Equal(new int[] { 1, 1, 0, 0 }, ((Tensor<float>)e0.Outputs![0]).Dimensions.ToArray());
        var e1 = CPUExecutionProvider.MaxPool(x, null, null, null, new int[] { 3, 3 }, null, null, new int[] { 2, 2 }, null);
        Assert.Equal(OpStatus.Success, e1.Status);
        Assert.Equal(new float[] { 4f }, ((Tensor<float>)e1.Outputs![0]).ToArray());
        var ep = CPUExecutionProvider.MaxPool(x, null, null, null, new int[] { 4, 4 }, new int[] { 1, 1, 1, 1 }, null, new int[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, ep.Status);
        Assert.Equal(new float[] { 4f }, ((Tensor<float>)ep.Outputs![0]).ToArray());
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.MaxPool(x, null, null, null, new int[] { 4, 4 }, null, null, new int[] { 1, 1 }, null));
        var v0 = CPUExecutionProvider.MaxPool(x, "VALID", null, null, new int[] { 3, 3 }, null, null, new int[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, v0.Status);
        Assert.Equal(new int[] { 1, 1, 0, 0 }, ((Tensor<float>)v0.Outputs![0]).Dimensions.ToArray());
        var v1 = CPUExecutionProvider.MaxPool(x, "VALID", null, null, new int[] { 3, 3 }, null, null, new int[] { 2, 2 }, null);
        Assert.Equal(OpStatus.Success, v1.Status);
        Assert.Equal(new float[] { 4f }, ((Tensor<float>)v1.Outputs![0]).ToArray());
        var x1 = F4(new float[1, 1, 1, 1] { { { { 7f } } } });
        var v2 = CPUExecutionProvider.MaxPool(x1, "VALID", null, null, new int[] { 4, 4 }, null, null, new int[] { 2, 2 }, null);
        Assert.Equal(OpStatus.Success, v2.Status);
        Assert.Equal(new int[] { 1, 1, 0, 0 }, ((Tensor<float>)v2.Outputs![0]).Dimensions.ToArray());
        var c0 = CPUExecutionProvider.MaxPool(x, null, 1, null, new int[] { 3, 3 }, null, null, new int[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, c0.Status);
        Assert.Equal(new int[] { 1, 1, 0, 0 }, ((Tensor<float>)c0.Outputs![0]).Dimensions.ToArray());
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.MaxPool(x, null, 1, null, new int[] { 4, 4 }, null, null, new int[] { 1, 1 }, null));
        var xd = DenseTensor<double>.OfValues(new double[1, 1, 2, 2] { { { { 1.0, 2.0 }, { 3.0, 4.0 } } } });
        var d0 = CPUExecutionProvider.MaxPool(xd, null, null, null, new int[] { 3, 3 }, null, null, new int[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, d0.Status);
        Assert.Equal(new int[] { 1, 1, 0, 0 }, ((Tensor<double>)d0.Outputs![0]).Dimensions.ToArray());
        var w = F4(new float[1, 1, 3, 3] { { { { 1f, 1f, 1f }, { 1f, 1f, 1f }, { 1f, 1f, 1f } } } });
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.Conv(x, w, null, null, null, 1, new int[] { 3, 3 }, null, null, null));
        var zc = DenseTensor<float>.OfShape(1, 0, 2, 2);
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.MaxPool(zc, null, null, null, new int[] { 1, 1 }, null, null, new int[] { 1, 1 }, null));
        var zw = DenseTensor<float>.OfShape(1, 1, 2, 0);
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.MaxPool(zw, "VALID", null, null, new int[] { 1, 1 }, null, null, new int[] { 1, 1 }, null));
    }
}
