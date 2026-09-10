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
}
