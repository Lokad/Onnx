using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Freezes empty-tensor behavior of shape ops against ORT 1.29 reference
/// shapes: empty index sets, empty operands, and empty results move zero
/// elements and succeed with exact shapes.
/// </summary>
public class EmptyShapeTests
{
    [Fact]
    public void GatherEmptyIndices_ReturnsEmpty()
    {
        // ORT 1.29: shape (0, 2).
        var data = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f }, { 5f, 6f } });
        var indices = DenseTensor<int>.OfValues(new int[0]);
        var r = CPU.Gather(data, indices, 0, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new[] { 0, 2 }, ((Tensor<float>)r.Outputs[0]).Dimensions.ToArray());
    }

    [Fact]
    public void ConcatEmptyOperand_KeepsRest()
    {
        // ORT 1.29: shape (3, 2).
        var empty = DenseTensor<float>.OfShape(0, 2);
        var full = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f }, { 5f, 6f } });
        var r = CPU.Concat(new ITensor[] { empty, full }, 0, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new[] { 3, 2 }, ((Tensor<float>)r.Outputs[0]).Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, ((Tensor<float>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void ReshapeEmpty_PreservesZero()
    {
        // ORT 1.29: shape (0, 5).
        var r = CPU.Reshape(DenseTensor<float>.OfShape(0), DenseTensor<long>.OfValues(new long[] { 0, 5 }), false, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new[] { 0, 5 }, ((Tensor<float>)r.Outputs[0]).Dimensions.ToArray());
    }

    [Fact]
    public void SliceEmptyWindow_ReturnsEmpty()
    {
        // ORT 1.29: [].
        var x = DenseTensor<float>.OfValues(new float[] { 0f, 1f, 2f, 3f });
        var r = CPU.Slice(x, DenseTensor<long>.OfValues(new long[] { 2 }), DenseTensor<long>.OfValues(new long[] { 2 }), DenseTensor<long>.OfValues(new long[] { 0 }), DenseTensor<long>.OfValues(new long[] { 1 }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Empty(((Tensor<float>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void ExpandToZero_ReturnsEmpty()
    {
        // ORT 1.29: [].
        var r = CPU.Expand(DenseTensor<float>.OfValues(new float[] { 7f }), DenseTensor<long>.OfValues(new long[] { 0 }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Empty(((Tensor<float>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void LayerNormEmptyNormalizedAxis_ThrowsDescriptive()
    {
        // ORT 1.29 fails the run (the normalized span must be at least 1);
        // the planner rejects a zero normalized extent up front, mirroring
        // the Conv zero-spatial refusal, instead of dividing the stats by
        // a zero extent product.
        foreach (var axis in new int[] { -1, 1 })
        {
            Assert.Throws<System.ArgumentException>(() => CPU.LayerNormalization(
                DenseTensor<float>.OfShape(2, 0), DenseTensor<float>.OfShape(0),
                null, axis, null, null, 1, null, null));
            Assert.Throws<System.ArgumentException>(() => CPU.LayerNormalization(
                DenseTensor<double>.OfShape(2, 0), DenseTensor<double>.OfShape(0),
                null, axis, null, null, 1, null, null));
        }
    }

    [Fact]
    public void LayerNormEmptyOuter_FlowsEmpty()
    {
        // ORT 1.29 float and double: an empty outer axis normalizes zero
        // rows and succeeds with the exact empty shape.
        var r = CPU.LayerNormalization(
            DenseTensor<float>.OfShape(0, 3),
            DenseTensor<float>.OfValues(new float[] { 1f, 1f, 1f }),
            null, -1, null, null, 1, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs[0];
        Assert.Equal(new int[] { 0, 3 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
        var rd = CPU.LayerNormalization(
            DenseTensor<double>.OfShape(0, 3),
            DenseTensor<double>.OfValues(new double[] { 1.0, 1.0, 1.0 }),
            null, -1, null, null, 1, null, null);
        Assert.Equal(OpStatus.Success, rd.Status);
        var yd = (Tensor<double>)rd.Outputs[0];
        Assert.Equal(new int[] { 0, 3 }, yd.Dimensions.ToArray());
        Assert.Empty(yd.ToArray());
    }
    [Fact]
    public void SliceHugeBounds_ClampsToEmpty()
    {
        // ORT 1.29: [] (bounds clamp to the dimension). The saturating
        // conversion already routes huge int64 bounds into clamping.
        var x = DenseTensor<float>.OfValues(new float[] { 0f, 1f, 2f, 3f, 4f });
        var r = CPU.Slice(x, DenseTensor<long>.OfValues(new long[] { 1099511627776L }), DenseTensor<long>.OfValues(new long[] { 1099511627779L }), DenseTensor<long>.OfValues(new long[] { 0L }), DenseTensor<long>.OfValues(new long[] { 1L }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Empty(((Tensor<float>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void SliceHugeStep_YieldsFirst()
    {
        // ORT 1.29: [0] (a step beyond the dimension takes one element).
        // The saturating conversion already routes the huge step here.
        var x = DenseTensor<float>.OfValues(new float[] { 0f, 1f, 2f, 3f, 4f });
        var r = CPU.Slice(x, DenseTensor<long>.OfValues(new long[] { 0L }), DenseTensor<long>.OfValues(new long[] { 5L }), DenseTensor<long>.OfValues(new long[] { 0L }), DenseTensor<long>.OfValues(new long[] { 1099511627776L }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 0f }, ((Tensor<float>)r.Outputs[0]).ToArray());
        // A huge negative step likewise takes one element when walking
        // toward the data, and none when walking away from it.
        var rn = CPU.Slice(x, DenseTensor<long>.OfValues(new long[] { 4L }), DenseTensor<long>.OfValues(new long[] { -6L }), DenseTensor<long>.OfValues(new long[] { 0L }), DenseTensor<long>.OfValues(new long[] { -1099511627776L }), null);
        Assert.Equal(OpStatus.Success, rn.Status);
        Assert.Equal(new float[] { 4f }, ((Tensor<float>)rn.Outputs[0]).ToArray());
        var re = CPU.Slice(x, DenseTensor<long>.OfValues(new long[] { 0L }), DenseTensor<long>.OfValues(new long[] { 5L }), DenseTensor<long>.OfValues(new long[] { 0L }), DenseTensor<long>.OfValues(new long[] { -1099511627776L }), null);
        Assert.Equal(OpStatus.Success, re.Status);
        Assert.Empty(((Tensor<float>)re.Outputs[0]).ToArray());
    }


}
