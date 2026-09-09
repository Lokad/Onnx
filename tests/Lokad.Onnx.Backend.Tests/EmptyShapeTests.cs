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
}
