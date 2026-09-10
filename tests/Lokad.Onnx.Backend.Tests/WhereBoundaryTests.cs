using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// First Where coverage anywhere: basic selection, multidirectional broadcast
/// with a higher-rank condition, scalar condition/branches, and clean rejection
/// of incompatible shapes, all against ORT 1.29 probe values.
/// </summary>
public class WhereBoundaryTests
{
    [Fact]
    public void DoubleBasic_MatchesOrt()
    {
        // ORT 1.29 double: [1.5, 20.5].
        var result = CPU.Where(
            DenseTensor<bool>.OfValues(new bool[] { true, false }),
            DenseTensor<double>.OfValues(new double[] { 1.5, 2.5 }),
            DenseTensor<double>.OfValues(new double[] { 10.5, 20.5 }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new double[] { 1.5, 20.5 }, ((Tensor<double>)result.Outputs![0]).ToArray());
    }

    [Fact]
    public void BasicSelect_MatchesOrt()
    {
        // ORT 1.29: [1, 20].
        var result = CPU.Where(
            DenseTensor<bool>.OfValues(new bool[] { true, false }),
            DenseTensor<float>.OfValues(new float[] { 1f, 2f }),
            DenseTensor<float>.OfValues(new float[] { 10f, 20f }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new float[] { 1f, 20f }, ((Tensor<float>)result.Outputs![0]).ToArray());
    }

    [Fact]
    public void HigherRankCondition_BroadcastsMultidirectionally()
    {
        // ORT 1.29: [[1, 20], [10, 2]]. Previously threw: the two-way x/y
        // broadcast blinded the kernel to a condition outranking both.
        var c = DenseTensor<bool>.OfShape(2, 2);
        new bool[] { true, false, false, true }.CopyTo(c.Buffer.Span);
        var result = CPU.Where(c,
            DenseTensor<float>.OfValues(new float[] { 1f, 2f }),
            DenseTensor<float>.OfValues(new float[] { 10f, 20f }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var output = (Tensor<float>)result.Outputs![0];
        Assert.Equal(new int[] { 2, 2 }, output.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 20f, 10f, 2f }, output.ToArray());
    }

    [Fact]
    public void ScalarCondition_Broadcasts()
    {
        var c = DenseTensor<bool>.OfShape();
        c.SetValue(0, true);
        var result = CPU.Where(c,
            DenseTensor<float>.OfValues(new float[] { 1f, 2f }),
            DenseTensor<float>.OfValues(new float[] { 10f, 20f }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new float[] { 1f, 2f }, ((Tensor<float>)result.Outputs![0]).ToArray());
    }

    [Fact]
    public void ScalarBranches_Broadcast()
    {
        var x = DenseTensor<float>.OfShape();
        x.SetValue(0, 5f);
        var y = DenseTensor<float>.OfShape();
        y.SetValue(0, 7f);
        var result = CPU.Where(
            DenseTensor<bool>.OfValues(new bool[] { true, false }),
            x, y, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new float[] { 5f, 7f }, ((Tensor<float>)result.Outputs![0]).ToArray());
    }

    [Fact]
    public void IncompatibleCondition_ThrowsDescriptively()
    {
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.Where(
            DenseTensor<bool>.OfValues(new bool[] { true, false, true }),
            DenseTensor<float>.OfValues(new float[] { 1f, 2f }),
            DenseTensor<float>.OfValues(new float[] { 10f, 20f })));
    }
}