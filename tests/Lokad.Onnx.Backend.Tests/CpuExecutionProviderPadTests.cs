using System;
using System.Linq;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

public class CpuExecutionProviderPadTests
{
    [Fact]
    public void Conv_HonorsExplicitPadsWhenAutoPadAbsent()
    {
        var x = DenseTensor<float>.OfShape(1, 1, 4, 4);
        x.Fill(1f);
        var w = DenseTensor<float>.OfShape(1, 1, 3, 3);
        w.Fill(1f);
        var r = CPU.Conv(x, w, null, null, null, 1, new int[] { 3, 3 }, new int[] { 1, 1, 1, 1 }, new int[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 1, 4, 4 }, y.Dimensions.ToArray());
        Assert.Equal(4f, y[0, 0, 0, 0], 5);
        Assert.Equal(9f, y[0, 0, 1, 1], 5);
    }

    [Fact]
    public void MaxPool_HonorsExplicitPadsWhenAutoPadAbsent()
    {
        var x = DenseTensor<float>.OfShape(1, 1, 4, 4);
        for (int i = 0; i < 16; i++) x.SetValue(i, i);
        var r = CPU.MaxPool(x, null, null, null, new int[] { 3, 3 }, new int[] { 1, 1, 1, 1 }, null, new int[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 1, 4, 4 }, y.Dimensions.ToArray());
    }
}
