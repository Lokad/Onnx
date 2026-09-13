using System;
using System.Linq;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

public class CpuExecutionProviderIdentityTests
{
    [Fact]
    public void Identity_PassesFloatThroughUnchanged()
    {
        var x = new DenseTensor<float>(new float[] { 1f, -2f, 3.5f }, new[] { 3 });
        var before = x.ToArray();
        var r = CPU.Identity(x, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 3 }, y.Dimensions.ToArray());
        Assert.Equal(before, y.ToArray());
        Assert.Equal(before, x.ToArray());
    }

    [Fact]
    public void Identity_PassesIntThroughUnchanged()
    {
        var x = new DenseTensor<int>(new int[] { 7, 8 }, new[] { 1, 2 });
        var r = CPU.Identity(x, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<int>)r.Outputs![0];
        Assert.Equal(new[] { 1, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new int[] { 7, 8 }, y.ToArray());
    }

    [Fact]
    public void Identity_MissingInput_Fails()
    {
        var r = CPU.Identity(null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }
}
