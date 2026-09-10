using System;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Float16 compute beyond movement stays refused (documented scope boundary,
/// like arithmetic): ORT 1.29 accepts every op below, but no half compute
/// kernels exist here. A probing attempt also showed Vector{Half}
/// throws at runtime, so any future kernel must use scalar paths (Tanh
/// precedent), never VectorizedApply.
/// </summary>
public class HalfComputeTests
{
    [Fact]
    public void HalfCosSin_RefusedCleanly()
    {
        // ORT 1.29 runs float16 Cos/Sin (probed values); refused here.
        var x = DenseTensor<Half>.OfValues(new Half[] { (Half)0.5f, (Half)1.5f });
        Assert.Equal(OpStatus.Failure, CPU.Cos(x, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Sin(x, null).Status);
    }

    [Fact]
    public void HalfEqualLess_RefusedCleanly()
    {
        // ORT 1.29 runs float16 Equal/Less including NaN rows (probed);
        // refused here with the arithmetic row.
        var x = DenseTensor<Half>.OfValues(new Half[] { (Half)float.NaN, (Half)1f });
        var y = DenseTensor<Half>.OfValues(new Half[] { (Half)float.NaN, (Half)1f });
        Assert.Equal(OpStatus.Failure, CPU.Equal(x, y, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Less(x, y, null).Status);
    }
}

