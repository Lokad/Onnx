using System;

namespace Lokad.Onnx.Tensors.Tests;

internal sealed class HardwareConfigScope : IDisposable
{
    private readonly bool useSimd;
    private readonly bool useIntrinsics;

    public HardwareConfigScope(bool useSimd, bool useIntrinsics)
    {
        this.useSimd = HardwareConfig.UseSimd;
        this.useIntrinsics = HardwareConfig.UseIntrinsics;
        HardwareConfig.UseSimd = useSimd;
        HardwareConfig.UseIntrinsics = useIntrinsics;
    }

    public void Dispose()
    {
        HardwareConfig.UseSimd = useSimd;
        HardwareConfig.UseIntrinsics = useIntrinsics;
    }
}
