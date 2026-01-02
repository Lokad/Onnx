using System;

namespace Lokad.Onnx.Tensors.Tests;

internal sealed class HardwareConfigScope : IDisposable
{
    private static readonly object sync = new();
    private readonly bool lockHeld;
    private readonly bool useSimd;
    private readonly bool useIntrinsics;

    public HardwareConfigScope(bool useSimd, bool useIntrinsics)
    {
        System.Threading.Monitor.Enter(sync);
        lockHeld = true;
        this.useSimd = HardwareConfig.UseSimd;
        this.useIntrinsics = HardwareConfig.UseIntrinsics;
        HardwareConfig.UseSimd = useSimd;
        HardwareConfig.UseIntrinsics = useIntrinsics;
    }

    public void Dispose()
    {
        HardwareConfig.UseSimd = useSimd;
        HardwareConfig.UseIntrinsics = useIntrinsics;
        if (lockHeld)
        {
            System.Threading.Monitor.Exit(sync);
        }
    }
}
