using System;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

public readonly record struct TensorExecutionOptions(bool UseSimd, bool UseIntrinsics)
{
    public static TensorExecutionOptions Scalar => new TensorExecutionOptions(false, false);

    public static TensorExecutionOptions Simd => new TensorExecutionOptions(true, false);

    public static TensorExecutionOptions Intrinsics => new TensorExecutionOptions(true, true);

    public static TensorExecutionOptions Auto
    {
        get
        {
#pragma warning disable CS0618 // Auto is defined as the legacy process-wide default.
            return new TensorExecutionOptions(HardwareConfig.UseSimd, HardwareConfig.UseIntrinsics && Fma.IsSupported);
#pragma warning restore CS0618
        }
    }

    public void Validate()
    {
        if (UseIntrinsics && !Fma.IsSupported)
        {
            throw new InvalidOperationException("Tensor intrinsics were explicitly requested but x86 FMA is not supported on this machine.");
        }
    }
}
