namespace Lokad.Onnx;

/// <summary>Process-wide execution defaults. Prefer immutable per-execution <see cref="Lokad.Onnx.TensorExecutionOptions"/> (and <see cref="Lokad.Onnx.ExecutionOptions"/> for graphs) for concurrent or mode-pinned work; <see cref="Lokad.Onnx.TensorExecutionOptions.Auto"/> resolves from these defaults.</summary>
public static class HardwareConfig
{
    public static bool UseSimd { get; set; } = true;

    public static bool UseIntrinsics { get; set; } = HardwareIntrinsics.IsX86FmaSupported;

    public static void EnableIntrinsics()
    {
        UseSimd = true;
        UseIntrinsics = true;
    }

    public static void EnableSimdOnly()
    {
        UseSimd = true;
        UseIntrinsics = false;
    }
}

