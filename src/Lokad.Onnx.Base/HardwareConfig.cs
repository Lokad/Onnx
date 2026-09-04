namespace Lokad.Onnx;

/// <summary>Process-wide execution defaults. Prefer immutable per-execution <see cref="Lokad.Onnx.TensorExecutionOptions"/> (and <see cref="Lokad.Onnx.ExecutionOptions"/> for graphs) for concurrent or mode-pinned work; <see cref="Lokad.Onnx.TensorExecutionOptions.Auto"/> resolves from these defaults.</summary>
public static class HardwareConfig
{
    [System.Obsolete("Process-wide execution policy is obsolete. Pass TensorExecutionOptions (or ExecutionOptions for graphs) explicitly instead.")]
    public static bool UseSimd { get; set; } = true;

    [System.Obsolete("Process-wide execution policy is obsolete. Pass TensorExecutionOptions (or ExecutionOptions for graphs) explicitly instead.")]
    public static bool UseIntrinsics { get; set; } = HardwareIntrinsics.IsX86FmaSupported;

    [System.Obsolete("Process-wide execution policy is obsolete. Pass TensorExecutionOptions explicitly instead.")]
    public static void EnableIntrinsics()
    {
#pragma warning disable CS0618 // Compatibility shim delegating to the obsolete setters.
        UseSimd = true;
        UseIntrinsics = true;
#pragma warning restore CS0618
    }

    [System.Obsolete("Process-wide execution policy is obsolete. Pass TensorExecutionOptions explicitly instead.")]
    public static void EnableSimdOnly()
    {
#pragma warning disable CS0618 // Compatibility shim delegating to the obsolete setters.
        UseSimd = true;
        UseIntrinsics = false;
#pragma warning restore CS0618
    }
}

