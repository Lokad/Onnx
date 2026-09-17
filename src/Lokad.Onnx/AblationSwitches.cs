namespace Lokad.Onnx;

using System;

/// <summary>
/// Measurement-only ablation switches. Each forces a legacy kernel path while
/// the shipped default stays current. Production code never sets these fields;
/// unit tests set them directly and diagnostic harnesses set them through the
/// documented environment variables before the process starts. The current
/// (default) paths keep a single perfectly-predicted branch.
/// </summary>
internal static class AblationSwitches
{
    static bool EnvIsSet(string name) =>
        string.Equals(Environment.GetEnvironmentVariable(name), "1", StringComparison.Ordinal);

    /// When true, softmax over contiguous float rows routes to the preserved
    /// legacy kernel instead of the default span kernel. Default off; set
    /// LOKAD_ONNX_SOFTMAX_LEGACY=1 before process start to enable.
    internal static bool ForceLegacySoftmax = EnvIsSet("LOKAD_ONNX_SOFTMAX_LEGACY");

    /// When true, TransposeInto skips the tiled (0,1,3,2) 4D face path so those
    /// shapes take the generic odometer loop. Default off; set
    /// LOKAD_ONNX_TRANSPOSE_LEGACY=1 before process start to enable.
    internal static bool ForceLegacyTransposeFace = EnvIsSet("LOKAD_ONNX_TRANSPOSE_LEGACY");

    /// Sticky latch: true once a forced legacy softmax has actually executed.
    internal static bool LegacySoftmaxUsed;

    /// Sticky latch: true once a skippable tiled transpose face was actually skipped.
    internal static bool LegacyTransposeFaceUsed;

    internal static void ResetLatches()
    {
        LegacySoftmaxUsed = false;
        LegacyTransposeFaceUsed = false;
    }
}
