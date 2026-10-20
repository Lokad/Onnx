namespace Lokad.Onnx;

using System;

/// <summary>Read-once comparison controls for qualified runtime optimizations.
/// Set an environment value to exactly 0 before process startup to disable it.</summary>
internal static class AblationSwitches
{
    internal static bool EnvDefaultOn(string name) =>
        !string.Equals(Environment.GetEnvironmentVariable(name), "0", StringComparison.Ordinal);

    internal static readonly bool EnablePackedAvx512Rows = EnvDefaultOn("LOKAD_ONNX_PACKED_AVX512_ROWS");
    internal static readonly bool EnableDeferredReleaseCache = EnvDefaultOn("LOKAD_ONNX_DEFERRED_RELEASE_CACHE");
    internal static readonly bool EnableReleasedBufferCache = EnvDefaultOn("LOKAD_ONNX_RELEASED_BUFFER_CACHE");
    internal static readonly bool EnableFusedTempRelease = EnvDefaultOn("LOKAD_ONNX_FUSED_TEMP_RELEASE");
    internal static readonly bool EnableSoftmaxExpPrune = EnvDefaultOn("LOKAD_ONNX_SOFTMAX_EXP_PRUNE");
    internal static readonly bool EnableSoftmaxExpInline = EnvDefaultOn("LOKAD_ONNX_SOFTMAX_EXP_INLINE");
    internal static readonly bool EnableSoftmaxNonpositive = EnvDefaultOn("LOKAD_ONNX_SOFTMAX_NONPOSITIVE");
    internal static readonly bool EnableBiasGeluInline = EnvDefaultOn("LOKAD_ONNX_BIAS_GELU_INLINE");
    internal static readonly bool EnableVectorTransposeFaces = EnvDefaultOn("LOKAD_ONNX_VECTOR_TRANSPOSE_FACES");
}
