namespace Lokad.Onnx;

using System;

/// <summary>
/// WARNING: process-wide mutable state. xUnit runs test classes concurrently,
/// so a set flag can reroute another thread public Softmax mid-test. Bitwise tests
/// must call kernels directly, never flag-affected publics; tolerance-based
/// agreement tests are immune. Test code setting these fields restores in finally.
/// Enabled defaults accept an explicit environment value 0 to restore comparison paths;
/// other experimental switches remain off by default and require value 1.
/// Production code never sets these fields;
/// unit tests set them directly and diagnostic harnesses set them through the
/// documented environment variables before the process starts. The current
/// (default) paths keep a single perfectly-predicted branch.
/// </summary>
internal static class AblationSwitches
{
    internal static bool EnvIsSet(string name) =>
        string.Equals(Environment.GetEnvironmentVariable(name), "1", StringComparison.Ordinal);

    internal static bool EnvDefaultOn(string name) =>
        !string.Equals(Environment.GetEnvironmentVariable(name), "0", StringComparison.Ordinal);

    /// Row sharing for prepared full-column-panel MatMul; set the variable to 0 to disable.
    internal static readonly bool EnablePackedAvx512Rows = EnvDefaultOn("LOKAD_ONNX_PACKED_AVX512_ROWS");

    /// Reuse wide tiles for existing per-call packed products with large row/reduction axes.
    internal static readonly bool EnablePackedAvx512Dynamic = EnvIsSet("LOKAD_ONNX_PACKED_AVX512_DYNAMIC");

    /// Divide a large private MatMul product in place, preserving exact division.
    internal static readonly bool EnableScaledMatMulInplace = EnvIsSet("LOKAD_ONNX_SCALED_MATMUL_INPLACE");

    /// Traverse each full packed panel across all row groups; requires packed AVX-512 rows.
    internal static readonly bool EnablePackedAvx512Panels = EnvIsSet("LOKAD_ONNX_PACKED_AVX512_PANELS");

    /// Use AVX-512 for prepared full-panel 2/3-row remainders; requires packed AVX-512 rows.
    internal static readonly bool EnablePackedAvx512Narrow = EnvIsSet("LOKAD_ONNX_PACKED_AVX512_NARROW");

    /// Memoization of failed deferred-release probes whose alias state is unchanged.
    internal static readonly bool EnableDeferredReleaseCache = EnvDefaultOn("LOKAD_ONNX_DEFERRED_RELEASE_CACHE");

    /// Retain only already-released arrays across serialized calls, within strict bounds.
    internal static readonly bool EnableReleasedBufferCache = EnvDefaultOn("LOKAD_ONNX_RELEASED_BUFFER_CACHE");

    /// Return the private MatMul result after the trailing Div composite finishes.
    internal static readonly bool EnableFusedTempRelease = EnvDefaultOn("LOKAD_ONNX_FUSED_TEMP_RELEASE");

    /// Skip exponential polynomial work that the existing underflow guard discards.
    internal static readonly bool EnableSoftmaxExpPrune = EnvDefaultOn("LOKAD_ONNX_SOFTMAX_EXP_PRUNE");

    /// Inline the pruned exponential without changing its arithmetic; requires exp pruning.
    internal static readonly bool EnableSoftmaxExpInline = EnvDefaultOn("LOKAD_ONNX_SOFTMAX_EXP_INLINE");

    /// Specialize exponentiation after subtracting a softmax row maximum.
    internal static readonly bool EnableSoftmaxNonpositive = EnvDefaultOn("LOKAD_ONNX_SOFTMAX_NONPOSITIVE");

    /// Widen long-row exponentiation while preserving eight-lane sums; requires nonpositive exp.
    internal static readonly bool EnableSoftmaxWideExp = EnvIsSet("LOKAD_ONNX_SOFTMAX_WIDE_EXP");

    /// Normalize SIMD float softmax rows with one reciprocal and multiplication.
    internal static readonly bool EnableSoftmaxReciprocal = EnvIsSet("LOKAD_ONNX_SOFTMAX_RECIPROCAL");

    // Preserve erf arithmetic while removing vector call spills.
    internal static readonly bool EnableBiasGeluInline = EnvDefaultOn("LOKAD_ONNX_BIAS_GELU_INLINE");

    /// Interleave four exact erf streams on AVX-512 hosts; requires inline BiasGelu.
    internal static readonly bool EnableBiasGeluInterleaved = EnvIsSet("LOKAD_ONNX_BIAS_GELU_INTERLEAVED");

    /// Use the existing exact-copy vector kernels for the two scalar attention faces.
    internal static readonly bool EnableVectorTransposeFaces = EnvDefaultOn("LOKAD_ONNX_VECTOR_TRANSPOSE_FACES");

    /// Drop dead dense Reshape view bindings before retrying their storage owners.
    internal static readonly bool EnableReshapeViewRelease = EnvIsSet("LOKAD_ONNX_RELEASE_RESHAPE_VIEWS");

    /// Bound float convolution patches and accumulate explicit 128-term partial products.
    internal static readonly bool EnableSegmentedConvolution = EnvIsSet("LOKAD_ONNX_SEGMENTED_CONV");

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
