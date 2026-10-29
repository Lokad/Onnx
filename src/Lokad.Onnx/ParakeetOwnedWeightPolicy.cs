namespace Lokad.Onnx;

/// <summary>
/// The exact export names and shapes qualified for the public owned-packing
/// opt-in. Keep this model policy separate from generic consumer/alias checks;
/// admitting additional exports or shapes requires performance qualification.
/// </summary>
internal static class ParakeetOwnedWeightPolicy
{
    internal static bool IsEligiblePointwise(string? name, int channels, int outputs) =>
        channels == 1024 && outputs is 1024 or 2048 && name is not null
        && name.StartsWith("/layers.", StringComparison.Ordinal)
        && name.EndsWith("/conv/Transpose_1", StringComparison.Ordinal);

    internal static bool IsEligible(string? name, int n, int k)
    {
        if (name is null) return false;
        if (((n == 1024 && k == 4096) || (n == 4096 && k == 1024))
            && name.Contains("/feed_forward", StringComparison.Ordinal)) return true;
        return n == 1024 && k == 1024
            && (name.EndsWith("/self_attn/linear_q/MatMul", StringComparison.Ordinal)
                || name.EndsWith("/self_attn/linear_k/MatMul", StringComparison.Ordinal)
                || name.EndsWith("/self_attn/linear_v/MatMul", StringComparison.Ordinal)
                || name.EndsWith("/self_attn/linear_out/MatMul", StringComparison.Ordinal)
                || name.EndsWith("/self_attn/linear_pos/MatMul", StringComparison.Ordinal));
    }
}
